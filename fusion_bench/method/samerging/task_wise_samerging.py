import logging
import os
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Mapping, TypeVar, Union, cast  # noqa: F401

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.compat.modelpool import ModelPool
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.models.wrappers.task_wise_fusion import (
    TaskWiseMergedModel,
    get_task_wise_weights,
)
from fusion_bench.utils.data import load_tensor_from_file
from fusion_bench.utils.type import TorchModelType

from .losses import compute_kl_loss, compute_jsd_loss, compute_ce_loss
from .utils import get_memory_usage, SAM

if TYPE_CHECKING:
    from fusion_bench.programs.fabric_fusion_program import FabricModelFusionProgram

log = logging.getLogger(__name__)


class TaskWiseSAMergingAlgorithm(
    ModelFusionAlgorithm,
    LightningFabricMixin,
    SimpleProfilerMixin,
):
    _program: "FabricModelFusionProgram"
    """The program that this algorithm is running on."""

    """
    Implements the Task-Wise SAMerging Algorithm.

    This class merges the parameters of a pretrained model with those of several fine-tuned models.
    The merging is controlled by task-wise weights, which can be initialized based on a provided configuration or loaded from a file.
    """

    def __init__(self, algorithm_config: DictConfig):
        """
        Initialize the TaskWiseSAMergingAlgorithm with the given configuration.

        Args:
            algorithm_config (DictConfig): The configuration for the algorithm.
        """
        super().__init__(algorithm_config)

    @torch.no_grad()
    def construct_task_wise_merged_model(self, modelpool: ModelPool):
        """
        Constructs a wrapped task-wise merged model from model pool.

        This method creates a new wrapped model by merging the parameters of a pretrained model with those of several fine-tuned models.
        The merging is controlled by task-wise weights, which is a `torch.Tensor` of the shape `(num_models,)`.
        The merging weights can be initialized based on a provided configuration or loaded from a file.

        Args:
            modelpool (ModelPool): An object containing the pretrained model and fine-tuned models to be merged.

        Returns:
            TaskWiseMergedModel: An instance of the merged model with task-wise weights applied.
        """
        if self.config.weights is None:
            task_wise_weight = get_task_wise_weights(
                num_models=len(modelpool.model_names),
                init_values=self.config.init_values,
            )
        else:
            if isinstance(self.config.weights, str):
                # self.config.weights is a path to a saved tensor
                task_wise_weight = load_tensor_from_file(self.config.weights)
            else:
                try:
                    task_wise_weight = torch.tensor(
                        list(self.config.weights), dtype=torch.float32
                    )
                except ValueError:
                    raise ValueError(
                        f"Unsupported weights format: {self.config.weights}"
                    )

        pretrained_model = modelpool.load_model("_pretrained_")
        finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]

        module = TaskWiseMergedModel(
            task_wise_weight=task_wise_weight,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            clamp_weights=self.config.clamp_weights,
            tie_weights=self.config.tie_weights,
            strict=self.config.strict,
        )
        print(f"{task_wise_weight.size()=}, {task_wise_weight.numel()=}")
        return module

    @rank_zero_only
    def save_merging_weights(self, file_path: str, merging_weights: torch.Tensor):
        """
        Save the merging weights to a file.

        Args:
            file_path (str): The path to save the merging weights.
            merging_weights (torch.Tensor): The merging weights to save.
        """
        if self.fabric.is_global_zero and self.config.get(
            "save_merging_weights", False
        ):
            if isinstance(file_path, str) and not file_path.startswith(("/", ".")):
                # if the file path is not absolute or relative to current working directory, save it in the log directory
                save_path = os.path.join(self.log_dir, file_path)
            else:
                save_path = file_path
            log.info(f"saving merging weights to {save_path}.")
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(merging_weights.detach().cpu(), save_path)

    def run(self, modelpool: ModelPool, **kwargs):
        """
        Run the Task-Wise SAMerging Algorithm.

        This method constructs the wrapped model and performs test-time adaptation if necessary.

        Args:
            modelpool (ModelPool): The model pool containing the pretrained and fine-tuned models.

        Returns:
            TaskWiseMergedModel: The merged model after test-time adaptation.
        """
        log.info("Fusing models using task-wise adaptive samerging.")
        self.modelpool = modelpool
        self.log_hyperparams(self.config)

        with self.profile("construct the wrapped model"):
            module = self.construct_task_wise_merged_model(modelpool)

        if self.config.weights is not None:
            # skip the test-time adaptation
            merged_model = module.merge_and_unload()
            return merged_model
        else:
            with self.profile("test-time adaptation"):
                module = self.test_time_adaptation(module)
            if self.config.get("save_merging_weights", False):
                self.save_merging_weights(
                    self.config.save_merging_weights, module.merge_weight
                )
            return module.merge_and_unload()

    def on_test_time_adaptation_start(self):
        """
        Something to do before the test-time adaptation starts. Such as setting up the task-specific heads.
        """
        pass

    @abstractmethod
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        """
        Loader of test dataset for test-time adaptation. labels are not needed.

        Args:
            task (str): The name of the task.

        Returns:
            DataLoader: The data loader for the test dataset.
        """
        pass

    @abstractmethod
    def compute_logits(self, module: nn.Module, images: Tensor, task: str) -> Tensor:
        """
        Compute the logits for the given images and task.

        Args:
            module (nn.Module): The model module.
            images (Tensor): The input images.
            task (str): The name of the task.

        Returns:
            Tensor: The computed logits.
        """
        pass

    def _sam_optimizer_step(self, module, optimizer, batches, expert_logits_dict):
        """
        Perform a single step of SAM optimization.

        Args:
            module: The model module to optimize
            optimizer: The SAM optimizer instance
            batches: Dictionary of batches for each task
            expert_logits_dict: Dictionary of pre-computed expert logits for each task

        Returns:
            float: The total loss value
        """
        optimizer.zero_grad()

        for task in self.modelpool.model_names:
            logits = self.compute_logits(
                module, batches[task].to(self.fabric.device), task
            )
            loss = compute_kl_loss(
                logits, expert_logits_dict[task].to(self.fabric.device)
            )
            self.fabric.backward(loss, retain_graph=True)

        optimizer.first_step(zero_grad=True)

        for task in self.modelpool.model_names:
            logits = self.compute_logits(
                module, batches[task].to(self.fabric.device), task
            )
            loss = compute_kl_loss(
                logits, expert_logits_dict[task].to(self.fabric.device)
            )
            self.fabric.backward(loss, retain_graph=True)

        optimizer.second_step(zero_grad=True)

        return loss

    def _precompute_expert_logits_and_batches(self, expert_models, num_steps):
        """
        Pre-compute expert logits for all tasks and steps to avoid redundant computation.

        Args:
            expert_models: Dictionary of expert models for each task
            num_steps: Number of optimization steps

        Returns:
            tuple: (all_batches, all_expert_logits) where each is a list of dictionaries
        """
        log.info("Pre-computing expert logits for all tasks and steps...")

        all_batches = []
        all_expert_logits = []

        for step_idx in tqdm(
            range(num_steps), desc="Pre-computing expert logits", dynamic_ncols=True
        ):
            step_batches = {}
            step_expert_logits = {}

            for task in self.modelpool.model_names:
                batch = next(self.get_shuffled_test_loader_iter(task))
                step_batches[task] = batch[0].clone().detach().cpu()

                with torch.no_grad():
                    # Move expert model to device just-in-time, then offload back to CPU
                    expert_model_on_device = expert_models[task].to(self.fabric.device)
                    expert_logits = self.compute_logits(
                        expert_model_on_device, batch[0].to(self.fabric.device), task
                    )
                    step_expert_logits[task] = expert_logits.detach().cpu()
                try:
                    expert_models[task].to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            all_batches.append(step_batches)
            all_expert_logits.append(step_expert_logits)

        return all_batches, all_expert_logits

    def test_time_adaptation(self, module: TaskWiseMergedModel):
        """
        Perform test-time adaptation on the merged model.

        This method adapts the merging weights during test-time to improve performance.

        Args:
            module (TaskWiseMergedModel): The merged model.

        Returns:
            TaskWiseMergedModel: The adapted merged model.
        """
        self.on_test_time_adaptation_start()

        # configure optimizer
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam([module.merge_weight], lr=self.config.lr)
            print(f"{optimizer=}")
            module, optimizer = self.fabric.setup(module, optimizer)
        elif self.config.optimizer == "sam":
            base_optimizer = torch.optim.SGD
            lambda_params = [module.merge_weight]
            other_params = [
                p for p in module.task_vectors.parameters() if p.requires_grad
            ]

            optim_groups = [
                dict(params=lambda_params, lr=self.config.lr),
                dict(params=other_params, lr=0.0),
            ]
            optimizer = SAM(
                optim_groups,
                base_optimizer,
                lr=self.config.lr,
                rho=0.07,
                adaptive=True,
                momentum=0.99,
                weight_decay=5e-4,
            )
            print(f"{optimizer=}")
            module, optimizer = self.fabric.setup(module, optimizer)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        module.train()
        module.merge_weights()

        expert_models = {}
        for task in self.modelpool.model_names:
            # Load on CPU first to reduce peak GPU memory; we will move per-use
            expert_models[task] = self.modelpool.load_model(task)

        # Pre-compute expert logits and batches for all steps
        num_steps = self.config.max_steps if not self.is_debug_mode else 1
        with self.profile("pre-computing expert logits"):
            all_batches, all_expert_logits = self._precompute_expert_logits_and_batches(
                expert_models, num_steps
            )

        del expert_models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # sanity check
        initial_task_vectors = [
            p.data.clone() for p in module.task_vectors.parameters()
        ]

        for step_idx in (
            pbar := tqdm(
                range(num_steps),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "SAMerging Test-time adaptation",
                dynamic_ncols=True,
            )
        ):
            # Use pre-computed batches and expert logits for this step
            batches = all_batches[step_idx % len(all_batches)]
            expert_logits_dict = all_expert_logits[step_idx % len(all_expert_logits)]

            with self.profile("optimizer step"):
                if self.config.optimizer == "sam":
                    loss = self._sam_optimizer_step(
                        module, optimizer, batches, expert_logits_dict
                    )
                else:
                    for task in self.modelpool.model_names:
                        with self.profile("forward pass"):
                            logits = self.compute_logits(
                                module, batches[task].to(self.fabric.device), task
                            )
                            loss = compute_kl_loss(
                                logits, expert_logits_dict[task].to(self.fabric.device)
                            )
                        with self.profile("backward pass"):
                            self.fabric.backward(loss, retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()

            with self.profile("merging weights"):
                module.merge_weights()

            metrics = {
                "train/loss": loss.item(),
                "train/weight_max": module.merge_weight.max().item(),
                "train/weight_min": module.merge_weight.min().item(),
                "train/weight_mean": module.merge_weight.mean().item(),
            }

            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)

        log.info(
            get_memory_usage(f"after task-wise samerging, the memory usage of GPU is:")
        )
        self.print_profile_summary()
        return module
