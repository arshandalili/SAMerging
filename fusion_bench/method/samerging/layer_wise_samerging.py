"""
fusion_bench \
    method=samerging \
        method.name=clip_layer_wise_samerging \
        method.save_merging_weights=merging_weights.pt \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
"""

import logging
import contextlib
import os
from abc import abstractmethod
from re import T
from typing import TYPE_CHECKING, Any, List, Mapping, TypeVar, Union, cast  # noqa: F401

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel


# from torch.autograd.functional import hessian

# from torchvision.transforms import RandAugment, ToPILImage, ToTensor

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.compat.modelpool import ModelPool
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.models.wrappers.layer_wise_fusion import (
    LayerWiseMergedModel,
    get_layer_wise_weights,
)
from fusion_bench.utils.data import load_tensor_from_file
from fusion_bench.utils.type import TorchModelType

from .losses import compute_kl_loss
from .utils import get_memory_usage, SAM
from .fsam import FisherSAM

if TYPE_CHECKING:
    from fusion_bench.programs.fabric_fusion_program import FabricModelFusionProgram

log = logging.getLogger(__name__)


class LayerWiseSAMergingAlgorithm(
    ModelFusionAlgorithm,
    LightningFabricMixin,
    SimpleProfilerMixin,
):
    _program: "FabricModelFusionProgram"
    """The program that this algorithm is running on."""

    """
    Implements the Layer-Wise SAMerging Algorithm.

    This class merges the layers of a pretrained model with those of several fine-tuned models.
    The merging is controlled by layer-wise weights, which can be initialized based on a provided configuration or loaded from a file.
    """

    def __init__(self, algorithm_config: DictConfig):
        """
        Initialize the LayerWiseSAMergingAlgorithm with the given configuration.

        Args:
            algorithm_config (DictConfig): The configuration for the algorithm.
        """
        super().__init__(algorithm_config)

    @torch.no_grad()
    def construct_layer_wise_merged_model(self, modelpool: "ModelPool"):
        """
        Constructs a wrapped layer-wise merged model from model pool.

        This method creates a new wrapped model by merging the layers of a pretrained model with those of several fine-tuned models.
        The merging is controlled by layer-wise weights, which is a `torch.Tensor` of the shape `(num_models, num_layers)`.
        The merging weights can be initialized based on a provided configuration or loaded from a file.

        Args:
            modelpool (ModelPool): An object containing the pretrained model and fine-tuned models to be merged.

        Returns:
            LayerWiseMergedModel: An instance of the merged model with layer-wise weights applied.
        """
        pretrained_model = modelpool.load_model("_pretrained_")
        finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]

        # initialize layer-wise weights using the provided configuration `init_values` or load from file if `weights` is provided
        if self.config.weights is None:
            layer_wise_weight = get_layer_wise_weights(
                num_models=len(modelpool.model_names),
                num_layers=len(
                    tuple(
                        filter(lambda p: p.requires_grad, pretrained_model.parameters())
                    )
                ),
                init_values=self.config.init_values,
            )
        else:
            if isinstance(self.config.weights, str):
                # self.config.weights is a path to a saved tensor
                layer_wise_weight = load_tensor_from_file(self.config.weights)
            else:
                raise ValueError(f"Unsupported weights format: {self.config.weights}")

        module = LayerWiseMergedModel(
            layer_wise_weight=layer_wise_weight,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            clamp_weights=self.config.clamp_weights,
            tie_weights=self.config.tie_weights,
            strict=self.config.strict,
            # sparsity_ratio=0.3,
        )
        print(f"{layer_wise_weight.size()=}, {layer_wise_weight.numel()=}")
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
        Run the Layer-Wise SAMerging Algorithm.

        This method constructs the wrapped model and performs test-time adaptation if necessary.

        Args:
            modelpool (ModelPool): The model pool containing the pretrained and fine-tuned models.

        Returns:
            LayerWiseMergedModel: The merged model after test-time adaptation.
        """
        log.info("Fusing models using layer-wise adaptive samerging.")
        self.modelpool = modelpool
        self.log_hyperparams(self.config)

        with self.profile("construct the wrapped model"):
            module = self.construct_layer_wise_merged_model(modelpool)

        if self.config.weights is not None:
            # skip the test-time adaptation
            merged_model = module.merge_and_unload()
            distance = 0
            pretrained_model = modelpool.load_model("_pretrained_").to(
                module.merge_weight.device
            )
            task_vectors = []
            for name, param in merged_model.state_dict().items():
                task_vectors.append(
                    param.data.flatten()
                    - pretrained_model.state_dict()[name].data.flatten()
                )
            task_vectors = torch.cat(task_vectors)
            distance = torch.norm(task_vectors, p=2)
            print(
                f"distance between the merged model and the pretrained model: {distance}"
            )
            for task in self.modelpool.model_names:
                expert_model = self.modelpool.load_model(task).to(
                    pretrained_model.device
                )
                expert_task_vectors = []
                for name, param in expert_model.state_dict().items():
                    expert_task_vectors.append(
                        param.data.flatten()
                        - pretrained_model.state_dict()[name].data.flatten()
                    )
                expert_task_vectors = torch.cat(expert_task_vectors)
                distance = torch.norm(expert_task_vectors, p=2)
                print(
                    f"distance between the expert model {task} and the merged model: {distance}"
                )
            return merged_model
        else:
            with self.profile("test-time adaptation"):
                module = self.test_time_adaptation(module)
            if self.config.get("save_merging_weights", False):
                self.save_merging_weights(
                    self.config.save_merging_weights, module.merge_weight
                )
            merged_model = module.merge_and_unload()
            # print the distance between the merged model and the pretrained model
            return merged_model

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
    def compute_logits(self, module, images: Tensor, task: str) -> Tensor:
        """
        Compute the logits for the given images and task.

        Args:
            module: The model module.
            images (Tensor): The input images.
            task (str): The name of the task.

        Returns:
            Tensor: The computed logits.
        """
        pass

    def _sam_optimizer_step(
        self, module, optimizer, expert_models, step_idx: int, global_step: int
    ):
        """
        Perform a single step of SAM optimization.

        Args:
            module: The model module to optimize
            optimizer: The SAM optimizer instance
            expert_models: Dictionary of expert models for each task
            step_idx: The current step index for logging purposes.

        Returns:
            float: The total loss value
        """

        optimizer.zero_grad()

        batches = {}
        expert_logits_dict = {}

        for task in self.modelpool.model_names:
            try:
                batch = next(self.get_shuffled_test_loader_iter(task))
                batches[task] = batch[0].to(self.fabric.device)

                with torch.no_grad():
                    expert_logits = self.compute_logits(
                        expert_models[task], batches[task], task
                    )
                    expert_logits_dict[task] = expert_logits.detach()
            except Exception as e:
                log.error(f"Error getting batch for task {task}: {e}")
                continue

        for task in self.modelpool.model_names:
            if task in batches:
                logits = self.compute_logits(module, batches[task], task)
                loss = compute_kl_loss(logits, expert_logits_dict[task], temperature=2)
                self.fabric.backward(loss, retain_graph=True)

        optimizer.first_step(zero_grad=True)

        total_loss = 0
        for task in self.modelpool.model_names:
            if task in batches:
                logits = self.compute_logits(module, batches[task], task)
                loss = compute_kl_loss(logits, expert_logits_dict[task], temperature=2)
                total_loss += loss
                self.fabric.backward(loss, retain_graph=True)

        optimizer.second_step(zero_grad=True)

        return total_loss

    def _sam_optimizer_step_accum(
        self,
        module,
        optimizer,
        expert_models,
        accum_steps: int,
        step_idx: int,
        global_step: int,
    ):
        """
        Perform a SAM optimization step with gradient accumulation over multiple micro-steps.

        This collects multiple micro-batches (per task), accumulates their gradients for the
        first SAM step, applies the perturbation, then replays the same micro-batches for the
        second SAM step, finally performing the update.

        Args:
            module: The model module to optimize
            optimizer: The SAM optimizer instance
            expert_models: Dictionary of expert models for each task
            accum_steps: Number of micro-steps to accumulate
        """

        optimizer.zero_grad()

        # Collect micro-batches on CPU to limit GPU memory usage
        collected_batches = []  # List[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]

        for accum_idx in range(accum_steps):
            batches = {}
            expert_logits_dict = {}

            for task in self.modelpool.model_names:
                try:
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    images_cpu = batch[
                        0
                    ]  # keep on CPU; move to device only when computing
                    batches[task] = images_cpu

                    with torch.no_grad():
                        expert_model = expert_models[task].to(self.fabric.device)
                        expert_logits = self.compute_logits(
                            expert_model, images_cpu.to(self.fabric.device), task
                        )
                        expert_logits_dict[task] = expert_logits.detach().cpu()
                except Exception as e:
                    log.error(f"Error getting batch for task {task}: {e}")
                    continue

            collected_batches.append((batches, expert_logits_dict))

            # First SAM pass: accumulate gradients across micro-steps
            for task in self.modelpool.model_names:
                if task in batches:
                    logits = self.compute_logits(
                        module, batches[task].to(self.fabric.device), task
                    )
                    loss = compute_kl_loss(
                        logits,
                        expert_logits_dict[task].to(self.fabric.device),
                        temperature=2,
                    )
                    # Average across accumulation steps to preserve gradient scale
                    self.fabric.backward(loss / accum_steps, retain_graph=True)

        # Apply the SAM perturbation after accumulating the first-pass gradients
        optimizer.first_step(zero_grad=True)

        total_loss = 0.0
        # Second SAM pass over the same collected micro-batches
        for batches, expert_logits_dict in collected_batches:
            for task in self.modelpool.model_names:
                if task in batches:
                    logits = self.compute_logits(
                        module, batches[task].to(self.fabric.device), task
                    )
                    loss = compute_kl_loss(
                        logits,
                        expert_logits_dict[task].to(self.fabric.device),
                        temperature=2,
                    )
                    total_loss += loss.detach()
                    # Average across accumulation steps to preserve gradient scale
                    self.fabric.backward(loss / accum_steps, retain_graph=True)

        optimizer.second_step(zero_grad=True)

        # Return mean loss across accumulated micro-steps for logging
        if accum_steps > 0:
            return total_loss / accum_steps
        return total_loss

    def test_time_adaptation(self, module: "LayerWiseMergedModel[TorchModelType]"):
        """
        Perform test-time adaptation on the merged model.

        This method adapts the merging weights during test-time to improve performance.

        Args:
            module (LayerWiseMergedModel): The merged model.

        Returns:
            LayerWiseMergedModel: The adapted merged model.
        """
        self.on_test_time_adaptation_start()

        # configure optimizer (SAM only)
        if self.config.optimizer == "sam":
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
                rho=self.config.rho,
                adaptive=True,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
            print(f"{optimizer=}")
            module, optimizer = self.fabric.setup(module, optimizer)
        else:
            raise ValueError(
                f"Unsupported optimizer: {self.config.optimizer}. Only 'sam' is supported."
            )

        module.train()
        module.merge_weights()

        expert_models = {}
        for task in self.modelpool.model_names:
            expert_models[task] = self.modelpool.load_model(task).to(self.fabric.device)

        num_steps = self.config.max_steps if not self.is_debug_mode else 1
        num_epochs = self.config.epochs
        global_step = 0

        grad_accum_steps = max(int(getattr(self.config, "grad_accum_steps", 1)), 1)

        for epoch_idx in range(num_epochs):
            for step_idx in (
                pbar := tqdm(
                    range(num_steps),
                    ("[DEBUG MODE] " if self.is_debug_mode else "")
                    + f"SAMerging Test-time adaptation (epoch {epoch_idx + 1}/{num_epochs})",
                    dynamic_ncols=True,
                )
            ):
                with self.profile("optimizer step"):
                    if self.config.optimizer == "sam":
                        if grad_accum_steps == 1:
                            loss = self._sam_optimizer_step(
                                module, optimizer, expert_models, step_idx, global_step
                            )
                        else:
                            loss = self._sam_optimizer_step_accum(
                                module,
                                optimizer,
                                expert_models,
                                grad_accum_steps,
                                step_idx,
                                global_step,
                            )
                    else:
                        raise ValueError(
                            f"Unsupported optimizer: {self.config.optimizer}. Only 'sam' is supported."
                        )

                        # no extra intermediates to clear for SAM

                with self.profile("merging weights"):
                    module.merge_weights()

                metrics = {
                    "train/loss": loss.item(),
                    "train/weight_max": module.merge_weight.max().item(),
                    "train/weight_min": module.merge_weight.min().item(),
                    "train/weight_mean": module.merge_weight.mean().item(),
                    "train/epoch": epoch_idx,
                }

                self.fabric.log_dict(metrics, step=global_step)
                pbar.set_postfix(metrics)
                global_step += 1

        log.info(get_memory_usage(f"after samerging, the memory usage of GPU is:"))
        self.print_profile_summary()
        return module
