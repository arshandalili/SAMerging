"""
Example Usage:

```bash
fusion_bench \
    method=samerging \
        method.name=clip_task_wise_samerging \
        method.save_merging_weights=merging_weights.pt \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8 \
    fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
    fabric.loggers.name=clip_task_wise_samerging_samerging
```
"""

import functools
import logging

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.utils.data import InfiniteDataLoader

from .task_wise_samerging import TaskWiseSAMergingAlgorithm


class CLIPTaskWiseSAMergingAlgorithm(
    CLIPClassificationMixin,
    TaskWiseSAMergingAlgorithm,
):
    def on_test_time_adaptation_start(self):
        """
        Here we load the CLIP processor and construct the zero-shot classification head for each task.
        """
        self.setup_zero_shot_classification_head()

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str):
        return super().get_shuffled_test_loader_iter(
            task,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
