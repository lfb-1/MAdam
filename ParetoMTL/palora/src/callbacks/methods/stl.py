from typing import TYPE_CHECKING, List, Tuple, Union

import torch
from torch import Tensor

from .algo_callback import AlgoCallback

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class STL(AlgoCallback):
    """Single task learning"""

    def __init__(self, num_tasks, main_task):
        super().__init__(num_tasks)
        self.main_task = main_task
        self.weights = torch.zeros(num_tasks)
        self.weights[main_task] = 1.0

    def on_before_fit(self, trainer: "BaseTrainer", *args, **kwargs):
        if len(trainer.benchmark.task_names) == 1:
            self.task_name = trainer.benchmark.task_names[0]
        else:
            self.task_name = trainer.benchmark.task_names[self.main_task]
        print("Main task:", self.task_name)

    def get_weighted_loss(self, losses: Tensor, *args, **kwargs) -> Tuple[Tensor, dict]:
        return losses[self.task_name], {}

    def __repr__(self) -> str:
        return f"STL(main_task={self.main_task})"
