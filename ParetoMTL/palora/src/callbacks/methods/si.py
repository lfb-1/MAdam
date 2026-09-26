from typing import List, Tuple, Union
from .algo_callback import AlgoCallback
import torch
from torch import Tensor


class ScaleInvariantLinearScalarization(AlgoCallback):
    def __init__(self, num_tasks, task_weights=None):
        super().__init__(num_tasks)
        if task_weights is None:
            task_weights = torch.ones((num_tasks,)) / num_tasks
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == num_tasks
        self.task_weights = task_weights

    def get_weighted_loss(self, losses: Tensor, *args, **kwargs) -> Tuple[Tensor, dict]:
        losses = self.cast_losses_to_correct_type(losses)
        self.task_weights = self.task_weights.to(losses.device)
        return (self.task_weights * torch.log(losses)).sum(), {}

    def __repr__(self) -> str:
        weights = self.task_weights.cpu().tolist()
        weights = [round(w, 3) for w in weights]
        return f"ScaleInvariantLinearScalarization(task_weights={weights})"
