from typing import Dict

import torch
import torch.nn as nn


class MultiForwardRegularizationLoss(nn.Module):
    def __init__(self, num_tasks: int, reg_coefficient: float):
        super().__init__()
        self.num_tasks = num_tasks
        self.reg_coefficient = reg_coefficient

    def forward(self, window_losses: Dict[str, Dict[float, float]]) -> torch.Tensor:
        assert self.num_tasks > 1, "Regularization loss is only computed when num_tasks > 1"

        losses = {k: dict(sorted(v.items(), reverse=True)) for k, v in window_losses.items()}

        mask = []
        reg_term = 0
        for k, v in losses.items():
            task_losses = list(v.values())
            diff = torch.stack(task_losses).diff()
            _reg_term = torch.nn.functional.relu(diff).exp().sum().div(self.num_tasks - 1).log()
            reg_term += _reg_term
            mask.append(torch.nn.functional.relu(diff) > 0)

        return reg_term * self.reg_coefficient
