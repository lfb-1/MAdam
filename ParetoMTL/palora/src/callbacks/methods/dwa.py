from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from .algo_callback import AlgoCallback
from .fim_support import FIMSupportMixin
import numpy as np


class DynamicWeightAverage(FIMSupportMixin, AlgoCallback):
    """Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    Modification of: https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_split.py#L242
    """

    def __init__(
        self,
        num_tasks,
        iteration_window: int = 25,
        temp=2.0,
        fim_method: str = "none",
        fim_ema_decay: float = 0.999,
        fim_eps_num: float = 1e-12,
        fim_eps_damp: float = 1e-6,
        fim_clip_grad_norm: float = 0.0,
        fim_bias_correction: bool = False,
        fim_stochastic_pairs: bool = True,
        exclude_bn_from_fim: Optional[bool] = None,
        exclude_head_from_fim: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(num_tasks)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, num_tasks), dtype=np.float32)
        self.weights = np.ones(num_tasks, dtype=np.float32)
        self.init_fim_support(
            fim_method=fim_method,
            fim_ema_decay=fim_ema_decay,
            fim_eps_num=fim_eps_num,
            fim_eps_damp=fim_eps_damp,
            fim_clip_grad_norm=fim_clip_grad_norm,
            fim_bias_correction=fim_bias_correction,
            fim_stochastic_pairs=fim_stochastic_pairs,
            exclude_bn_from_fim=exclude_bn_from_fim,
            exclude_head_from_fim=exclude_head_from_fim,
            log_prefix="DWA",
            extra_kwargs=kwargs,
        )

    def get_weighted_loss(self, losses: Tensor, **kwargs) -> Tuple[Tensor, dict]:
        if isinstance(losses, dict):
            losses = tuple(losses.values())
        if not isinstance(losses, Tensor):
            losses = torch.stack(losses)
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[: self.iteration_window, :].mean(0)
            self.weights = (self.num_tasks * np.exp(ws / self.temp)) / (np.exp(ws / self.temp)).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(losses.device)
        loss = (task_weights * losses).mean()

        if self._trainer is not None:
            self._trainer.ray = task_weights.detach()
            self._trainer.method.ray = self._trainer.ray

        self.running_iterations += 1

        return loss, dict(weights=task_weights)

    def __repr__(self) -> str:
        return (
            f"DynamicWeightAverage(iteration_window={self.iteration_window}, "
            f"temp={self.temp}, fim_method={self.fim_method})"
        )
