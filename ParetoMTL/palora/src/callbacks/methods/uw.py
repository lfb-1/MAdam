from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from .algo_callback import AlgoCallback
from .fim_support import FIMSupportMixin


class UncertaintyWeighting(FIMSupportMixin, AlgoCallback):
    """Implementation of `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics`
    Source: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
    """

    def __init__(
        self,
        num_tasks,
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
        self.logsigma = Parameter(torch.zeros((num_tasks,), requires_grad=True))
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
            log_prefix="UW",
            extra_kwargs=kwargs,
        )

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        loss = sum([0.5 * (torch.exp(-logs) * loss + logs) for loss, logs in zip(losses, self.logsigma)])

        if self._trainer is not None:
            weights = torch.exp(-self.logsigma.detach())
            weights = weights / weights.sum().clamp_min(1e-12)
            self._trainer.ray = weights.to(losses.device)
            self._trainer.method.ray = self._trainer.ray

        return loss, dict(weights=torch.exp(-self.logsigma))  # NOTE: not exactly task weights

    def parameters(self) -> List[torch.Tensor]:
        return [self.logsigma]
