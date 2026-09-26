import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional

from .algo_callback import AlgoCallback
from .fim_support import FIMSupportMixin

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class RandomLossWeighting(FIMSupportMixin, AlgoCallback):
    def __init__(
        self,
        num_tasks: int,
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
        super().__init__(num_tasks=num_tasks)
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
            log_prefix="RLW",
            extra_kwargs=kwargs,
        )

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.num_tasks
        weight = F.softmax(torch.randn(self.num_tasks), dim=-1).to(self.device)
        if isinstance(losses, dict):
            losses = torch.stack(list(losses.values()))
        loss = torch.sum(losses * weight)

        # FIM hooks read trainer.ray after get_weighted_loss() runs.
        if self._trainer is not None:
            self._trainer.ray = weight
            self._trainer.method.ray = weight

        return loss, dict(weights=weight)
