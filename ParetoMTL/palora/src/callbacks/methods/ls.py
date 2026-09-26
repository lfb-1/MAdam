import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch import Tensor

from .algo_callback import AlgoCallback
from .fim_support import FIMSupportMixin

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class LinearScalarization(FIMSupportMixin, AlgoCallback):
    def __init__(
        self,
        num_tasks,
        task_weights=None,
        fim_method: str = "none",
        fim_ema_decay: float = 0.999,
        fim_eps_num: float = 1e-12,
        fim_eps_damp: float = 1e-6,
        fim_clip_grad_norm: float = 0.0,
        fim_bias_correction: bool = False,
        fim_stochastic_pairs: bool = True,
        fim_start_fraction: float = 0.0,
        fim_end_fraction: Optional[float] = None,
        fim_ablation_name: Optional[str] = None,
        fim_freeze_after_fraction: Optional[float] = None,
        fim_reduce_denom: Optional[str] = None,
        fim_normalize_denom_mean: bool = False,
        exclude_bn_from_fim: Optional[bool] = None,
        exclude_head_from_fim: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(num_tasks)
        if task_weights is None:
            task_weights = torch.ones((num_tasks,)) / num_tasks
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == num_tasks
        self.task_weights = task_weights

        self.fim_start_fraction = float(fim_start_fraction)
        self.fim_end_fraction = 1.0 if fim_end_fraction is None else float(fim_end_fraction)
        self.fim_ablation_name = fim_ablation_name
        self.fim_freeze_after_fraction = (
            None if fim_freeze_after_fraction is None else float(fim_freeze_after_fraction)
        )
        self.fim_reduce_denom = None if fim_reduce_denom in (None, "none") else str(fim_reduce_denom)
        self.fim_normalize_denom_mean = bool(fim_normalize_denom_mean)

        self._train_step_index = 0
        self._logged_fim_window = False
        self._frozen_curvature: Optional[List[Tensor]] = None
        self._step_curvature_tensors: Optional[List[Tensor]] = None

        if not (0.0 <= self.fim_start_fraction <= self.fim_end_fraction <= 1.0):
            raise ValueError(
                "FIM activation window must satisfy 0.0 <= fim_start_fraction <= "
                "fim_end_fraction <= 1.0."
            )
        if self.fim_freeze_after_fraction is not None and not (
            0.0 <= self.fim_freeze_after_fraction <= 1.0
        ):
            raise ValueError("fim_freeze_after_fraction must lie in [0.0, 1.0].")
        if self.fim_reduce_denom not in (None, "global", "tensor"):
            raise ValueError("fim_reduce_denom must be one of None, 'global', or 'tensor'.")

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
            log_prefix="LS",
            extra_kwargs=kwargs,
        )

        if fim_method != "none" and (
            self.fim_freeze_after_fraction is not None
            or self.fim_reduce_denom is not None
            or self.fim_normalize_denom_mean
        ):
            logging.info(
                "[LS] FIM ablation enabled: freeze_after=%s, reduce_denom=%s, "
                "normalize_denom_mean=%s",
                self.fim_freeze_after_fraction,
                self.fim_reduce_denom,
                self.fim_normalize_denom_mean,
            )

    def get_weighted_loss(self, losses: Tensor, *args, **kwargs) -> Tuple[Tensor, dict]:
        losses = self.cast_losses_to_correct_type(losses)
        self.task_weights = self.task_weights.to(device=losses.device, dtype=losses.dtype)

        if self._trainer is not None:
            lam = self.task_weights.detach()
            self._trainer.ray = lam
            self._trainer.method.ray = lam

        return (self.task_weights * losses).sum(), {"weights": self.task_weights}

    def _get_lam(self, trainer: "BaseTrainer") -> Tensor:
        ray = trainer.ray
        if ray is None:
            ray = self.task_weights.to(trainer.device)
            trainer.ray = ray
            trainer.method.ray = ray
        return ray.detach().unsqueeze(0)

    def _begin_fim_step(self, trainer: "BaseTrainer") -> bool:
        total_steps = max(int(getattr(trainer, "total_train_steps", 1)), 1)
        progress = self._train_step_index / total_steps
        self._step_curvature_tensors = None
        self._fim_active_this_step = (
            self.fim_method != "none"
            and self.fim_start_fraction <= progress < self.fim_end_fraction
        )
        if self.fim_method != "none" and not self._logged_fim_window:
            logging.info(
                "[LS] FIM window configured: start=%.2f end=%.2f total_steps=%d",
                self.fim_start_fraction,
                self.fim_end_fraction,
                total_steps,
            )
            self._logged_fim_window = True
        self._train_step_index += 1
        return self._fim_active_this_step

    def _current_progress(self, trainer: "BaseTrainer") -> float:
        total_steps = max(int(getattr(trainer, "total_train_steps", 1)), 1)
        return min(max(self._train_step_index - 1, 0) / total_steps, 1.0)

    def _clone_curvature_tensors(self, curvature_tensors: List[Tensor]) -> List[Tensor]:
        return [curvature.detach().clone() for curvature in curvature_tensors]

    def _resolve_curvature_tensors(self, trainer: "BaseTrainer", precond) -> List[Tensor]:
        if self._step_curvature_tensors is not None:
            return self._step_curvature_tensors

        current = [
            curvature.to(param.device, param.dtype).detach().clone()
            for param, curvature in zip(precond.params, precond.get_curvature_per_param())
        ]
        selected = self._clone_curvature_tensors(current)

        progress = self._current_progress(trainer)
        if (
            self.fim_freeze_after_fraction is not None
            and self._frozen_curvature is None
            and progress >= self.fim_freeze_after_fraction
        ):
            self._frozen_curvature = self._clone_curvature_tensors(selected)
        if self._frozen_curvature is not None:
            selected = self._clone_curvature_tensors(self._frozen_curvature)

        self._step_curvature_tensors = selected
        return selected

    def _transform_denom_tensors(self, denom_tensors: List[Tensor]) -> List[Tensor]:
        values = [denom.detach().clone() for denom in denom_tensors]
        if self.fim_normalize_denom_mean:
            flat = torch.cat([value.reshape(-1) for value in values])
            mean = flat.mean().clamp_min(1e-12)
            values = [value / mean for value in values]

        if self.fim_reduce_denom == "global":
            flat = torch.cat([value.reshape(-1) for value in values])
            scalar = flat.mean().detach()
            values = [torch.full_like(value, scalar) for value in values]
        elif self.fim_reduce_denom == "tensor":
            values = [torch.full_like(value, value.mean().detach()) for value in values]
        return values

    def _uses_custom_denom_path(self) -> bool:
        return (
            self.fim_freeze_after_fraction is not None
            or self.fim_reduce_denom is not None
            or self.fim_normalize_denom_mean
        )

    def _build_custom_denom_tensors(
        self,
        trainer: "BaseTrainer",
        precond,
        curvature_tensors: List[Tensor],
    ) -> Optional[List[Tensor]]:
        if not self._uses_custom_denom_path():
            return None
        denom_tensors = precond.build_denominator_per_param(curvature_tensors)
        return self._transform_denom_tensors(denom_tensors)

    def __repr__(self) -> str:
        weights = self.task_weights.cpu().tolist()
        weights = [round(weight, 3) for weight in weights]
        return f"LinearScalarization(task_weights={weights}, fim_method={self.fim_method})"
