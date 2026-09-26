"""ExcessMTL task weighting adapted to the PaLoRA callback API.

Algorithm source: https://github.com/yifei-he/ExcessMTL
The upstream LibMTL implementation is MIT licensed; see
``THIRD_PARTY_NOTICES.md``.
"""

from typing import List, Optional

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from .algo_callback import AlgoCallback
from .adaptive_fim_support import AdaptiveFIMSupportMixin


class ExcessMTL(AdaptiveFIMSupportMixin, AlgoCallback):
    """Balance tasks using their estimated excess risks.

    Preserves the official all-history diagonal curvature estimate, first-batch
    reference, exponentiated update, weights summing to num_tasks, and weighted
    shared/head gradients.
    """

    def __init__(
        self,
        num_tasks: int,
        robust_step_size: float = 1e-2,
        estimator_eps: float = 1e-7,
        fim_method: str = "none",
        fim_ema_decay: float = 0.999,
        fim_eps_num: float = 1e-12,
        fim_eps_damp: float = 1e-6,
        fim_clip_grad_norm: float = 0.0,
        fim_bias_correction: bool = False,
        fim_stochastic_pairs: bool = True,
        fim_lambda_policy: str = "aggregation_raw",
        fim_preserve_shared_grad_norm: bool = False,
        fim_solver_geometry: str = "raw",
        exclude_bn_from_fim: Optional[bool] = None,
        exclude_head_from_fim: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(num_tasks)
        if not torch.isfinite(torch.tensor(float(robust_step_size))):
            raise ValueError("ExcessMTL robust_step_size must be finite.")
        if estimator_eps <= 0:
            raise ValueError("ExcessMTL estimator_eps must be positive.")
        if fim_solver_geometry != "raw":
            raise ValueError(
                "ExcessMTL supports only fim_solver_geometry='raw'; its "
                "official excess-risk estimator is not a Gram-matrix solver."
            )
        self.robust_step_size = float(robust_step_size)
        self.estimator_eps = float(estimator_eps)
        self.loss_weight: Optional[Tensor] = None
        self.grad_sum: Optional[Tensor] = None
        self.initial_excess: Optional[Tensor] = None
        self.last_excess: Optional[Tensor] = None
        self.last_relative_excess: Optional[Tensor] = None
        self.estimator_updates = 0
        self.weight_updates = 0
        self.init_fim_support(
            fim_method=fim_method, fim_ema_decay=fim_ema_decay,
            fim_eps_num=fim_eps_num, fim_eps_damp=fim_eps_damp,
            fim_clip_grad_norm=fim_clip_grad_norm, fim_bias_correction=fim_bias_correction,
            fim_stochastic_pairs=fim_stochastic_pairs, fim_lambda_policy=fim_lambda_policy,
            fim_preserve_shared_grad_norm=fim_preserve_shared_grad_norm,
            fim_solver_geometry=fim_solver_geometry,
            exclude_bn_from_fim=exclude_bn_from_fim, exclude_head_from_fim=exclude_head_from_fim,
            log_prefix="ExcessMTL", extra_kwargs=kwargs,
        )

    def __repr__(self) -> str:
        return f"ExcessMTL(robust_step_size={self.robust_step_size}, fim_method={self.fim_method})"

    @staticmethod
    def _flatten_task_gradients(losses: Tensor, shared_parameters: List[Parameter]) -> Tensor:
        rows = []
        for loss in losses:
            grad_parts = torch.autograd.grad(loss, shared_parameters, retain_graph=True, allow_unused=True)
            rows.append(torch.cat([
                grad.detach().reshape(-1) if grad is not None else torch.zeros_like(param).reshape(-1)
                for param, grad in zip(shared_parameters, grad_parts)
            ]))
        return torch.stack(rows)

    def _update_weights(self, gradients: Tensor) -> Tensor:
        if self.loss_weight is None:
            self.loss_weight = torch.ones(self.num_tasks, device=gradients.device, dtype=gradients.dtype)
        else:
            self.loss_weight = self.loss_weight.to(device=gradients.device, dtype=gradients.dtype)
        if self.grad_sum is None:
            self.grad_sum = torch.zeros_like(gradients)
        self.grad_sum = self.grad_sum.to(device=gradients.device, dtype=gradients.dtype)
        self.grad_sum.add_(gradients.square())
        denom = torch.sqrt(self.grad_sum + self.estimator_eps)
        excess = (gradients.square() / denom).sum(dim=1)
        if not torch.isfinite(excess).all():
            raise RuntimeError("ExcessMTL produced a non-finite excess-risk estimate.")
        self.estimator_updates += 1
        self.last_excess = excess.detach().clone()
        if self.initial_excess is None:
            if torch.any(excess <= 0):
                raise RuntimeError("ExcessMTL requires a positive first-batch excess-risk estimate for every task.")
            self.initial_excess = excess.detach().clone()
            self.last_relative_excess = None
        else:
            relative_excess = excess / self.initial_excess
            updated = self.loss_weight * torch.exp(relative_excess * self.robust_step_size)
            updated = updated / updated.sum() * self.num_tasks
            if not torch.isfinite(updated).all() or torch.any(updated < 0):
                raise RuntimeError("ExcessMTL produced invalid task weights.")
            self.loss_weight = updated.detach().clone()
            self.last_relative_excess = relative_excess.detach().clone()
            self.weight_updates += 1
        return self.loss_weight

    def get_weighted_loss(self, losses: Tensor, shared_parameters: List[Parameter], **kwargs):
        losses = self.cast_losses_to_correct_type(losses)
        shared_parameters = [shared_parameters] if isinstance(shared_parameters, Tensor) else list(shared_parameters)
        gradients = self._flatten_task_gradients(losses, shared_parameters)
        weights = self._update_weights(gradients)
        self.set_fim_preference_from_aggregation(weights)
        outputs = {
            "weights": weights.detach().clone(),
            "excess_estimate": self.last_excess.detach().clone(),
            "relative_excess": None if self.last_relative_excess is None else self.last_relative_excess.detach().clone(),
        }
        return torch.sum(losses * weights), outputs

    @staticmethod
    def _tensor_list(value: Optional[Tensor]):
        if value is None:
            return None
        return [float(item) for item in value.detach().cpu().reshape(-1)]

    def get_diagnostics(self) -> dict:
        weights, grad_sum = self.loss_weight, self.grad_sum
        return {
            "robust_step_size": self.robust_step_size,
            "estimator_eps": self.estimator_eps,
            "estimator_updates": self.estimator_updates,
            "weight_updates": self.weight_updates,
            "loss_weight": self._tensor_list(weights),
            "loss_weight_sum": None if weights is None else float(weights.sum().detach().cpu()),
            "loss_weight_min": None if weights is None else float(weights.min().detach().cpu()),
            "loss_weight_max": None if weights is None else float(weights.max().detach().cpu()),
            "initial_excess_estimate": self._tensor_list(self.initial_excess),
            "last_excess_estimate": self._tensor_list(self.last_excess),
            "last_relative_excess": self._tensor_list(self.last_relative_excess),
            "grad_sum_min": None if grad_sum is None else float(grad_sum.min().detach().cpu()),
            "grad_sum_mean": None if grad_sum is None else float(grad_sum.mean().detach().cpu()),
            "grad_sum_max": None if grad_sum is None else float(grad_sum.max().detach().cpu()),
            **self.get_fim_lambda_diagnostics(),
            **self.get_fim_composition_diagnostics(),
        }
