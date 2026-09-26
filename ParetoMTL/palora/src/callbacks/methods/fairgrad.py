"""FairGrad task-gradient aggregation adapted to the PaLoRA callback API.

Algorithm source: https://github.com/OptMN-Lab/fairgrad
The upstream implementation is MIT licensed; see ``THIRD_PARTY_NOTICES.md``.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import least_squares
from torch import Tensor
from torch.nn.parameter import Parameter

from .algo_callback import AlgoCallback
from .adaptive_fim_support import AdaptiveFIMSupportMixin


class FairGrad(AdaptiveFIMSupportMixin, AlgoCallback):
    """Allocate the shared update according to alpha-fair utilities."""

    def __init__(
        self,
        num_tasks: int,
        alpha: float = 1.0,
        max_norm: float = 1.0,
        max_norm_clip_order: str = "before_madam",
        solver_rtol: float = 1e-4,
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
        if alpha <= 0:
            raise ValueError("FairGrad alpha must be positive.")
        if max_norm_clip_order not in {"before_madam", "after_madam"}:
            raise ValueError("max_norm_clip_order must be 'before_madam' or 'after_madam'.")
        if solver_rtol <= 0:
            raise ValueError("FairGrad solver_rtol must be positive.")
        self.alpha = float(alpha)
        self.max_norm = float(max_norm)
        self.max_norm_clip_order = str(max_norm_clip_order)
        self.solver_rtol = float(solver_rtol)
        self.solver_attempts = 0
        self.last_solver_relative_residual: Optional[float] = None
        self.last_solver_max_weight: Optional[float] = None
        self._task_specific_grads: List[Tuple[Parameter, Optional[Tensor]]] = []
        self.init_fim_support(
            fim_method=fim_method, fim_ema_decay=fim_ema_decay,
            fim_eps_num=fim_eps_num, fim_eps_damp=fim_eps_damp,
            fim_clip_grad_norm=fim_clip_grad_norm, fim_bias_correction=fim_bias_correction,
            fim_stochastic_pairs=fim_stochastic_pairs, fim_lambda_policy=fim_lambda_policy,
            fim_preserve_shared_grad_norm=fim_preserve_shared_grad_norm,
            fim_solver_geometry=fim_solver_geometry,
            exclude_bn_from_fim=exclude_bn_from_fim, exclude_head_from_fim=exclude_head_from_fim,
            log_prefix="FairGrad", extra_kwargs=kwargs,
        )

    def __repr__(self) -> str:
        return f"FairGrad(alpha={self.alpha}, max_norm={self.max_norm}, fim_method={self.fim_method})"

    @staticmethod
    def _flatten_task_gradients(losses: Tensor, shared_parameters: List[Parameter]) -> Tensor:
        rows = []
        for loss in losses:
            grad_parts = torch.autograd.grad(loss, shared_parameters, retain_graph=True, allow_unused=True)
            rows.append(torch.cat([
                grad.reshape(-1) if grad is not None else torch.zeros_like(param).reshape(-1)
                for param, grad in zip(shared_parameters, grad_parts)
            ]))
        return torch.stack(rows)

    def _solve_weights(self, gram: Tensor) -> np.ndarray:
        matrix = np.asarray(gram.detach().cpu().numpy(), dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise RuntimeError("FairGrad received a non-finite task-gradient Gram matrix.")
        if np.linalg.norm(matrix) == 0:
            raise RuntimeError("FairGrad has no finite solution for a zero task-gradient Gram matrix.")
        self.solver_attempts += 1
        initial = np.ones(self.num_tasks, dtype=np.float64) / self.num_tasks

        def residual(weights: np.ndarray) -> np.ndarray:
            return matrix @ weights - np.power(1.0 / weights, 1.0 / self.alpha)

        result = least_squares(residual, initial, bounds=(0.0, np.inf))
        solved_weights = np.asarray(result.x, dtype=np.float64)
        if (not result.success or not np.isfinite(solved_weights).all() or np.any(solved_weights <= 0)):
            raise RuntimeError(f"FairGrad's nonnegative least-squares solve failed: {result.message}")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            residual_value = residual(solved_weights)
            target = np.power(1.0 / solved_weights, 1.0 / self.alpha)
            relative_residual = float(
                np.linalg.norm(residual_value) / max(np.linalg.norm(target), np.finfo(np.float64).eps)
            )
        self.last_solver_relative_residual = relative_residual
        self.last_solver_max_weight = float(np.max(solved_weights))
        if not np.isfinite(relative_residual) or relative_residual > self.solver_rtol:
            raise RuntimeError(
                "FairGrad solver reported success without satisfying the alpha-fair system: relative residual "
                f"{relative_residual:.3e}; required <= {self.solver_rtol:.3e}."
            )
        weights = np.asarray(solved_weights, dtype=np.float32)
        if not np.isfinite(weights).all():
            raise RuntimeError("FairGrad weights overflowed the training dtype.")
        return weights

    def _cache_unweighted_task_specific_grads(
        self, losses: Tensor, task_specific_parameters: List[Parameter] | Tensor | None,
    ) -> None:
        self._task_specific_grads = []
        if task_specific_parameters is None:
            return
        parameters = ([task_specific_parameters] if isinstance(task_specific_parameters, Tensor)
                      else list(task_specific_parameters))
        if not parameters:
            return
        grad_parts = torch.autograd.grad(losses.sum(), parameters, retain_graph=True, allow_unused=True)
        self._task_specific_grads = [
            (parameter, None if grad is None else grad.detach().clone())
            for parameter, grad in zip(parameters, grad_parts)
        ]

    def get_weighted_loss(
        self, losses: Tensor, shared_parameters: List[Parameter] | Tensor,
        task_specific_parameters: List[Parameter] | Tensor | None = None, **kwargs,
    ) -> Tuple[Tensor, dict]:
        losses = self.cast_losses_to_correct_type(losses)
        shared_parameters = [shared_parameters] if isinstance(shared_parameters, Tensor) else list(shared_parameters)
        gradients = self._flatten_task_gradients(losses, shared_parameters)
        solver_gradients = self.transform_task_gradients_for_solver(gradients, shared_parameters)
        gram = solver_gradients @ solver_gradients.t()
        raw_weights = self._solve_weights(gram)
        # Official overwrite_grad() multiplies only the shared gradient by num_tasks.
        weights = torch.as_tensor(raw_weights * self.num_tasks, device=losses.device, dtype=losses.dtype)
        self._cache_unweighted_task_specific_grads(losses, task_specific_parameters)
        weighted_loss = torch.sum(losses * weights)
        self.set_fim_preference_from_aggregation(weights)
        if self._trainer is not None:
            self._trainer.ray = weights.detach()
            self._trainer.method.ray = self._trainer.ray
        return weighted_loss, {
            "weights": weights.detach(), "raw_weights": torch.as_tensor(raw_weights), "GTG": gram.detach(),
        }

    def get_diagnostics(self) -> dict:
        return {
            "alpha": self.alpha,
            "solver_rtol": self.solver_rtol,
            "solver_attempts": self.solver_attempts,
            "max_norm_clip_order": self.max_norm_clip_order,
            "last_solver_relative_residual": self.last_solver_relative_residual,
            "last_solver_max_weight": self.last_solver_max_weight,
            **self.get_fim_lambda_diagnostics(),
            **self.get_fim_composition_diagnostics(),
        }

    def _restore_unweighted_task_specific_grads(self, trainer) -> None:
        scale = 1.0 if self._grads_unscaled else float(trainer.scaler.get_scale())
        for parameter, grad in self._task_specific_grads:
            parameter.grad = None if grad is None else grad * scale
        self._task_specific_grads = []

    def on_after_backward(self, trainer, *args, **kwargs) -> None:
        # Restore heads before unscale; the official max-norm constraint applies
        # to the shared method direction before composing MAdam.
        if self.max_norm_clip_order == "before_madam":
            self._restore_unweighted_task_specific_grads(trainer)
            self.clip_shared_gradients(trainer, self.max_norm)
            super().on_after_backward(trainer, *args, **kwargs)
        else:
            super().on_after_backward(trainer, *args, **kwargs)
            self._restore_unweighted_task_specific_grads(trainer)
            self.clip_shared_gradients(trainer, self.max_norm)
