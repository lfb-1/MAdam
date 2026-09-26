"""Nash-MTL task weighting adapted to the PaLoRA callback API.

Algorithm source: https://github.com/AvivNavon/nash-mtl
The upstream implementation is MIT licensed; see ``THIRD_PARTY_NOTICES.md``.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter

try:
    import cvxpy as cp
except ImportError:  # Keep unrelated methods importable without the optional solver.
    cp = None

from .algo_callback import AlgoCallback
from .adaptive_fim_support import AdaptiveFIMSupportMixin


class NashMTL(AdaptiveFIMSupportMixin, AlgoCallback):
    """Approximate the Nash bargaining solution for task-gradient weights."""

    def __init__(
        self,
        num_tasks: int,
        max_norm: float = 1.0,
        max_norm_clip_order: str = "before_madam",
        update_weights_every: int = 1,
        optim_niter: int = 20,
        solver: str = "ECOS",
        max_solver_failure_rate: Optional[float] = None,
        min_solver_attempts_before_failure: int = 100,
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
        if cp is None:
            raise ImportError(
                "NashMTL requires cvxpy and the configured convex solver. "
                "Install the dependencies from requirements.txt."
            )
        if max_norm_clip_order not in {"before_madam", "after_madam"}:
            raise ValueError("max_norm_clip_order must be 'before_madam' or 'after_madam'.")
        if update_weights_every <= 0:
            raise ValueError("update_weights_every must be positive.")
        if optim_niter <= 0:
            raise ValueError("optim_niter must be positive.")
        if max_solver_failure_rate is not None and not 0.0 <= max_solver_failure_rate <= 1.0:
            raise ValueError("max_solver_failure_rate must be null or lie in [0, 1].")
        if min_solver_attempts_before_failure <= 0:
            raise ValueError("min_solver_attempts_before_failure must be positive.")

        self.optim_niter = int(optim_niter)
        self.update_weights_every = int(update_weights_every)
        self.max_norm = float(max_norm)
        self.max_norm_clip_order = str(max_norm_clip_order)
        self.solver = str(solver)
        self.max_solver_failure_rate = None if max_solver_failure_rate is None else float(max_solver_failure_rate)
        self.min_solver_attempts_before_failure = int(min_solver_attempts_before_failure)
        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = np.eye(self.num_tasks)
        self.step = 0
        self.prvs_alpha = np.ones(self.num_tasks, dtype=np.float32)
        self.solver_attempts = 0
        self.solver_failures = 0
        self.solver_status_counts: dict[str, int] = {}
        self.last_solver_status: Optional[str] = None
        self.last_solver_residual: Optional[float] = None
        self.init_fim_support(
            fim_method=fim_method, fim_ema_decay=fim_ema_decay,
            fim_eps_num=fim_eps_num, fim_eps_damp=fim_eps_damp,
            fim_clip_grad_norm=fim_clip_grad_norm, fim_bias_correction=fim_bias_correction,
            fim_stochastic_pairs=fim_stochastic_pairs, fim_lambda_policy=fim_lambda_policy,
            fim_preserve_shared_grad_norm=fim_preserve_shared_grad_norm,
            fim_solver_geometry=fim_solver_geometry,
            exclude_bn_from_fim=exclude_bn_from_fim, exclude_head_from_fim=exclude_head_from_fim,
            log_prefix="NashMTL", extra_kwargs=kwargs,
        )

    def __repr__(self) -> str:
        return (
            "NashMTL(max_norm={}, update_weights_every={}, optim_niter={}, "
            "solver={}, fim_method={})"
        ).format(self.max_norm, self.update_weights_every, self.optim_niter, self.solver, self.fim_method)

    def _stop_criteria(self, gtg: np.ndarray, alpha_t: np.ndarray) -> bool:
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value) < 1e-6)
        )

    def _record_solver_status(self, status: str) -> None:
        self.last_solver_status = status
        self.solver_status_counts[status] = self.solver_status_counts.get(status, 0) + 1

    def _check_solver_failure_rate(self) -> None:
        if self.max_solver_failure_rate is None:
            return
        failure_rate = self.solver_failures / max(self.solver_attempts, 1)
        if (self.solver_attempts >= self.min_solver_attempts_before_failure
                and failure_rate > self.max_solver_failure_rate):
            raise RuntimeError(
                "NashMTL solver failure rate exceeded the configured limit: "
                f"{self.solver_failures}/{self.solver_attempts} "
                f"({failure_rate:.2%}) > {self.max_solver_failure_rate:.2%}."
            )

    def _use_previous_weights(self, reason: str, *, status: str) -> np.ndarray:
        self._record_solver_status(status)
        self.solver_failures += 1
        if self.solver_failures <= 3 or self.solver_failures % 100 == 0:
            logging.warning(
                "NashMTL solver fallback #%d/%d: %s; reusing previous weights.",
                self.solver_failures, self.solver_attempts, reason,
            )
        self._check_solver_failure_rate()
        return self.prvs_alpha

    def solve_optimization(self, gtg: np.ndarray) -> np.ndarray:
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor
        if self.solver not in cp.installed_solvers():
            raise RuntimeError(
                f"NashMTL solver '{self.solver}' is unavailable; installed CVXPY "
                f"solvers: {', '.join(cp.installed_solvers()) or 'none'}."
            )
        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t
            try:
                self.prob.solve(solver=self.solver, warm_start=True, max_iters=100)
            except cp.error.SolverError as exc:
                return self._use_previous_weights(
                    f"{self.solver} raised {type(exc).__name__}", status="solver_error",
                )
            status = str(self.prob.status)
            if status not in {str(cp.OPTIMAL), str(cp.OPTIMAL_INACCURATE)}:
                return self._use_previous_weights(
                    f"{self.solver} returned status={status}", status=status,
                )
            if self.alpha_param.value is None:
                return self._use_previous_weights(
                    f"{self.solver} returned no weights", status="empty_solution",
                )
            if self._stop_criteria(gtg, alpha_t):
                break
            alpha_t = self.alpha_param.value
        if alpha_t is not None:
            alpha_t = np.asarray(alpha_t, dtype=np.float32)
            if not np.isfinite(alpha_t).all() or np.any(alpha_t < 0):
                return self._use_previous_weights(
                    "solver produced non-finite or negative task weights", status="invalid_weights",
                )
            self.prvs_alpha = alpha_t
        self._record_solver_status(str(self.prob.status))
        self.last_solver_residual = float(
            np.linalg.norm(gtg @ self.prvs_alpha - 1 / (self.prvs_alpha + 1e-10))
        )
        self._check_solver_failure_rate()
        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        return prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)

    def _init_optim_problem(self) -> None:
        self.alpha_param = cp.Variable(shape=(self.num_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(shape=(self.num_tasks,), value=self.prvs_alpha)
        self.G_param = cp.Parameter(shape=(self.num_tasks, self.num_tasks), value=self.init_gtg)
        self.normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0]))
        self.phi_alpha = self._calc_phi_alpha_linearization()
        G_alpha = self.G_param @ self.alpha_param
        constraints = [
            -cp.log(self.alpha_param[i] * self.normalization_factor_param) - cp.log(G_alpha[i]) <= 0
            for i in range(self.num_tasks)
        ]
        objective = cp.Minimize(cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param)
        self.prob = cp.Problem(objective, constraints)

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

    def get_weighted_loss(
        self, losses: Tensor, shared_parameters: List[Parameter] | Tensor, **kwargs,
    ) -> Tuple[Tensor, dict]:
        losses = self.cast_losses_to_correct_type(losses)
        shared_parameters = [shared_parameters] if isinstance(shared_parameters, Tensor) else list(shared_parameters)
        if self.step == 0:
            self._init_optim_problem()
        if (self.step % self.update_weights_every) == 0:
            self.solver_attempts += 1
            gradients = self._flatten_task_gradients(losses, shared_parameters)
            solver_gradients = self.transform_task_gradients_for_solver(gradients, shared_parameters)
            gtg = solver_gradients @ solver_gradients.t()
            norm = torch.linalg.vector_norm(gtg)
            if not torch.isfinite(norm) or norm <= 0:
                alpha_np = self._use_previous_weights(
                    "zero or non-finite task-gradient Gram matrix", status="invalid_gram",
                )
            else:
                self.normalization_factor = norm.detach().cpu().numpy().reshape((1,))
                normalized_gtg = gtg / norm
                alpha_np = self.solve_optimization(normalized_gtg.detach().cpu().numpy())
        else:
            alpha_np = self.prvs_alpha
        self.step += 1
        alpha = torch.as_tensor(alpha_np, device=losses.device, dtype=losses.dtype)
        weighted_loss = torch.sum(losses * alpha)
        self.set_fim_preference_from_aggregation(alpha)
        if self._trainer is not None:
            self._trainer.ray = alpha.detach()
            self._trainer.method.ray = self._trainer.ray
        return weighted_loss, {"weights": alpha.detach()}

    def get_diagnostics(self) -> dict:
        return {
            "solver": self.solver,
            "solver_attempts": self.solver_attempts,
            "solver_failures": self.solver_failures,
            "solver_failure_rate": self.solver_failures / max(self.solver_attempts, 1),
            "solver_status_counts": dict(self.solver_status_counts),
            "last_solver_status": self.last_solver_status,
            "last_solver_residual": self.last_solver_residual,
            "last_successful_solver_residual": self.last_solver_residual,
            "max_solver_failure_rate": self.max_solver_failure_rate,
            "max_norm_clip_order": self.max_norm_clip_order,
            "update_weights_every": self.update_weights_every,
            "optim_niter": self.optim_niter,
            **self.get_fim_lambda_diagnostics(),
            **self.get_fim_composition_diagnostics(),
        }

    def on_after_backward(self, trainer, *args, **kwargs) -> None:
        if self.max_norm_clip_order == "before_madam":
            self.clip_shared_gradients(trainer, self.max_norm)
            super().on_after_backward(trainer, *args, **kwargs)
        else:
            super().on_after_backward(trainer, *args, **kwargs)
            self.clip_shared_gradients(trainer, self.max_norm)
