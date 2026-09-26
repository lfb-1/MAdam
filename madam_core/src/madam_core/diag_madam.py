"""Diagonal MAdam preconditioner — diagonal-only variant without cross-task terms.

Same two-phase API as :class:`MAdamPreconditioner`::

    from madam_core.diag_madam import DiagonalMAdamPreconditioner

    params = [p for p in model.parameters() if p.requires_grad]
    precond = DiagonalMAdamPreconditioner(params, num_tasks=T)
    optimizer = torch.optim.Adam(params, lr=1e-3)

    for x, y in loader:
        optimizer.zero_grad()
        losses = [loss_fn(model(x), y, task=t) for t in range(T)]
        lam = sample_ray()

        precond.update(losses, lam)           # FIM update — before backward
        scalarized = sum(lam[t] * losses[t] for t in range(T))
        scalarized.backward()                 # normal backward
        precond.precondition()                # modify param.grad in-place
        optimizer.step()

Diagonal ablation of MAdam: only stores per-task Fisher diagonals F_{ii},
dropping all cross-task terms F_{ij} (i != j).  Assembly reduces to
F(λ) = Σ_i λ_i² F_{ii}.  Storage is O(T·D) instead of O(T(T+1)/2 · D).

Use this to isolate the contribution of cross-task terms in MAdam.
On same-loss datasets (e.g. SARCOS, all MSE), cross-terms help and
full MAdam > diagonal MAdam.  On mixed-loss datasets, results depend on eps_damp tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable, List, Optional, Sequence, Tuple


class DiagonalMAdamPreconditioner:
    """Diagonal-only Conditional Fisher preconditioner (no cross-task terms).

    Args:
        params: Iterable of ``nn.Parameter`` to precondition.
        num_tasks: Number of tasks ``T``.
        ema_decay: EMA decay factor β for F_{ii} estimates.
        eps_num: Numerical floor used only to clamp the final denominator.
        eps_damp: Curvature damping inside ``sqrt(F(λ) + eps_damp)``.
        stochastic: If True, sample one random task per step and only
            update that task's F_{ii} (1 backward pass instead of T).
        bias_correct: Adam-style EMA bias correction ``F/(1-β^t)``.
        gradient_balance: If True, normalize each task-gradient row before
            updating the diagonal Fisher statistics.
        gradient_balance_eps: Stabilizer used by ``gradient_balance``.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        num_tasks: int,
        ema_decay: float = 0.999,
        eps_num: float = 1e-12,
        eps_damp: float = 1e-6,
        stochastic: bool = False,
        bias_correct: bool = False,
        gradient_balance: bool = False,
        gradient_balance_eps: float = 1e-3,
    ) -> None:
        self.params: List[nn.Parameter] = list(params)
        self.T = num_tasks
        self.beta = ema_decay
        self.eps_num = float(eps_num)
        self.eps_damp = float(eps_damp)
        self.stochastic = stochastic
        self.bias_correct = bias_correct
        self.gradient_balance = bool(gradient_balance)
        self.gradient_balance_eps = float(gradient_balance_eps)

        self.D = sum(p.numel() for p in self.params)

        # Diagonal pairs only: (0,0), (1,1), ..., (T-1,T-1)  →  T pairs
        self.pair_i = torch.arange(num_tasks)
        self.pair_j = torch.arange(num_tasks)
        self.num_pairs = num_tasks

        # EMA storage — lazily initialised on first update.
        self.fim: Optional[Tensor] = None  # [T, D]
        self._step = 0
        self._cached_lam: Optional[Tensor] = None

        # Warmup: 1/(1-β) steps for the EMA to converge
        self.warmup = (
            0
            if bias_correct
            else int(1.0 / (1.0 - ema_decay))
        )

    def _build_denom(self, f_lam: Tensor, alpha: float) -> Tensor:
        denom = alpha * (f_lam.clamp(min=0.0) + self.eps_damp).sqrt() + (1.0 - alpha)
        return denom.clamp_min(self.eps_num)

    # ------------------------------------------------------------------
    # Two-phase API
    # ------------------------------------------------------------------

    def update(self, task_losses: Sequence[Tensor], lam: Tensor) -> None:
        """Phase 1: compute per-task gradients and update diagonal FIM EMA."""
        self._step += 1
        if lam.dim() == 1:
            lam = lam.unsqueeze(0)
        self._cached_lam = lam

        task_grads, tasks_computed = self._compute_task_grads(task_losses)
        self._update_fim(task_grads, tasks_computed)

    def precondition(self) -> None:
        """Phase 2: read ``param.grad``, divide by the split MAdam denominator."""
        lam = self._cached_lam
        K = lam.shape[0]

        if self.fim is not None and self.pair_i.device != self.fim.device:
            self.pair_i = self.pair_i.to(self.fim.device)
            self.pair_j = self.pair_j.to(self.fim.device)

        fim = self._corrected_fim()
        f_lam = self._assemble(fim, lam)

        all_grads = torch.cat(
            [
                p.grad.flatten()
                if p.grad is not None
                else torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
                for p in self.params
            ]
        )

        alpha = 1.0 if self.bias_correct else min(1.0, self._step / max(self.warmup, 1))
        denom = self._build_denom(f_lam, alpha)
        g_precond = (all_grads / K) / denom

        self._unflatten_to_grad(g_precond)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def sample_ray(self) -> Tensor:
        """Sample a preference vector from Dir(1, ..., 1)."""
        device = self.params[0].device if self.params else torch.device("cpu")
        return torch.distributions.Dirichlet(
            torch.ones(self.T, device=device)
        ).sample()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_task_grads(
        self, task_losses: Sequence[Tensor]
    ) -> Tuple[Tensor, Optional[set]]:
        """Return [T, D] flat per-task gradients."""
        T, D = self.T, self.D
        device = self.params[0].device

        if self.stochastic and self.fim is not None:
            # Sample one random task (diagonal pair = single task)
            t_idx = torch.randint(T, (1,)).item()
            tasks_needed = {t_idx}

            grads = torch.zeros(T, D, device=device)
            g_t = torch.autograd.grad(
                task_losses[t_idx], self.params,
                retain_graph=True, allow_unused=True,
            )
            grads[t_idx] = self._flat_task_grad(g_t, device)
            return self._balance_task_grads_for_fim(grads, tasks_needed), tasks_needed
        else:
            grads = torch.zeros(T, D, device=device)
            for t in range(T):
                g_t = torch.autograd.grad(
                    task_losses[t], self.params,
                    retain_graph=True, allow_unused=True,
                )
                grads[t] = self._flat_task_grad(g_t, device)
            return self._balance_task_grads_for_fim(grads, None), None

    def _flat_task_grad(self, grads, device: torch.device) -> Tensor:
        return torch.cat(
            [
                g.flatten()
                if g is not None
                else torch.zeros(p.numel(), device=device, dtype=p.dtype)
                for g, p in zip(grads, self.params)
            ]
        )

    def _balance_task_grads_for_fim(
        self, grads: Tensor, tasks_computed: Optional[set]
    ) -> Tensor:
        if tasks_computed is None:
            tasks = list(range(self.T))
        else:
            tasks = sorted(int(task_idx) for task_idx in tasks_computed)

        if not self.gradient_balance:
            return grads

        balanced = grads.clone()
        for task_idx in tasks:
            norm = grads[task_idx].norm()
            scale = 1.0 / norm.add(self.gradient_balance_eps).clamp_min(self.eps_num)
            balanced[task_idx] = grads[task_idx] * scale

        return balanced

    def _update_fim(self, grads: Tensor, tasks_computed: Optional[set]) -> None:
        """EMA update of diagonal Fisher terms F_{ii} = EMA[g_i²]."""
        device = grads.device

        if self.fim is None:
            self.fim = torch.zeros(self.T, self.D, device=device)
            self.pair_i = self.pair_i.to(device)
            self.pair_j = self.pair_j.to(device)

        beta = self.beta
        with torch.no_grad():
            if tasks_computed is not None:
                # Stochastic: update only the sampled task's diagonal
                for t in tasks_computed:
                    self.fim[t].mul_(beta).add_(grads[t].square(), alpha=1 - beta)
            else:
                # Full update: all T diagonals at once
                self.fim.mul_(beta).add_(grads.square(), alpha=1 - beta)

    def _corrected_fim(self) -> Tensor:
        """Return bias-corrected FIM if enabled, else raw."""
        if self.bias_correct and self._step > 0:
            correction = max(1.0 - self.beta ** self._step, 1e-8)
            return self.fim / correction
        return self.fim

    def _assemble(self, fim: Tensor, lam: Tensor) -> Tensor:
        """Assemble F(λ) = Σ_i λ̄_i² F_{ii} (diagonal only, no cross-terms)."""
        lam_bar = lam.mean(dim=0).to(fim.device)  # [T]
        w = lam_bar.square()  # [T]  — λ_i² weights
        return fim.T @ w  # [D] = [D, T] @ [T]

    def _unflatten_to_grad(self, flat: Tensor) -> None:
        """Write a flat [D] vector back into each param.grad."""
        offset = 0
        for p in self.params:
            n = p.numel()
            p.grad = flat[offset : offset + n].view_as(p).clone()
            offset += n

    def _unflatten(self, flat: Tensor) -> List[Tensor]:
        values: List[Tensor] = []
        offset = 0
        for p in self.params:
            n = p.numel()
            values.append(flat[offset : offset + n].view_as(p).detach().clone())
            offset += n
        return values

    def _assembled_curvature(self) -> Tensor:
        if self.fim is None or self._cached_lam is None:
            raise RuntimeError("Diagonal MAdam curvature requested before update().")
        lam = self._cached_lam
        if lam.dim() == 1:
            lam = lam.unsqueeze(0)
        fim = self._corrected_fim()
        return self._assemble(fim, lam)

    def get_curvature_per_param(self) -> List[Tensor]:
        return self._unflatten(self._assembled_curvature())

    def build_denominator_per_param(
        self,
        curvature_per_param: Optional[List[Tensor]] = None,
    ) -> List[Tensor]:
        if curvature_per_param is None:
            f_lam = self._assembled_curvature()
        else:
            if len(curvature_per_param) != len(self.params):
                raise ValueError(
                    f"Expected {len(self.params)} curvature tensors, got {len(curvature_per_param)}."
                )
            flat_parts: List[Tensor] = []
            for param, curvature in zip(self.params, curvature_per_param):
                if curvature.numel() != param.numel():
                    raise ValueError(
                        "Curvature tensor shape does not match parameter shape: "
                        f"expected {tuple(param.shape)}, got {tuple(curvature.shape)}."
                    )
                flat_parts.append(curvature.to(device=param.device, dtype=param.dtype).reshape(-1))
            f_lam = torch.cat(flat_parts)

        alpha = 1.0 if self.bias_correct else min(1.0, self._step / max(self.warmup, 1))
        denom = self._build_denom(f_lam, alpha)
        return self._unflatten(denom)

    def get_denominator_per_param(self) -> List[Tensor]:
        return self.build_denominator_per_param()
