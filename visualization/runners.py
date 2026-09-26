"""Adam and MAdam runners on a 2D bi-objective toy problem.

Both runners use the same scalarized loss ``lam_1 f_1 + lam_2 f_2`` and the
same vanilla ``torch.optim.Adam`` underneath; the only difference is that the
MAdam runner inserts the real ``MAdamPreconditioner`` from ``madam_core`` between
``backward()`` and ``optimizer.step()`` to rescale the gradient by
``1 / sqrt(F(lam) + eps_damp)`` instead of Adam's coordinate-wise
``1 / sqrt(v_t + eps)``.

Both runners log per-iteration:
  - x (length-2 parameter)
  - f_1, f_2 (objective values)
  - g_1, g_2 (per-task gradients, length-2 each)
  - denom (the per-coordinate preconditioning denominator that was actually
    applied) — for Adam this is ``sqrt(v_hat + eps)``; for MAdam this is the
    output of ``precond.build_denominator_per_param()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MADAM_CORE_SRC = REPO_ROOT / "madam_core" / "src"
if str(MADAM_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(MADAM_CORE_SRC))

from madam_core.madam import MAdamPreconditioner

from typing import Union

from toy_problems import _BiQuadratic2D, BiQuadraticGeneral2D

# Either toy variant works: only `f_torch(x) -> (f_1, f_2)` is required.
BiObjective2D = Union[_BiQuadratic2D, BiQuadraticGeneral2D]


@dataclass
class RunLog:
    """Per-iteration logs for a single optimizer run."""

    xs: np.ndarray  # (steps + 1, 2) — includes initial x
    f1: np.ndarray  # (steps + 1,)
    f2: np.ndarray  # (steps + 1,)
    g1: np.ndarray  # (steps, 2) — task-1 gradient at the iterate before step t
    g2: np.ndarray  # (steps, 2)
    denom: np.ndarray  # (steps, 2) — per-coord preconditioner denominator at step t
    lam: np.ndarray  # (2,)
    optimizer: str  # "adam" or "madam"
    extra: dict = field(default_factory=dict)


def _per_task_grads_numpy(
    problem: BiObjective2D, x: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-task gradients via autograd, return as detached numpy."""
    x_param = x.detach().clone().requires_grad_(True)
    f1, f2 = problem.f_torch(x_param)
    g1 = torch.autograd.grad(f1, x_param, retain_graph=True)[0].detach().numpy()
    g2 = torch.autograd.grad(f2, x_param, retain_graph=False)[0].detach().numpy()
    return g1, g2


def run_adam(
    problem: BiObjective2D,
    lam: Sequence[float],
    x0: Sequence[float],
    lr: float = 1e-2,
    steps: int = 800,
    seed: int = 0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    noise_std: float = 0.0,
) -> RunLog:
    """Vanilla torch.optim.Adam on the scalarized loss lam_1 f_1 + lam_2 f_2.

    No preconditioner. Adam's coordinate-wise second moment serves as its
    metric, which is the baseline this visualization compares against.

    If ``noise_std > 0``, a linear perturbation ``eps_i^T x`` is added to
    each task's loss at every step, with ``eps_i ~ N(0, sigma^2 I)`` drawn
    fresh per step. This shifts each per-task gradient by an unbiased
    Gaussian -- the simplest model of mini-batch SGD noise.
    """
    torch.manual_seed(seed)
    lam_t = torch.tensor(lam, dtype=torch.float64)
    x = torch.nn.Parameter(torch.tensor(list(x0), dtype=torch.float64))
    optimizer = torch.optim.Adam([x], lr=lr, betas=betas, eps=eps)

    xs = np.zeros((steps + 1, 2))
    f1_log = np.zeros(steps + 1)
    f2_log = np.zeros(steps + 1)
    g1_log = np.zeros((steps, 2))
    g2_log = np.zeros((steps, 2))
    denom_log = np.zeros((steps, 2))

    xs[0] = x.detach().numpy()
    with torch.no_grad():
        f1_init, f2_init = problem.f_torch(x)
        f1_log[0], f2_log[0] = float(f1_init), float(f2_init)

    for t in range(steps):
        # Per-task gradients for logging only (not used by Adam)
        g1_log[t], g2_log[t] = _per_task_grads_numpy(problem, x)

        optimizer.zero_grad()
        f1, f2 = problem.f_torch(x)
        if noise_std > 0.0:
            eps1 = torch.randn(x.shape, dtype=x.dtype) * noise_std
            eps2 = torch.randn(x.shape, dtype=x.dtype) * noise_std
            f1 = f1 + eps1 @ x
            f2 = f2 + eps2 @ x
        scalarized = lam_t[0] * f1 + lam_t[1] * f2
        scalarized.backward()
        optimizer.step()

        # Adam's effective per-coord denominator at step t+1: sqrt(v_hat) + eps,
        # where v_hat = v / (1 - beta2^(t+1))
        state = optimizer.state[x]
        v = state["exp_avg_sq"].detach().numpy()
        bias_correction2 = 1.0 - betas[1] ** state["step"].item()
        denom_log[t] = np.sqrt(v / bias_correction2) + eps

        xs[t + 1] = x.detach().numpy()
        with torch.no_grad():
            f1_v, f2_v = problem.f_torch(x)
            f1_log[t + 1], f2_log[t + 1] = float(f1_v), float(f2_v)

    return RunLog(
        xs=xs, f1=f1_log, f2=f2_log,
        g1=g1_log, g2=g2_log, denom=denom_log,
        lam=np.array(lam, dtype=float), optimizer="adam",
    )


def run_madam(
    problem: BiObjective2D,
    lam: Sequence[float],
    x0: Sequence[float],
    lr: float = 1e-2,
    steps: int = 800,
    seed: int = 0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    ema_decay: float = 0.999,
    eps_damp: float = 1e-8,
    stochastic: bool = False,
    gradient_balance: bool = False,
    noise_std: float = 0.0,
) -> RunLog:
    """MAdam = MAdamPreconditioner + torch.optim.Adam on the same scalarized loss.

    Two-phase API per the docstring at madam_core/madam.py:
        precond.update(losses, lam)   -> per-task grads + FIM EMA update
        scalarized.backward()          -> populates param.grad
        precond.precondition()         -> divides param.grad by sqrt(F(lam)+eps_damp)
        optimizer.step()               -> Adam step on the preconditioned gradient
    """
    torch.manual_seed(seed)
    lam_t = torch.tensor(lam, dtype=torch.float64)
    x = torch.nn.Parameter(torch.tensor(list(x0), dtype=torch.float64))
    optimizer = torch.optim.Adam([x], lr=lr, betas=betas, eps=eps)
    precond = MAdamPreconditioner(
        params=[x],
        num_tasks=2,
        ema_decay=ema_decay,
        eps_damp=eps_damp,
        stochastic=stochastic,
        gradient_balance=gradient_balance,
    )

    xs = np.zeros((steps + 1, 2))
    f1_log = np.zeros(steps + 1)
    f2_log = np.zeros(steps + 1)
    g1_log = np.zeros((steps, 2))
    g2_log = np.zeros((steps, 2))
    denom_log = np.zeros((steps, 2))

    xs[0] = x.detach().numpy()
    with torch.no_grad():
        f1_init, f2_init = problem.f_torch(x)
        f1_log[0], f2_log[0] = float(f1_init), float(f2_init)

    for t in range(steps):
        g1_log[t], g2_log[t] = _per_task_grads_numpy(problem, x)

        optimizer.zero_grad()
        f1, f2 = problem.f_torch(x)
        if noise_std > 0.0:
            eps1 = torch.randn(x.shape, dtype=x.dtype) * noise_std
            eps2 = torch.randn(x.shape, dtype=x.dtype) * noise_std
            f1 = f1 + eps1 @ x
            f2 = f2 + eps2 @ x
        precond.update([f1, f2], lam_t)
        scalarized = lam_t[0] * f1 + lam_t[1] * f2
        scalarized.backward()
        # Capture the MAdam denominator that will divide param.grad
        denom_per_param = precond.build_denominator_per_param()
        denom_log[t] = denom_per_param[0].detach().numpy()
        precond.precondition()
        optimizer.step()

        xs[t + 1] = x.detach().numpy()
        with torch.no_grad():
            f1_v, f2_v = problem.f_torch(x)
            f1_log[t + 1], f2_log[t + 1] = float(f1_v), float(f2_v)

    return RunLog(
        xs=xs, f1=f1_log, f2=f2_log,
        g1=g1_log, g2=g2_log, denom=denom_log,
        lam=np.array(lam, dtype=float), optimizer="madam",
    )


def _smoke() -> None:
    from toy_problems import AnisoQuadratic2D

    p = AnisoQuadratic2D()
    lam = [0.5, 0.5]
    x0 = [3.0, 3.0]

    log_a = run_adam(p, lam, x0, lr=1e-2, steps=50, seed=0)
    log_m = run_madam(p, lam, x0, lr=1e-2, steps=50, seed=0)

    # Both should drive the scalarized loss down
    s0 = log_a.lam[0] * log_a.f1[0] + log_a.lam[1] * log_a.f2[0]
    s_a = log_a.lam[0] * log_a.f1[-1] + log_a.lam[1] * log_a.f2[-1]
    s_m = log_m.lam[0] * log_m.f1[-1] + log_m.lam[1] * log_m.f2[-1]
    print(f"adam scalarized: {s0:.4f} -> {s_a:.4f}")
    print(f"madam scalarized: {s0:.4f} -> {s_m:.4f}")
    assert s_a < s0, "Adam did not decrease scalarized loss"
    assert s_m < s0, "MAdam did not decrease scalarized loss"

    # MAdam's denom should track F(lam); not equal to sqrt(g_scalarized^2)
    print(f"adam denom[-1]:  {log_a.denom[-1]}")
    print(f"madam denom[-1]: {log_m.denom[-1]}")

    # Per-task gradient logs should match the analytic gradient
    g_an = p.grads_numpy(log_a.xs[0])
    np.testing.assert_allclose(log_a.g1[0], g_an[0], atol=1e-10)
    np.testing.assert_allclose(log_a.g2[0], g_an[1], atol=1e-10)
    print("runners smoke ok")


if __name__ == "__main__":
    _smoke()
