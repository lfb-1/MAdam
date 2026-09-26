from pathlib import Path
import sys

import torch


def _ensure_madam_core() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "madam_core" / "src"
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return
    raise ImportError("Could not locate madam_core/src")


_ensure_madam_core()

from madam_core.madam import MAdamPreconditioner
from madam_core.diag_madam import DiagonalMAdamPreconditioner


class _BasePreconditionedAdam(torch.optim.Adam):
    def __init__(
        self,
        params,
        num_tasks,
        lr=1e-3,
        ema_decay=0.999,
        stochastic=False,
        preconditioned_params=None,
        **kwargs,
    ):
        param_list = list(params)
        super().__init__(param_list, lr=lr, **kwargs)
        self._num_tasks = num_tasks
        self._ema_decay = ema_decay
        self._stochastic = stochastic
        self._preconditioned_params = list(preconditioned_params) if preconditioned_params is not None else param_list
        self.losses = None
        self.precond = self._build_preconditioner(self._preconditioned_params)

    def _build_preconditioner(self, params):
        raise NotImplementedError

    def _task_losses(self):
        if self.losses is None:
            raise RuntimeError("MAdam optimizer expected closure to populate optimizer.losses.")
        if not torch.is_tensor(self.losses):
            raise RuntimeError(f"MAdam optimizer expected optimizer.losses to be a tensor, got {type(self.losses)!r}.")
        if self.losses.ndim != 1:
            raise RuntimeError(f"MAdam optimizer expected optimizer.losses to be 1D, got shape {tuple(self.losses.shape)}.")
        if len(self.losses) != self._num_tasks:
            raise RuntimeError(
                f"MAdam optimizer expected {self._num_tasks} loss components, got {len(self.losses)}."
            )
        return tuple(self.losses.unbind())

    def step(self, closure):
        if closure is None:
            raise RuntimeError("MAdam optimizer requires a closure that supports skip_backward=True.")

        self.losses = None
        with torch.enable_grad():
            closure(skip_backward=True)
            task_losses = self._task_losses()
            lam = torch.ones(self._num_tasks, device=self.losses.device, dtype=self.losses.dtype) / self._num_tasks

            self.precond.update(task_losses, lam)

            self.zero_grad()
            scalarized = torch.sum(self.losses * lam)
            scalarized.backward()
        self.precond.precondition()
        super().step()

        return scalarized.detach()


class MAdam(_BasePreconditionedAdam):
    def _build_preconditioner(self, params):
        return MAdamPreconditioner(
            params,
            num_tasks=self._num_tasks,
            ema_decay=self._ema_decay,
            stochastic=self._stochastic,
        )


class DiagMAdam(_BasePreconditionedAdam):
    def _build_preconditioner(self, params):
        return DiagonalMAdamPreconditioner(
            params,
            num_tasks=self._num_tasks,
            ema_decay=self._ema_decay,
            stochastic=self._stochastic,
        )
