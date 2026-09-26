from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from .algo_callback import AlgoCallback
from .fim_support import FIMSupportMixin
import numpy as np
from scipy.optimize import minimize


class CAGrad(FIMSupportMixin, AlgoCallback):
    def __init__(
        self,
        num_tasks,
        c=0.4,
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
        self.c = c
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
            log_prefix="CAGrad",
            extra_kwargs=kwargs,
        )

    def __repr__(self) -> str:
        return f"CAGrad(c={self.c}, fim_method={self.fim_method})"

    def get_weighted_loss(
        self,
        losses: Tensor,
        shared_parameters: List[Parameter] | Tensor,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        losses = self.cast_losses_to_correct_type(losses)
        if isinstance(shared_parameters, torch.Tensor):
            shared_parameters = [shared_parameters]
        else:
            shared_parameters = list(shared_parameters)

        grads_per_task = []
        for loss in losses:
            grads = torch.autograd.grad(loss, shared_parameters, retain_graph=True, allow_unused=True)
            flat_grads = []
            for param, grad in zip(shared_parameters, grads):
                if grad is None:
                    flat_grads.append(torch.zeros_like(param).reshape(-1))
                else:
                    flat_grads.append(grad.reshape(-1))
            grads_per_task.append(torch.cat(flat_grads))

        grads = torch.stack(grads_per_task, dim=1)
        weights = self.cagrad(grads, alpha=self.c, rescale=1).to(device=losses.device, dtype=losses.dtype)

        if self._trainer is not None:
            self._trainer.ray = weights.detach()
            self._trainer.method.ray = self._trainer.ray

        return torch.sum(losses * weights), dict(weights=weights.detach())

    def cagrad(self, grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.num_tasks) / self.num_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, self.num_tasks).dot(A).dot(b.reshape(self.num_tasks, 1))
                + c * np.sqrt(x.reshape(1, self.num_tasks).dot(A).dot(x.reshape(self.num_tasks, 1)) + 1e-8)
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        coeffs = torch.full((self.num_tasks,), 1.0 / self.num_tasks, device=grads.device, dtype=grads.dtype)
        coeffs = coeffs + lmbda * ww
        if rescale == 0:
            return coeffs
        elif rescale == 1:
            return coeffs / (1 + alpha**2)
        else:
            return coeffs / (1 + alpha)
