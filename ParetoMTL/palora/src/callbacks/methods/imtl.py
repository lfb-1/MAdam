from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from .algo_callback import AlgoCallback
from .fim_support import FIMSupportMixin


class IMTLG(FIMSupportMixin, AlgoCallback):
    """TOWARDS IMPARTIAL MULTI-TASK LEARNING: https://openreview.net/pdf?id=IMPnRXEWpvr"""

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
            log_prefix="IMTL",
            extra_kwargs=kwargs,
        )

    def __repr__(self) -> str:
        return f"IMTLG(fim_method={self.fim_method})"

    def get_weighted_loss(
        self,
        losses: Tensor,
        shared_parameters: List[Parameter] | Tensor,
        task_specific_parameters: List[Parameter] | Tensor,
        last_shared_parameters: List[Parameter] | Tensor,
        representation: Parameter | Tensor,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        grads = {}
        norm_grads = {}

        if isinstance(losses, dict):
            losses = torch.stack(tuple(losses.values()))

        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    loss,
                    shared_parameters,
                    retain_graph=True,
                )
            )
            grad = torch.cat([torch.flatten(grad) for grad in g])
            norm_term = torch.norm(grad)

            grads[i] = grad
            norm_grads[i] = grad / norm_term

        G = torch.stack(tuple(v for v in grads.values()))
        D = G[0,] - G[1:,]

        U = torch.stack(tuple(v for v in norm_grads.values()))
        U = U[0,] - U[1:,]
        first_element = torch.matmul(
            G[0,],
            U.t(),
        )
        try:
            second_element = torch.inverse(torch.matmul(D, U.t()))
        except:
            # workaround for cases where matrix is singular
            second_element = torch.inverse(
                torch.eye(self.num_tasks - 1, device=self.device) * 1e-8 + torch.matmul(D, U.t())
            )

        alpha_ = torch.matmul(first_element, second_element)
        alpha = torch.cat((torch.tensor(1 - alpha_.sum(), device=self.device).unsqueeze(-1), alpha_))

        loss = torch.sum(losses * alpha)

        if self._trainer is not None:
            self._trainer.ray = alpha.detach()
            self._trainer.method.ray = self._trainer.ray

        return loss, dict(weights=alpha)
