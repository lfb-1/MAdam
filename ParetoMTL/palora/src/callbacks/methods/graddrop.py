from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from .algo_callback import AlgoCallback


class GradDrop(AlgoCallback):
    @staticmethod
    def graddrop(grads):
        P = 0.5 * (1.0 + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
        U = torch.rand_like(grads[:, 0])
        M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
        g = (grads * M.float()).mean(1)
        return g

    @staticmethod
    def reshape_gradients(grads, shared_parameters):
        from itertools import accumulate

        n = [p.numel() for p in shared_parameters]
        n = [0] + list(accumulate(n))

        grads_reshaped = []
        for a, b, p in zip(n[:-1], n[1:], shared_parameters):
            grads_reshaped.append(grads[a:b].view(p.shape))

        return grads_reshaped

    def set_graddrop_gradients(self, losses, shared_parameters, task_specific_parameters):
        # adapted from PCGRAD implementation
        shared_grads = []
        for l in losses:
            grads = torch.autograd.grad(l, shared_parameters, retain_graph=True)
            grads = torch.cat([g.view(-1) for g in grads])
            shared_grads.append(grads)

        # compute gradients for shared parameters
        shared_grads = torch.stack(shared_grads, dim=1)
        shared_grads = self.graddrop(shared_grads)
        shared_grads = self.reshape_gradients(shared_grads, shared_parameters)

        # compute task specific gradients
        losses.mean().backward(retain_graph=True)

        # overwrite gradients for shared parameters
        for p, g in zip(shared_parameters, shared_grads):
            p.grad = g

    def backward(
        self,
        losses: Tensor,
        shared_parameters: List[Parameter] | Tensor = None,
        task_specific_parameters: List[Parameter] | Tensor = None,
        last_shared_parameters: List[Parameter] | Tensor = None,
        representation: List[Parameter] | Tensor = None,
        **kwargs,
    ) -> Tuple[Tensor | None, dict | None]:
        if isinstance(losses, dict):
            losses = torch.stack(tuple(losses.values()))
        self.set_graddrop_gradients(losses, shared_parameters, task_specific_parameters)
        return torch.mean(losses), {}  # NOTE: to align with all other weight methods
