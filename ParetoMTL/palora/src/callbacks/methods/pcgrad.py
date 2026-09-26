from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import copy
import random
from .algo_callback import AlgoCallback


class PCGrad(AlgoCallback):
    """Modification of: https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py

    @misc{Pytorch-PCGrad,
      author = {Wei-Cheng Tseng},
      title = {WeiChengTseng/Pytorch-PCGrad},
      url = {https://github.com/WeiChengTseng/Pytorch-PCGrad.git},
      year = {2020}
    }

    """

    def __init__(self, num_tasks: int, reduction="sum"):
        super().__init__(num_tasks)
        assert reduction in ["mean", "sum"]
        self.reduction = reduction

    def get_weighted_loss(self, *args, **kwargs) -> Tuple[Tensor, dict]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"PCGrad(reduction={self.reduction})"

    def _set_pc_grads(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[Parameter], torch.Tensor] = None,
    ):
        # shared part
        shared_grads = []
        for i, l in enumerate(losses):
            grads = torch.autograd.grad(l, shared_parameters, retain_graph=True)
            shared_grads.append(grads)

        if isinstance(shared_parameters, torch.Tensor):
            shared_parameters = [shared_parameters]
        non_conflict_shared_grads = self._project_conflicting(shared_grads)
        for p, g in zip(shared_parameters, non_conflict_shared_grads):
            p.grad = g

        # task specific part
        if task_specific_parameters is not None:
            task_specific_grads = torch.autograd.grad(losses.sum(), task_specific_parameters)
            if isinstance(task_specific_parameters, torch.Tensor):
                task_specific_parameters = [task_specific_parameters]
            for p, g in zip(task_specific_parameters, task_specific_grads):
                p.grad = g

    def _project_conflicting(self, grads: List[Tuple[torch.Tensor]]):
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = sum(
                    [torch.dot(torch.flatten(grad_i), torch.flatten(grad_j)) for grad_i, grad_j in zip(g_i, g_j)]
                )
                if g_i_g_j < 0:
                    g_j_norm_square = torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2
                    for grad_i, grad_j in zip(g_i, g_j):
                        grad_i -= g_i_g_j * grad_j / g_j_norm_square

        merged_grad = [sum(g) for g in zip(*pc_grad)]
        if self.reduction == "mean":
            merged_grad = [g / self.num_tasks for g in merged_grad]

        return merged_grad

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[Parameter], torch.Tensor] = None,
        shared_parameters: Union[List[Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[Parameter], torch.Tensor] = None,
        **kwargs,
    ):
        if isinstance(losses, dict):
            losses = torch.stack(tuple(losses.values()))
        self._set_pc_grads(losses, shared_parameters, task_specific_parameters)
        return torch.mean(losses), {}  # NOTE: to align with all other weight methods
