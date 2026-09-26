from typing import List, Tuple, Union
from .algo_callback import AlgoCallback
import torch
from torch import Tensor
from src.callbacks.methods.utils.min_norm_solvers import (
    MinNormSolver,
    gradient_normalizers,
)
from torch.nn.parameter import Parameter
import numpy as np


class MGDA(AlgoCallback):
    """Based on the official implementation of: Multi-Task Learning as Multi-Objective Optimization
    Ozan Sener, Vladlen Koltun
    Neural Information Processing Systems (NeurIPS) 2018
    https://github.com/intel-isl/MultiObjectiveOptimization
    """

    def __init__(self, num_tasks, params="shared", normalization="none"):
        super().__init__(num_tasks)
        self.solver = MinNormSolver()
        assert params in ["shared", "last", "rep"]
        self.params = params
        assert normalization in ["norm", "loss", "loss+", "none"]
        self.normalization = normalization

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1) for i, g in enumerate(grad)), dim=0)

    def get_weighted_loss(
        self,
        losses: Tensor,
        shared_parameters: List[Parameter] | Tensor,
        task_specific_parameters: List[Parameter] | Tensor,
        last_shared_parameters: List[Parameter] | Tensor,
        representation: Parameter | Tensor,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        if isinstance(losses, dict):
            losses = torch.stack(tuple(losses.values()))
        # Our code
        grads = {}
        params = dict(rep=representation, shared=shared_parameters, last=last_shared_parameters)[self.params]
        for i, loss in enumerate(losses):
            g = list(torch.autograd.grad(loss, params, retain_graph=True))
            # Normalize all gradients, this is optional and not included in the paper.

            grads[i] = [torch.flatten(grad) for grad in g]

        gn = gradient_normalizers(grads, losses, self.normalization)
        for t in range(self.num_tasks):
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        sol, min_norm = self.solver.find_min_norm_element([grads[t] for t in range(len(grads))])
        sol = sol * self.num_tasks  # make sure it sums to self.num_tasks
        weighted_loss = sum([losses[i] * sol[i] for i in range(len(sol))])

        return weighted_loss, dict(weights=torch.from_numpy(sol.astype(np.float32)))

    def __repr__(self) -> str:
        return f"MGDA({self.params}, {self.normalization})"
