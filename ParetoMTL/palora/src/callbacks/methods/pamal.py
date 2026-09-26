import logging
from typing import TYPE_CHECKING, Optional, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

from src.callbacks.methods.ll.palora_modules import PaSequential
from src.callbacks.methods.ll.subspace_modules import (
    SubspaceBatchNorm2d,
    SubspaceConv,
    SubspaceLinear,
)
from src.utils.utils import DumbWrapper

from .algo_callback import ParetoFrontApproximationAlgoCallback
from .fim_support import FIMSupportMixin
from .utils.samplers import Sampler

# class PaMaL(ParetoFrontApproximationAlgoCallback):
#     def __init__(
#         self,
#         num_tasks,
#         validate_models,
#         reinit_flag=True,
#         p=1,
#         num=1,
#         reg_coefficient=0,
#         inner_method="ls",
#         **kwargs,
#     ):
#         super().__init__(num_tasks, validate_models)
#         self.reinit_flag = reinit_flag
#         self.p = p
#         self.num = num
#         self.reg_coefficient = reg_coefficient
#         self.inner_method = inner_method
#         logging.error(f"The following keywork arguments are not used: {kwargs}")

#         self.task_weights_list = self.generate_single_task_weights_list(num_tasks)

#     def configure_model(self, model, data_module=None):
#         self.make_subspace_compatible(model)
#         return model

#     def make_subspace_compatible(self, module: nn.Module, name: str = ""):
#         for name, immediate_child_module in module.named_children():
#             if isinstance(immediate_child_module, nn.Conv2d):
#                 m = getattr(module, name)
#                 setattr(module, name, SubspaceConv(m, self.num_tasks, self.reinit_flag))
#             elif isinstance(immediate_child_module, nn.Linear):
#                 m = getattr(module, name)
#                 setattr(module, name, SubspaceLinear(m, self.num_tasks, self.reinit_flag))
#             elif isinstance(immediate_child_module, nn.BatchNorm2d):
#                 m = getattr(module, name)
#                 setattr(module, name, SubspaceBatchNorm2d(m, self.num_tasks, self.reinit_flag))
#             elif isinstance(immediate_child_module, SubspaceConv):
#                 break
#             elif isinstance(immediate_child_module, SubspaceLinear):
#                 break
#             else:
#                 self.make_subspace_compatible(immediate_child_module, name)

#     def set_alpha(self, alpha, model):
#         """Sets alpha on task weights and then traverses the model like a tree to set the alpha on all modules."""
#         self.alpha = alpha
#         self.task_weights = sum([tw * a for tw, a in zip(self.task_weights_list, self.alpha)])
#         self.set_alpha_recursively(alpha=alpha, module=model)

#     def set_alpha_recursively(self, alpha, module: nn.Module):
#         for k, v in module.named_children():
#             if isinstance(v, (SubspaceConv, SubspaceLinear, SubspaceBatchNorm2d)):
#                 setattr(v, f"alpha", alpha)
#             else:
#                 self.set_alpha_recursively(alpha, v)

#     @staticmethod
#     def generate_single_task_weights_list(num_tasks):
#         task_weights_list = np.eye(num_tasks).tolist()
#         task_weights_list = torch.vstack([torch.Tensor(t) for t in task_weights_list])
#         return task_weights_list

#     def get_weighted_loss(
#         self,
#         losses: Tensor,
#         shared_parameters: List[Parameter] | Tensor,
#         task_specific_parameters: List[Parameter] | Tensor,
#         last_shared_parameters: List[Parameter] | Tensor,
#         representation: Parameter | Tensor,
#         **kwargs,
#     ) -> Tuple[Tensor, dict]:
#         losses = self.cast_losses_to_correct_type(losses)
#         self.task_weights = self.task_weights.to(losses.device)
#         return (self.task_weights * losses).sum(), {}


class PaMaL(FIMSupportMixin, ParetoFrontApproximationAlgoCallback):

    def __init__(
        self,
        num_tasks: int,
        ray_sampler: Sampler,
        num: int,
        reg_coefficient: float,
        reweight_lr=True,
        reinit_flag=True,
        fim_method: str = "none",
        fim_ema_decay: float = 0.999,
        fim_eps_num: float = 1e-12,
        fim_eps_damp: float = 1e-6,
        fim_clip_grad_norm: float = 0.0,
        exclude_bn_from_fim: Optional[bool] = None,
        exclude_head_from_fim: Optional[bool] = None,
        fim_bias_correction: bool = False,
        fim_stochastic_pairs: bool = True,
        **kwargs,
    ):
        super().__init__(num_tasks=num_tasks, ray_sampler=ray_sampler)
        self.reinit_flag = reinit_flag
        self.num = num
        self.reg_coefficient = reg_coefficient
        self.reweight_lr = reweight_lr
        self.inner_method = kwargs.pop("inner_method", "ls")
        self.p = kwargs.pop("p", None)
        self.validate_models = kwargs.pop("validate_models", None)
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
            log_prefix="PaMaL",
            extra_kwargs=kwargs,
        )

        # logging.error(f"The following keywork arguments are not used: {kwargs}")
        # self.task_weights_list = self.generate_single_task_weights_list(num_tasks)

    def configure_model(self, model, data_module=None, keep_original_weights=False) -> nn.Module:
        self.make_subspace_compatible(model)

        return model

    def make_subspace_compatible(self, module: nn.Module, name: str = ""):
        for name, immediate_child_module in module.named_children():
            if isinstance(immediate_child_module, nn.Conv2d):
                m = getattr(module, name)
                setattr(module, name, SubspaceConv(m, self.num_tasks, self.reinit_flag))
            elif isinstance(immediate_child_module, nn.Linear):
                m = getattr(module, name)
                setattr(module, name, SubspaceLinear(m, self.num_tasks, self.reinit_flag))
            elif isinstance(immediate_child_module, nn.BatchNorm2d):
                m = getattr(module, name)
                setattr(
                    module,
                    name,
                    SubspaceBatchNorm2d(m, self.num_tasks, self.reinit_flag),
                )
            elif isinstance(immediate_child_module, SubspaceConv):
                break
            elif isinstance(immediate_child_module, SubspaceLinear):
                break
            elif isinstance(immediate_child_module, nn.Sequential):
                layer = getattr(module, name)
                setattr(module, name, PaSequential.from_module(layer))
                self.make_subspace_compatible(getattr(module, name), name)
            else:
                self.make_subspace_compatible(immediate_child_module, name)

        def fix_relus(module: nn.Module):
            if isinstance(module, DumbWrapper):
                return
            for name, immediate_child_module in module.named_children():
                if isinstance(immediate_child_module, nn.ReLU):
                    layer = getattr(module, name)
                    if not isinstance(layer, DumbWrapper):
                        setattr(module, name, DumbWrapper(module=layer))
                else:
                    fix_relus(immediate_child_module)

        fix_relus(module)

    def _resolve_task_weights(self, losses: Tensor, ray: Optional[Tensor] = None) -> Tensor:
        task_weights = getattr(self, "task_weights", None)
        if task_weights is None:
            if ray is None:
                ray = getattr(self, "ray", None)
            if ray is None:
                raise RuntimeError("PaMaL requires a task ray or task_weights before scalarization.")
            task_weights = ray

        if isinstance(task_weights, torch.Tensor):
            task_weights = task_weights.to(losses.device)
        else:
            task_weights = torch.as_tensor(task_weights, device=losses.device)

        self.task_weights = task_weights
        return task_weights

    def get_weighted_loss(
        self,
        losses: Tensor,
        ray: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        losses = self.cast_losses_to_correct_type(losses)
        task_weights = self._resolve_task_weights(losses, ray)
        assert task_weights.shape[0] == self.num_tasks == len(losses)
        return (task_weights * losses).sum(), {}


class PaMaL_GB(PaMaL):
    def get_weighted_loss(self, losses, shared_parameters, **kwargs):
        losses = self.cast_losses_to_correct_type(losses)
        task_weights = self._resolve_task_weights(losses, kwargs.get("ray"))

        grads = {}
        norm_grads = {}
        norm_terms = {}

        for i, loss in enumerate(losses):
            g = list(torch.autograd.grad(loss, shared_parameters, retain_graph=True, allow_unused=True))
            grad = torch.cat([torch.flatten(grad) for grad in g])
            norm_term = torch.norm(grad)
            norm_terms[i] = norm_term

            grads[i] = grad
            norm_grads[i] = grad / norm_term

        xi = 0.001
        loss = sum([losses[i] * task_weights[i] / (norm_terms[i] + xi) for i in range(len(losses))])
        return loss, dict(weights=task_weights)


class PaMaL_LB(PaMaL):
    def __init__(self, num_tasks, iteration_window: int = 25, temp=2.0, **kwargs):
        super().__init__(num_tasks=num_tasks, **kwargs)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, num_tasks), dtype=np.float32)
        self.weights = np.ones(num_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, **kwargs):
        losses = self.cast_losses_to_correct_type(losses)
        task_weights = self._resolve_task_weights(losses, kwargs.get("ray"))
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[: self.iteration_window, :].mean(0)
            ws = self.costs[self.iteration_window :, :].mean(0)
            self.weights = 1 - np.abs(ws) / np.abs(ws).sum()

        runnin_avg_task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(losses.device)
        loss = (runnin_avg_task_weights * losses * task_weights).sum()

        self.running_iterations += 1

        return loss, dict(weights=task_weights)
