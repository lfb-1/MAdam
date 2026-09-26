import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import wandb
from src.callbacks.methods.ll.palora_modules import (
    PaConv2d,
    PaLinear,
    PaLoRALayer,
    PaSequential,
)
from src.utils.utils import DumbWrapper

from .algo_callback import ParetoFrontApproximationAlgoCallback
from .fim_support import FIMSupportMixin
from .utils.samplers import Sampler

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class PaLoRA(FIMSupportMixin, ParetoFrontApproximationAlgoCallback):

    def __init__(
        self,
        num_tasks: int,
        ray_sampler: Sampler,
        rank: int,
        num: int,
        reg_coefficient: float = 0,
        reweight_lr=True,
        reinit_flag=True,
        lora_alpha=1,
        fim_method: str = 'none',
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
        self.rank = rank
        self.r = rank
        self.reweight_lr = reweight_lr
        self.lora_alpha = lora_alpha
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
            log_prefix="PaLoRA",
            extra_kwargs=kwargs,
        )


    def configure_model(self, model, data_module=None, keep_original_weights=False) -> nn.Module:
        logging.info(f"Configuring model for PaLoRA with lora_alpha={self.lora_alpha}")
        self.lorafy_model(model, keep_original_weights=keep_original_weights)
        if self.exclude_bn_from_fim and self.fim_method != 'none':
            bn_params = [n for n, p in model.named_parameters() if p.requires_grad and self._is_bn_param(n)]
            logging.info(f"BN exclusion from FIM: {len(bn_params)} params excluded: {bn_params[:5]}...")
        return model

    def configure_param_groups(self, model: torch.nn.Module, lr: Optional[float] = None):
        # return super().configure_param_groups(model)
        if self.reweight_lr:
            logging.info("Using different learning rates for LoRA parameters")
            param_groups = [{"params": m} for k, m in model.named_parameters() if "lora" not in k]
            param_groups += [
                {"params": m, "lr": lr * self.num_tasks} for k, m in model.named_parameters() if "lora" in k
            ]
            return param_groups

        else:
            return super().configure_param_groups(model, lr=lr)

    def compute_cosine_loss(self, trainer: "BaseTrainer") -> torch.Tensor:
        model = trainer.model

        loss = 0

        num_layers = 0
        for name, module in model.named_modules():
            if isinstance(module, PaLoRALayer):
                num_layers += 2
                i, j = 0, 1
                i, j = str(i), str(j)
                loss += F.cosine_similarity(module.lora_A[i].flatten(), module.lora_A[j].flatten(), dim=0)
                loss += F.cosine_similarity(module.lora_B[i].flatten(), module.lora_B[j].flatten(), dim=0)

        loss /= num_layers
        return loss

    def get_weighted_loss(
        self,
        losses: Tensor,
        ray: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        losses = self.cast_losses_to_correct_type(losses)
        assert ray.shape[0] == self.num_tasks == len(losses)
        return (ray * losses).sum(), {}

    def lorafy_model(self, module: nn.Module, name="", keep_original_weights=False):
        kwargs = dict(num_members=self.num_tasks, r=self.r, lora_alpha=self.lora_alpha)
        for name, immediate_child_module in module.named_children():
            if isinstance(immediate_child_module, nn.Conv2d):
                layer = getattr(module, name)
                setattr(module, name, PaConv2d.from_module(module=layer, **kwargs))
                if keep_original_weights:
                    getattr(module, name).conv.weight.data = layer.weight.data
                    getattr(module, name).conv.bias.data = layer.bias.data
            elif isinstance(immediate_child_module, nn.Linear):
                layer = getattr(module, name)
                setattr(module, name, PaLinear.from_module(layer, **kwargs))
                if keep_original_weights:
                    getattr(module, name).weight.data = layer.weight.data
                    getattr(module, name).bias.data = layer.bias.data
            elif isinstance(immediate_child_module, nn.Sequential):
                layer = getattr(module, name)
                setattr(module, name, PaSequential.from_module(layer))
                for i, child in enumerate(immediate_child_module):
                    self.lorafy_model(
                        getattr(module, name),
                        name,
                        keep_original_weights=keep_original_weights,
                    )
            elif isinstance(immediate_child_module, PaConv2d):
                break
            elif isinstance(immediate_child_module, PaLinear):
                break
            elif isinstance(immediate_child_module, nn.ReLU):
                layer = getattr(module, name)
                setattr(module, name, DumbWrapper(module=layer))
            else:
                self.lorafy_model(
                    immediate_child_module,
                    name,
                    keep_original_weights=keep_original_weights,
                )

    def on_before_backward(self, trainer: "BaseTrainer", *args, **kwargs):
        if self.reg_coefficient > 0:
            cosine_loss = self.compute_cosine_loss(trainer)
            trainer.loss += self.reg_coefficient * cosine_loss
            trainer.cosine_loss = cosine_loss
            wandb.log({"cosine_loss": cosine_loss.item(), "mystep": trainer.current_step})

        super().on_before_backward(trainer, *args, **kwargs)

class PaLoRA_LB(PaLoRA):
    def __init__(self, num_tasks, iteration_window: int = 25, temp=2.0, **kwargs):
        super().__init__(num_tasks=num_tasks, **kwargs)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, num_tasks), dtype=np.float32)
        self.weights = np.ones(num_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, shared_parameters, ray: Tensor, **kwargs):
        losses = self.cast_losses_to_correct_type(losses)
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[: self.iteration_window, :].mean(0)
            ws = self.costs[self.iteration_window :, :].mean(0)
            self.weights = 1 - np.abs(ws) / np.abs(ws).sum()

        runnin_avg_task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(losses.device)
        loss = (runnin_avg_task_weights * losses * ray).sum()

        self.running_iterations += 1

        return loss, dict(weights=ray)


class PaLoRA_GB(PaLoRA):

    # def get_weighted_loss(self, losses: Tensor, ray: Tensor, **kwargs) -> Tuple[Tensor, dict]:
    #     losses = self.cast_losses_to_correct_type(losses)
    #     assert ray.shape[0] == self.num_tasks == len(losses)
    #     return (ray * losses).sum(), {}

    def get_weighted_loss(self, losses, shared_parameters, ray: Tensor, **kwargs):
        losses = self.cast_losses_to_correct_type(losses)

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
        loss = sum([losses[i] * ray[i] / (norm_terms[i] + xi) for i in range(len(losses))])
        return loss, dict(weights=ray)


class PaLoRAFull(PaLoRA_LB):

    def get_weighted_loss(self, losses, shared_parameters, ray: Tensor, **kwargs):
        losses = self.cast_losses_to_correct_type(losses)

        # STEP 1: loss balancing
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[: self.iteration_window, :].mean(0)
            ws = self.costs[self.iteration_window :, :].mean(0)
            self.weights = 1 - np.abs(ws) / np.abs(ws).sum()

        runnin_avg_task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(losses.device)
        losses = runnin_avg_task_weights * losses

        self.running_iterations += 1

        # STEP 2: gradient balancing
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
        loss = sum([losses[i] * ray[i] / (norm_terms[i] + xi) for i in range(len(losses))])
        return loss, dict(weights=ray)
