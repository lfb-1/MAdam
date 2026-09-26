import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .algo_callback import ParetoFrontApproximationAlgoCallback
from .utils.samplers import Sampler
from peft import LoraConfig, get_peft_model, LoHaConfig, LoKrConfig
import numpy as np

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class PaDoRA(ParetoFrontApproximationAlgoCallback):
    import peft

    peft.tuners.lora.LoraLayer

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
        peft="lora",
        lora_alpha_multiplier=None,
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
        if lora_alpha_multiplier is not None:
            self.lora_alpha = lora_alpha_multiplier * self.rank

        logging.info(f"Using lora_alpha={lora_alpha}")
        assert peft in ["lora", "loha", "dora", "lokr"]
        self.peft = peft

    def configure_model(self, model, data_module=None, keep_original_weights=False) -> nn.Module:
        self.task_names = data_module.task_names
        modules = {k: v for k, v in model.named_modules() if isinstance(v, (torch.nn.Conv2d, torch.nn.Linear))}

        if self.peft == "loha":
            config = LoHaConfig(
                r=self.rank,
                alpha=self.lora_alpha,
                target_modules=list(modules.keys()),
            )
        elif self.peft in ["dora", "lora"]:
            config = LoraConfig(
                r=self.rank,
                lora_alpha=self.lora_alpha,
                target_modules=list(modules.keys()),
                lora_dropout=0.0,
                use_dora=self.peft == "dora",
            )
        elif self.peft == "lokr":
            config = LoKrConfig(
                r=self.rank,
                alpha=self.lora_alpha,
                target_modules=list(modules.keys()),
                module_dropout=0.0,
            )
        else:
            raise ValueError(f"Unknown peft method {self.peft}")

        import pprint

        logging.info(f"I am using the following config:\n{pprint.pformat(config)}")

        logging.info(f"Configuring model for PaDoRA with lora_alpha={self.lora_alpha}")
        model = get_peft_model(model, config, adapter_name=data_module.task_names[0])
        for task in data_module.task_names[1:]:
            model.add_adapter(task, config)

        logging.info("Activating adapters for all tasks: {}".format(data_module.task_names))

        model.base_model.set_adapter(data_module.task_names)

        import copy

        for _, layer in model.named_modules():
            if hasattr(layer, "scaling"):
                setattr(layer, "original_scaling", copy.deepcopy(layer.scaling))

        for p in model.parameters():
            p.requires_grad = True

        return model

    # def configure_param_groups(self, model: torch.nn.Module, lr: Optional[float] = None):
    #     # return super().configure_param_groups(model)
    #     if self.reweight_lr:
    #         logging.info("Using different learning rates for LoRA parameters")
    #         param_groups = [{"params": m} for k, m in model.named_parameters() if "lora" not in k]
    #         param_groups += [
    #             {"params": m, "lr": lr * self.num_tasks} for k, m in model.named_parameters() if "lora" in k
    #         ]
    #         return param_groups

    #     else:
    #         return super().configure_param_groups(model, lr=lr)

    def on_before_forward(self, trainer: "BaseTrainer"):
        ray = self.ray
        ray = dict(zip(self.task_names, ray))
        setattr(trainer.model, "ray", ray)
        for _, layer in trainer.model.named_modules():
            if hasattr(layer, "scaling"):
                assert ray.keys() == layer.original_scaling.keys()
                # previous_scaling = layer.scaling
                layer.scaling = {k: (layer.original_scaling[k] * ray[k]).item() for k in ray.keys()}

                # logging.info(layer.original_scaling, layer.scaling)

    def on_before_eval_epoch(self, trainer: "BaseTrainer", *args, **kwargs):
        module = next(trainer.model.modules())
        if hasattr(module, "ray"):
            ray = module.ray
            # logging.info(ray)
        return super().on_before_eval_epoch(*args, **kwargs)

    # def compute_cosine_loss(self, trainer: "BaseTrainer") -> torch.Tensor:
    #     model = trainer.model

    #     loss = 0

    #     num_layers = 0
    #     for name, module in model.named_modules():
    #         if isinstance(module, PaLoRALayer):
    #             num_layers += 2
    #             i, j = 0, 1
    #             i, j = str(i), str(j)
    #             loss += F.cosine_similarity(module.lora_A[i].flatten(), module.lora_A[j].flatten(), dim=0)
    #             loss += F.cosine_similarity(module.lora_B[i].flatten(), module.lora_B[j].flatten(), dim=0)

    #     loss /= num_layers
    #     return loss

    # def on_before_backward(self, trainer: "BaseTrainer", *args, **kwargs):
    #     if self.reg_coefficient > 0:
    #         cosine_loss = self.compute_cosine_loss(trainer)
    #         trainer.loss += self.reg_coefficient * cosine_loss
    #         trainer.cosine_loss = cosine_loss

    #         wandb.log({"cosine_loss": cosine_loss.item(), "mystep": trainer.current_step})

    def get_weighted_loss(
        self,
        losses: Tensor,
        ray: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        losses = self.cast_losses_to_correct_type(losses)
        assert ray.shape[0] == self.num_tasks == len(losses)
        return (ray * losses).sum(), {}


class PaDoRA_LB(PaDoRA):
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


class PaDoRA_GB(PaDoRA):

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
            g = [torch.zeros_like(p) if grad is None else grad for p, grad in zip(shared_parameters, g)]
            grad = torch.cat([torch.flatten(grad) for grad in g])
            norm_term = torch.norm(grad)
            norm_terms[i] = norm_term

            grads[i] = grad
            norm_grads[i] = grad / norm_term

        xi = 0.001
        loss = sum([losses[i] * ray[i] / (norm_terms[i] + xi) for i in range(len(losses))])
        return loss, dict(weights=ray)


class PaDoRAFull(PaDoRA_LB):

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
