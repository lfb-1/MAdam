import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.models.base_model import SharedBottom
from src.models.factory.cosmos.upsampler import Upsampler

from .algo_callback import ParetoFrontApproximationAlgoCallback
from .utils.samplers import Sampler

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class COSMOS(ParetoFrontApproximationAlgoCallback):
    def __init__(self, num_tasks, ray_sampler: Sampler, num: int, reg_coefficient=0):
        super().__init__(num_tasks=num_tasks, ray_sampler=ray_sampler)
        self.reg_coefficient = reg_coefficient
        self.num = num

    def get_weighted_loss(self, losses, **kwargs):
        if not isinstance(losses, torch.Tensor):
            losses = torch.stack(losses)
        self.alpha = self.alpha.to(losses.device)
        loss = torch.sum(losses * self.alpha)
        cossim = torch.nn.functional.cosine_similarity(losses, self.alpha, dim=0)
        loss -= self.reg_coefficient * cossim
        return loss, dict(weights=self.alpha, cossim=cossim)

    @staticmethod
    def get_inner_module(module: nn.Module):
        try:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                return module
            else:
                return COSMOS.get_inner_module(list(module.children())[0])
        except:
            print("The first nested module must be either a Linear or Conv2d module.")

    def configure_model(self, model, data_module):
        # augment first layer dimensions and add Upsampler
        # print(model)
        # assert isinstance(model, SharedBottom)
        module = self.get_inner_module(model)
        logging.info(f"Augmenting {module} input dimension with {self.num_tasks} tasks.")
        if isinstance(module, nn.Linear):
            input_dim = 1
            out_features, in_features = module.weight.shape
            module.in_features = in_features + self.num_tasks
            module.weight.data = torch.randn([out_features, in_features + self.num_tasks])
            module.reset_parameters()
        elif isinstance(module, nn.Conv2d):
            input_dim = 3
            out_channels, in_channels, kernel_size, _ = module.weight.shape
            module.in_channels = in_channels + self.num_tasks
            module.weight.data = torch.randn([out_channels, in_channels + self.num_tasks, kernel_size, kernel_size])
            module.reset_parameters()

        return Upsampler(K=self.num_tasks, child_model=model, input_dim=data_module.input_dims)

    # def set_alpha(self, alpha, model=None):
    #     self.alpha = alpha

    # def on_before_forward(self, trainer: "BaseTrainer", *args, **kwargs):
    #     super().on_before_forward(trainer, *args, **kwargs)
    #     trainer.model.alpha = self.alpha
