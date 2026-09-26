from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from src.callbacks.base_callback import BaseCallback
from src.utils.utils import DumbWrapper

from .utils.samplers import Sampler

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class AlgoCallback(BaseCallback):
    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks

    def connect(self, trainer: "BaseTrainer"):
        self.device = trainer.device

    @staticmethod
    def cast_losses_to_correct_type(losses: Union[dict, List[Tensor], Tensor]):
        if isinstance(losses, Tensor):
            return losses
        elif isinstance(losses, (list, tuple)):
            return torch.stack(losses)
        elif isinstance(losses, dict):
            return torch.stack(list(losses.values()))
        else:
            raise NotImplementedError

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[Parameter], torch.Tensor],
        task_specific_parameters: Union[List[Parameter], torch.Tensor],
        last_shared_parameters: Union[List[Parameter], torch.Tensor],
        representation: Union[Parameter, torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        pass

    def configure_model(self, model, data_module):
        # import torch.nn as nn

        # def augment_model(module: torch.nn.Module, name=""):
        #     for name, immediate_child_module in module.named_children():
        #         if isinstance(immediate_child_module, nn.ModuleDict):
        #             for k, v in immediate_child_module.items():
        #                 if not isinstance(v, DumbWrapper):
        #                     immediate_child_module[k] = DumbWrapper(module=v)
        #                     augment_model(immediate_child_module[k], k)

        #         elif not isinstance(immediate_child_module, DumbWrapper):
        #             layer = getattr(module, name)
        #             setattr(module, name, DumbWrapper(module=layer))
        #             augment_model(immediate_child_module, name)

        # augment_model(model)
        # print(model)
        return model

    def compute_loss(self):
        raise NotImplementedError

    def count_parameters(self):
        # also calculate number of additional parameters
        raise NotImplementedError

    def on_before_fit(self, *args, **kwargs):
        return super().on_before_fit(*args, **kwargs)

    def configure_param_groups(self, model: torch.nn.Module, lr=None):
        return [{"params": model.parameters()}] + [{"params": self.parameters()}]

    def parameters(self):
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[Parameter], torch.Tensor] = None,
        representation: Union[List[Parameter], torch.Tensor] = None,
        regularization_loss: Optional[torch.Tensor] = None,
        scaler=None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        losses = self.cast_losses_to_correct_type(losses)

        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        scaler.scale(loss).backward()
        return loss, extra_outputs


class ParetoFrontApproximationAlgoCallback(AlgoCallback):
    def __init__(self, num_tasks, ray_sampler: Sampler):
        super().__init__(num_tasks)
        self.ray_sampler = ray_sampler
