import copy
from typing import Iterator

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """The BaseModel class has the functionality of the typical nn.Module from PyTorch as
    well as callback functionality provided from BaseCallback."""

    def __init__(self):
        super().__init__()


class BaseModelWrapper(BaseModel):
    def __init__(self, model, config, *args, **kwargs):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, x):
        return self.model(x)


class SharedBottom(BaseModel):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, task_names: list):
        super().__init__()
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.encoder = encoder
        if isinstance(decoder, list):
            assert len(decoder) == self.num_tasks
            self.decoders = nn.ModuleList(decoder)
        elif isinstance(decoder, dict):
            assert len(decoder) == self.num_tasks
            self.decoders = nn.ModuleDict(decoder)
        else:
            self.decoders = nn.ModuleDict({k: self._create_new(decoder) for k in task_names})

    def _create_new(self, module: nn.Module, reinit=True):
        new_module = copy.deepcopy(module)
        if reinit:
            if hasattr(new_module, "reset_parameters"):
                new_module.reset_parameters()
            else:
                for m in new_module.children():
                    m.reset_parameters()
        return new_module

    def forward(self, x, ray: torch.Tensor = None, return_embedding=False):
        if ray is None:
            embedding = self.encoder(x)
        else:
            embedding = self.encoder(x, ray=ray)

        if ray is None:
            task_outs = {k: d(embedding) for k, d in self.decoders.items()}
        else:
            task_outs = {k: d(embedding, ray=ray) for k, d in self.decoders.items()}

        if return_embedding:
            return task_outs, embedding
        else:
            return task_outs

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.encoder.parameters()

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.decoders.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        # TODO: fix this for MGDA
        return self.encoder.get_last_layer().parameters()
