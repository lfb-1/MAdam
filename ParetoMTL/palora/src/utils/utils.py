import torch
from torch.nn.modules.module import Module


def safe_pop(config, key="name"):
    if getattr(config, key, None) is not None:
        config = dict(config)
        config.pop(key)
        return config
    return config


class DumbWrapper(torch.nn.Module):
    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x, ray=None, *args, **kwargs):
        if isinstance(self.module, DumbWrapper):
            return self.module(x, ray, *args, **kwargs)

        return self.module(x, *args, **kwargs)

    def __repr__(self) -> str:
        return "DUMB" + self.module.__repr__()

    def get_last_layer(self):
        if isinstance(self.module, DumbWrapper):
            return self.module.get_last_layer()
        return self.module

    def __getitem__(self, key):
        return self.module[key]
