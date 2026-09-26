import logging


def install_logging(*args, **kwargs):
    from .logging_utils import install_logging as _install_logging

    return _install_logging(*args, **kwargs)


def initialize_wandb(*args, **kwargs):
    from .logging_utils import initialize_wandb as _initialize_wandb

    return _initialize_wandb(*args, **kwargs)


def set_seed(seed=-1):
    """for reproducibility
    :param seed:
    :return:
    """
    import random

    import numpy as np
    import torch

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    if seed != -1:
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    logging.info(f"Setting seed to {torch.initial_seed()}")


def safe_pop(config, key="name"):
    if getattr(config, key, None) is not None:
        config = dict(config)
        config.pop(key)
        return config
    return config


__all__ = ["set_seed", "safe_pop", "install_logging", "initialize_wandb"]
