import os

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from src.utils.loggers.base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self,
        entity=None,
        project=None,
        config=None,
        disabled=True,
    ) -> None:
        super().__init__()

        if disabled:
            wandb.init(mode="disabled")
        else:
            if config:
                config = OmegaConf.to_container(config)
            wandb.init(
                entity=entity,
                project=project,
                config=config,
                mode="online",
            )
            # if HydraConfig.initialized():
            #     orig_cwd = hydra.utils.get_original_cwd()
            #     wandb.run.log_code(orig_cwd, exclude_fn=lambda path, root: "logs" in os.path.relpath(path, root))

    def log(self, key, value):
        wandb.log({key: value})

    def terminate(self):
        wandb.finish(quiet=True)
