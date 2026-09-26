import logging
import os

import coloredlogs
import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def install_logging(level=logging.DEBUG):
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES
    level_styles["info"] = {"color": "yellow"}

    coloredlogs.install(
        level=level,
        fmt="%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s",
        level_styles=level_styles,
    )

    # hide the gazillion cryptic debug messages coming from PIL
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.category").disabled = True
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # plt.set_loglevel("WARNING")


def initialize_wandb(config):
    tags = getattr(config.wandb, "tags", None)
    if tags is None:
        tags = []
    elif isinstance(tags, str):
        tags = [tags]
    else:
        tags = list(tags)

    slurm_job_id = os.environ.get("SLURM_JOB_ID") or getattr(config.wandb, "slurm_job_id", None)
    if slurm_job_id is not None:
        slurm_tag = f"slurm:{slurm_job_id}"
        if slurm_tag not in tags:
            tags.append(slurm_tag)

    init_kwargs = dict(
        entity=getattr(config.wandb, "entity", None),
        project=getattr(config.wandb, "project", None),
        config=OmegaConf.to_container(config),
        # group=getattr(config.wandb, "group", None),
        group=getattr(config.data, "dataset", None),
        mode=getattr(config.wandb, "mode", "online"),
        tags=tags or None,
    )
    for key in ("name", "id", "dir", "resume"):
        value = getattr(config.wandb, key, None)
        if value is not None:
            init_kwargs[key] = value

    wandb.init(**init_kwargs)
    if HydraConfig.initialized():
        orig_cwd = hydra.utils.get_original_cwd()
        wandb.run.log_code(orig_cwd, exclude_fn=lambda x: "logs/" in x)
