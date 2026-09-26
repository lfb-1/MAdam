import logging
from typing import Iterator

import hydra
import torch
import torch.nn as nn
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.callbacks import (
    NYUMetricCallback,
    SaveModelCallback,
    SchedulerCallback,
    get_default_callbacks,
)
from src.callbacks.methods import METHODS, PFL_METHODS, get_method
from src.callbacks.methods.algo_callback import ParetoFrontApproximationAlgoCallback
from src.callbacks.methods.palora import PaLoRA, PaLoRA_GB, PaLoRA_LB
from src.callbacks.methods.pamal import PaMaL, PaMaL_GB, PaMaL_LB
from src.datasets.nyuv2 import NYUv2DataModule
from src.models.base_model import BaseModel
from src.models.factory.deeplab import MTLDeepLabv3
from src.models.factory.segnet_nyu import SegNet, SegNetMtan
from src.trainer import BaseTrainer, EnsembleTrainer, MultiForwardEnsembleTrainer
from src.utils import initialize_wandb, install_logging, safe_pop, set_seed
from src.utils.losses import NYUv2Loss
from src.utils.results_io import save_test_results


class Wrapper(BaseModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, return_embedding=False):
        return self.model(x), None

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.model.shared_parameters()

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.model.task_specific_parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.model.last_shared_parameters()


LOGGING_FREQ = 3


@hydra.main(config_path="configs/experiment/nyuv2", config_name="nyuv2")
def my_app(config: DictConfig) -> None:
    import warnings

    initialize_wandb(config)

    warnings.filterwarnings(
        "ignore",
        message="Note that order of the arguments: ceil_mode and return_indices will change",
    )
    install_logging()
    logging.info(OmegaConf.to_yaml(config))
    set_seed(config.seed)

    dm = NYUv2DataModule(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    if config.method.name in PFL_METHODS:
        ray_sampler = instantiate(safe_pop(config.ray_sampler), num=config.method.num, num_tasks=dm.num_tasks)

        weight_method = get_method(
            config.method.name, **safe_pop(config.method), ray_sampler=ray_sampler, num_tasks=dm.num_tasks
        )
    else:

        weight_method = get_method(config.method.name, dm.num_tasks, **safe_pop(config.method))
    logging.info(f"I am using the following method {weight_method}")

    logging.info(f"I am using the following benchmark {dm.name}")
    tasks = {"sem": 13, "depth": 1, "normal": 3}
    model = dict(segnetmtan=SegNetMtan(), segnet=SegNet(), deeplab=MTLDeepLabv3(tasks=tasks))[config.model.type]

    model = weight_method.configure_model(model, dm)
    param_groups = weight_method.configure_param_groups(model, lr=config.optimizer.lr)

    # logging.info(model)
    optimizer = torch.optim.Adam(param_groups, lr=config.optimizer.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler.step,
        gamma=config.scheduler.gamma,
    )
    logging.info(scheduler)

    callbacks = [
        NYUMetricCallback(),
        *get_default_callbacks(),
        SaveModelCallback(epoch_frequency=20),
        SchedulerCallback(scheduler=scheduler),
    ]

    trainer_kwargs = dict(
        model=model,
        benchmark=dm,
        optimizer=optimizer,
        loss_fn=NYUv2Loss(),
        gpu=0,
        callbacks=callbacks,
    )

    if isinstance(weight_method, ParetoFrontApproximationAlgoCallback):
        trainer_kwargs["validate_every_n"] = config.validate_every_n

        if getattr(weight_method, "num", 1) > 1:
            trainer = MultiForwardEnsembleTrainer(method=weight_method, **trainer_kwargs)
        else:
            trainer = EnsembleTrainer(method=weight_method, **trainer_kwargs)
    else:
        trainer = BaseTrainer(method=weight_method, **trainer_kwargs)

    logging.info(trainer)
    trainer.fit(epochs=config.training.epochs)
    test_metrics = trainer.predict(test_loader=dm.test_dataloader())
    sigma = trainer.method.compute_sigma() if hasattr(trainer.method, "compute_sigma") else None
    save_test_results(test_metrics, config, "nyuv2", sigma=sigma, trainer=trainer)

    wandb.finish(quiet=True)


if __name__ == "__main__":
    my_app()
