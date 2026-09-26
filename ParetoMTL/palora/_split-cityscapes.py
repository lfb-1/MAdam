import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import os
import wandb
from src.callbacks import (
    CityscapesMetricCallback,
    ParetoFrontVisualizerCallback,
    SchedulerCallback,
    get_default_callbacks,
)
from src.callbacks.methods import METHODS, PFL_METHODS, ParetoFrontApproximationAlgoCallback, get_method
from src.callbacks.methods.palora import PaLoRA, PaLoRA_GB, PaLoRA_LB, PaLoRAFull
from src.callbacks.utils.save_model import SaveModelCallback
from src.datasets import Cityscapes2SplitDataModule
from src.models.factory.segnet_cityscapes import SegNet, SegNetMtan
from src.trainer import BaseTrainer, EnsembleTrainer, MultiForwardEnsembleTrainer
from src.utils import initialize_wandb, install_logging, safe_pop, set_seed
from src.utils.losses import CityscapesTwoTaskLoss
from src.utils.results_io import save_test_results


@hydra.main(config_path="configs/experiment/cityscapes", config_name="cityscapes", version_base="1.3")
def my_app(config: DictConfig) -> None:
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Note that order of the arguments: ceil_mode and return_indices will change",
    )
    install_logging()
    set_seed(config.seed)

    initialize_wandb(config)
    logging.info(OmegaConf.to_yaml(config))

    logging.info("I am logging in the following directory {}".format(hydra.utils.get_original_cwd()))
    logging.info("I am logging in the following directory {}".format(os.getcwd()))

    path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logging.info("I am logging in the following directory {}".format(path))

    dm = Cityscapes2SplitDataModule(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        apply_augmentation=config.data.apply_augmentation,
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
    model = dict(segnetmtan=SegNetMtan(), segnet=SegNet())[config.model.type]

    model = weight_method.configure_model(model, data_module=dm)

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
        CityscapesMetricCallback(),
        *get_default_callbacks(logging_frequency=3),
        SaveModelCallback(epoch_frequency=20),
        SchedulerCallback(scheduler=scheduler),
        ParetoFrontVisualizerCallback(),
    ]

    trainer_kwargs = dict(
        model=model,
        benchmark=dm,
        optimizer=optimizer,
        loss_fn=CityscapesTwoTaskLoss(),
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

    trainer.fit(epochs=config.training.epochs)
    test_metrics = trainer.predict(test_loader=dm.test_dataloader())
    sigma = trainer.method.compute_sigma() if hasattr(trainer.method, "compute_sigma") else None
    save_test_results(test_metrics, config, "cityscapes", sigma=sigma, trainer=trainer)

    logging.info("Logs can be found in the following directory {}".format(hydra.utils.get_original_cwd()))

    wandb.finish(quiet=True)


if __name__ == "__main__":
    my_app()
