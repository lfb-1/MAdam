import logging
from inspect import signature

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.callbacks import get_default_callbacks
from src.callbacks.methods import METHODS, PFL_METHODS, get_method
from src.callbacks.methods.algo_callback import ParetoFrontApproximationAlgoCallback
from src.callbacks.methods.palora import PaLoRA
from src.callbacks.metrics.mtl_metric_callback import (
    ClassificationMultiTaskMetricCallback,
)
from src.callbacks.utils.pareto_front_visualizer import ParetoFrontVisualizerCallback
from src.callbacks.utils.pareto_front_visualizer3d import ParetoFrontVisualizer3dCallback
from src.callbacks.utils.save_model import SaveModelCallback
from src.datasets import MultiMnistThreeDataModule
from src.models.base_model import SharedBottom
from src.models.factory.lenet import MultiLeNetO, MultiLeNetR
from src.trainer.base_trainer import BaseTrainer
from src.trainer.ensemble_trainer import EnsembleTrainer
from src.trainer.multi_forward_ensemble_trainer import MultiForwardEnsembleTrainer
from src.utils import initialize_wandb, install_logging, safe_pop, set_seed
from src.utils.losses import MultiTaskCrossEntropyLoss


@hydra.main(config_path="configs/experiment/multimnist3", config_name="multimnist3")
def my_app(config: DictConfig) -> None:
    install_logging()
    logging.info(OmegaConf.to_yaml(config))
    set_seed(config.seed)

    initialize_wandb(config)
    wandb.run.tags = ("ablation3",)

    dm: MultiMnistThreeDataModule = instantiate(safe_pop(config.data, key="dataset"))
    logging.info(f"I am using the following benchmark {dm.name}")

    if config.method.name in PFL_METHODS:
        ray_sampler = instantiate(safe_pop(config.ray_sampler), num=config.method.num, num_tasks=dm.num_tasks)

        weight_method = get_method(
            config.method.name, **safe_pop(config.method), ray_sampler=ray_sampler, num_tasks=dm.num_tasks
        )
    else:
        weight_method = get_method(config.method.name, dm.num_tasks, **safe_pop(config.method))
    logging.info(f"I am using the following method {weight_method}")

    model = SharedBottom(
        encoder=MultiLeNetR(in_channels=1),
        decoder=MultiLeNetO(),
        task_names=dm.task_names,
    )
    model = weight_method.configure_model(model, data_module=dm)

    param_groups = weight_method.configure_param_groups(model, lr=config.optimizer.lr)
    optimizer = torch.optim.Adam(param_groups, lr=config.optimizer.lr)

    metric_cb = ClassificationMultiTaskMetricCallback(use_task_names=True)
    callbacks = [
        metric_cb,
        *get_default_callbacks(),
        SaveModelCallback(epoch_frequency=20),
        ParetoFrontVisualizer3dCallback(),
        # *get_verbose_callbacks(),
    ]

    trainer_kwargs = dict(
        model=model,
        benchmark=dm,
        optimizer=optimizer,
        gpu=0,
        callbacks=callbacks,
        loss_fn=MultiTaskCrossEntropyLoss(),
    )

    if isinstance(weight_method, ParetoFrontApproximationAlgoCallback):
        trainer_kwargs["validate_every_n"] = config.validate_every_n
        trainer_kwargs["validate_models"] = config.method.validate_models
        if getattr(weight_method, "num", 1) > 1:
            trainer = MultiForwardEnsembleTrainer(method=weight_method, **trainer_kwargs)
        else:
            trainer = EnsembleTrainer(method=weight_method, **trainer_kwargs)
    else:
        trainer = BaseTrainer(method=weight_method, **trainer_kwargs)

    trainer.fit(epochs=config.training.epochs)
    trainer.predict(dm.test_dataloader())

    wandb.finish(quiet=True)


if __name__ == "__main__":
    my_app()
