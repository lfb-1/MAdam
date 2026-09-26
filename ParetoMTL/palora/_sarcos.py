import logging

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pymoo.indicators.hv import HV

from src.callbacks import get_default_callbacks
from src.callbacks.callback import Callback
from src.callbacks.methods import PFL_METHODS, get_method
from src.callbacks.methods.algo_callback import ParetoFrontApproximationAlgoCallback
from src.callbacks.metrics.mtl_metric_callback import (
    ClassificationMultiTaskMetricCallback,
    RegressionMultiTaskMetricCallback,
)
from src.callbacks.utils.local_training_dynamics import LocalTrainingDynamicsCallback
from src.callbacks.utils.pareto_front_visualizer import ParetoFrontVisualizerCallback
from src.callbacks.utils.save_model import SaveModelCallback
from src.datasets.sarcos import SarcosDataModule
from src.models.factory.mlp import MultiTaskMLP
from src.trainer.base_trainer import BaseTrainer
from src.trainer.ensemble_trainer import EnsembleTrainer
from src.trainer.multi_forward_ensemble_trainer import MultiForwardEnsembleTrainer
from src.utils import safe_pop, set_seed
from src.utils.logging_utils import initialize_wandb, install_logging
from src.utils.losses import MultiTaskMSELoss
from src.utils.results_io import save_test_results


class SarcosCallback(Callback):

    def __init__(self):
        super().__init__()
        self.ref_point = np.array([1, 5, 1, 1, 10, 10, 1])
        self.num_tasks = 7
        self.indicator = HV(ref_point=np.array(self.ref_point))

    def on_after_validating_interpolations(self, trainer: "BaseTrainer", *args, **kwargs):
        results = trainer.results
        results = [[v for k, v in r.items() if "Task" in k] for r in results.values()]
        results = np.array(results)
        hv = self.indicator(results)

        rays = np.array([v.numpy() if hasattr(v, 'numpy') else np.array(v) for v in trainer.eval_protocol.points.values()])
        losses = results * rays
        losses = losses / np.sum(losses, axis=1, keepdims=True)
        m = self.num_tasks
        non_uniformity = np.sum(losses * np.log2(losses * m), axis=1).mean()

        print("VALIDATION", hv, non_uniformity)
        wandb.log(
            {
                "val/HV": hv,
                "val/non_uniformity": non_uniformity,
                "epoch": trainer.current_epoch,
            }
        )

    def on_after_eval_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        if not isinstance(trainer.method, ParetoFrontApproximationAlgoCallback):
            results = trainer.val_metrics
            results = {k: v for k, v in results.items() if "Task" in k}
            results = dict(sorted(results.items()))
            results = np.array([v for k, v in results.items()])
            hv = self.indicator(results)
            wandb.log({"val/HV": hv, "epoch": trainer.current_epoch})
            print("VALIDATION", hv)


@hydra.main(config_path="configs/experiment/sarcos", config_name="sarcos")
def my_app(config: DictConfig) -> None:
    install_logging()
    logging.info(OmegaConf.to_yaml(config))
    set_seed(config.seed)
    initialize_wandb(config)

    dm = SarcosDataModule(batch_size=config.data.batch_size, num_workers=config.data.num_workers)
    logging.info(f"I am using the following benchmark {dm.name}")

    if config.method.name in PFL_METHODS:
        ray_sampler = instantiate(safe_pop(config.ray_sampler), num=config.method.num, num_tasks=dm.num_tasks)
        weight_method = get_method(
            config.method.name, **safe_pop(config.method), ray_sampler=ray_sampler, num_tasks=dm.num_tasks
        )
    else:
        weight_method = get_method(config.method.name, dm.num_tasks, **safe_pop(config.method))
    logging.info(f"I am using the following method {weight_method}")

    model = MultiTaskMLP(
        in_features=dm.num_features,
        task_names=dm.task_names,
        encoder_specs=config.model.encoder_specs,
        decoder_specs=config.model.decoder_specs,
    )

    model = weight_method.configure_model(model, data_module=dm)
    param_groups = weight_method.configure_param_groups(model, lr=config.optimizer.lr)


    _opt_type = getattr(config.optimizer, 'type', 'Adam')
    _opt_lr = config.optimizer.lr
    _opt_momentum = getattr(config.optimizer, 'momentum', 0.9)
    if str(_opt_type).upper() == 'SGD':
        optimizer = torch.optim.SGD(param_groups, lr=_opt_lr, momentum=_opt_momentum)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=_opt_lr)

    metric_cb = RegressionMultiTaskMetricCallback(use_task_names=True)
    callbacks = [
        metric_cb,
        *get_default_callbacks(),
        LocalTrainingDynamicsCallback("sarcos", config),
        SarcosCallback(),
    ]

    trainer_kwargs = dict(
        model=model,
        benchmark=dm,
        optimizer=optimizer,
        gpu=0,
        callbacks=callbacks,
        loss_fn=MultiTaskMSELoss(),
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
    test_metrics = trainer.predict(dm.test_dataloader())
    print(test_metrics)
    sigma = trainer.method.compute_sigma() if hasattr(trainer.method, 'compute_sigma') else None
    save_test_results(test_metrics, config, "sarcos", sigma=sigma, trainer=trainer)

    wandb.finish(quiet=True)


if __name__ == "__main__":
    my_app()
