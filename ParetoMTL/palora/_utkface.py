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
from src.callbacks.metrics.mtl_metric_callback import UTKFaceMultiTaskMetricCallback
from src.callbacks.utils.pareto_front_visualizer3d import ParetoFrontVisualizer3dCallback
from src.callbacks.utils.save_model import SaveModelCallback
from src.datasets import UTKFaceDataModule
from src.models.base_model import SharedBottom
from src.models.factory.resnet import UtkFaceResnet
from src.trainer.base_trainer import BaseTrainer
from src.trainer.ensemble_trainer import EnsembleTrainer
from src.trainer.multi_forward_ensemble_trainer import MultiForwardEnsembleTrainer
from src.utils import initialize_wandb, install_logging, safe_pop, set_seed
from src.utils.losses import UTKFaceMultiTaskLoss
from src.utils.results_io import save_test_results


@hydra.main(config_path="configs/experiment/utkface", config_name="utkface")
def my_app(config: DictConfig) -> None:
    install_logging()
    logging.info(OmegaConf.to_yaml(config))
    set_seed(config.seed)

    initialize_wandb(config)

    dm: UTKFaceDataModule = instantiate(safe_pop(config.data, key="dataset"))
    logging.info(f"I am using the following benchmark {dm.name}")

    if config.method.name in PFL_METHODS:
        ray_sampler = instantiate(safe_pop(config.ray_sampler), num=config.method.num, num_tasks=dm.num_tasks)

        weight_method = get_method(
            config.method.name, **safe_pop(config.method), ray_sampler=ray_sampler, num_tasks=dm.num_tasks
        )
    else:
        weight_method = METHODS[config.method.name](dm.num_tasks, **safe_pop(config.method))
    logging.info(f"I am using the following method {weight_method}")

    model = UtkFaceResnet()
    model = weight_method.configure_model(model, data_module=dm)

    param_groups = weight_method.configure_param_groups(model, lr=config.optimizer.lr)

    _opt_type = getattr(config.optimizer, 'type', 'Adam')
    _opt_lr = config.optimizer.lr
    _opt_momentum = getattr(config.optimizer, 'momentum', 0.9)
    if str(_opt_type).upper() == 'SGD':
        optimizer = torch.optim.SGD(param_groups, lr=_opt_lr, momentum=_opt_momentum)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=_opt_lr)

    metric_cb = UTKFaceMultiTaskMetricCallback()
    callbacks = [
        metric_cb,
        *get_default_callbacks(),
        SaveModelCallback(epoch_frequency=25),
        ParetoFrontVisualizer3dCallback(),
    ]

    label_smoothing = OmegaConf.select(config, 'label_smoothing', default=0.0)
    fim_label_smoothing = OmegaConf.select(config, 'fim_label_smoothing', default=0.0)
    adaptive_label_smoothing = OmegaConf.select(config, 'adaptive_label_smoothing', default=0.0)
    if label_smoothing > 0:
        logging.info(f"Global label smoothing enabled: {label_smoothing}")
    if fim_label_smoothing > 0:
        logging.info(f"FIM-only label smoothing enabled: {fim_label_smoothing}")
    if adaptive_label_smoothing > 0:
        logging.info(f"Lambda-adaptive label smoothing enabled: {adaptive_label_smoothing}")

    trainer_kwargs = dict(
        model=model,
        benchmark=dm,
        optimizer=optimizer,
        gpu=0,
        callbacks=callbacks,
        loss_fn=UTKFaceMultiTaskLoss(
            label_smoothing=label_smoothing,
            fim_label_smoothing=fim_label_smoothing,
            adaptive_label_smoothing=adaptive_label_smoothing,
            entropy_reg=config.get("entropy_reg", 0.0),
            elr_lambda=config.get("elr_lambda", 0.0),
            elr_beta=config.get("elr_beta", 0.7),
            fim_temperature=config.get("fim_temperature", 1.0),
        ),
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

    print(trainer)
    trainer.fit(epochs=config.training.epochs)
    test_metrics = trainer.predict(dm.test_dataloader())
    sigma = trainer.method.compute_sigma() if hasattr(trainer.method, 'compute_sigma') else None
    save_test_results(test_metrics, config, "utkface", sigma=sigma, trainer=trainer)

    wandb.finish(quiet=True)


if __name__ == "__main__":
    my_app()
