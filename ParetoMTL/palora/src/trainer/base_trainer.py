import logging
from typing import List, Optional

import torch

import wandb
from src.callbacks.base_callback import BaseCallback
from src.callbacks.methods.algo_callback import (
    AlgoCallback,
    ParetoFrontApproximationAlgoCallback,
)
from src.callbacks.methods.padora import PaDoRA
from src.datasets.base_data_module import BaseDataModule
from src.models.base_model import SharedBottom
from src.utils.loggers.base_logger import BaseLogger
from src.utils.logging_utils import install_logging

from .callback_hooks import TrainerCallbackHookMixin
from .state_manager import TrainerMode, TrainerStateManagerMixin


class BaseTrainer(TrainerStateManagerMixin, TrainerCallbackHookMixin, BaseCallback):
    validate_zeroshot = False

    def __init__(
        self,
        model: SharedBottom,
        benchmark: BaseDataModule,
        method: AlgoCallback,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        callbacks: Optional[List[BaseCallback]] = None,
        loggers: Optional[BaseLogger] = None,
        use_amp=False,
        gpu=None,
        max_norm=None,
    ) -> None:
        install_logging()
        self.device = "cpu" if gpu is None else f"cuda:{gpu}"

        self.model: SharedBottom = model.to(self.device)
        self.benchmark = benchmark

        self.method = method
        if self.method is not None:
            # method is None in case of ensemble training
            self.method.connect(self)
        # ensure callbacks is a list and do not insert None
        self.callbacks = [] if callbacks is None else list(callbacks)
        if self.method is not None:
            # place method first in the callbacks list
            self.callbacks.insert(0, self.method)
        self.loggers = loggers
        self.setup_callbacks()

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        model_parameter_count = self.count_parameters()
        logging.info(f"Running on {self.device}")
        logging.info(f"The model has {model_parameter_count/1e6:.3f}m parameters")
        if getattr(wandb, "run", None) is not None:
            wandb.config.update(
                {"num_model_parameters": model_parameter_count},
                allow_val_change=True,
            )

        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.max_norm = max_norm

        self.num_tasks = benchmark.num_tasks
        self.task_names = benchmark.task_names

    @property
    def total_train_steps(self) -> int:
        return self.epochs * len(self.train_loader)

    @property
    def t(self) -> float:
        t = (self.current_step + 1) / self.total_train_steps
        if t > 1:
            logging.info(f"Setting t to 1. It was {t}")
            t = 1

        return t

    def count_parameters(self):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])

    def log(self, key: str, value):
        wandb.log({key: value})

    def record_training_dynamics(self, payload: dict):
        recorder = getattr(self, "training_dynamics_recorder", None)
        if recorder is not None:
            recorder.record(payload)

    def setup_callbacks(self):
        if self.callbacks is None:
            self.callbacks = []
        for cb in self.callbacks:
            cb.connect(self)

    def setup(self):
        pass

    def _parse_config(self, config):
        self.config = config
        self.logging_freq = config.logging.freq
        self.epochs = config.training.epochs
        self.num_tasks = config.data.num_tasks

    def forward(self, ray: Optional[torch.Tensor] = None):
        """Calls the forward function of the model"""
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            if isinstance(self.method, ParetoFrontApproximationAlgoCallback) and not isinstance(self.method, PaDoRA):
                output, self.features = self.model(self.x, ray=ray, return_embedding=True)
            else:
                output, self.features = self.model(self.x, return_embedding=True)

        return output

    @staticmethod
    def _unpack_batch_functional(batch, device):
        x = batch[0]
        if isinstance(batch[1], (list, tuple, dict)):
            y = batch[1]
        else:
            y = batch[1:]
        x = x.to(device)

        if isinstance(y, torch.Tensor):
            y = y.to(device)
        elif isinstance(y, tuple) or isinstance(y, list):
            y = tuple(yy.to(device) for yy in y)
        elif isinstance(y, dict):
            y = {k: v.to(device) for k, v in y.items()}
        else:
            raise NotImplementedError

        return x, y

    def _unpack_batch(self, batch):
        self.x, self.y = self._unpack_batch_functional(batch, self.device)

    def zero_grad_optimizer(self):
        self.optimizer.zero_grad()

    def step_optimizer(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def training_step(self):
        """The training step, i.e. training for each batch. Goes through the usual hoops of zeroing out the optimizer,
        forwarding the input, computing the loss, backpropagating and updating the weights. For each different steps,
        callbacks are offered for maximum versatility and ease of use."""
        self.zero_grad_optimizer()

        ray = None
        if isinstance(self.method, ParetoFrontApproximationAlgoCallback):
            t = self.t
            if self.t > 1:
                logging.info("Setting t to 1")
                t = 1
            ray = self.method.ray_sampler.sample(t).to(self.device)
        self.ray = ray
        self.method.ray = ray

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            self.on_before_forward()
            self.on_before_forward_callbacks()
            self.y_hat = self.forward(ray=self.ray)
            self.on_after_forward()
            self.on_after_forward_callbacks()

            self.loss_fn._current_lam = getattr(self, 'ray', None)
            _loss_result = self.loss_fn(self.y_hat, self.y)
            if isinstance(_loss_result, tuple):
                self.losses, self.fim_losses = _loss_result
            else:
                self.losses, self.fim_losses = _loss_result, None
            losses = self.method.cast_losses_to_correct_type(self.losses)

            self.loss, _ = self.method.get_weighted_loss(
                losses,
                shared_parameters=list(self.model.shared_parameters()),
                task_specific_parameters=list(self.model.task_specific_parameters()),
                last_shared_parameters=list(self.model.last_shared_parameters()),
                representation=self.features,
                ray=self.ray,
            )

            # Add entropy regularization penalty OUTSIDE scalarization
            # This preserves MAdam decomposition (entropy is not per-task-weighted)
            if hasattr(self.loss_fn, "_entropy_penalty") and self.loss_fn._entropy_penalty is not None:
                self.loss = self.loss - self.loss_fn._entropy_penalty

            self.on_before_backward()
            self.on_before_backward_callbacks()
            self.scaler.scale(self.loss).backward()
            self.on_after_backward()
            self.on_after_backward_callbacks()

        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        self.on_before_optimizer_step()
        self.on_before_optimizer_step_callbacks()
        self.step_optimizer()
        self.on_after_optimizer_step()
        self.on_after_optimizer_step_callbacks()

    def validation_step(self):
        """Performs the validation step. Callbacks are offered for each step of the process."""
        with torch.no_grad():
            self.on_before_forward()
            self.on_before_forward_callbacks()
            self.y_hat = self.forward(ray=self.ray)
            self.on_after_forward()
            self.on_after_forward_callbacks()

            # for validation we simply compute the avg loss, we don't need to compute the weighted loss
            # which incurs more computational costs for gradient balancing methods
            self.loss_fn._current_lam = None  # always hard CE for validation
            _loss_result = self.loss_fn(self.y_hat, self.y)
            if isinstance(_loss_result, tuple):
                self.losses, _ = _loss_result
            else:
                self.losses = _loss_result
            losses = list(self.losses.values())
            self.loss = sum(losses) / len(losses)

    def testing_step(self):
        self.validation_step()

    def train_epoch(self):
        """Trains the model for a single epoch. Callbacks are offered for each method."""
        self.model.train()
        self.on_before_setting_dataloader()
        self.on_before_setting_dataloader_callbacks()
        self.dataloader = self.train_loader
        self.on_after_setting_dataloader()
        self.on_after_setting_dataloader_callbacks()
        for self.batch_idx, batch in enumerate(self.dataloader):
            self._unpack_batch(batch)
            self.on_before_training_step()
            self.on_before_training_step_callbacks()
            self._tick_step()
            self.training_step()
            self.on_after_training_step()
            self.on_after_training_step_callbacks()

    def eval_epoch(self):
        """Performs the evaluation of the model on the validation set. If no validation dataloader is provided, the
        method returns without any computation."""
        self.model.eval()
        if self.val_loader is None:
            return

        if not isinstance(self.method, ParetoFrontApproximationAlgoCallback):
            self.ray = None

        self.on_before_setting_dataloader()
        self.on_before_setting_dataloader_callbacks()
        self.dataloader = self.val_loader
        self.on_after_setting_dataloader()
        self.on_after_setting_dataloader_callbacks()
        for self.batch_idx, batch in enumerate(self.dataloader):
            self._unpack_batch(batch)
            self.on_before_validation_step()
            self.on_before_validation_step_callbacks()
            # self._tick_step()
            self.validation_step()
            self.on_after_validation_step()
            self.on_after_validation_step_callbacks()

    def test_epoch(self):
        """Performs the evaluation of the model on the validation set."""
        if not isinstance(self.method, ParetoFrontApproximationAlgoCallback):
            self.ray = None
        self._set_test()
        self.model.eval()
        self.on_before_setting_dataloader()
        self.on_before_setting_dataloader_callbacks()
        self.dataloader = self.test_loader
        self.on_after_setting_dataloader()
        self.on_after_setting_dataloader_callbacks()
        for self.batch_idx, batch in enumerate(self.dataloader):
            self._unpack_batch(batch)
            self.on_before_testing_step()
            self.on_before_testing_step_callbacks()
            # self._tick_step()
            self.testing_step()
            self.on_after_testing_step()
            self.on_after_testing_step_callbacks()

    def predict(self, test_loader):
        """Performs the evaluation of the provided test dataloader.

        Args:
            test_dataloader (DataLoader): the dataloader to be evaluated.
        """
        if test_loader is None:
            # some datasets (e.g. Cityscapes) do not have predefined test datasets.
            print("No test loader provided.")
            return
        self.test_loader = test_loader
        self._set_test()
        self.on_before_testing_epoch()
        self.on_before_testing_epoch_callbacks()
        self.test_epoch()
        self.on_after_testing_epoch()
        self.on_after_testing_epoch_callbacks()

        return self.test_metrics

    def _train_loop(self):
        self.on_before_training_epoch()
        self.on_before_training_epoch_callbacks()
        self._tick_epoch()
        self.log("epoch", self.current_epoch)
        self.train_epoch()
        self.on_after_training_epoch()
        self.on_after_training_epoch_callbacks()

    def _val_loop(self):
        self.on_before_eval_epoch()
        self.on_before_eval_epoch_callbacks()
        self.eval_epoch()
        self.on_after_eval_epoch()
        self.on_after_eval_epoch_callbacks()

        return self.val_metrics

    def _fit(self):
        self.epoch = 0
        if self.validate_zeroshot:
            print("Validating zeroshot")
            self._set_val()
            self._val_loop()

        for self.epoch in range(1, self.epochs + 1):
            if self.STOP_TRAINING:
                break
            self._set_train()
            self._train_loop()
            if self.val_loader is not None:
                self._set_val()
                self._val_loop()

    def fit(self, epochs):
        """The fit method of the Trainer."""
        self.epochs = epochs
        self.train_loader = self.benchmark.train_dataloader()
        self.val_loader = self.benchmark.val_dataloader()

        logging.info(
            "Training will last for {} epochs that correpond to {} steps.".format(epochs, self.total_train_steps)
        )
        self.setup()

        self.on_before_fit()
        self.on_before_fit_callbacks()
        self._fit()
        self.on_after_fit()
        self.on_after_fit_callbacks()
