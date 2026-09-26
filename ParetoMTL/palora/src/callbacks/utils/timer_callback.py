import time
from typing import Any

import wandb

from src.trainer.base_trainer import BaseTrainer

from src.callbacks.callback import Callback


class Timer:
    def __init__(self) -> None:
        self.old_value = None
        self.counter = 0
        self.intervals = []

    def compute(self):
        if len(self.intervals) == 0:
            return -1
        return sum(self.intervals) / len(self.intervals)

    def start(self):
        self.old_value = time.perf_counter()

    def reset_old_value(self):
        self.old_value = None

    def stop(self):
        return time.perf_counter() - self.old_value

    def update(self):
        self.intervals.append(self.stop())
        self.counter += 1
        self.reset_old_value()


class TimerCallback(Callback):
    """WARNING: These timers are not exact because they do not take into account the time spent in the other
    callbacks."""

    def __init__(self):
        super().__init__()

        self.train_step_timer = Timer()
        self.val_step_timer = Timer()
        self.train_epoch_timer = Timer()
        self.val_epoch_timer = Timer()

    def on_before_training_step(self, trainer: BaseTrainer, *args, **kwargs):
        self.train_step_timer.start()

    def on_after_training_step(self, trainer: BaseTrainer, *args, **kwargs):
        self.train_step_timer.update()

    def on_before_validation_step(self, trainer: BaseTrainer, *args, **kwargs):
        self.val_step_timer.start()

    def on_after_validation_step(self, trainer: BaseTrainer, *args, **kwargs):
        self.val_step_timer.update()

    def on_before_training_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        self.train_epoch_timer.start()

    def on_after_training_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        self.train_epoch_timer.update()

    def on_before_eval_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        self.val_epoch_timer.start()

    def on_after_eval_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        self.val_epoch_timer.update()

    def on_after_fit(self, trainer: BaseTrainer, *args, **kwargs):
        if wandb.run is not None:
            wandb.log(
                {
                    "timer/train_step_time": self.train_step_timer.compute(),
                    "timer/val_step_time": self.val_step_timer.compute(),
                    "timer/train_epoch_time": self.train_epoch_timer.compute(),
                    "timer/val_epoch_time": self.val_epoch_timer.compute(),
                }
            )
