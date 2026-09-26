import logging

from src.trainer.base_trainer import BaseTrainer


from src.callbacks.callback import Callback
from torch.optim.lr_scheduler import LRScheduler


def format_lr(lr):
    if isinstance(lr, (list, tuple)):
        return [format_lr(l) for l in lr]
    else:
        return round(lr, 6)


class SchedulerCallback(Callback):
    def __init__(self, scheduler: LRScheduler, scheduler_step_on_epoch=True):
        super().__init__()
        self.scheduler = scheduler
        self.scheduler_step_on_epoch = scheduler_step_on_epoch

    def on_after_training_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        if self.scheduler_step_on_epoch:
            old_lr = self.scheduler.get_last_lr()
            self.scheduler.step()
            new_lr = self.scheduler.get_last_lr()
            if new_lr != old_lr:
                if isinstance(new_lr, (list, tuple)) and len(new_lr) > 3:
                    logging.info("The LR changed.")
                else:
                    logging.info(f"The LR changed from {format_lr(old_lr)} to {format_lr(new_lr)}.")

    def on_after_training_step(self, *args, **kwargs):
        if not self.scheduler_step_on_epoch:
            self.scheduler.step()
            # new_lr = self.scheduler.get_last_lr()
            # desc = self.tqdm_dl.desc.split(",")[0]
            # self.tqdm_dl.set_description(f"{desc}, lr={new_lr[0]:.4f}")
