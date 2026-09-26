from src.callbacks.callback import Callback
from tqdm import tqdm
import sys
import time
from src.trainer import BaseTrainer, EnsembleTrainer

BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


class TqdmCallback(Callback):
    def __init__(self, leave_val=True):
        super().__init__()
        self.bar_format = BAR_FORMAT
        self.leave_val = leave_val

    def _tqdm(self, dataloader, mode="train", epoch=-1):
        if mode == "train":
            msg = f"Epoch {epoch}"
            leave = True
        elif mode == "val":
            msg = f"Validating epoch {epoch}"
            leave = self.leave_val
        elif mode == "test":
            msg = f"TESTING"
            leave = self.leave_val
        return tqdm(
            dataloader,
            position=0,
            desc=msg,
            # file=sys.stdout,  # so that tqdm does not print out of order,
            leave=leave,
            bar_format=self.bar_format,
            # disable=True,
        )

    def on_before_fit(self, trainer: BaseTrainer, *args, **kwargs):
        if isinstance(trainer, EnsembleTrainer):
            self.leave_val = False

    def on_after_setting_dataloader(self, trainer: BaseTrainer, *args, **kwargs):
        mode = trainer.current_mode.value
        epoch = trainer.current_epoch
        self.tqdm = self._tqdm(trainer.dataloader, mode=mode, epoch=epoch)

    def on_after_training_step(self, trainer: BaseTrainer, *args, **kwargs):
        self.tqdm.update(1)
        msg = getattr(trainer, "progress_bar_message", "")
        self.update_tqdm(msg)

    def on_after_validation_step(self, trainer: BaseTrainer, *args, **kwargs):
        self.tqdm.update(1)
        msg = getattr(trainer, "progress_bar_message", "")
        self.update_tqdm(msg)

    def on_after_eval_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        msg = getattr(trainer, "progress_bar_message", "")
        self.update_tqdm(msg)
        # self.tqdm.close()

    def on_after_testing_step(self, trainer: BaseTrainer, *args, **kwargs):
        self.tqdm.update(1)
        msg = getattr(trainer, "progress_bar_message", "")
        self.update_tqdm(msg)

    def update_tqdm(self, msg):
        if len(msg) > 5:
            new_msg = {}
            for key in msg:
                if "avg" in key:
                    new_msg[key] = msg[key]
            msg = new_msg
        self.tqdm.set_postfix(msg)
