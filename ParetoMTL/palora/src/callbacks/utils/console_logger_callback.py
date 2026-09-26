import math
import time

from src.callbacks.callback import Callback
from src.callbacks.methods.algo_callback import ParetoFrontApproximationAlgoCallback
from src.trainer import BaseTrainer


class ConsoleLoggerCallback(Callback):
    def __init__(self, logging_frequency=3, logging_interval=None):
        super().__init__()
        if logging_interval is not None:
            assert isinstance(logging_interval, int) and logging_interval > 0

        assert isinstance(logging_frequency, int) and logging_frequency > 0
        self.logging_frequency = logging_frequency
        self.logging_interval = logging_interval
        self.start_time = time.time()

    def on_after_setting_dataloader(self, trainer: BaseTrainer, *args, **kwargs):
        mode = trainer.current_mode.value
        epoch = trainer.current_epoch
        if mode == "train":
            if self.logging_interval is not None:
                self.log_every = self.logging_interval
            else:
                self.log_every = int(len(trainer.dataloader) // self.logging_frequency)

            print(f"------ Epoch {epoch} ------")

    def format_msg(self, msg):
        time_elapsed = time.time() - self.start_time
        if isinstance(msg, dict):
            if len(msg) > 5:
                msg = {k: v for k, v in msg.items() if "avg" not in k}
            msg1 = " ".join([f"{k}: {v:.4f}" for k, v in msg.items() if "samples" not in k])
            msg2 = " ".join([f"{k}: {v}" for k, v in msg.items() if "samples" in k])
            msg = f"{msg1} {msg2}"
        return f"{msg} ({time_elapsed:.2f}s)"

    def on_after_training_step(self, trainer: BaseTrainer, *args, **kwargs):
        batch_idx = trainer.batch_idx
        if (batch_idx + 1) % self.log_every == 0:
            msg = getattr(trainer, "progress_bar_message", "")
            num_batches = len(trainer.dataloader)
            a = int(math.log10(num_batches)) + 1
            index = "{0}".format(f"{batch_idx + 1}".zfill(a))
            print(f"t={trainer.t:.4f}, Step {index}/{num_batches}:", self.format_msg(msg))
            self.start_time = time.time()

    def on_before_training_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        print("\n")
        self.epoch_start_time = time.time()
        self.start_time = time.time()

    def on_after_training_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        elapsed_time = time.time() - self.epoch_start_time
        print(f"Epoch {trainer.current_epoch} finished in {elapsed_time:.2f}s")

    def on_before_eval_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        self.start_time = time.time()

    def on_after_eval_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        if not isinstance(trainer.method, ParetoFrontApproximationAlgoCallback):
            msg = getattr(trainer, "progress_bar_message", "")
            print(f"Validating epoch {trainer.current_epoch}:", self.format_msg(msg))

    def on_before_validating_interpolations(self, *args, **kwargs):
        self.start_time_interpolations = time.time()

    def on_after_validating_interpolations(self, *args, **kwargs):
        elapsed_time = time.time() - self.start_time_interpolations
        print(f"Validating interpolations took {elapsed_time:.2f}s")
