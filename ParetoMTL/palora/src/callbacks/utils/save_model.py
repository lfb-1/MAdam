import logging
import os
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch

import wandb
from src.trainer.base_trainer import BaseTrainer

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer

from src.callbacks.callback import Callback


class SaveModelCallback(Callback):
    def __init__(self, wandb=False, inputs=-1, epoch_frequency=None, step_frequency=None):
        super().__init__()
        self.wandb = wandb
        self.filename = "model.pth"
        self.inputs = inputs
        assert isinstance(self.inputs, int)
        if self.inputs > 10:
            logging.info("Will save 10 images and not more.")
            self.inputs = 10

        self.epoch_frequency = epoch_frequency
        self.step_frequency = step_frequency

    def on_before_fit(self, trainer: "BaseTrainer", *args, **kwargs):
        assert isinstance(self.inputs, int)
        if self.inputs > 0 and self.wandb:
            train_dataset = trainer.train_loader.dataset
            random_indices = np.random.choice(len(train_dataset), size=self.inputs, replace=False).tolist()
            for index in random_indices:
                img = train_dataset[index][0]
                # logging.info(img)
                # logging.info(img.dim())
                if img.dim() == 1:
                    # not an image
                    return
                img = wandb.Image(img)
                wandb.log({f"inputs/{index}": img})

    def get_ckpt(self, trainer: "BaseTrainer", include_optimizer: bool = False) -> Dict[str, Any]:
        ckpt = {
            "method": trainer.method.__repr__(),
            "epoch": trainer.current_epoch,
            "step": trainer.current_step,
            "state_dict": trainer.model.state_dict(),
        }
        if include_optimizer:
            ckpt["optimizer_state_dict"] = trainer.optimizer.state_dict()

        if getattr(trainer, "val_metrics", None) is not None:
            ckpt["val_metrics"] = trainer.val_metrics

        if getattr(trainer, "train_metrics", None) is not None:
            ckpt["train_metrics"] = trainer.train_metrics

        if getattr(trainer, "test_metrics", None) is not None:
            ckpt["test_metrics"] = trainer.test_metrics

        # if wandb.run is not None:
        #     ckpt["wandb"] = wandb.run.id
        #     ckpt["wandb_url"] = wandb.run.get_url()

        return ckpt

    def on_after_training_step(self, trainer: BaseTrainer, *args, **kwargs):
        if self.step_frequency is not None and trainer.current_step % self.step_frequency == 0:
            filename = f"ckpt_step={trainer.current_step:04d}.pt"
            torch.save(f=filename, obj=self.get_ckpt(trainer))
            ckpt_size = os.path.getsize(filename) / 1e6
            logging.info(f"Saved checkpoint as {os.getcwd()}/{filename}. The file size is {ckpt_size:.2f} MB.")

    def on_after_training_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        if self.epoch_frequency is not None and trainer.current_epoch % self.epoch_frequency == 0:
            filename = f"ckpt_epoch={trainer.current_epoch:03d}.pt"
            torch.save(f=filename, obj=self.get_ckpt(trainer))
            ckpt_size = os.path.getsize(filename) / 1e6
            logging.info(f"Saved checkpoint as {os.getcwd()}/{filename}. The file size is {ckpt_size:.2f} MB.")

    def on_after_fit(self, trainer: "BaseTrainer", *args, **kwargs):
        torch.save(f=self.filename, obj=self.get_ckpt(trainer, include_optimizer=True))
        ckpt_size = os.path.getsize(self.filename) / 1e6
        logging.info(f"Saved model as {os.getcwd()}/{self.filename}. The file size is {ckpt_size:.2f} MB.")
        if self.wandb:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(self.filename)
            wandb.run.log_artifact(artifact)
            logging.info("Saved model to Weights&Biases!")
