import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import wandb
from src.callbacks.callback import Callback
from src.callbacks.methods.ll.palora_modules import PaLoRALayer
from src.callbacks.methods.palora import PaLoRA
from src.trainer.base_trainer import BaseTrainer

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


class CosineSimilarityCallback(Callback):
    def __init__(self):
        super().__init__()

    def connect(self, trainer: BaseTrainer, *args, **kwargs):
        self.task_names = trainer.benchmark.task_names
        return super().connect(trainer, *args, **kwargs)

    def compute_similarities(self, trainer: "BaseTrainer"):
        # for each layer extract the LoRA parameters
        model = trainer.model

        weights_A = {}
        weights_B = {}

        for name, module in model.named_modules():
            if isinstance(module, PaLoRALayer):
                weights_A[name] = [v.data.detach().flatten() for v in module.lora_A.values()]
                weights_B[name] = [v.data.detach().flatten() for v in module.lora_B.values()]

        for layer_name in weights_A.keys():
            for i in range(len(weights_A[layer_name])):
                for j in range(i + 1, len(weights_A[layer_name])):
                    value = F.cosine_similarity(weights_A[layer_name][i], weights_A[layer_name][j], dim=0).item()
                    wandb.log({f"similarities/cosine_A_{layer_name}_{i}-{j}": value})

                    value = F.cosine_similarity(weights_B[layer_name][i], weights_B[layer_name][j], dim=0).item()
                    wandb.log({f"similarities/cosine_B_{layer_name}_{i}-{j}": value})

        # compute global similarity
        num_tasks = len(self.task_names)
        task_vectors = {
            i: torch.cat(
                [
                    torch.cat([v[i] for v in weights_A.values()]),
                    torch.cat([v[i] for v in weights_B.values()]),
                ]
            )
            for i in range(num_tasks)
        }
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                value = F.cosine_similarity(task_vectors[i], task_vectors[j], dim=0).item()
                wandb.log({f"similarities/cosine_A_{i}-{j}": value})
                logging.info(f"Cosine similarity of tasks {i} and {j} is {value:.6f}")

    def on_after_training_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        if isinstance(self.trainer.method, PaLoRA):
            self.compute_similarities(self.trainer)
