from collections import defaultdict
from typing import Dict, List

import torch
from src.models.base_model import SharedBottom
from src.utils.moo_losses import MultiForwardRegularizationLoss

from .ensemble_trainer import EnsembleTrainer


class MultiForwardEnsembleTrainer(EnsembleTrainer):
    model: SharedBottom

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num = self.method.num
        self.reg_coefficient = self.method.reg_coefficient

        self.num_tasks = self.benchmark.num_tasks
        self.population_loss = MultiForwardRegularizationLoss(self.num_tasks, self.reg_coefficient)

    def training_step(self):
        return self.training_step1()

    def compute_multiforward_regularization_loss(
        self, losses: List[Dict[str, torch.Tensor]], rays: List[torch.Tensor]
    ):
        task_losses = {}
        for i, loss in enumerate(losses):
            ray = rays[i]
            for j, key in enumerate(loss.keys()):
                if key not in task_losses:
                    task_losses[key] = {}
                task_losses[key][ray[j].item()] = loss[key]

        return self.population_loss(task_losses)

    def training_step1(self):
        rays = self.method.ray_sampler.sample(self.t)
        if rays.ndim == 1:
            rays = rays.unsqueeze(0)
        if rays.shape[0] < self.num:
            raise ValueError(
                f"Ray sampler returned {rays.shape[0]} rays, but {self.num} forwards are required."
            )
        rays = rays[: self.num].to(self.device)
        self.rays = rays  # Store active rays for FIM preconditioning.
        self.zero_grad_optimizer()
        self.loss = 0

        self.losses = []
        self._fim_losses_list = []
        self.fim_losses = None
        for i in range(self.num):
            self.ray = rays[i]
            self.method.ray = rays[i]
            self.method.task_weights = rays[i]
            self.on_before_forward()
            self.on_before_forward_callbacks()
            self.y_hat = self.forward(self.ray)
            self.on_after_forward()
            self.on_after_forward_callbacks()

            self.loss_fn._current_lam = self.ray
            _loss_result = self.loss_fn(self.y_hat, self.y)
            if isinstance(_loss_result, tuple):
                losses, fim_losses = _loss_result
            else:
                losses, fim_losses = _loss_result, None
            self.losses.append(losses)
            if fim_losses is not None:
                if not hasattr(self, '_fim_losses_list'):
                    self._fim_losses_list = []
                self._fim_losses_list.append(fim_losses)

            loss, _ = self.method.get_weighted_loss(
                losses,
                ray=self.ray,
                shared_parameters=list(self.model.shared_parameters()),
                task_specific_parameters=list(self.model.task_specific_parameters()),
                last_shared_parameters=list(self.model.last_shared_parameters()),
                representation=self.features,
                scaler=self.scaler,
            )

            self.loss += loss

        if self._fim_losses_list:
            self.fim_losses = self._fim_losses_list
        self.on_before_backward()
        self.on_before_backward_callbacks()
        self.loss.backward()
        self.on_after_backward()
        self.on_after_backward_callbacks()

        self.on_before_optimizer_step()
        self.on_before_optimizer_step_callbacks()
        self.step_optimizer()
        self.on_after_optimizer_step()
        self.on_after_optimizer_step_callbacks()

        # fix losses to be a dictionary for compatibility with the metric callbacks
        sum_dict = defaultdict(lambda: 0)
        for d in self.losses:
            for key, value in d.items():
                sum_dict[key] += value
        sum_dict = dict(sum_dict)
        self.losses = {key: value / self.num for key, value in sum_dict.items()}
