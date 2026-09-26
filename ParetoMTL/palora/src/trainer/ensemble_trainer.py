import dataclasses
import logging
import os
from pprint import pprint
from typing import Dict, List

import meshzoo
import numpy as np
import pandas as pd
import torch

import wandb
from src.callbacks.methods.algo_callback import ParetoFrontApproximationAlgoCallback

from .base_trainer import BaseTrainer

BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


@dataclasses.dataclass
class ParetoSetLearningEvaluationProtocol:
    num_tasks: int
    validate_models: int

    def __post_init__(self):

        self.points = self.create_interpolated_points()
        self._counter = 0
        # if self.num_tasks == 2:
        #     self.points["vanilla"] = torch.tensor([0.0, 0.0])
        #     self.points["minus-t1"] = torch.tensor([-1.0, 1.0])
        #     self.points["minus-t2"] = torch.tensor([1.0, -1.0])
        #     self.points["minus"] = torch.tensor([-1.0, -1.0])

        #     b = 0.2
        #     self.points["minus-t1b"] = torch.tensor([-b, 1.0])
        #     self.points["minus-t2b"] = torch.tensor([1.0, -b])

        #     b = 0.5
        #     self.points["minus-t1c"] = torch.tensor([-b, 1.0])
        #     self.points["minus-t2c"] = torch.tensor([1.0, -b])
        pass

        if self.num_tasks == 40:
            celeba_points_path = os.environ.get("CELEBA_POINTS_PATH")
            if celeba_points_path:
                self.points = torch.load(os.path.expanduser(celeba_points_path))
            self.points = {k: self.points[k] for k in range(len(self.points))}

        pprint(self.points, width=1, compact=True)

    def reset_counter(self):
        self._counter = 0

    def next(self):
        # set alpha
        index = self._counter
        index = list(self.points.keys())[index]
        alpha = self.points[index]

        # increment counter
        num_points = len(self.points)
        self._counter += 1
        self._counter = self._counter % num_points

        return index, alpha

    def create_interpolated_points(self) -> Dict[str, List[float]]:
        midpoint = np.ones(self.num_tasks) / self.num_tasks
        if self.validate_models == -1 or self.num_tasks >= 20:
            logging.warn("We only evaluate the midpoint model.")
            # only validate the midpoint
            return dict(midpoint=midpoint.tolist())

        match self.num_tasks:
            case 2:
                points = np.linspace(0, 1, self.validate_models).tolist()
                points = [[p, 1 - p] for p in points]
                points = [torch.tensor(p) for p in points]
                return dict(zip(range(len(points)), points[::-1]))
            case 3:
                if self.validate_models == 7:
                    # 7 representative rays: 3 vertices + 3 edge midpoints + 1 near-center
                    representative = [
                        [1.0, 0.0, 0.0],   # task 0 only
                        [0.5, 0.5, 0.0],   # task 0-1 edge
                        [0.0, 1.0, 0.0],   # task 1 only
                        [0.5, 0.0, 0.5],   # task 0-2 edge
                        [0.0, 0.5, 0.5],   # task 1-2 edge
                        [0.0, 0.0, 1.0],   # task 2 only
                        [0.4, 0.3, 0.3],   # near-center
                    ]
                    return {i: torch.tensor(r) for i, r in enumerate(representative)}
                points, _ = meshzoo.triangle(self.validate_models - 1)
                points = points.T.tolist()
                points = dict(zip(range(len(points)), points))
                points = {k: torch.tensor(v) for k, v in points.items()}
                return points
            case _:
                logging.warn("For more than 3 members, we only evaluate the 'clean' and the midpoint models.")
                points = torch.eye(self.num_tasks).tolist()
                points.append(midpoint.tolist())
                return dict(zip(range(len(points)), points))


class EnsembleTrainer(BaseTrainer):
    method: ParetoFrontApproximationAlgoCallback

    def __init__(self, validate_every_n=20, validate_models=11, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.validate_every_n = validate_every_n
        self.validate_models = validate_models
        self.eval_protocol = ParetoSetLearningEvaluationProtocol(
            num_tasks=self.method.num_tasks, validate_models=self.validate_models
        )

    def _check_validation_condition(self):
        return self.epoch == self.epochs or self.epoch % self.validate_every_n == 0

    def _val_loop(self):
        if self.val_loader is not None and self._check_validation_condition():
            self._set_val()
            self.on_before_validating_interpolations()
            self.on_before_validating_interpolations_callbacks()
            self.results = self._validate_interpolations()
            self.on_after_validating_interpolations()
            self.on_after_validating_interpolations_callbacks()

    def _validate_interpolations(self):
        self.eval_protocol.reset_counter()
        results = {}
        iterator = range(len(self.eval_protocol.points))
        if self.current_epoch == 0:
            print("=" * 50, "> Validating zeroshot, Fixing lambdas.")
            iterator = [0]
        # iterator = tqdm(iterator, disable=(self.method.num_tasks == 1), bar_format=BAR_FORMAT)
        for i, _ in enumerate(iterator):
            index, alpha = self.eval_protocol.next()

            self.ray = alpha
            self.method.ray = alpha
            super()._val_loop()
            results[index] = self.val_metrics

        for k, v in results.items():
            _remove_keys = ["loss/top-left", "loss/bottom-right"]
            _res = {kk: round(vv, 4) for kk, vv in v.items() if all([key not in kk for key in _remove_keys])}
            logging.info(f"{k}: {_res}")

        wandb.log(
            {
                "epoch": self.current_epoch,
                "val/results-epoch={}".format(self.current_epoch): wandb.Table(dataframe=pd.DataFrame(results).T),
            }
        )

        return results

    def on_after_validating_interpolations(self, *args, **kwargs):
        self.log_interpolations(prefix="val")

    def log_interpolations(self, prefix):
        for key, val in self.results.items():
            if isinstance(key, float):
                key = round(key, 2)
            if "avg_loss" in val:
                val.pop("avg_loss")
            for metric_name, metric_value in val.items():
                wandb.log({f"interpolations/{prefix}/{metric_name}-{key}": metric_value})

    def predict_single_ray(self, test_loader, ray):
        self.ray = ray
        self.method.ray = ray
        return super().predict(test_loader)

    def predict(self, test_loader):
        return self.predict_interpolations(test_loader)

    def predict_interpolations(self, dataloader):
        self._set_test()
        self.on_before_predicting_interpolations()
        self.on_before_predicting_interpolations_callbacks()
        self.results = self._predict_interpolations(dataloader)
        self.on_after_predicting_interpolations()
        self.on_after_predicting_interpolations_callbacks()

        wandb.log(
            {
                "epoch": self.current_epoch,
                f"test/results-epoch={self.current_epoch}": wandb.Table(dataframe=pd.DataFrame(self.results).T),
            }
        )

        return self.results

    def _predict_interpolations(self, dataloader, split="test"):
        self.eval_protocol.reset_counter()
        results = {}
        iterator = range(len(self.eval_protocol.points))
        # iterator = tqdm(iterator, disable=(self.method.num_tasks == 1), bar_format=BAR_FORMAT)
        for i, _ in enumerate(iterator):
            index, alpha = self.eval_protocol.next()

            self.ray = alpha
            self.method.ray = alpha

            results[index] = super().predict(dataloader)

        for k, v in results.items():
            _res = {kk: round(vv, 4) for kk, vv in v.items()}
            logging.info(f"TEST -- {k}: {_res}")

        wandb.log(
            {
                "epoch": self.current_epoch,
                "test/results-epoch={}".format(self.current_epoch): wandb.Table(dataframe=pd.DataFrame(results).T),
                "test/results": wandb.Table(dataframe=pd.DataFrame(results).T),
            }
        )

        return results

    def on_after_predicting_interpolations(self, *args, **kwargs):
        self.log_interpolations(prefix="test")
