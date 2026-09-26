import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torchmetrics import (
    Accuracy,
    F1Score,
    MeanMetric,
    MeanSquaredError,
    MetricCollection,
    Precision,
    Recall,
    SumMetric,
)

from src.callbacks.callback import Callback
from src.trainer.base_trainer import BaseTrainer
from src.utils.metrics import CrossEntropyLossMetric, HuberLossMetric, compute_hv


class MultiTaskMetricCallback(Callback):
    """Handles the computation and logging of metrics. Callback hooks after train/val/test steps/epochs etc. Inherits from Callback."""

    num_tasks: int
    task_names: List[str]

    def __init__(self, metrics: MetricCollection, use_task_names=False, logging_interval=0.1):
        super().__init__()
        self.use_task_names = use_task_names
        self.orig_metrics = metrics
        self.logging_interval = logging_interval

    def on_after_setting_dataloader(self, trainer: BaseTrainer):
        # self.log_every = int(self.logging_frequency * len(trainer.dataloader))
        self.log_every = int(self.logging_interval)
        if self.log_every == 0:
            self.log_every = 1

    def on_before_fit(self, trainer: BaseTrainer, *args, **kwargs):
        super().connect(trainer, *args, **kwargs)
        self.num_tasks = trainer.benchmark.num_tasks
        if self.use_task_names:
            self.task_names = trainer.benchmark.task_names
        else:
            self.task_names = [f"task-{i}" for i in range(self.num_tasks)]
        self.setup_metrics(self.orig_metrics)
        self.move_to_device(trainer.device)

    def setup_metrics(self, metrics):
        if isinstance(metrics, MetricCollection):
            # all tasks have the same metrics
            metrics = {k: metrics for k in self.task_names}

        assert set(metrics.keys()) == set(self.task_names)
        self.metrics = nn.ModuleDict({k: v.clone(postfix=f"/{k}") for k, v in metrics.items()})
        self.avg_loss = MeanMetric(compute_on_step=False)

    def move_to_device(self, device):
        self.metrics = self.metrics.to(device)
        self.avg_loss = self.avg_loss.to(device)

    def _reset_metrics(self):
        # logging.info("Reseting metrics")
        self.avg_loss.reset()
        for m in self.metrics.values():
            m.reset()

    def log(self, trainer: BaseTrainer, key, value):
        trainer.log(key, value)

    def msg_process(self, msg):
        msg = {k: v for k, v in msg.items() if "loss/" not in k}
        return msg

    def update_metrics(self, metrics, trainer: BaseTrainer):
        # for task_id, (yy_hat, y) in enumerate(zip(trainer.y_hat, trainer.y)):
        #     metrics[task_id](yy_hat, y)
        insta_results = {}
        for k in trainer.y_hat.keys():
            insta_results[k] = metrics[k](trainer.y_hat[k], trainer.y[k])

        insta_results = {k: {kk: vv.item() for kk, vv in v.items()} for k, v in insta_results.items()}
        insta_results = {f"{kk}": vv for k, v in insta_results.items() for kk, vv in v.items()}

        return insta_results

    def compute_metrics(self, metrics, avg_metric):
        msg = dict()
        msg["avg_loss"] = avg_metric.compute().item()

        for k, v in metrics.items():
            res = v.compute()
            res = {kk: vv.item() for kk, vv in res.items()}
            msg.update(res)

        # ------------
        metric_keys = [a.split("/")[0] for a in msg.keys()]
        distinct_metric_keys = set(metric_keys)
        for key in distinct_metric_keys:
            if metric_keys.count(key) == self.num_tasks:
                msg[f"avg_{key}"] = sum([v for k, v in msg.items() if key in k]) / self.num_tasks

        # ------------
        return msg

    # ------- STEPS -------
    def compute_hv(self):
        return compute_hv(
            num_tasks=self.num_tasks,
            task_names=self.task_names,
            results=self.trainer.results,
        )

    def _after_step(self, trainer: BaseTrainer, prefix: Optional[str] = None):
        self.avg_loss(trainer.loss)
        insta_results = self.update_metrics(self.metrics, trainer)

        if (trainer.batch_idx + 1) % self.log_every == 0:
            msg = self.compute_metrics(self.metrics, self.avg_loss)
            if prefix is not None:
                record = {
                    "epoch": trainer.current_epoch,
                    "mystep": trainer.current_step,
                }
                for k, v in msg.items():
                    metric_key = f"train/{k}"
                    trainer.log(metric_key, v)
                    record[metric_key] = v
                for k, v in insta_results.items():
                    record[f"insta/train/{k}"] = v
                trainer.record_training_dynamics(record)
            msg = self.msg_process(msg)
            setattr(trainer, "progress_bar_message", msg)

    def on_after_training_step(self, trainer: BaseTrainer):
        self._after_step(trainer, prefix="train")

    def on_after_validation_step(self, trainer: BaseTrainer):
        self._after_step(trainer)

    def on_after_testing_step(self, trainer: BaseTrainer):
        self._after_step(trainer)

    # ------- EPOCHS - before -------
    def on_before_training_epoch(self, trainer):
        self._reset_metrics()

    def on_before_eval_epoch(self, trainer):
        self._reset_metrics()

    def on_before_testing_epoch(self, trainer):
        self._reset_metrics()

    # ------- EPOCHS - after -------
    def _after_epoch(self, trainer: BaseTrainer, prefix: str):
        self.results = self.compute_metrics(self.metrics, self.avg_loss)
        self.register_results_to_trainer(trainer, f"{prefix}_metrics", self.results)
        for k, v in self.results.items():
            trainer.log(f"{prefix}/{k}", v)
        msg = self.msg_process(self.results)
        setattr(trainer, "progress_bar_message", msg)

    def on_after_eval_epoch(self, trainer: BaseTrainer):
        self._after_epoch(trainer, prefix="val")

    def on_after_testing_epoch(self, trainer: BaseTrainer):
        self._after_epoch(trainer, prefix="test")

    #
    # ------- INTERPOLATION STUFF -------
    #
    def log_best_interpolation_results(self, trainer, prefix: str):
        keys = trainer.results[next(iter(trainer.results))].keys()
        res = {key: [m[key] for m in trainer.results.values()] for key in keys}
        best_results = {}
        for key, values in res.items():
            if "loss" in key or "huber" in key:
                best = min(values)
            elif "acc" in key:
                best = max(values)
            elif "error" in key:
                best = min(values)
            else:
                raise NotImplementedError

            best_results[key] = round(best, 3)
            trainer.log(f"{prefix}/best/{key}", best)

        logging.info(f"Best results for {prefix.upper()} dataset: {dict(sorted(best_results.items()))}")

    def on_after_validating_interpolations(self, trainer: BaseTrainer):
        logging.info("Logging best results out of interpolations for validation dataset.")
        self.log_best_interpolation_results(trainer, prefix="val")

        if trainer.benchmark.num_tasks == 2:
            trainer.log(f"HV/val", self.compute_hv())

    def on_after_predicting_interpolations(self, trainer: BaseTrainer):
        logging.info("Logging best results out of interpolations for test dataset.")
        self.log_best_interpolation_results(trainer, prefix="test")
        if trainer.benchmark.num_tasks == 2:
            trainer.log(f"HV/test", self.compute_hv())

    def register_results_to_trainer(self, trainer: BaseTrainer, results_name, results_dict):
        setattr(trainer, results_name, results_dict)


class CounterMetric(SumMetric):

    def update(self, x, *args, **kwargs):
        self.value += len(x)

    def compute(self) -> torch.Tensor:
        return super().compute().int()


acc_metrics = MetricCollection(
    {
        "acc": Accuracy(compute_on_step=False),
        # "samples": CounterMetric(compute_on_step=False),
        "loss": CrossEntropyLossMetric(compute_on_step=False),
    }
)


class HackedMeanSquaredError(MeanSquaredError):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        return super().update(preds.squeeze(), target.squeeze())


reg_metrics = MetricCollection(
    {
        "error": HackedMeanSquaredError(compute_on_step=False),
    }
)

binary_acc_metrics = MetricCollection(
    {
        "acc": Accuracy(compute_on_step=False),
        "precision": Precision(compute_on_step=False, average="micro", ignore_index=0),
        "recall": Recall(compute_on_step=False, average="micro", ignore_index=0),
        "f1": F1Score(compute_on_step=False, average="micro", ignore_index=0),
        "loss": CrossEntropyLossMetric(compute_on_step=False, average="micro", ignore_index=0),
    }
)


class ClassificationMultiTaskMetricCallback(MultiTaskMetricCallback):
    def __init__(self, use_task_names=True, logging_interval=0.1):
        super().__init__(
            metrics=acc_metrics,
            use_task_names=use_task_names,
            logging_interval=logging_interval,
        )


class RegressionMultiTaskMetricCallback(MultiTaskMetricCallback):
    def __init__(self, use_task_names=True, logging_interval=0.1):
        super().__init__(
            metrics=reg_metrics,
            use_task_names=use_task_names,
            logging_interval=logging_interval,
        )


class BinaryClassificationMetricCallback(MultiTaskMetricCallback):
    def __init__(self, use_task_names=False):
        super().__init__(metrics=binary_acc_metrics, use_task_names=use_task_names)


class UTKFaceMultiTaskMetricCallback(MultiTaskMetricCallback):
    utk_face_metrics = {
        "age": MetricCollection({"huber": HuberLossMetric()}),
        "gender": acc_metrics,
        "race": acc_metrics,
    }

    def __init__(self):
        super().__init__(metrics=self.utk_face_metrics, use_task_names=True)
