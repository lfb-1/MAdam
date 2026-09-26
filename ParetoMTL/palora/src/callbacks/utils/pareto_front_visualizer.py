import copy
import json
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV
from scipy.stats import spearmanr

import wandb
from src.callbacks.callback import Callback
from src.trainer.base_trainer import BaseTrainer

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


def compute_uniformity(loss1: list[float], loss2: list[float]) -> np.float32:
    losses = np.array([loss1, loss2]).T
    losses = losses / losses.sum(axis=0)
    m = len(losses)
    return np.sum(losses * np.log(losses * m))


def compute_hypervolume(points, xmax=True, ymax=True):
    points = copy.deepcopy(points)

    ref = [1, 1]
    if xmax:
        ref[0] = 0
        points[:, 0] = -points[:, 0]
    if ymax:
        ref[1] = 1
        points[:, 1] = -points[:, 1]
    ind = HV(ref_point=np.array(ref))
    return ind(points)


MAPPING = {
    "loss/top-left": False,
    "loss/bottom-right": False,
    "acc/top-left": True,
    "acc/bottom-right": True,
    "sem/loss": False,
    "depth/loss": False,
    "sem/iou": True,
    "depth/abs_err": False,
    "loss/age": False,
    "loss/education": False,
    "acc/age": True,
    "acc/education": True,
}


def get_num_non_dominated(points, xmax, ymax):
    def is_dominated2d(p1, p2):
        """Check if point p1 is dominated by point p2."""
        if xmax:
            p1 = -p1
        if ymax:
            p2 = -p2
        return (p1[0] >= p2[0] and p1[1] >= p2[1]) and (p1[0] > p2[0] or p1[1] > p2[1])

    def pareto_front(points):
        """Compute the Pareto front of a set of 3D points.

        Parameters:
        points (ndarray): An array of shape (n, 3) representing the coordinates of the points.

        Returns:
        int: The number of nondominated points.
        """
        num_points = len(points)
        is_nondominated = np.ones(num_points, dtype=bool)

        for i in range(num_points):
            for j in range(num_points):
                if i != j and is_dominated2d(points[i], points[j]):
                    is_nondominated[i] = False
                    break

        return np.sum(is_nondominated)

    return pareto_front(points)


class ParetoFrontVisualizerCallback(Callback):

    def __init__(self):
        super().__init__()
        self.discounted_metric = 0
        self.discounted_metric_normalized = 0

    def on_before_fit(self, trainer: BaseTrainer, *args, **kwargs):

        match trainer.benchmark.name:
            case "multimnist":
                self.pairs = [
                    ("loss/top-left", "loss/bottom-right"),
                    ("acc/top-left", "acc/bottom-right"),
                ]
                self.baselines = json.load(
                    open(
                        os.path.join(os.path.dirname(__file__), "results/multimnist.json"),
                        "r",
                    )
                )
            case "Cityscapes2":
                self.pairs = [
                    ("sem/loss", "depth/loss"),
                    ("sem/iou", "depth/abs_err"),
                ]
                self.baselines = None
            case "census_age_education":
                self.pairs = [
                    ("loss/age", "loss/education"),
                    ("acc/age", "acc/education"),
                ]
                self.baselines = None
            case _:
                raise ValueError(f"benchmark {trainer.benchmark.name} not supported")

        self.previous_results = {}
        for pair in self.pairs:
            x = pair[0]
            y = pair[1]
            self.previous_results[x] = {}
            self.previous_results[y] = {}
        return super().on_before_fit(trainer, *args, **kwargs)

    def on_after_validating_interpolations(self, trainer: "BaseTrainer", *args, **kwargs):
        try:
            population_results = self.dotheplots(trainer, prefix="val/")
            population_results = {f"pop/{k}": v for k, v in population_results.items()}
            results = trainer.results
            results["population"] = population_results
            wandb.log({"epoch": trainer.current_epoch, **population_results})
        except Exception as e:
            print(e)

    def on_after_predicting_interpolations(self, trainer: "BaseTrainer", *args, **kwargs):
        population_results = self.dotheplots(trainer, prefix="test/")
        results = trainer.results
        results["population"] = population_results
        # json.dump(results, open(f"test.json", "w"), indent=4)

    def dotheplots(self, trainer: BaseTrainer, prefix):
        results = trainer.results
        population_results = {}

        for x_label, y_label in self.pairs:
            x = self.extract_metric(results, x_label)
            y = self.extract_metric(results, y_label)
            # table = wandb.Table(data=np.array([x, y]).T.tolist(), columns=[x_label, y_label])
            self.previous_results[x_label][trainer.current_epoch] = x
            self.previous_results[y_label][trainer.current_epoch] = y
            if trainer.benchmark.name != "census_age_education":
                try:
                    self.plot_figure(
                        x_label, y_label, x, y, epoch=trainer.current_epoch, prefix=prefix, use_previous=True
                    )
                except Exception as e:
                    print(e)
            self.plot_figure(
                x_label,
                y_label,
                x,
                y,
                epoch=trainer.current_epoch,
                prefix=prefix,
                use_previous=False,
            )
            if self.baselines is not None and "test" not in prefix:
                self.plot_with_baselines(x_label, y_label, x, y, epoch=trainer.current_epoch, prefix=prefix)

            if "loss" in x_label and "loss" in y_label:
                uniformity = compute_uniformity(x, y)
                print(f"Uniformity: {uniformity}")
                hypervolume = compute_hypervolume(np.array([x, y]).T, xmax=MAPPING[x_label], ymax=MAPPING[y_label])
                print(f"Hypervolume: {hypervolume}")
                spearman = spearmanr(x, y)
                print(f"Spearman: {spearman.statistic}")

                num_non_dominated = get_num_non_dominated(
                    np.array([x, y]).T, xmax=MAPPING[x_label], ymax=MAPPING[y_label]
                )

                self.discounted_metric = 0.9 * self.discounted_metric + num_non_dominated
                self.discounted_metric_normalized = 0.9 * self.discounted_metric_normalized + num_non_dominated / len(
                    x
                )

                self.new_metric = hypervolume + self.discounted_metric_normalized / len(x)
                if trainer.benchmark.name == "Cityscapes2":
                    self.new_metric = hypervolume - 10 * int(num_non_dominated / len(x) > 0.5)

                _prefix = prefix.replace("/", "")
                wandb.log(
                    {
                        f"val_epoch": trainer.current_epoch,
                        f"population/{_prefix}/uniformity/{x_label}-{y_label}": uniformity.item(),
                        f"population/{_prefix}/hypervolume/{x_label}-{y_label}": hypervolume,
                        f"population/{_prefix}/spearman/{x_label}-{y_label}": spearman.statistic,
                        f"population/{_prefix}/num_non_dominated/{x_label}-{y_label}": num_non_dominated,
                        f"population/{_prefix}/discounted_metric/{x_label}-{y_label}": self.discounted_metric,
                    }
                )

                if "val" in _prefix:
                    wandb.log(
                        {
                            "num_non_dominated_discounted": self.discounted_metric,
                            "num_non_dominated_discounted_normalized": self.discounted_metric_normalized,
                            "combined_metric": self.new_metric,
                            f"val_epoch": trainer.current_epoch,
                        }
                    )

                population_results["uniformity"] = uniformity
                population_results["hypervolume"] = hypervolume
                population_results["spearman"] = spearman.statistic
                population_results["num_non_dominated"] = num_non_dominated

                print(population_results)

        return population_results

    @staticmethod
    def extract_metric(results: dict, metric: str):

        output = {
            k: v[kk] for k, v in results.items() for kk in v if isinstance(v, dict) and metric in kk and type(k) == int
        }
        output = sorted(output.items(), key=lambda x: x[0])
        output = [v for k, v in output]
        return output

    def plot_figure(self, x_label, y_label, x, y, epoch, prefix="val/", use_previous=True):
        plt.figure()
        if use_previous:
            if self.previous_results[x_label]:
                num_epochs = len(self.previous_results[x_label].keys())
                alphas = np.linspace(0.107, 0.9, num_epochs)
                for i, epoch in enumerate(self.previous_results[x_label].keys()):
                    xx = self.previous_results[x_label][epoch]
                    yy = self.previous_results[y_label][epoch]
                    plt.plot(xx, yy, label=f"epoch {epoch}", color="blue", alpha=alphas[i])
                    if epoch == 0:
                        plt.scatter(xx, yy, label=f"epoch {epoch}", color="red", alpha=alphas[i])

        plt.plot(x, y, "-o")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        xx_label = x_label.replace("/", " ")
        yy_label = y_label.replace("/", " ")
        _prefix = prefix.replace("/", "")
        filename = f"{_prefix}-{xx_label}-{yy_label}--{epoch}"
        if use_previous:
            plt.legend()
        if not use_previous:
            filename += "-no-previous"
        plt.title(filename)
        plt.savefig(filename + ".png", bbox_inches="tight")

        if "loss" in x_label and "loss" in y_label:
            log_name = f"PF-{_prefix}-loss"

        else:
            log_name = f"PF-{_prefix}-metric"

        if use_previous:
            log_name += "-progression"

        wandb.log(
            {
                "epoch": epoch,
                log_name: wandb.Image(filename + ".png", caption=f"{_prefix}-{xx_label}-{yy_label}"),
                # log_name+ "2": plt,
            }
        )

    def plot_with_baselines(self, x_label, y_label, x, y, epoch, prefix="val/"):
        # print("Missing ftm")
        pass

    # x_label = prefix + x_label
    # y_label = prefix + y_label
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # plot_baselines(ax, x_label, y_label, self.baselines)
    # ax.plot(x, y, color="blue", label="PaLoRA")
    # fix_legend(ax, x_label, y_label)
    # xx_label = x_label.replace("/", " ")
    # yy_label = y_label.replace("/", " ")
    # _prefix = prefix.replace("/", "")
    # filename = f"{_prefix}-baselines-{xx_label}-{yy_label}--{epoch}"
    # plt.title(epoch)
    # plt.savefig(filename + ".png", bbox_inches="tight")
    # plt.close(fig)
