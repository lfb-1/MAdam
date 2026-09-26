import copy
import json
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV

import wandb
from src.callbacks.callback import Callback
from src.trainer.base_trainer import BaseTrainer
from src.trainer.ensemble_trainer import EnsembleTrainer

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer
import meshzoo
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import Delaunay
import plotly.express as px


def compute_hypervolume(points, xmax=True, ymax=True, zmax=True):
    points = copy.deepcopy(points)

    ref = [1, 1, 1]
    if xmax:
        ref[0] = 0
        points[:, 0] = -points[:, 0]
    if ymax:
        ref[1] = 0
        points[:, 1] = -points[:, 1]
    if zmax:
        ref[2] = 0
        points[:, 2] = -points[:, 2]
    ind = HV(ref_point=np.array(ref))
    return ind(points)


MAPPING = {
    "loss/top-left": False,
    "loss/bottom-right": False,
    "loss/Task-1": False,
    "loss/Task-2": False,
    "loss/Task-3": False,
    "acc/Task-1": True,
    "acc/Task-2": True,
    "acc/Task-3": True,
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
    #
    "loss/age": False,
    "huber/age": False,
    "loss/race": False,
    "loss/gender": False,
    "age/race": False,
    "age/gender": False,
}


def is_dominated(p1, p2, maximize=True):
    """Check if point p1 is dominated by point p2."""
    if maximize:
        p1 = -p1
        p2 = -p2
    return (p1[0] >= p2[0] and p1[1] >= p2[1] and p1[2] >= p2[2]) and (p1[0] > p2[0] or p1[1] > p2[1] or p1[2] > p2[2])


def pareto_front(points, maximize=True, return_indices=False):
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
            if i != j and is_dominated(points[i], points[j], maximize=maximize):
                is_nondominated[i] = False
                break

    if return_indices:
        return np.sum(is_nondominated), is_nondominated

    return np.sum(is_nondominated)


def show_points(
    x,
    y,
    z,
    xlabel,
    ylabel,
    zlabel,
    epoch,
    prefix="val/",
    figsize=(7, 7),
    num_points=10,
):
    prefix = prefix.replace("/", "")
    vpoints = np.array([x, y, z]).T

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    points, _ = meshzoo.triangle(num_points)
    print("num points", points.shape)
    points = points.T

    np.save(f"vpoints_{prefix}_{epoch}.npy", vpoints)
    np.save(f"points_{prefix}_{epoch}.npy", points)

    metric = "loss" if "loss" in xlabel and "loss" in ylabel and "loss" in zlabel else "acc"
    n, is_non_dominated = pareto_front(vpoints, maximize=metric == "acc", return_indices=True)
    assert len(is_non_dominated) == len(x) == len(y) == len(z)
    colors = ["red" if is_non_dominated[i] else "blue" for i in range(len(x))]

    wandb.log({"{}/{}/num_non_dominated1".format(prefix, metric): n, "epoch": epoch})

    ax.scatter(vpoints[:, 0], vpoints[:, 1], vpoints[:, 2], c=colors, s=20)

    try:
        tri = Delaunay(points[:, :2])
        for simplex in tri.simplices:
            simplex = np.append(simplex, simplex[0])
            ax.plot(vpoints[simplex, 0], vpoints[simplex, 1], vpoints[simplex, 2], "b-")
    except (IndexError, Exception):
        pass  # triangulation may fail with few evaluation points

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_title("Non-dominated points: {} out of {}".format(n, len(x)))

    log_name = f"PF-{prefix}-{metric}"
    filename = log_name + f"-{epoch}.png"
    html_filename = log_name + f"-{epoch}.html"
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saving {os.path.abspath(filename)}")
    wandb.log(
        {
            "epoch": epoch,
            log_name: wandb.Image(filename, caption=log_name),
        }
    )

    fig = px.scatter_3d(
        x=x,
        y=y,
        z=z,
        color=colors,
        title=f"{xlabel} vs {ylabel} vs {zlabel}",
        labels={xlabel: xlabel, ylabel: ylabel, zlabel: zlabel},
    )

    fig.write_html(html_filename)
    print(f"Saving {os.path.abspath(html_filename)} in {os.path.pardir}")
    # Add Plotly figure as HTML file into Table
    # Create a table
    # table = wandb.Table(columns=["plotly_figure"])
    # table.add_data(wandb.Html(f"{log_name}.html"))
    # wandb.log({log_name + "_plotly": table})


class ParetoFrontVisualizer3dCallback(Callback):

    def __init__(self):
        super().__init__()
        self.discounted_metric = 0
        self.discounted_metric_normalized = 0

    def on_before_fit(self, trainer: BaseTrainer, *args, **kwargs):

        match trainer.benchmark.name:
            case "multimnist3":
                self.pairs = [
                    tuple("loss/" + t for t in trainer.task_names),
                    tuple("acc/" + t for t in trainer.task_names),
                ]
            case "UTKFace":
                self.pairs = [
                    ("huber/age", "loss/gender", "loss/race"),
                    ("huber/age", "acc/gender", "acc/race"),
                ]
            case _:
                raise ValueError(f"benchmark {trainer.benchmark.name} not supported")

        return super().on_before_fit(trainer, *args, **kwargs)

    def on_after_validating_interpolations(self, trainer: "BaseTrainer", *args, **kwargs):
        population_results = self.dotheplots(trainer, prefix="val/")
        population_results = {f"pop/{k}": v for k, v in population_results.items()}
        results = trainer.results
        results["population"] = population_results
        wandb.log({"epoch": trainer.current_epoch, **population_results})

    def on_after_predicting_interpolations(self, trainer: "BaseTrainer", *args, **kwargs):
        population_results = self.dotheplots(trainer, prefix="test/")
        results = trainer.results
        results["population"] = population_results
        # json.dump(results, open(f"test.json", "w"), indent=4)

    def dotheplots(self, trainer: EnsembleTrainer, prefix):
        results = trainer.results
        population_results = {}

        for x_label, y_label, z_label in self.pairs:
            print(f"Plotting {x_label} vs {y_label} vs {z_label}")
            x = self.extract_metric(results, x_label)
            y = self.extract_metric(results, y_label)
            z = self.extract_metric(results, z_label)

            if "acc" in z_label and "huber" in x_label:
                x = [-v for v in x]

            show_points(
                x,
                y,
                z,
                x_label,
                y_label,
                z_label,
                epoch=trainer.current_epoch,
                prefix=prefix,
                num_points=trainer.validate_models - 1,
            )

            if ("loss" in x_label or "huber" in x_label) and "loss" in y_label and "loss" in z_label:
                hypervolume = compute_hypervolume(
                    np.array([x, y, z]).T, xmax=MAPPING[x_label], ymax=MAPPING[y_label], zmax=MAPPING[z_label]
                )
                print(f"Hypervolume: {hypervolume}")

                points = np.array([x, y, z]).T
                num_non_dominated = pareto_front(points, maximize=False)

                self.discounted_metric = 0.9 * self.discounted_metric + num_non_dominated
                self.discounted_metric_normalized = 0.9 * self.discounted_metric_normalized + num_non_dominated / len(
                    x
                )
                self.new_metric = hypervolume + self.discounted_metric_normalized / len(x)
                if trainer.benchmark.name == "multimnist3":
                    self.new_metric = num_non_dominated + sum(
                        [
                            -(max(self.extract_metric(results, label)) < 0.9) * 100
                            for label in ["acc/Task-1", "acc/Task-2", "acc/Task-3"]
                        ]
                    )
                    print("New metric", self.new_metric)
                if trainer.benchmark.name == "UTKFace":
                    # self.new_metric = num_non_dominated + sum(
                    #     [
                    #         -(max(self.extract_metric(results, label)) < 0.9) * 100
                    #         for label in ["acc/race", "acc/gender", "acc/Task-3"]
                    #     ]
                    # )
                    print("New metric", self.new_metric)

                _prefix = prefix.replace("/", "")
                wandb.log(
                    {
                        f"val_epoch": trainer.current_epoch,
                        f"population/{_prefix}/hypervolume/{x_label}-{y_label}": hypervolume,
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

                population_results["hypervolume"] = hypervolume
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
