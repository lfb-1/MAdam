"""Decision-space grid figure on TiltedPair2D.

Generate one grid figure:
  - rows are lambda combinations
  - columns are step snapshots T, default {200, 300, 400, 500, 1000}
  - each cell overlays trajectories from three starting points
  - Adam is red, MAdam is green, and line style identifies the start

Default sweep:
  5 fixed lambda combinations:
  (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5)

Run from the MTL root:

    MPLCONFIGDIR=/tmp/mplconfig python visualization/plot_decision_space_perlam.py

Outputs ``decision_space_perlam.{pdf,png}`` with an embedded legend, plus
``decision_space_perlam_legend.{pdf,png}`` next to this file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toy_problems import TiltedPair2D  # noqa: E402
from runners import run_adam, run_madam  # noqa: E402


ADAM_C = "#d62728"
MADAM_C = "#00cc00"
PARETO_C = "#111111"
START_C = "#222222"
F1_MIN_C = "#9467bd"
F2_MIN_C = "#00d4ff"
TARGET_C = "#ff7f0e"

START_STYLES = ["-", "--", ":"]
START_MARKERS = ["o", "s", "^"]


def parse_start_point(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"start point must be formatted as x,y; got {value!r}"
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"start point must contain numeric x,y values; got {value!r}"
        ) from exc


def output_prefix_for_legend(base: Path) -> Path:
    if base.suffix in {".pdf", ".png"}:
        base = base.with_suffix("")
    return base.with_name(f"{base.name}_legend")


def build_rays(lambda_start: float, lambda_end: float, num_rays: int) -> np.ndarray:
    lambda_1 = np.linspace(lambda_start, lambda_end, num_rays)
    return np.stack([lambda_1, 1.0 - lambda_1], axis=-1)


def validate_args(args: argparse.Namespace) -> None:
    if not args.snapshots:
        raise ValueError("--snapshots must contain at least one step.")
    if min(args.snapshots) < 0:
        raise ValueError("--snapshots must be nonnegative.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1).")
    if args.num_rays <= 0:
        raise ValueError("--num-rays must be positive.")
    if not 0.0 <= args.lambda_start <= 1.0:
        raise ValueError("--lambda-start must be in [0, 1].")
    if not 0.0 <= args.lambda_end <= 1.0:
        raise ValueError("--lambda-end must be in [0, 1].")
    if not args.start_points:
        raise ValueError("--start-points must contain at least one point.")
    if len(args.start_points) > len(START_STYLES):
        raise ValueError(f"at most {len(START_STYLES)} start points are supported.")


def add_landscape(
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    f_lam: np.ndarray,
    cmap: str,
) -> None:
    f_floor = max(float(f_lam.min()), 1e-3)
    shifted = f_lam + f_floor
    norm = LogNorm(vmin=f_floor + 1e-3, vmax=float(shifted.max()))
    levels = np.geomspace(f_floor + 1e-2, float(shifted.max()), 12)
    ax.pcolormesh(X, Y, shifted, cmap=cmap, norm=norm,
                  shading="auto", alpha=0.95, zorder=0, rasterized=True)
    contours = ax.contour(X, Y, shifted, levels=levels, colors="white",
                          linewidths=0.5, alpha=0.6, zorder=1)
    for collection in getattr(contours, "collections", []):
        collection.set_rasterized(True)


def build_legend_handles(starts: list[tuple[float, float]]) -> list[plt.Line2D]:
    handles = [
        plt.Line2D([0], [0], color=ADAM_C, lw=2.0, label="Adam trajectory"),
        plt.Line2D([0], [0], color=MADAM_C, lw=2.0, label="MAdam trajectory"),
        plt.Line2D([0], [0], color=PARETO_C, lw=2.0, label="Pareto set"),
        plt.Line2D([0], [0], marker="x", color=TARGET_C, lw=0,
                   ms=10, mew=2.5, label=r"target $x^*(\lambda)$"),
    ]
    start_labels = ["start point (solid)", "start point (dashed)",
                    "start point (dotted)"]
    for idx, start in enumerate(starts):
        handles.append(
            plt.Line2D(
                [0], [0], color=START_C, lw=2.0,
                linestyle=START_STYLES[idx], marker=START_MARKERS[idx], ms=7,
                label=start_labels[idx],
            )
        )
    handles.extend([
        plt.Line2D([0], [0], marker="X", color=F1_MIN_C, lw=0, ms=10,
                   label=r"$f_1$ minimum"),
        plt.Line2D([0], [0], marker="X", color=F2_MIN_C, lw=0, ms=10,
                   label=r"$f_2$ minimum"),
    ])
    return handles


def render_legend_figure(
    starts: list[tuple[float, float]],
    out_prefix: Path,
) -> None:
    handles = build_legend_handles(starts)
    fig = plt.figure(figsize=(8.6, 1.4), constrained_layout=True)
    fig.legend(handles=handles, loc="center", ncol=4, fontsize=10,
               framealpha=0.92)
    out_pdf = out_prefix.with_suffix(".pdf")
    out_png = out_prefix.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}\nwrote {out_png}")


def render_grid_figure(
    *,
    rays: np.ndarray,
    targets: np.ndarray,
    starts: list[tuple[float, float]],
    all_logs: list[dict[str, list]],
    problem: TiltedPair2D,
    snapshots: list[int],
    X: np.ndarray,
    Y: np.ndarray,
    f1_grid: np.ndarray,
    f2_grid: np.ndarray,
    pareto_xs: np.ndarray,
    grid_lo: float,
    grid_hi: float,
    cmap: str,
    out_prefix: Path,
) -> None:
    n_rows = len(rays)
    n_cols = len(snapshots)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.9 * n_cols, 3.75 * n_rows + 1.0),
        constrained_layout=True,
        squeeze=False,
    )

    for r, ray in enumerate(rays):
        f_lam = ray[0] * f1_grid + ray[1] * f2_grid
        for c, T in enumerate(snapshots):
            ax = axes[r, c]
            add_landscape(ax, X, Y, f_lam, cmap)

            ax.plot(pareto_xs[:, 0], pareto_xs[:, 1], "-", color=PARETO_C,
                    lw=1.8, zorder=4)
            ax.plot(targets[r, 0], targets[r, 1], "x", color=TARGET_C,
                    ms=13, mew=3, zorder=8)
            ax.plot(problem.c1[0], problem.c1[1], "X", color=F1_MIN_C,
                    ms=10, mec="white", mew=0.8, zorder=7)
            ax.plot(problem.c2[0], problem.c2[1], "X", color=F2_MIN_C,
                    ms=10, mec="white", mew=0.8, zorder=7)

            adam_distances = []
            madam_distances = []
            for idx, start in enumerate(starts):
                style = START_STYLES[idx]
                marker = START_MARKERS[idx]
                a = all_logs[r]["adam"][idx]
                m = all_logs[r]["madam"][idx]

                ax.plot(a.xs[: T + 1, 0], a.xs[: T + 1, 1], linestyle=style,
                        color=ADAM_C, lw=1.8, alpha=0.95, zorder=3)
                ax.plot(m.xs[: T + 1, 0], m.xs[: T + 1, 1], linestyle=style,
                        color=MADAM_C, lw=1.8, alpha=0.95, zorder=3)

                ax.plot(start[0], start[1], marker=marker, color=START_C, ms=7,
                        mec="white", mew=0.8, zorder=7)
                ax.plot(a.xs[T, 0], a.xs[T, 1], "*", color=ADAM_C, ms=12,
                        mec="white", mew=0.8, zorder=6)
                ax.plot(m.xs[T, 0], m.xs[T, 1], "o", color=MADAM_C, ms=9,
                        mec="white", mew=0.8, zorder=6)

                adam_distances.append(float(np.linalg.norm(a.xs[T] - targets[r])))
                madam_distances.append(float(np.linalg.norm(m.xs[T] - targets[r])))

            ax.set_xlim(grid_lo, grid_hi)
            ax.set_ylim(grid_lo, grid_hi)
            ax.set_aspect("equal")
            ax.tick_params(axis="both", labelsize=10)

            if r == n_rows - 1:
                ax.set_xlabel(r"$x_1$", fontsize=12)
            if c == 0:
                ax.set_ylabel(
                    fr"$\lambda=({ray[0]:.2f},{ray[1]:.2f})$"
                    "\n"
                    r"$x_2$",
                    fontsize=12,
                )

            title_lines = []
            if r == 0:
                title_lines.append(f"T = {T}")
            title_lines.append(
                f"A {np.mean(adam_distances):.3f} | "
                f"M {np.mean(madam_distances):.3f}"
            )
            ax.set_title("\n".join(title_lines), fontsize=11, pad=6)

    handles = build_legend_handles(starts)
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.035), ncol=5, fontsize=11,
               framealpha=0.92)

    out_pdf = out_prefix.with_suffix(".pdf")
    out_png = out_prefix.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}\nwrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=int, nargs="+",
                        default=[200, 300, 400, 500, 1000])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--out", type=str,
                        default=str(HERE / "decision_space_perlam"))
    parser.add_argument("--cmap", type=str, default="viridis_r")
    parser.add_argument("--lambda-start", type=float, default=0.1)
    parser.add_argument("--lambda-end", type=float, default=0.5)
    parser.add_argument("--num-rays", type=int, default=5)
    parser.add_argument("--start-points", type=parse_start_point, nargs="+",
                        default=[(0.0, -2.0), (-2.0, -2.0), (2.0, -2.0)])
    args = parser.parse_args()
    validate_args(args)

    t0 = time.time()
    problem = TiltedPair2D()
    rays = build_rays(args.lambda_start, args.lambda_end, args.num_rays)
    starts = list(args.start_points)
    snapshots = list(args.snapshots)
    max_steps = max(snapshots)

    print(
        f"Running {len(rays)} rays x {len(starts)} starts x "
        f"2 optimizers x {max_steps} steps..."
    )
    all_logs = []
    for ray in rays:
        ray_logs = {"adam": [], "madam": []}
        for start in starts:
            ray_logs["adam"].append(
                run_adam(problem, ray.tolist(), start,
                         lr=args.lr, steps=max_steps, seed=0)
            )
            ray_logs["madam"].append(
                run_madam(problem, ray.tolist(), start,
                          lr=args.lr, steps=max_steps, seed=0,
                          ema_decay=args.ema_decay,
                          gradient_balance=True)
            )
        all_logs.append(ray_logs)
    print(f"  done in {time.time()-t0:.1f}s")

    angles = np.linspace(1e-4, np.pi / 2 - 1e-4, 400)
    lam_set = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    lam_set = lam_set / lam_set.sum(axis=-1, keepdims=True)
    pareto_xs = problem.pareto_x(lam_set)
    targets = problem.pareto_x(rays)

    grid_lo, grid_hi = -3.5, 3.5
    grid = np.linspace(grid_lo, grid_hi, 300)
    X, Y = np.meshgrid(grid, grid)
    pts = np.stack([X, Y], axis=-1)
    f1_grid, f2_grid = problem.f_numpy(pts)

    out_prefix = Path(args.out)
    if out_prefix.suffix in {".pdf", ".png"}:
        out_prefix = out_prefix.with_suffix("")
    render_grid_figure(
            rays=rays,
            targets=targets,
            starts=starts,
            all_logs=all_logs,
            problem=problem,
            snapshots=snapshots,
            X=X,
            Y=Y,
            f1_grid=f1_grid,
            f2_grid=f2_grid,
            pareto_xs=pareto_xs,
            grid_lo=grid_lo,
            grid_hi=grid_hi,
            cmap=args.cmap,
            out_prefix=out_prefix,
    )

    render_legend_figure(starts, output_prefix_for_legend(Path(args.out)))


if __name__ == "__main__":
    main()
