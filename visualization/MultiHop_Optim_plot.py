import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ============================================================
# 0. Configuration
# ============================================================

OUT_PATH = "madam_multiphase_demo_v3_per_phase_grid.pdf"


# ============================================================
# 1. Problem: two tasks with ray-dependent curvature
# ============================================================

KAPPA = 120.0

Q1 = np.diag([KAPPA, 1.0])   # task 1: stiff in x
Q2 = np.diag([1.0, KAPPA])   # task 2: stiff in y

MU1 = np.array([2.2, -1.8])
MU2 = np.array([-1.8, 2.2])


def task_losses(theta):
    d1 = theta - MU1
    d2 = theta - MU2
    l1 = 0.5 * d1 @ Q1 @ d1
    l2 = 0.5 * d2 @ Q2 @ d2
    return np.array([l1, l2])


def task_grads(theta):
    g1 = Q1 @ (theta - MU1)
    g2 = Q2 @ (theta - MU2)
    return np.stack([g1, g2], axis=0)


def scalar_loss(theta, lam):
    return float(lam @ task_losses(theta))


def scalar_grad(theta, lam):
    return lam @ task_grads(theta)


def scalar_hessian(lam):
    return lam[0] * Q1 + lam[1] * Q2


def scalar_minimizer(lam):
    H = scalar_hessian(lam)
    b = lam[0] * Q1 @ MU1 + lam[1] * Q2 @ MU2
    return np.linalg.solve(H, b)


# ============================================================
# 2. Multi-phase lambda schedule
# ============================================================

def make_multiphase_schedule():
    lam_A = np.array([0.95, 0.05], dtype=np.float64)
    lam_B = np.array([0.05, 0.95], dtype=np.float64)

    phases = [
        (r"\lambda_A", lam_A, 70),
        (r"\lambda_B", lam_B, 45),
        (r"\lambda_A", lam_A, 45),
        (r"\lambda_B", lam_B, 45),
        (r"\lambda_A", lam_A, 45),
        (r"\lambda_B", lam_B, 90),
    ]

    lams = []
    phase_names = []
    phase_lams = []
    phase_lengths = []

    for name, lam, n_steps in phases:
        phase_names.append(name)
        phase_lams.append(lam)
        phase_lengths.append(n_steps)

        for _ in range(n_steps):
            lams.append(lam.copy())

    lams = np.array(lams, dtype=np.float64)

    cum = np.concatenate([[0], np.cumsum(phase_lengths)])
    phase_starts = cum[:-1]   # length n_phases
    phase_ends = cum[1:]      # length n_phases

    return {
        "lams": lams,
        "phase_names": phase_names,
        "phase_lams": phase_lams,
        "phase_lengths": phase_lengths,
        "phase_starts": phase_starts,
        "phase_ends": phase_ends,
        "lam_A": lam_A,
        "lam_B": lam_B,
        "final_lam": phase_lams[-1],
        "final_name": phase_names[-1],
    }


# ============================================================
# 3. Optimizers
# ============================================================

def run_optimizer(
    method,
    x0,
    lams,
    lr=0.02,
    beta1=0.9,
    beta2=0.98,
    eps=1e-8,
):
    theta = np.array(x0, dtype=np.float64)

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    T, D = 2, 2
    F = np.zeros((T, T, D), dtype=np.float64)

    path = [theta.copy()]

    for step, lam in enumerate(lams, start=1):
        G = task_grads(theta)
        g = lam @ G

        m = beta1 * m + (1.0 - beta1) * g
        m_hat = m / (1.0 - beta1 ** step)

        if method == "adam":
            v = beta2 * v + (1.0 - beta2) * (g * g)
            denom_state = v / (1.0 - beta2 ** step)
        elif method == "madam":
            pair_products = G[:, None, :] * G[None, :, :]
            F = beta2 * F + (1.0 - beta2) * pair_products
            F_hat = F / (1.0 - beta2 ** step)
            denom_state = np.einsum("i,j,ijd->d", lam, lam, F_hat)
            denom_state = np.abs(denom_state)
        else:
            raise ValueError(f"Unknown method: {method}")

        denom = np.sqrt(np.maximum(denom_state, 0.0)) + eps

        theta = theta - lr * m_hat / denom
        path.append(theta.copy())

    return {"path": np.array(path)}


# ============================================================
# 4. Plotting helpers
# ============================================================

def compute_contour(lam, x_min, x_max, y_min, y_max, n=400):
    xs = np.linspace(x_min, x_max, n)
    ys = np.linspace(y_min, y_max, n)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            theta = np.array([X[i, j], Y[i, j]])
            Z[i, j] = scalar_loss(theta, lam)

    log_Z = np.log10(Z + 1.0)
    vmin = log_Z.min()
    vmax = vmin + 0.78 * (log_Z.max() - vmin)
    levels = np.linspace(vmin, log_Z.max(), 80)

    return X, Y, log_Z, vmin, vmax, levels


def compute_active_loss_gap(path, lams):
    """L_{lambda_t}(theta_t) - L_{lambda_t}^* per step.

    path[0] is the initial point; path[k] for k>=1 is the state after the
    k-th update made under lams[k-1]. We attribute path[0] to lams[0] so the
    pre-update gap is meaningful.
    """
    n = len(path)
    gap = np.zeros(n)
    for t in range(n):
        lam = lams[0] if t == 0 else lams[t - 1]
        star = scalar_minimizer(lam)
        L_star = scalar_loss(star, lam)
        gap[t] = scalar_loss(path[t], lam) - L_star
    return gap


def add_switch_lines(ax, boundaries, label_first=True):
    for k, b in enumerate(boundaries):
        ax.axvline(b, color="black", linestyle="--", lw=1.0, alpha=0.55)
        if label_first and k == 0:
            ax.text(
                b + 3, 0.95, "ray switch",
                transform=ax.get_xaxis_transform(),
                fontsize=9, style="italic", va="top",
            )


def add_phase_labels(ax, phase_names, phase_lengths):
    starts = np.concatenate([[0], np.cumsum(phase_lengths)[:-1]])
    ends = np.cumsum(phase_lengths)
    for name, s, e in zip(phase_names, starts, ends):
        mid = 0.5 * (s + e)
        ax.text(
            mid, 0.96, rf"${name}$",
            transform=ax.get_xaxis_transform(),
            fontsize=10, ha="center", va="top",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="black",
                linewidth=0.4,
                alpha=0.85,
            ),
        )


def draw_contour(ax, X, Y, log_Z, vmin, vmax, levels):
    ax.contourf(
        X, Y, log_Z,
        levels=levels,
        cmap="RdBu_r",
        vmin=vmin, vmax=vmax,
        extend="max",
        alpha=0.97,
    )
    ax.contour(
        X, Y, log_Z,
        levels=levels[::3],
        colors="black",
        linewidths=0.25,
        alpha=0.22,
    )


def draw_phase_panel(
    ax,
    path,
    phase_idx,
    phase_starts,
    phase_ends,
    solid_color,
    faint_color,
    contour_args,
    active_star,
    inactive_star,
    active_label,
    inactive_label,
    x_min, x_max, y_min, y_max,
    show_start_marker,
):
    """Draw the trajectory for a single phase on the given axis.

    Past phases are drawn as dashed/faint lines; the current phase is drawn solid.
    The contour landscape and highlighted optimum reflect the active phase's ray.
    """
    draw_contour(ax, *contour_args)

    start = phase_starts[phase_idx]
    end = phase_ends[phase_idx]

    # Past trajectory (steps 0 .. start) drawn dashed/faint
    if start > 0:
        prev = path[:start + 1]
        ax.plot(
            prev[:, 0],
            prev[:, 1],
            color=faint_color,
            lw=1.5,
            linestyle="--",
            alpha=0.45,
            zorder=4,
        )

    # Current phase drawn solid
    cur = path[start:end + 1]
    ax.plot(
        cur[:, 0],
        cur[:, 1],
        color=solid_color,
        lw=2.3,
        zorder=5,
    )

    # Start marker (only on phase 0)
    if show_start_marker and phase_idx == 0:
        x0 = path[0]
        ax.scatter(
            x0[0], x0[1],
            s=70,
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=9,
        )

    # End-of-current-phase marker
    end_pt = cur[-1]
    ax.scatter(
        end_pt[0], end_pt[1],
        s=55,
        color=solid_color,
        edgecolor="black",
        linewidth=0.8,
        zorder=10,
    )

    # Inactive ray optimum: faded reference
    ax.scatter(
        inactive_star[0], inactive_star[1],
        marker="*",
        s=70,
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.55,
        zorder=7,
    )
    # Active ray optimum: highlighted target for this phase
    ax.scatter(
        active_star[0], active_star[1],
        marker="*",
        s=140,
        color="#E71D36",
        edgecolor="black",
        linewidth=0.7,
        zorder=8,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])


# ============================================================
# 5. Plotting
# ============================================================

def plot_results(adam, madam, schedule, out_path=OUT_PATH):
    rcParams["font.family"] = "serif"
    rcParams["mathtext.fontset"] = "cm"

    phase_names = schedule["phase_names"]
    phase_lams = schedule["phase_lams"]
    phase_starts = schedule["phase_starts"]
    phase_ends = schedule["phase_ends"]
    lam_A = schedule["lam_A"]
    lam_B = schedule["lam_B"]

    theta_star_A = scalar_minimizer(lam_A)
    theta_star_B = scalar_minimizer(lam_B)

    # Colors matched to trial-4.py
    ADAM_COLOR = "#4B0082"
    Madam_COLOR = "#FFD23F"
    ADAM_FAINT = "#4B0082"
    Madam_FAINT = "#B8860B"

    n_phases = len(phase_names)

    # Plotting bounds (matched to trial-4.py)
    y_min = 0.0
    y_max = 5.8
    span = y_max - y_min
    x_center = -0.25
    x_min = x_center - span / 2.0
    x_max = x_center + span / 2.0

    # Precompute contours for both unique rays (lam_A and lam_B).
    contour_A = compute_contour(lam_A, x_min, x_max, y_min, y_max)
    contour_B = compute_contour(lam_B, x_min, x_max, y_min, y_max)

    # Layout: 2 x n_phases trajectory grid on the left,
    # one tall loss panel on the right spanning both rows.
    mosaic = [
        [f"a{i}" for i in range(n_phases)] + ["loss"],
        [f"m{i}" for i in range(n_phases)] + ["loss"],
    ]
    side_units = 3.0
    fig, ax_dict = plt.subplot_mosaic(
        mosaic,
        figsize=(2.4 * n_phases + 2.4 * side_units, 5.0),
        gridspec_kw={
            "wspace": 0.05,
            "hspace": 0.08,
            "width_ratios": [1] * n_phases + [side_units],
        },
    )
    axes = np.array([
        [ax_dict[f"a{i}"] for i in range(n_phases)],
        [ax_dict[f"m{i}"] for i in range(n_phases)],
    ])
    ax_loss = ax_dict["loss"]

    label_A = r"$\theta^\star_{\lambda_A}$"
    label_B = r"$\theta^\star_{\lambda_B}$"

    # Rows: 0 = Adam, 1 = MAdam
    for phase_idx in range(n_phases):
        active_lam = phase_lams[phase_idx]
        is_A = np.allclose(active_lam, lam_A)
        contour_args = contour_A if is_A else contour_B
        active_star = theta_star_A if is_A else theta_star_B
        inactive_star = theta_star_B if is_A else theta_star_A
        active_label = label_A if is_A else label_B
        inactive_label = label_B if is_A else label_A

        # Adam panel
        ax = axes[0, phase_idx]
        draw_phase_panel(
            ax,
            path=adam["path"],
            phase_idx=phase_idx,
            phase_starts=phase_starts,
            phase_ends=phase_ends,
            solid_color=ADAM_COLOR,
            faint_color=ADAM_FAINT,
            contour_args=contour_args,
            active_star=active_star,
            inactive_star=inactive_star,
            active_label=active_label,
            inactive_label=inactive_label,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            show_start_marker=True,
        )

        # MAdam panel
        ax = axes[1, phase_idx]
        draw_phase_panel(
            ax,
            path=madam["path"],
            phase_idx=phase_idx,
            phase_starts=phase_starts,
            phase_ends=phase_ends,
            solid_color=Madam_COLOR,
            faint_color=Madam_FAINT,
            contour_args=contour_args,
            active_star=active_star,
            inactive_star=inactive_star,
            active_label=active_label,
            inactive_label=inactive_label,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            show_start_marker=True,
        )
        # Phase label below the column (under the MAdam panel).
        ax.set_xlabel(
            rf"phase {phase_idx + 1}: ${phase_names[phase_idx]}$",
            fontsize=10,
            labelpad=6,
        )

    # Row labels on the leftmost column
    axes[0, 0].set_ylabel("Adam", fontsize=12, labelpad=10)
    axes[1, 0].set_ylabel("MAdam", fontsize=12, labelpad=10)

    # Title spanning just the per-phase grid (not the side loss panel).
    fig.canvas.draw()
    left = axes[0, 0].get_position().x0
    right = axes[0, n_phases - 1].get_position().x1
    top = axes[0, 0].get_position().y1
    fig.text(
        0.5 * (left + right),
        top + 0.025,
        r"MAdam rapidly reconditions update geometry across repeated ray switches",
        ha="center", va="bottom",
        fontsize=12,
    )

    # ------------------------------------------------------------
    # Side panel: active-ray loss gap (sawtooth)
    # ------------------------------------------------------------
    boundaries = phase_starts[1:]

    adam_gap = np.maximum(compute_active_loss_gap(
        adam["path"], schedule["lams"]), 1e-12)
    madam_gap = np.maximum(compute_active_loss_gap(
        madam["path"], schedule["lams"]), 1e-12)
    steps_loss = np.arange(len(adam_gap))

    ax_loss.plot(steps_loss, adam_gap, lw=2.25, color=ADAM_COLOR, label="Adam")
    ax_loss.plot(steps_loss, madam_gap, lw=2.45,
                 color=Madam_COLOR, label="MAdam")

    add_switch_lines(ax_loss, boundaries, label_first=False)
    add_phase_labels(ax_loss, phase_names, schedule["phase_lengths"])

    ax_loss.scatter(
        [0], [adam_gap[0]],
        s=55, facecolor="white", edgecolor="black", zorder=6,
    )
    ax_loss.annotate(
        "start",
        xy=(0, adam_gap[0]),
        xytext=(8, adam_gap[0] * 0.35),
        fontsize=10, style="italic",
    )

    ax_loss.scatter(
        [steps_loss[-1]], [adam_gap[-1]],
        s=62, color=ADAM_COLOR, edgecolor="black", zorder=8,
    )
    ax_loss.scatter(
        [steps_loss[-1]], [madam_gap[-1]],
        s=68, color=Madam_COLOR, edgecolor="black", zorder=8,
    )

    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Optimization step")
    ax_loss.set_ylabel(r"$L_{\lambda_t}(\theta_t)-L_{\lambda_t}^{\star}$")
    ax_loss.set_title(r"Convergence under active ray $\lambda_t$", fontsize=12)
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(frameon=True, fontsize=10, loc="lower left")

    # Pad the loss panel away from the grid so the y-label is visible.
    pos = ax_loss.get_position()
    shift = 0.05
    ax_loss.set_position([pos.x0 + shift, pos.y0, pos.width - shift, pos.height])

    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

    # Console diagnostics: end-of-phase active loss gap
    print("\nEnd-of-phase active-ray gaps:")
    print(f"{'phase':<8}{'name':<10}{'Adam':>14}{'MAdam':>14}")
    for i, name in enumerate(phase_names):
        end_idx = phase_ends[i]
        lam = schedule["phase_lams"][i]
        star = scalar_minimizer(lam)
        L_star = scalar_loss(star, lam)
        a_gap = scalar_loss(adam["path"][end_idx], lam) - L_star
        m_gap = scalar_loss(madam["path"][end_idx], lam) - L_star
        print(f"{i+1:<8}{name:<10}{a_gap:>14.3e}{m_gap:>14.3e}")


# ============================================================
# 6. Main
# ============================================================

def main():
    schedule = make_multiphase_schedule()

    x0 = np.array([-3.0, 3.0])

    lr = 0.02
    beta1 = 0.9
    beta2 = 0.98

    adam = run_optimizer(
        method="adam",
        x0=x0,
        lams=schedule["lams"],
        lr=lr,
        beta1=beta1,
        beta2=beta2,
    )

    madam = run_optimizer(
        method="madam",
        x0=x0,
        lams=schedule["lams"],
        lr=lr,
        beta1=beta1,
        beta2=beta2,
    )

    plot_results(adam=adam, madam=madam, schedule=schedule, out_path=OUT_PATH)


if __name__ == "__main__":
    main()
