import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


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
    return np.stack([g1, g2], axis=0)  # [T=2, D=2]


def scalar_loss(theta, lam):
    return lam @ task_losses(theta)


def scalar_grad(theta, lam):
    return lam @ task_grads(theta)


def scalar_hessian(lam):
    return lam[0] * Q1 + lam[1] * Q2


def scalar_minimizer(lam):
    H = scalar_hessian(lam)
    b = lam[0] * Q1 @ MU1 + lam[1] * Q2 @ MU2
    return np.linalg.solve(H, b)


# ============================================================
# 2. Optimizers
# ============================================================

def run_optimizer(
    method,
    x0,
    lams,
    lr=0.02,
    beta1=0.9,
    beta2=0.98,
    eps=1e-8,
    target_lam=None,
):
    """
    method:
      "adam"  : Adam first moment + Adam scalar-gradient second moment.
      "madam" : Adam first moment + ray-conditioned pairwise F_ij denominator.

    target_lam:
      Ray used for diagnostics. Usually lambda_B, so we can show how the
      new-ray objective behaves before and after the switch.
    """

    theta = np.array(x0, dtype=np.float64)

    # Adam-style numerator
    m = np.zeros_like(theta)

    # Adam denominator state
    v = np.zeros_like(theta)

    # MAdam denominator state:
    # F[i, j, d] stores EMA of g_i[d] * g_j[d]
    T, D = 2, 2
    F = np.zeros((T, T, D), dtype=np.float64)

    path = [theta.copy()]
    denom_path = []

    # Include initial target-ray loss at optimization step 0.
    target_loss_path = []
    if target_lam is not None:
        target_loss_path.append(scalar_loss(theta, target_lam))

    # One value per update step, so length is n_steps.
    eff_kappa_path = []

    for step, lam in enumerate(lams, start=1):
        G = task_grads(theta)       # [T, D]
        g = lam @ G                 # scalarized gradient, [D]

        # Adam numerator, shared by both Adam and MAdam
        m = beta1 * m + (1.0 - beta1) * g
        m_hat = m / (1.0 - beta1 ** step)

        if method == "adam":
            # Adam's ray-agnostic second moment
            v = beta2 * v + (1.0 - beta2) * (g * g)
            denom_state = v / (1.0 - beta2 ** step)

        elif method == "madam":
            # Pairwise task-gradient Fisher / second-moment terms
            pair_products = G[:, None, :] * G[None, :, :]  # [T, T, D]
            F = beta2 * F + (1.0 - beta2) * pair_products
            F_hat = F / (1.0 - beta2 ** step)

            # Assemble ray-conditioned denominator:
            # F(lambda)[d] = sum_ij lambda_i lambda_j F_ij[d]
            denom_state = np.einsum("i,j,ijd->d", lam, lam, F_hat)

            # Cross terms can be negative; mirror MAdam abs-before-sqrt.
            denom_state = np.abs(denom_state)

        else:
            raise ValueError(f"Unknown method: {method}")

        denom = np.sqrt(np.maximum(denom_state, 0.0)) + eps

        # Parameter update
        theta = theta - lr * m_hat / denom

        path.append(theta.copy())
        denom_path.append(denom.copy())

        # Diagnostics under the target ray, usually lambda_B.
        if target_lam is not None:
            target_loss_path.append(scalar_loss(theta, target_lam))

            H_target = scalar_hessian(target_lam)

            # Effective preconditioned Hessian:
            # D^{-1/2} H_lambda_B D^{-1/2}
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(denom, 1e-12)))
            H_eff = D_inv_sqrt @ H_target @ D_inv_sqrt

            eigvals = np.linalg.eigvalsh(H_eff)
            eigvals = np.maximum(eigvals, 1e-12)
            eff_kappa = eigvals[-1] / eigvals[0]
            eff_kappa_path.append(eff_kappa)

    return {
        "path": np.array(path),                    # length n_steps + 1
        "denom": np.array(denom_path),             # length n_steps
        "target_loss": np.array(target_loss_path),  # length n_steps + 1
        "eff_kappa": np.array(eff_kappa_path),     # length n_steps
    }


# ============================================================
# 3. Experiment: train on one ray, then switch rays
# ============================================================

def make_ray_schedule(warmup_steps=120, switch_steps=120):
    lam_A = np.array([0.95, 0.05])
    lam_B = np.array([0.05, 0.95])

    lams = []
    for _ in range(warmup_steps):
        lams.append(lam_A.copy())
    for _ in range(switch_steps):
        lams.append(lam_B.copy())

    return np.array(lams), lam_A, lam_B


# ============================================================
# 4. Plotting
# ============================================================

def plot_results(
    adam,
    madam,
    lam_A,
    lam_B,
    warmup_steps,
    out_path="madam_ray_switch_demo_v4_two_panel.pdf",
):
    rcParams["font.family"] = "serif"
    rcParams["mathtext.fontset"] = "cm"

    theta_star_A = scalar_minimizer(lam_A)
    theta_star_B = scalar_minimizer(lam_B)

    # Colors chosen to remain visible on RdBu_r.
    ADAM_COLOR = "#4B0082"      # dark indigo / purple
    Madam_COLOR = "#FFD23F"     # gold
    ADAM_FAINT = "#4B0082"
    Madam_FAINT = "#B8860B"     # darker gold for dashed pre-switch

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2))

    # ------------------------------------------------------------
    # Panel A: full trajectory, before and after lambda switch
    # ------------------------------------------------------------

    # Square crop:
    # Removes most negative-y red space while keeping equal x/y span.
    y_min = 0.0
    y_max = 5.8
    span = y_max - y_min

    x_center = -0.25
    x_min = x_center - span / 2.0
    x_max = x_center + span / 2.0

    xs = np.linspace(x_min, x_max, 500)
    ys = np.linspace(y_min, y_max, 500)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            theta = np.array([X[i, j], Y[i, j]])
            Z[i, j] = scalar_loss(theta, lam_B)

    # Log landscape: spreads the basin and controls red saturation.
    log_Z = np.log10(Z + 1.0)
    vmin = log_Z.min()
    vmax = vmin + 0.78 * (log_Z.max() - vmin)
    levels = np.linspace(vmin, log_Z.max(), 80)

    ax.contourf(
        X,
        Y,
        log_Z,
        levels=levels,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        extend="max",
        alpha=0.97,
    )
    ax.contour(
        X,
        Y,
        log_Z,
        levels=levels[::3],
        colors="black",
        linewidths=0.28,
        alpha=0.25,
    )

    adam_path = adam["path"]
    madam_path = madam["path"]

    # Include the switch point in both phases.
    adam_pre = adam_path[:warmup_steps + 1]
    adam_post = adam_path[warmup_steps:]

    madam_pre = madam_path[:warmup_steps + 1]
    madam_post = madam_path[warmup_steps:]

    # Pre-switch: dashed and transparent
    ax.plot(
        adam_pre[:, 0],
        adam_pre[:, 1],
        color=ADAM_FAINT,
        lw=1.8,
        linestyle="--",
        alpha=0.42,
        label=r"Adam before switch",
    )
    ax.plot(
        madam_pre[:, 0],
        madam_pre[:, 1],
        color=Madam_FAINT,
        lw=1.8,
        linestyle="--",
        alpha=0.48,
        label=r"MAdam before switch",
    )

    # Post-switch: solid and dominant
    ax.plot(
        adam_post[:, 0],
        adam_post[:, 1],
        color=ADAM_COLOR,
        lw=2.35,
        label=r"Adam after switch",
        zorder=5,
    )
    ax.plot(
        madam_post[:, 0],
        madam_post[:, 1],
        color=Madam_COLOR,
        lw=2.55,
        label=r"MAdam after switch",
        zorder=6,
    )

    # Start point
    x0 = adam_path[0]
    ax.scatter(
        x0[0],
        x0[1],
        s=90,
        facecolor="white",
        edgecolor="black",
        linewidth=1.25,
        zorder=9,
    )
    ax.annotate(
        "start",
        xy=x0,
        xytext=(x0[0] + 0.12, x0[1] - 0.22),
        fontsize=11,
        style="italic",
        zorder=10,
    )

    # Switch points: smaller boxes
    adam_switch = adam_path[warmup_steps]
    madam_switch = madam_path[warmup_steps]

    ax.scatter(
        adam_switch[0],
        adam_switch[1],
        s=44,
        marker="s",
        color=ADAM_COLOR,
        edgecolor="black",
        linewidth=0.85,
        zorder=9,
    )
    ax.scatter(
        madam_switch[0],
        madam_switch[1],
        s=44,
        marker="s",
        color=Madam_COLOR,
        edgecolor="black",
        linewidth=0.85,
        zorder=9,
    )

    ax.annotate(
        r"switch to $\lambda_B$",
        xy=madam_switch,
        xytext=(madam_switch[0] + 0.85, madam_switch[1] - 0.70),
        fontsize=10,
        style="italic",
        arrowprops=dict(arrowstyle="->", lw=0.9, color="black"),
        zorder=10,
    )

    # Endpoints for Adam and MAdam
    adam_end = adam_path[-1]
    madam_end = madam_path[-1]

    ax.scatter(
        adam_end[0],
        adam_end[1],
        s=78,
        marker="o",
        color=ADAM_COLOR,
        edgecolor="black",
        linewidth=1.0,
        zorder=10,
    )
    ax.scatter(
        madam_end[0],
        madam_end[1],
        s=88,
        marker="o",
        color=Madam_COLOR,
        edgecolor="black",
        linewidth=1.0,
        zorder=11,
    )

    ax.annotate(
        "Adam end",
        xy=adam_end,
        xytext=(adam_end[0] - 1.05, adam_end[1] + 0.32),
        fontsize=9,
        arrowprops=dict(arrowstyle="-", lw=0.8, color="black", alpha=0.7),
        zorder=11,
    )
    ax.annotate(
        "MAdam end",
        xy=madam_end,
        xytext=(madam_end[0] - 0.40, madam_end[1] + 0.55),
        fontsize=9,
        arrowprops=dict(arrowstyle="-", lw=0.8, color="black", alpha=0.7),
        zorder=11,
    )

    # Optimum under old ray
    ax.scatter(
        theta_star_A[0],
        theta_star_A[1],
        marker="*",
        s=140,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=8,
    )
    ax.annotate(
        r"$\theta^\star_{\lambda_A}$",
        xy=theta_star_A,
        xytext=(theta_star_A[0] - 0.20, theta_star_A[1] - 0.55),
        fontsize=10,
        arrowprops=dict(arrowstyle="-", lw=0.8, color="black", alpha=0.6),
        zorder=9,
    )

    # Optimum under new ray
    ax.scatter(
        theta_star_B[0],
        theta_star_B[1],
        marker="*",
        s=205,
        color="#E71D36",
        edgecolor="black",
        linewidth=0.8,
        zorder=9,
    )
    ax.annotate(
        r"$\theta^\star_{\lambda_B}$",
        xy=theta_star_B,
        xytext=(theta_star_B[0] - 1.05, theta_star_B[1] + 0.20),
        fontsize=10,
        arrowprops=dict(arrowstyle="-", lw=0.8, color="black", alpha=0.6),
        zorder=9,
    )

    ax.set_title(
        r"Trajectory before and after $\lambda_A \rightarrow \lambda_B$",
        fontsize=12,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend(frameon=True, fontsize=7, loc="upper right")

    # Right panel disabled:
    # The former "Convergence measured on new ray lambda_B" plot is omitted
    # so this output only contains the trajectory panel.

    fig.text(
        0.5,
        0.03,
        "Weighting Mismatch",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


# ============================================================
# 5. Main
# ============================================================

def main():
    warmup_steps = 120
    switch_steps = 120

    lams, lam_A, lam_B = make_ray_schedule(
        warmup_steps=warmup_steps,
        switch_steps=switch_steps,
    )

    x0 = np.array([-3.0, 3.0])

    # Same hyperparameters for both methods in this mechanistic figure.
    lr = 0.02
    beta1 = 0.9
    beta2 = 0.98

    adam = run_optimizer(
        method="adam",
        x0=x0,
        lams=lams,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        target_lam=lam_B,
    )

    madam = run_optimizer(
        method="madam",
        x0=x0,
        lams=lams,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        target_lam=lam_B,
    )

    plot_results(
        adam=adam,
        madam=madam,
        lam_A=lam_A,
        lam_B=lam_B,
        warmup_steps=warmup_steps,
        out_path="madam_ray_switch_demo_v4_two_panel.pdf",
    )


if __name__ == "__main__":
    main()
