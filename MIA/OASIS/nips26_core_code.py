
# For reproducing the strongest result, use lambda_mode='fixed'.
# The UW mode is included as an implementation demonstration and may collapse on this task.

"""
MAdam integration snippet — OASIS3 3D super-resolution (RFL codebase).

This is the minimum working example of the MAdam four-phase training
loop as used in our RFL experiments. Drop into any existing trainer and
replace the backward/step block.

Required:
    - madam_core
    - a model that outputs the same shape as the target (here SwinUNETR-3D)
    - per-task filter losses: Sobel-3D, Laplacian-3D, FFT-3D, SSIM(2D mid-slice)

Three lambda regimes are supported:
    - 'fixed'     : lambda_t = 1/T
    - 'dirichlet' : lambda ~ Dir(1, ..., 1)  resampled every step
    - 'uw'        : lambda_t = softmax(-log_var_t) with learnable log_var

The 'uw' regime adds a new param group (log_vars) to the optimizer.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


def _ensure_madam_core() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "madam_core" / "src"
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return
    raise ImportError("Could not locate madam_core/src")


_ensure_madam_core()

from madam_core.madam import MAdamPreconditioner
# L1 = MAE. Each filter module returns a filtered tensor; loss = MAE on filtered tensors.
from losses3d import SobelFilterLoss3D, LaplacianFilterLoss3D, FFTLoss3D
# SSIM returns a scalar in [0,1]; loss = 1 - SSIM on the mid-depth 2D slice.
from losses import SSIM, MAELoss


# =============================================================================
# One-time setup (call once before the training loop)
# =============================================================================
def setup_madam(model, optimizer, device, lambda_mode='fixed', filter_names=('sobel','fft','laplacian','ssim')):
    """Build the preconditioner and, for UW mode, the learnable log_vars."""
    T = 1 + len(filter_names)                         # T tasks: L1 + filters
    params = [p for p in model.parameters() if p.requires_grad]
    precond = MAdamPreconditioner(
        params, num_tasks=T,
        ema_decay=0.999,
        eps_damp=1e-6,
        stochastic=True,            # O(3) backward per step instead of O(T)
    )

    log_vars = None
    if lambda_mode == 'uw':
        log_vars = nn.Parameter(torch.zeros(T, device=device))
        optimizer.add_param_group({'params': [log_vars], 'lr': optimizer.defaults['lr']})

    # Per-filter modules (stateless 3×3×3 kernels), moved to device once.
    filters = {
        'sobel':     SobelFilterLoss3D().to(device),
        'laplacian': LaplacianFilterLoss3D(in_channels=1).to(device),
        'fft':       FFTLoss3D(remove_dc=True).to(device),
        'ssim':      SSIM().to(device),   # applied to mid-depth 2D slice
    }
    return precond, log_vars, filters, filter_names


# =============================================================================
# Per-step task losses (graph must stay alive — no .detach())
# =============================================================================
def compute_task_losses(pred, target, filters, filter_names):
    """Return a list [L1, *filter_losses] — length T."""
    mae = MAELoss()
    losses = [mae(pred, target)]                                     # L1 anchor
    for name in filter_names:
        f = filters[name]
        if name == 'ssim':
            mid = target.shape[2] // 2                               # middle depth slice
            ssim_val = f(pred[:, :, mid], target[:, :, mid])
            losses.append(1.0 - ssim_val)
        else:
            losses.append(mae(f(pred), f(target)))
    return losses


# =============================================================================
# Per-step lambda (preference ray)
# =============================================================================
def compute_lambda(mode, T, device, log_vars=None):
    if mode == 'fixed':
        return torch.ones(T, device=device) / T
    if mode == 'dirichlet':
        return Dirichlet(torch.ones(T, device=device)).sample()
    if mode == 'uw':
        # Kendall-style precision weighting: lambda ∝ exp(-s); normalise to 1.
        s = torch.clamp(log_vars, min=-4.0, max=4.0)
        precision = torch.exp(-s)
        return precision / precision.sum()
    raise ValueError(mode)


# =============================================================================
# Training step — the four MAdam phases
# =============================================================================
def train_step(model, optimizer, precond, log_vars, filters, filter_names,
               lr_img, hr_img, lambda_mode='fixed'):
    T = 1 + len(filter_names)
    device = hr_img.device

    optimizer.zero_grad(set_to_none=True)
    pred = model(lr_img)
    task_losses = compute_task_losses(pred, hr_img, filters, filter_names)
    lam = compute_lambda(lambda_mode, T, device, log_vars=log_vars)

    # ---------- Phase 1: FIM EMA update (graph still alive) -------------------
    precond.update(task_losses, lam)

    # ---------- Phase 2: scalarised backward ----------------------------------
    if lambda_mode == 'uw':
        # Kendall UW objective: sum_t exp(-s_t) * L_t + s_t
        s = torch.clamp(log_vars, -4.0, 4.0)
        total = sum(torch.exp(-s[t]) * task_losses[t] + s[t] for t in range(T))
    else:
        total = sum(lam[t] * task_losses[t] for t in range(T))
    total.backward()

    # ---------- Phase 3: precondition param.grad in-place ---------------------
    precond.precondition()

    # ---------- Phase 4: optimizer step ---------------------------------------
    optimizer.step()

    return total.detach(), task_losses, lam


# =============================================================================
# Usage
# =============================================================================
if __name__ == '__main__':
    # Minimal example — replace model/loader with yours.
    from monai.networks.nets import SwinUNETR

    device = torch.device('cuda')
    model = SwinUNETR(
        img_size=(80, 128, 128), in_channels=1, out_channels=1,
        feature_size=48, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    LAMBDA_MODE = 'fixed'     # 'fixed' | 'dirichlet' | 'uw'
    precond, log_vars, filters, filter_names = setup_madam(
        model, optimizer, device,
        lambda_mode=LAMBDA_MODE,
        filter_names=('sobel', 'fft', 'laplacian', 'ssim'),
    )

    for epoch in range(200):
        for lr_img, hr_img in loader:                       # noqa: F821
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            total, per_task, lam = train_step(
                model, optimizer, precond, log_vars, filters, filter_names,
                lr_img, hr_img, lambda_mode=LAMBDA_MODE,
            )
