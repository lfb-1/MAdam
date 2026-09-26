"""Full MAdam preconditioner.

This module implements the full MAdam Fisher preconditioner used by the
benchmark adapters.  The public method/config name is ``madam``.  The
implementation is the full conditional Fisher metric:

    F(λ) = Σ_{ij} λ_i λ_j F_{ij}

The metric adapts to the current preference vector λ, giving each ray its own
effective curvature landscape.

Two-phase API (recommended — works with any training loop)::

    from madam_core.madam import MAdamPreconditioner

    params = [p for p in model.parameters() if p.requires_grad]
    precond = MAdamPreconditioner(params, num_tasks=T)
    optimizer = torch.optim.Adam(params, lr=1e-3)

    for x, y in loader:
        optimizer.zero_grad()
        losses = [loss_fn(model(x), y, task=t) for t in range(T)]
        lam = sample_ray()                   # [T] preference vector

        precond.update(losses, lam)           # (1) FIM update — before backward
        scalarized = sum(lam[t] * losses[t] for t in range(T))
        scalarized.backward()                 # (2) normal backward
        precond.precondition()                # (3) modify param.grad in-place
        optimizer.step()                      # (4) optimizer sees preconditioned grads

Features:
  - **EMA ramp**: blends from identity to full MAdam over the EMA
    convergence window, avoiding noisy preconditioning during early training.
  - **Stochastic pair sampling**: ``stochastic=True`` updates only one
    random (i,j) pair plus both diagonals per step. In multi-ray mode,
    the sampled pair statistics are averaged ray-by-ray and the full
    denominator is assembled from the EMA bank.
  - **Multi-ray**: when ``lam`` is ``[K, T]`` and per-ray task losses are
    provided as ``[K][T]``, MAdam statistics are computed ray-by-ray and the
    assembled curvatures are averaged into one step-level denominator.
  - **Gradient-balanced FIM**: optionally normalizes each task-gradient row
    before the FIM EMA, useful for mixed-loss benchmarks whose task gradients
    are measured in different units.

Curvature-proxy contract:
  ``task_losses`` are scalar curvature-proxy losses. They often match the
  training losses, but adapters may pass separate ``fim_losses`` when the
  training objective and curvature metric should differ. MAdam does not infer
  loss types or automatically make task metrics commensurate.

  The ideal full object is ``J.T @ M @ J`` for task-gradient rows ``J`` and an
  objective-space metric ``M``. Pair products can be negative when task
  gradients oppose each other, but the exact assembled diagonal is nonnegative
  for positive-semidefinite ``M``.

See also:
  ``diag_madam.py`` — diagonal-only MAdam variant with no cross-task terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable, List, Optional, Sequence, Tuple


class MAdamPreconditioner:
    """Conditional pullback-metric preconditioner for multi-task learning.

    Args:
        params: Iterable of ``nn.Parameter`` to precondition.  The user
            controls which parameters are included (e.g. exclude BatchNorm).
        num_tasks: Number of tasks ``T``.
        ema_decay: EMA decay factor β for F_{ij} estimates.
        eps_num: Numerical floor used only to clamp the final denominator.
        eps_damp: Curvature damping inside ``sqrt(F(λ) + eps_damp)``.
        stochastic: If True, each update samples one random (i,j) pair and
            only updates the FIM EMA for that pair + both diagonals.
        gradient_balance: If True, normalize each task-gradient row by its
            flat L2 norm before writing FIM EMA statistics.  The scalarized
            training gradient remains unchanged.
        gradient_balance_eps: Numerical stabilizer in the task-gradient norm
            denominator used by ``gradient_balance``.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        num_tasks: int,
        ema_decay: float = 0.999,
        eps_num: float = 1e-12,
        eps_damp: float = 1e-8,
        stochastic: bool = True,
        gradient_balance: bool = False,
        gradient_balance_eps: float = 1e-3,
    ) -> None:
        self.params: List[nn.Parameter] = list(params)
        self.T = num_tasks
        self.beta = ema_decay
        self.eps_num = float(eps_num)
        self.eps_damp = float(eps_damp)
        self.stochastic = stochastic
        self.gradient_balance = bool(gradient_balance)
        self.gradient_balance_eps = float(gradient_balance_eps)

        self.D = sum(p.numel() for p in self.params)

        # Build (i, j) pair indices with i <= j  →  T(T+1)/2 pairs
        pi, pj = [], []
        self._pair_to_idx: dict[Tuple[int, int], int] = {}
        idx = 0
        for i in range(num_tasks):
            for j in range(i, num_tasks):
                pi.append(i)
                pj.append(j)
                self._pair_to_idx[(i, j)] = idx
                idx += 1
        self.pair_i = torch.tensor(pi)
        self.pair_j = torch.tensor(pj)
        self.num_pairs = len(pi)

        # Diagonal pair index for each task (used by stochastic mode)
        self._diag_idx = [self._pair_to_idx[(t, t)] for t in range(num_tasks)]

        # EMA storage — lazily initialised on first .step()
        self.fim: Optional[Tensor] = None  # [num_pairs, D]
        self._step = 0
        self._cached_lam: Optional[Tensor] = None
        self._cached_curvature: Optional[Tensor] = None

        # Warmup: 1/(1-beta) steps for the EMA to converge.
        self.warmup = int(1.0 / (1.0 - ema_decay))

    def _validate_single_loss(self, loss, location: str) -> Tensor:
        """Validate one scalar curvature-proxy loss before any state mutation."""
        if not isinstance(loss, Tensor):
            raise TypeError(f"MAdam {location} must be a torch.Tensor, got {type(loss)!r}.")
        if loss.numel() != 1:
            raise ValueError(
                f"MAdam {location} must be scalar with numel()==1, got shape {tuple(loss.shape)}."
            )
        if not torch.isfinite(loss.detach()).all().item():
            raise ValueError(f"MAdam {location} is non-finite; the FIM EMA bank would be corrupted.")
        if not loss.requires_grad:
            raise ValueError(
                f"MAdam {location} does not require grad. Pass graph-attached scalar losses "
                "to update() before the training backward pass."
            )
        if loss.is_leaf:
            raise ValueError(
                f"MAdam {location} is a leaf tensor. Pass a graph-attached scalar computed "
                "from model outputs to update() before the training backward pass."
            )
        return loss

    def _validate_task_losses(
        self,
        task_losses: Sequence[Tensor],
        lam: Tensor,
    ) -> Tuple[List[Tensor], Optional[List[List[Tensor]]]]:
        """Validate flat ``[T]`` or multi-ray ``[K][T]`` curvature losses."""
        if len(task_losses) == 0:
            raise ValueError(f"Expected {self.T} MAdam task losses, got 0.")

        first = task_losses[0]
        if isinstance(first, Tensor):
            if len(task_losses) != self.T:
                raise ValueError(f"Expected {self.T} MAdam task losses, got {len(task_losses)}.")
            losses = [
                self._validate_single_loss(loss, f"task loss {idx}")
                for idx, loss in enumerate(task_losses)
            ]
            return losses, None

        if lam.shape[0] <= 1:
            raise ValueError(
                "Nested MAdam task losses require multi-ray lambda shaped [K, T] with K > 1."
            )
        if len(task_losses) != lam.shape[0]:
            raise ValueError(
                f"Expected {lam.shape[0]} per-ray MAdam loss groups, got {len(task_losses)}."
            )

        losses_by_ray: List[List[Tensor]] = []
        for ray_idx, ray_losses in enumerate(task_losses):
            if not isinstance(ray_losses, (list, tuple)):
                raise TypeError(
                    f"MAdam ray {ray_idx} losses must be a list or tuple, got {type(ray_losses)!r}."
                )
            if len(ray_losses) != self.T:
                raise ValueError(
                    f"Ray {ray_idx} supplied {len(ray_losses)} MAdam task losses, expected {self.T}."
                )
            losses_by_ray.append(
                [
                    self._validate_single_loss(loss, f"ray {ray_idx} task loss {task_idx}")
                    for task_idx, loss in enumerate(ray_losses)
                ]
            )
        return [], losses_by_ray

    def _build_denom(self, f_lam: Tensor, alpha: float) -> Tensor:
        """Construct the preconditioning denominator.

        ``eps_damp`` controls how aggressively curvature is trusted.
        ``eps_num`` is a pure numerical floor on the final denominator.
        """
        denom = alpha * (f_lam.clamp(min=0.0) + self.eps_damp).sqrt() + (1.0 - alpha)
        return denom.clamp_min(self.eps_num)

    def _resolve_lam(
        self,
        lam: Optional[Tensor],
        *,
        allow_cached: bool = True,
        allow_uniform: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Normalize preference vectors to ``[K, T]`` and handle defaults."""
        if lam is None:
            if allow_cached and self._cached_lam is not None:
                lam = self._cached_lam
            elif allow_uniform:
                if device is None or dtype is None:
                    raise ValueError("Uniform lambda requires explicit device and dtype.")
                lam = torch.ones(self.T, device=device, dtype=dtype) / self.T
            else:
                raise RuntimeError("MAdam lambda requested before update().")

        if lam.dim() == 1:
            lam = lam.unsqueeze(0)
        if lam.dim() != 2 or lam.shape[1] != self.T:
            raise ValueError(f"Expected lambda shaped [T] or [K, T] with T={self.T}, got {tuple(lam.shape)}.")

        if device is not None or dtype is not None:
            target_device = device if device is not None else lam.device
            target_dtype = dtype if dtype is not None else lam.dtype
            lam = lam.to(device=target_device, dtype=target_dtype)
        return lam

    # ------------------------------------------------------------------
    # Two-phase API (for integration with training loops that call
    # loss.backward() between the FIM update and preconditioning)
    # ------------------------------------------------------------------

    def update(self, task_losses: Sequence[Tensor], lam: Tensor) -> None:
        """Phase 1: compute per-task gradients and update the FIM EMA.

        Must be called **before** ``loss.backward()`` while the computation
        graph is still alive.  Caches ``lam`` for the subsequent
        :meth:`precondition` call.

        Args:
            task_losses: Length-T scalar curvature-proxy losses (graph
                attached). For multi-ray ray-conditioned forwards, pass
                per-ray task losses shaped ``[K][T]``. See the module
                docstring for the training-loss / ``fim_losses`` contract.
            lam: ``[T]`` or ``[K, T]`` preference vector(s).
        """
        lam = self._resolve_lam(
            lam,
            allow_cached=False,
            allow_uniform=False,
            device=self.params[0].device,
            dtype=self.params[0].dtype,
        )
        task_losses, losses_by_ray = self._validate_task_losses(task_losses, lam)

        self._step += 1
        self._cached_lam = lam

        if losses_by_ray is not None:
            use_stochastic_multi_ray = self.stochastic and self.fim is not None
            if use_stochastic_multi_ray:
                pair_idx, cross_prod, diag_products = self._compute_multi_ray_sampled_products(
                    losses_by_ray
                )
                self._update_fim_sampled_products(pair_idx, cross_prod, diag_products)
                fim = self._corrected_fim()
                self._cached_curvature = self._assemble(fim, lam)
            else:
                products_mean, f_lam = self._compute_multi_ray_statistics(losses_by_ray, lam)
                self._update_fim_from_products(products_mean)
                self._cached_curvature = f_lam
        else:
            self._cached_curvature = None
            task_grads, tasks_computed = self._compute_task_grads(task_losses)
            fim_task_grads = self._balance_task_grads_for_fim(task_grads, tasks_computed)
            self._update_fim(fim_task_grads, tasks_computed)

    def precondition(self) -> None:
        """Phase 2: read ``param.grad``, divide by the split MAdam denominator.

        Must be called **after** ``loss.backward()`` has populated
        ``param.grad``.  Uses ``lam`` cached from the preceding
        :meth:`update` call.
        """
        lam = self._resolve_lam(None, allow_cached=True, allow_uniform=False)
        K = lam.shape[0]

        # Ensure pair indices are on the same device as FIM
        if self.fim is not None and self.pair_i.device != self.fim.device:
            self.pair_i = self.pair_i.to(self.fim.device)
            self.pair_j = self.pair_j.to(self.fim.device)

        # Read param.grad written by loss.backward()
        all_grads = torch.cat([p.grad.flatten() for p in self.params
                               if p.grad is not None])  # [D]

        denom = self._flat_denominator()
        g_precond = (all_grads / K) / denom

        self._unflatten_to_grad(g_precond)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def sample_ray(self) -> Tensor:
        """Sample a preference vector from Dir(1, ..., 1)."""
        device = self.params[0].device if self.params else torch.device("cpu")
        return torch.distributions.Dirichlet(
            torch.ones(self.T, device=device)
        ).sample()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _multi_ray_task_losses(
        self,
        task_losses: Sequence[Tensor],
        lam: Tensor,
    ) -> Optional[List[List[Tensor]]]:
        """Return per-ray task losses when the caller provides ``[K][T]`` input."""
        if lam.shape[0] <= 1:
            return None
        if len(task_losses) == 0:
            return None
        first = task_losses[0]
        if isinstance(first, Tensor):
            return None

        losses_by_ray = [list(ray_losses) for ray_losses in task_losses]
        if len(losses_by_ray) != lam.shape[0]:
            raise ValueError(
                f"Expected {lam.shape[0]} per-ray loss groups, got {len(losses_by_ray)}."
            )
        for ray_idx, ray_losses in enumerate(losses_by_ray):
            if len(ray_losses) != self.T:
                raise ValueError(
                    f"Ray {ray_idx} supplied {len(ray_losses)} task losses, expected {self.T}."
                )
        return losses_by_ray

    def _pair_weights_single(self, lam: Tensor, device: torch.device) -> Tensor:
        """Return per-pair assembly weights for one preference vector."""
        lam = lam.to(device)
        pair_i = self.pair_i.to(device)
        pair_j = self.pair_j.to(device)
        outer = torch.outer(lam, lam)  # [T, T]
        w = outer[pair_i, pair_j]  # [num_pairs]
        off = pair_i != pair_j
        w[off] += outer[pair_j[off], pair_i[off]]
        return w

    def _pair_weights(self, lam: Tensor, device: torch.device) -> Tensor:
        """Return per-pair assembly weights for one or more rays."""
        if lam.dim() == 1:
            return self._pair_weights_single(lam, device)

        weights = torch.zeros(self.num_pairs, device=device, dtype=lam.dtype)
        for ray in lam:
            weights.add_(self._pair_weights_single(ray, device))
        return weights / max(lam.shape[0], 1)

    def _assemble_from_products(self, products: Tensor, lam: Tensor) -> Tensor:
        """Assemble curvature directly from same-ray pair products."""
        w = self._pair_weights(lam, products.device).to(products.dtype)
        return products.T @ w  # [D]

    def _compute_multi_ray_statistics(
        self,
        task_losses_by_ray: Sequence[Sequence[Tensor]],
        lam: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return averaged pair products and assembled curvature."""
        device = self.params[0].device
        dtype = self.params[0].dtype
        K = lam.shape[0]
        products_sum = torch.zeros(self.num_pairs, self.D, device=device, dtype=dtype)
        f_sum = torch.zeros(self.D, device=device, dtype=dtype)
        pair_i = self.pair_i.to(device)
        pair_j = self.pair_j.to(device)

        for ray_idx, ray_losses in enumerate(task_losses_by_ray):
            ray_grads, _ = self._compute_task_grads(ray_losses, force_full=True)
            fim_ray_grads = self._balance_task_grads_for_fim(ray_grads, None)
            products = fim_ray_grads[pair_i] * fim_ray_grads[pair_j]  # [num_pairs, D]
            ray_lam = lam[ray_idx].to(device)
            products_sum.add_(products)
            f_sum.add_(self._assemble_from_products(products, ray_lam))

        scale = 1.0 / max(K, 1)
        return products_sum * scale, f_sum * scale

    def _compute_selected_task_grads(
        self, task_losses: Sequence[Tensor], tasks_needed: Sequence[int]
    ) -> Tensor:
        """Return flat gradients for a specific task subset."""
        device = self.params[0].device
        grads = torch.zeros(self.T, self.D, device=device)
        for t in tasks_needed:
            g_t = torch.autograd.grad(
                task_losses[t], self.params,
                retain_graph=True, allow_unused=True,
            )
            grads[t] = self._flat_task_grad(g_t, device)
        return grads

    def _compute_multi_ray_sampled_products(
        self,
        task_losses_by_ray: Sequence[Sequence[Tensor]],
    ) -> Tuple[int, Tensor, dict[int, Tensor]]:
        """Return averaged sampled-pair products across rays."""
        device = self.params[0].device
        dtype = self.params[0].dtype
        pair_idx = torch.randint(self.num_pairs, (1,), device=device).item()
        i = self.pair_i[pair_idx].item()
        j = self.pair_j[pair_idx].item()
        tasks_needed = sorted({i, j})
        cross_prod = torch.zeros(self.D, device=device, dtype=dtype)
        diag_products = {
            t: torch.zeros(self.D, device=device, dtype=dtype)
            for t in tasks_needed
        }

        for ray_losses in task_losses_by_ray:
            ray_grads = self._compute_selected_task_grads(ray_losses, tasks_needed)
            fim_ray_grads = self._balance_task_grads_for_fim(ray_grads, set(tasks_needed))
            cross_prod.add_(fim_ray_grads[i] * fim_ray_grads[j])
            for t in tasks_needed:
                diag_products[t].add_(fim_ray_grads[t].square())

        scale = 1.0 / max(len(task_losses_by_ray), 1)
        cross_prod.mul_(scale)
        for t in tasks_needed:
            diag_products[t].mul_(scale)
        return pair_idx, cross_prod, diag_products

    def _update_fim_from_products(self, products: Tensor) -> None:
        """EMA update from precomputed pair products."""
        self._ensure_fim_storage(products.device)
        beta = self.beta
        with torch.no_grad():
            self.fim.mul_(beta).add_(products, alpha=1 - beta)

    def _update_fim_sampled_products(
        self,
        pair_idx: int,
        cross_prod: Tensor,
        diag_products: dict[int, Tensor],
    ) -> None:
        """EMA update for one sampled pair plus the corresponding diagonals."""
        self._ensure_fim_storage(cross_prod.device)
        beta = self.beta
        with torch.no_grad():
            self.fim[pair_idx].mul_(beta).add_(cross_prod, alpha=1 - beta)
            for t, diag_prod in diag_products.items():
                d_idx = self._diag_idx[t]
                if d_idx != pair_idx:
                    self.fim[d_idx].mul_(beta).add_(diag_prod, alpha=1 - beta)

    def _ensure_fim_storage(self, device: torch.device) -> None:
        """Lazily initialize pair storage on the target device."""
        if self.fim is None:
            self.fim = torch.zeros(self.num_pairs, self.D, device=device)
            self.pair_i = self.pair_i.to(device)
            self.pair_j = self.pair_j.to(device)

    def _compute_task_grads(
        self, task_losses: Sequence[Tensor], force_full: bool = False
    ) -> Tuple[Tensor, Optional[set]]:
        """Return ``[T, D]`` flat per-task gradients.

        When ``stochastic=True``, only two tasks are differentiated (the
        sampled pair).  The remaining rows are zero — they are not used for
        the FIM update, and the scalarised gradient ``g(λ)`` is computed
        from *all* T losses via a single combined backward instead.
        """
        T, D = self.T, self.D
        device = self.params[0].device

        if self.stochastic and self.fim is not None and not force_full:
            # Sample one (i,j) pair
            p = torch.randint(self.num_pairs, (1,)).item()
            i, j = self.pair_i[p].item(), self.pair_j[p].item()
            tasks_needed = {i, j}

            grads = torch.zeros(T, D, device=device)
            for t in tasks_needed:
                g_t = torch.autograd.grad(
                    task_losses[t], self.params,
                    retain_graph=True, allow_unused=True,
                )
                grads[t] = self._flat_task_grad(g_t, device)
            return grads, tasks_needed
        else:
            grads = torch.zeros(T, D, device=device)
            for t in range(T):
                g_t = torch.autograd.grad(
                    task_losses[t], self.params,
                    retain_graph=True, allow_unused=True,
                )
                grads[t] = self._flat_task_grad(g_t, device)
            return grads, None  # None = all computed

    def _flat_task_grad(
        self, grad_parts: Sequence[Optional[Tensor]], device: torch.device
    ) -> Tensor:
        return torch.cat(
            [
                (grad.flatten() if grad is not None else torch.zeros(param.numel(), device=device))
                for grad, param in zip(grad_parts, self.params)
            ]
        )

    def _balance_task_grads_for_fim(
        self, grads: Tensor, tasks_computed: Optional[set]
    ) -> Tensor:
        """Return task gradients used for FIM statistics.

        Gradient balancing is deliberately applied only to the curvature rows;
        the scalarized training/update gradient remains the original method
        gradient.
        """
        if tasks_computed is None:
            tasks = list(range(self.T))
        else:
            tasks = sorted(int(task_idx) for task_idx in tasks_computed)

        if not self.gradient_balance:
            return grads

        balanced = grads.clone()
        for task_idx in tasks:
            norm = grads[task_idx].norm()
            scale = 1.0 / norm.add(self.gradient_balance_eps).clamp_min(self.eps_num)
            balanced[task_idx] = grads[task_idx] * scale

        return balanced

    def _update_fim(self, grads: Tensor, tasks_computed: Optional[set]) -> None:
        """EMA update of cross-task Fisher terms."""
        device = grads.device
        self._ensure_fim_storage(device)

        beta = self.beta
        with torch.no_grad():
            if tasks_computed is not None:
                # Stochastic: update sampled pair + diagonals
                tasks_list = sorted(tasks_computed)
                i, j = tasks_list[0], tasks_list[-1]
                # Cross pair (i, j)
                p_idx = self._pair_to_idx[(i, j)]
                self.fim[p_idx].mul_(beta).addcmul_(
                    grads[i], grads[j], value=1 - beta
                )
                # Diagonals for both tasks (skip if same as cross pair)
                for t in tasks_computed:
                    d_idx = self._diag_idx[t]
                    if d_idx != p_idx:
                        self.fim[d_idx].mul_(beta).add_(
                            grads[t].square(), alpha=1 - beta
                        )
            else:
                # Full vectorised update: all pairs at once
                products = grads[self.pair_i] * grads[self.pair_j]  # [num_pairs, D]
                self.fim.mul_(beta).add_(products, alpha=1 - beta)

    def _corrected_fim(self) -> Tensor:
        """Return the current EMA estimate."""
        return self.fim

    def _assemble(self, fim: Tensor, lam: Tensor) -> Tensor:
        """Assemble F(λ) from the FIM cross-terms.

        Args:
            fim: ``[num_pairs, D]`` EMA storage.
            lam: ``[K, T]`` preference vectors (already normalised to 2-D).

        For single-ray (K=1): ``F(λ) = Σ_{ij} λ_i λ_j F_{ij}``.
        For multi-ray (K>1): the pair weights use the averaged second moment
        ``E_k[sym(λ_i^k λ_j^k)]``. Exact multi-ray mode bypasses this helper
        and caches the averaged assembled curvature from same-ray statistics.

        """
        w = self._pair_weights(lam, fim.device).to(fim.dtype)
        return fim.T @ w  # [D]

    def _unflatten_to_grad(self, flat: Tensor) -> None:
        """Write a flat [D] vector back into each param.grad."""
        offset = 0
        for p in self.params:
            n = p.numel()
            p.grad = flat[offset : offset + n].view_as(p).clone()
            offset += n

    def _unflatten(self, flat: Tensor) -> List[Tensor]:
        """Return a list of tensors matching params' shapes from flat vector."""
        values: List[Tensor] = []
        offset = 0
        for p in self.params:
            n = p.numel()
            values.append(flat[offset : offset + n].view_as(p).detach().clone())
            offset += n
        return values

    def _flat_denominator(self, lam: Optional[Tensor] = None) -> Tensor:
        """Return the exact flat denominator used by ``precondition()``."""
        if lam is None:
            f_lam = self._assembled_curvature()
        else:
            lam = self._resolve_lam(
                lam,
                allow_cached=False,
                allow_uniform=False,
                device=self.params[0].device,
                dtype=self.params[0].dtype,
            )
            if lam.shape[0] != 1:
                raise ValueError(
                    "Explicit multi-ray lambda is not supported here; use cached state from update()."
                )
            if self.fim is None:
                raise RuntimeError("MAdam denominator requested before update().")
            fim = self._corrected_fim()
            f_lam = self._assemble(fim, lam)
        alpha = min(1.0, self._step / max(self.warmup, 1))
        return self._build_denom(f_lam, alpha)

    def _assembled_curvature(self) -> Tensor:
        if self._cached_curvature is not None:
            return self._cached_curvature
        if self.fim is None:
            raise RuntimeError("MAdam curvature requested before update().")
        lam = self._resolve_lam(None, allow_cached=True, allow_uniform=False)
        fim = self._corrected_fim()
        return self._assemble(fim, lam)

    def get_curvature_per_param(self) -> List[Tensor]:
        """Return assembled F(λ) reshaped per parameter (no preconditioning)."""
        return self._unflatten(self._assembled_curvature())

    def build_denominator_per_param(
        self,
        curvature_per_param: Optional[List[Tensor]] = None,
    ) -> List[Tensor]:
        """Return the exact preconditioning denominator reshaped per parameter.

        When ``curvature_per_param`` is provided, it must contain assembled
        ``F(lambda)`` tensors already aligned with ``self.params``.
        """
        if curvature_per_param is None:
            return self._unflatten(self._flat_denominator())

        if len(curvature_per_param) != len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} curvature tensors, got {len(curvature_per_param)}."
            )

        flat_parts: List[Tensor] = []
        for param, curvature in zip(self.params, curvature_per_param):
            if curvature.numel() != param.numel():
                raise ValueError(
                    "Curvature tensor shape does not match parameter shape: "
                    f"expected {tuple(param.shape)}, got {tuple(curvature.shape)}."
                )
            flat_parts.append(curvature.to(device=param.device, dtype=param.dtype).reshape(-1))

        f_lam = torch.cat(flat_parts)
        alpha = min(1.0, self._step / max(self.warmup, 1))
        denom = self._build_denom(f_lam, alpha)
        return self._unflatten(denom)

    def get_denominator_per_param(self) -> List[Tensor]:
        """Return the full preconditioning denominator per parameter."""
        return self.build_denominator_per_param()
