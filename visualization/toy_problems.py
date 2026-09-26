"""Toy 2D bi-objective problems for the MAdam vs Adam visualization.

Each problem provides:
  - ``f_torch(x)`` returns ``(f_1, f_2)`` as graph-attached scalar tensors,
    suitable for the MAdamPreconditioner two-phase API.
  - ``f_numpy(x)`` for vectorized batched evaluation (e.g. contour grids).
  - ``pareto_x(lam)`` closed-form minimizer of the scalarized loss
    ``lam_1 f_1 + lam_2 f_2``.
  - ``pareto_front(n)`` samples ``n`` points along the analytic front in
    objective space.
  - ``distance_to_front(point_obj)`` for per-ray quality metrics.

Convention: ``x`` is a length-2 vector ``(x_1, x_2)``. ``lam`` is normalized
(sums to 1) on the simplex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class _BiQuadratic2D:
    """Two diagonal-Hessian quadratics with offset minima on the x-axis.

    f_1(x_1, x_2) = a_1 (x_1 + 1)^2 + b_1 x_2^2     (min at c_1 = (-1, 0))
    f_2(x_1, x_2) = a_2 (x_1 - 1)^2 + b_2 x_2^2     (min at c_2 = (+1, 0))

    Hessian H_i = diag(2 a_i, 2 b_i). The Pareto set is the segment
    {(x_1, 0) : x_1 in [-1, 1]} (both objectives prefer x_2 = 0).
    """

    a1: float
    b1: float
    a2: float
    b2: float

    def __post_init__(self) -> None:
        self.c1 = np.array([-1.0, 0.0])
        self.c2 = np.array([+1.0, 0.0])
        self.H1 = np.diag([2.0 * self.a1, 2.0 * self.b1])
        self.H2 = np.diag([2.0 * self.a2, 2.0 * self.b2])

    def f_torch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.a1 * (x[0] + 1.0) ** 2 + self.b1 * x[1] ** 2
        f2 = self.a2 * (x[0] - 1.0) ** 2 + self.b2 * x[1] ** 2
        return f1, f2

    def f_numpy(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x)
        x0, x1 = x[..., 0], x[..., 1]
        f1 = self.a1 * (x0 + 1.0) ** 2 + self.b1 * x1 ** 2
        f2 = self.a2 * (x0 - 1.0) ** 2 + self.b2 * x1 ** 2
        return f1, f2

    def grads_numpy(self, x: np.ndarray) -> np.ndarray:
        """Per-task gradients. Returns shape (..., 2 tasks, 2 dims)."""
        x = np.asarray(x)
        g1 = np.stack(
            [2.0 * self.a1 * (x[..., 0] + 1.0), 2.0 * self.b1 * x[..., 1]], axis=-1
        )
        g2 = np.stack(
            [2.0 * self.a2 * (x[..., 0] - 1.0), 2.0 * self.b2 * x[..., 1]], axis=-1
        )
        return np.stack([g1, g2], axis=-2)

    def pareto_x(self, lam: np.ndarray) -> np.ndarray:
        """Closed-form minimizer of lam_1 f_1 + lam_2 f_2.

        For diagonal Hessians and centers (-1, 0), (+1, 0), only the x_1
        component depends on lam:

            x_1*(lam) = (lam_2 a_2 - lam_1 a_1) / (lam_1 a_1 + lam_2 a_2)
            x_2*(lam) = 0
        """
        lam = np.asarray(lam, dtype=float)
        lam1 = lam[..., 0]
        lam2 = lam[..., 1]
        denom = lam1 * self.a1 + lam2 * self.a2
        x1_star = (lam2 * self.a2 - lam1 * self.a1) / denom
        return np.stack([x1_star, np.zeros_like(x1_star)], axis=-1)

    def pareto_front(self, n: int = 400) -> np.ndarray:
        """Sample n points on the analytic Pareto front in objective space."""
        xs = np.linspace(-1.0, 1.0, n)
        pts = np.stack([xs, np.zeros_like(xs)], axis=-1)
        f1, f2 = self.f_numpy(pts)
        return np.stack([f1, f2], axis=-1)

    def distance_to_front(self, point_obj: np.ndarray) -> float:
        front = self.pareto_front(2000)
        d = np.linalg.norm(front - np.asarray(point_obj)[None, :], axis=-1)
        return float(d.min())


class AnisoQuadratic2D(_BiQuadratic2D):
    """Anisotropic conflicting quadratics — main flagship-figure problem.

    Default (a_1, b_1) = (4, 0.25), (a_2, b_2) = (0.25, 4): each objective is
    sharp along one axis and shallow along the other, with the two objectives
    swapping roles. This is the geometry that makes Adam's coordinate-wise
    second moment a *wrong* metric for the active preference, while MAdam's
    F(lam) tracks the right one.
    """

    def __init__(
        self,
        a1: float = 4.0,
        b1: float = 0.25,
        a2: float = 0.25,
        b2: float = 4.0,
    ) -> None:
        super().__init__(a1=a1, b1=b1, a2=a2, b2=b2)


class MetricMismatch2D(_BiQuadratic2D):
    """Sign-flip / metric-mismatch toy — appendix figure.

    Uses extreme axis-asymmetry between the two objectives so that the
    Euclidean cosine between the two task gradients has the OPPOSITE sign
    of the Adam-rescaled cosine. Concretely:

      f_1: x_1 cheap, x_2 expensive  (b_1 = 100)
      f_2: x_1 expensive, x_2 cheap  (a_2 = 100)

    Under the Euclidean metric, the gradients are dominated by their large
    components and look "compatible" (similar direction). After Adam-style
    coordinate scaling by 1/sqrt(diag(g_i^2)), the small components dominate
    and the gradients flip to look "conflicting". This is the metric
    sign-flip phenomenon the paper names.
    """

    def __init__(
        self,
        a1: float = 1.0,
        b1: float = 100.0,
        a2: float = 100.0,
        b2: float = 1.0,
    ) -> None:
        super().__init__(a1=a1, b1=b1, a2=a2, b2=b2)


class BiQuadraticGeneral2D:
    """Two convex quadratics with FULL (possibly off-diagonal) Hessians.

    f_i(x) = 0.5 * (x - c_i)^T H_i (x - c_i)

    Hessians H_i must be symmetric positive definite. Pareto set is a
    smooth curve in 2D given by the closed form

        x*(lam) = (lam_1 H_1 + lam_2 H_2)^{-1} (lam_1 H_1 c_1 + lam_2 H_2 c_2).

    This generalizes ``_BiQuadratic2D`` (which assumed diagonal Hessians and
    axis-aligned centers) to support classic 2D test functions such as
    Booth and Matyas whose Hessians have non-zero off-diagonals.
    """

    def __init__(self, H1: np.ndarray, c1: np.ndarray,
                 H2: np.ndarray, c2: np.ndarray) -> None:
        self.H1 = np.asarray(H1, dtype=float)
        self.H2 = np.asarray(H2, dtype=float)
        self.c1 = np.asarray(c1, dtype=float)
        self.c2 = np.asarray(c2, dtype=float)
        assert self.H1.shape == (2, 2) and self.H2.shape == (2, 2)
        assert self.c1.shape == (2,) and self.c2.shape == (2,)
        # As torch tensors for f_torch
        self._H1_t = torch.tensor(self.H1, dtype=torch.float64)
        self._H2_t = torch.tensor(self.H2, dtype=torch.float64)
        self._c1_t = torch.tensor(self.c1, dtype=torch.float64)
        self._c2_t = torch.tensor(self.c2, dtype=torch.float64)

    def f_torch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d1 = x - self._c1_t
        d2 = x - self._c2_t
        f1 = 0.5 * d1 @ self._H1_t @ d1
        f2 = 0.5 * d2 @ self._H2_t @ d2
        return f1, f2

    def f_numpy(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        d1 = x - self.c1
        d2 = x - self.c2
        # Vectorize: d^T H d using einsum
        f1 = 0.5 * np.einsum("...i,ij,...j->...", d1, self.H1, d1)
        f2 = 0.5 * np.einsum("...i,ij,...j->...", d2, self.H2, d2)
        return f1, f2

    def pareto_x(self, lam: np.ndarray) -> np.ndarray:
        """Closed-form minimizer of lam_1 f_1 + lam_2 f_2.

        Supports a single lam shape (2,) or batched (..., 2).
        """
        lam = np.asarray(lam, dtype=float)
        lam1 = lam[..., 0:1, None]  # (..., 1, 1) for broadcasting against (2,2)
        lam2 = lam[..., 1:2, None]
        # H_combined: (..., 2, 2)
        H = lam1 * self.H1 + lam2 * self.H2
        # rhs: lam_1 H_1 c_1 + lam_2 H_2 c_2 -> (..., 2)
        rhs = (lam[..., 0:1] * (self.H1 @ self.c1)
               + lam[..., 1:2] * (self.H2 @ self.c2))
        # Solve H x* = rhs for each lam in the batch
        return np.linalg.solve(H, rhs[..., None])[..., 0]

    def pareto_front(self, n: int = 400) -> np.ndarray:
        """Sample n points on the analytic Pareto front (n simplex rays)."""
        # Sweep angles in [0, pi/2] then normalize -- simple but works
        angles = np.linspace(1e-4, np.pi / 2 - 1e-4, n)
        lam = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        lam = lam / lam.sum(axis=-1, keepdims=True)
        x_stars = self.pareto_x(lam)  # (n, 2)
        f1, f2 = self.f_numpy(x_stars)
        return np.stack([f1, f2], axis=-1)

    def distance_to_front(self, point_obj: np.ndarray) -> float:
        front = self.pareto_front(2000)
        d = np.linalg.norm(front - np.asarray(point_obj)[None, :], axis=-1)
        return float(d.min())


class BoothMatyas2D(BiQuadraticGeneral2D):
    """Booth + Matyas as the two objectives.

    f_1 (Booth)  = (x + 2y - 7)^2 + (2x + y - 5)^2,  min at (1, 3),
                   Hessian H_1 = [[10, 8], [8, 10]],  eigvals 18 / 2
                   along (1,1) / (1,-1).
    f_2 (Matyas) = 0.26 (x^2 + y^2) - 0.48 x y,       min at (0, 0),
                   Hessian H_2 = [[0.52, -0.48], [-0.48, 0.52]],
                   eigvals 1.0 / 0.04 along (1,-1) / (1,1).

    The two Hessians have *opposite* principal axes -- Booth is stiff along
    (1,1), Matyas is stiff along (1,-1). Adam's axis-aligned coordinate
    scaling is structurally mismatched with both, while MAdam's F(lam)
    rotates with the active preference and tracks the right metric.
    """

    def __init__(self) -> None:
        super().__init__(
            H1=np.array([[10.0, 8.0], [8.0, 10.0]]),
            c1=np.array([1.0, 3.0]),
            H2=np.array([[0.52, -0.48], [-0.48, 0.52]]),
            c2=np.array([0.0, 0.0]),
        )


class TiltedPair2D(BiQuadraticGeneral2D):
    """Matched-scale quadratics with opposite +/-45 deg principal axes.

    Both objectives share the same eigenvalue spectrum (alpha, beta), so
    neither dominates the combined loss in scale. Their stiff directions
    are perpendicular: f_1 stiff along (1, 1), f_2 stiff along (1, -1).

    Centers sit on the x-axis at (+/- d, 0) so the two minima are offset
    along a neutral direction (neither H_i is stiff/soft along it). The
    balanced (lam = (0.5, 0.5)) Hessian is *isotropic*, so the scalarized
    landscape is visibly different from the lam = (0.9, 0.1) and
    (0.1, 0.9) ones -- different curvature per ray, exactly the property
    that the single-heatmap view obscures.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 0.2,
                 d: float = 1.5) -> None:
        # +45 deg rotation: stiff along (1, 1)
        H1 = np.array([
            [(alpha + beta) / 2, (alpha - beta) / 2],
            [(alpha - beta) / 2, (alpha + beta) / 2],
        ])
        # -45 deg rotation: stiff along (1, -1)
        H2 = np.array([
            [(alpha + beta) / 2, -(alpha - beta) / 2],
            [-(alpha - beta) / 2, (alpha + beta) / 2],
        ])
        super().__init__(H1=H1, c1=np.array([+d, 0.0]),
                         H2=H2, c2=np.array([-d, 0.0]))
        self.alpha = alpha
        self.beta = beta
        self.d = d


def _smoke() -> None:
    """Quick self-check for the analytic forms."""
    p = AnisoQuadratic2D()
    # Pure ray lam=(1,0) should give x* = c_1 = (-1, 0)
    np.testing.assert_allclose(p.pareto_x(np.array([1.0, 0.0])), [-1.0, 0.0])
    np.testing.assert_allclose(p.pareto_x(np.array([0.0, 1.0])), [+1.0, 0.0])
    # Distance from a Pareto-set point to the front is ~0
    on_front = np.array([0.0, 0.0])
    f1_v, f2_v = p.f_numpy(on_front)
    obj_pt = np.array([float(f1_v), float(f2_v)])
    assert p.distance_to_front(obj_pt) < 1e-2
    # Torch and numpy must agree
    x_torch = torch.tensor([0.5, -0.3], requires_grad=False)
    f1_t, f2_t = p.f_torch(x_torch)
    f1_n, f2_n = p.f_numpy(np.array([0.5, -0.3]))
    assert abs(float(f1_t) - float(f1_n)) < 1e-6
    assert abs(float(f2_t) - float(f2_n)) < 1e-6

    # ----- BoothMatyas2D checks -----
    bm = BoothMatyas2D()
    # Booth's minimum is (1, 3), value 0
    f1_min, _ = bm.f_numpy(np.array([1.0, 3.0]))
    assert abs(float(f1_min)) < 1e-9, f"Booth at (1,3) = {f1_min}, expected 0"
    # Matyas's minimum is (0, 0), value 0
    _, f2_min = bm.f_numpy(np.array([0.0, 0.0]))
    assert abs(float(f2_min)) < 1e-9, f"Matyas at (0,0) = {f2_min}, expected 0"
    # Booth value at (0, 0) should equal (0+0-7)^2 + (0+0-5)^2 = 74
    f1_orig, _ = bm.f_numpy(np.array([0.0, 0.0]))
    assert abs(float(f1_orig) - 74.0) < 1e-9
    # Pareto extremes
    np.testing.assert_allclose(bm.pareto_x(np.array([1.0, 0.0])), [1.0, 3.0],
                               atol=1e-9)
    np.testing.assert_allclose(bm.pareto_x(np.array([0.0, 1.0])), [0.0, 0.0],
                               atol=1e-9)
    # torch and numpy agree on a generic point
    x_torch = torch.tensor([0.7, 1.4], dtype=torch.float64)
    f1_t, f2_t = bm.f_torch(x_torch)
    f1_n, f2_n = bm.f_numpy(np.array([0.7, 1.4]))
    assert abs(float(f1_t) - float(f1_n)) < 1e-9
    assert abs(float(f2_t) - float(f2_n)) < 1e-9
    # ----- TiltedPair2D checks -----
    tp = TiltedPair2D()
    # f_1 min at (+d, 0), f_2 min at (-d, 0) — both should be 0
    f1_at_c1, _ = tp.f_numpy(np.array([+tp.d, 0.0]))
    _, f2_at_c2 = tp.f_numpy(np.array([-tp.d, 0.0]))
    assert abs(float(f1_at_c1)) < 1e-12
    assert abs(float(f2_at_c2)) < 1e-12
    # Pareto extremes
    np.testing.assert_allclose(tp.pareto_x(np.array([1.0, 0.0])),
                               [+tp.d, 0.0], atol=1e-9)
    np.testing.assert_allclose(tp.pareto_x(np.array([0.0, 1.0])),
                               [-tp.d, 0.0], atol=1e-9)
    # Matched scale: f_1 and f_2 at the origin should be equal
    f1_o, f2_o = tp.f_numpy(np.array([0.0, 0.0]))
    assert abs(float(f1_o) - float(f2_o)) < 1e-12, \
        f"matched-scale check failed: f_1(0)={f1_o}, f_2(0)={f2_o}"
    # Balanced ray Hessian should be isotropic (off-diagonals cancel)
    H_balanced = 0.5 * (tp.H1 + tp.H2)
    assert abs(H_balanced[0, 1]) < 1e-12 and abs(H_balanced[1, 0]) < 1e-12
    print("toy_problems smoke ok (incl. BoothMatyas2D, TiltedPair2D)")


if __name__ == "__main__":
    _smoke()
