"""Hypervolume, Pareto dominance, and stability computations.

All computations operate on aggregated result dicts from loader.py.
"""
import numpy as np

from .config import HV_CONFIG, DATASET_SPECS

# ---------------------------------------------------------------------------
# Pareto diversity / population quality
# ---------------------------------------------------------------------------

def compute_num_non_dominated(per_ray_metrics, dataset):
    """Compute number of non-dominated solutions from per-ray metrics.

    A solution at ray i is non-dominated if no other ray j produces a point
    that is at least as good on ALL objectives and strictly better on at least one.

    Args:
        per_ray_metrics: {ray_idx: {metric_key: value}}
        dataset: Dataset name.

    Returns:
        (num_non_dominated, total_rays)
    """
    dom_keys = _get_dominance_keys(dataset)

    rays = sorted(per_ray_metrics.keys())
    if not rays:
        return 0, 0

    # Build objective matrix (all minimization)
    points = []
    for ray_idx in rays:
        point = []
        valid = True
        for key, negate in dom_keys:
            if key not in per_ray_metrics[ray_idx]:
                valid = False
                break
            val = per_ray_metrics[ray_idx][key]
            if negate:
                val = -val
            point.append(val)
        if valid:
            points.append(point)

    if not points:
        return 0, 0

    points = np.array(points)
    n = len(points)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j <= i on all and j < i on at least one
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_dominated[i] = True
                break

    num_nd = int(np.sum(~is_dominated))
    return num_nd, n


def compute_diversity_stats(agg, dataset):
    """Compute diversity/population quality statistics from aggregated results.

    Uses stored population metrics where available, computes from per-ray
    data otherwise.

    Returns:
        dict with keys (each as {mean, std} dicts or None):
            num_non_dominated, num_total_rays,
            uniformity (2D only), spearman (2D only)
    """
    pop_mean = agg.get("population_mean", {})
    pop_std = agg.get("population_std", {})
    raw_results = agg.get("raw_results", [])

    result = {}

    # num_non_dominated: prefer stored, else compute from per-ray
    if "num_non_dominated" in pop_mean:
        result["num_non_dominated"] = {
            "mean": pop_mean["num_non_dominated"],
            "std": pop_std.get("num_non_dominated", 0.0),
        }
        # Total rays: infer from per-ray results
        sample_rays = agg.get("per_ray_mean", {})
        result["num_total_rays"] = len(sample_rays)
    elif raw_results:
        # Compute from per-ray results per seed
        nd_per_seed = []
        total = 0
        for r in raw_results:
            per_ray = {int(k): v for k, v in r.get("per_ray_results", {}).items() if k != "population"}
            nd, tot = compute_num_non_dominated(per_ray, dataset)
            nd_per_seed.append(nd)
            total = tot
        arr = np.array(nd_per_seed, dtype=float)
        result["num_non_dominated"] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
        result["num_total_rays"] = total
    else:
        result["num_non_dominated"] = None
        result["num_total_rays"] = 0

    # uniformity (2D datasets only)
    if "uniformity" in pop_mean:
        result["uniformity"] = {
            "mean": pop_mean["uniformity"],
            "std": pop_std.get("uniformity", 0.0),
        }
    else:
        result["uniformity"] = None

    # spearman (2D datasets only)
    if "spearman" in pop_mean:
        result["spearman"] = {
            "mean": pop_mean["spearman"],
            "std": pop_std.get("spearman", 0.0),
        }
    else:
        result["spearman"] = None

    return result


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------

def compute_hypervolume_from_results(per_ray_metrics, dataset):
    """Compute hypervolume from per-ray metric dict.

    Args:
        per_ray_metrics: {ray_idx: {metric_key: value}}
        dataset: Dataset name.

    Returns:
        float or None (if dataset has no HV config).
    """
    if dataset not in HV_CONFIG:
        return None

    hv_cfg = HV_CONFIG[dataset]
    ref_point = hv_cfg["ref_point"]
    obj_keys = hv_cfg["objective_keys"]

    try:
        from pymoo.indicators.hv import HV
    except ImportError:
        raise ImportError("pymoo is required for HV computation: pip install pymoo")

    negate = hv_cfg.get("negate", [False] * len(obj_keys))

    points = []
    for ray_idx in sorted(per_ray_metrics.keys()):
        metrics = per_ray_metrics[ray_idx]
        point = []
        for key, neg in zip(obj_keys, negate):
            if key not in metrics:
                return None
            val = -metrics[key] if neg else metrics[key]
            point.append(val)
        points.append(point)

    points = np.array(points)

    # Apply same negation to reference point (it's already in the target space)
    # Filter points inside reference box (dominated by ref point)
    mask = np.all(points < ref_point, axis=1)
    if mask.sum() == 0:
        return 0.0
    points = points[mask]

    ind = HV(ref_point=ref_point)
    return float(ind(points))


def compute_hypervolume_per_seed(raw_results, dataset):
    """Compute HV for each seed independently.

    Args:
        raw_results: List of result dicts (one per seed).
        dataset: Dataset name.

    Returns:
        List of floats (one HV per seed).
    """
    hvs = []
    for result in raw_results:
        per_ray = {}
        for k, v in result.get("per_ray_results", {}).items():
            if k == "population":
                continue
            per_ray[int(k)] = v
        hv = compute_hypervolume_from_results(per_ray, dataset)
        if hv is not None:
            hvs.append(hv)
    return hvs


def compute_hv_stats(raw_results, dataset):
    """Compute HV mean, std, and CV across seeds.

    Returns:
        dict with keys: mean, std, cv, per_seed, ref_point, space
        or None if dataset has no HV config.
    """
    if dataset not in HV_CONFIG:
        return None

    hvs = compute_hypervolume_per_seed(raw_results, dataset)
    if not hvs:
        return None

    mean = float(np.mean(hvs))
    std = float(np.std(hvs))
    cv = std / mean if mean > 0 else float("inf")

    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "per_seed": hvs,
        "ref_point": HV_CONFIG[dataset]["ref_point"].tolist(),
        "space": HV_CONFIG[dataset]["space"],
    }


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------

def _get_dominance_keys(dataset):
    """Get metric keys for dominance comparison.

    All metrics are converted to minimization (lower is better).
    Returns list of (metric_key, negate) tuples.
    """
    spec = DATASET_SPECS[dataset]
    keys = []
    for task_name, metric_info in spec["primary_metrics"].items():
        key = metric_info["key"]
        # If direction is "up" (higher is better), we negate for dominance check
        negate = metric_info["direction"] == "up"
        keys.append((key, negate))
    return keys


def pareto_dominates(point_a, point_b):
    """Check if point_a Pareto-dominates point_b (all minimization).

    Returns True if a <= b on all objectives and a < b on at least one.
    """
    a = np.array(point_a)
    b = np.array(point_b)
    return bool(np.all(a <= b) and np.any(a < b))


def compute_dominance_rate(agg_a, agg_b, dataset):
    """Compute per-ray Pareto dominance rate between two methods.

    Uses mean values across seeds for comparison.

    Args:
        agg_a: Aggregated results for method A.
        agg_b: Aggregated results for method B.
        dataset: Dataset name.

    Returns:
        dict with:
            a_dominates: count of rays where A dominates B
            b_dominates: count of rays where B dominates A
            neither: count of rays where neither dominates
            total: total rays compared
            a_pct: percentage A dominates
            b_pct: percentage B dominates
            neither_pct: percentage neither
    """
    dom_keys = _get_dominance_keys(dataset)
    mean_a = agg_a["per_ray_mean"]
    mean_b = agg_b["per_ray_mean"]

    common_rays = sorted(set(mean_a.keys()) & set(mean_b.keys()))

    a_dom = 0
    b_dom = 0
    neither = 0

    for ray_idx in common_rays:
        point_a = []
        point_b = []
        valid = True
        for key, negate in dom_keys:
            if key not in mean_a[ray_idx] or key not in mean_b[ray_idx]:
                valid = False
                break
            va = mean_a[ray_idx][key]
            vb = mean_b[ray_idx][key]
            if negate:
                va = -va
                vb = -vb
            point_a.append(va)
            point_b.append(vb)

        if not valid:
            continue

        if pareto_dominates(point_a, point_b):
            a_dom += 1
        elif pareto_dominates(point_b, point_a):
            b_dom += 1
        else:
            neither += 1

    total = a_dom + b_dom + neither
    return {
        "a_dominates": a_dom,
        "b_dominates": b_dom,
        "neither": neither,
        "total": total,
        "a_pct": a_dom / total * 100 if total > 0 else 0,
        "b_pct": b_dom / total * 100 if total > 0 else 0,
        "neither_pct": neither / total * 100 if total > 0 else 0,
    }


def compute_dominance_per_seed(raw_a_list, raw_b_list, dataset):
    """Compute dominance rate per seed, then aggregate.

    Args:
        raw_a_list: List of result dicts for method A (per seed).
        raw_b_list: List of result dicts for method B (per seed).

    Returns:
        dict with per-seed and aggregated dominance stats.
    """
    dom_keys = _get_dominance_keys(dataset)

    per_seed = []
    for result_a, result_b in zip(raw_a_list, raw_b_list):
        per_ray_a = {int(k): v for k, v in result_a["per_ray_results"].items() if k != "population"}
        per_ray_b = {int(k): v for k, v in result_b["per_ray_results"].items() if k != "population"}
        common = sorted(set(per_ray_a.keys()) & set(per_ray_b.keys()))

        a_dom = b_dom = neither_count = 0
        for ray_idx in common:
            point_a = []
            point_b = []
            valid = True
            for key, negate in dom_keys:
                if key not in per_ray_a[ray_idx] or key not in per_ray_b[ray_idx]:
                    valid = False
                    break
                va = per_ray_a[ray_idx][key]
                vb = per_ray_b[ray_idx][key]
                if negate:
                    va, vb = -va, -vb
                point_a.append(va)
                point_b.append(vb)

            if not valid:
                continue
            if pareto_dominates(point_a, point_b):
                a_dom += 1
            elif pareto_dominates(point_b, point_a):
                b_dom += 1
            else:
                neither_count += 1

        total = a_dom + b_dom + neither_count
        per_seed.append({
            "a_dominates": a_dom,
            "b_dominates": b_dom,
            "neither": neither_count,
            "total": total,
        })

    # Aggregate across seeds
    total_a = sum(s["a_dominates"] for s in per_seed)
    total_b = sum(s["b_dominates"] for s in per_seed)
    total_n = sum(s["neither"] for s in per_seed)
    total_all = total_a + total_b + total_n

    return {
        "per_seed": per_seed,
        "total_a_dominates": total_a,
        "total_b_dominates": total_b,
        "total_neither": total_n,
        "total": total_all,
        "a_pct": total_a / total_all * 100 if total_all > 0 else 0,
        "b_pct": total_b / total_all * 100 if total_all > 0 else 0,
        "neither_pct": total_n / total_all * 100 if total_all > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Per-task improvement (delta)
# ---------------------------------------------------------------------------

def compute_per_task_delta(agg_a, agg_b, dataset):
    """Compute per-task improvement of method A over method B.

    Positive delta means A is better:
    - For "up" metrics (accuracy): delta = A - B
    - For "down" metrics (error): delta = B - A

    Returns:
        Dict[task_name, dict] with keys: mean_a, mean_b, delta, delta_pct
        Averaged across all rays.
    """
    spec = DATASET_SPECS[dataset]
    mean_a = agg_a["per_ray_mean"]
    mean_b = agg_b["per_ray_mean"]
    common_rays = sorted(set(mean_a.keys()) & set(mean_b.keys()))

    task_deltas = {}
    for task_name, metric_info in spec["primary_metrics"].items():
        key = metric_info["key"]
        scale = metric_info["scale"]
        direction = metric_info["direction"]

        vals_a = []
        vals_b = []
        for ray_idx in common_rays:
            if key in mean_a[ray_idx] and key in mean_b[ray_idx]:
                vals_a.append(mean_a[ray_idx][key] * scale)
                vals_b.append(mean_b[ray_idx][key] * scale)

        if not vals_a:
            continue

        mean_val_a = np.mean(vals_a)
        mean_val_b = np.mean(vals_b)

        # Positive delta = method B improved over baseline A
        if direction == "up":
            delta = mean_val_b - mean_val_a  # higher is better, so improvement = B - A
        else:
            delta = mean_val_a - mean_val_b  # lower is better, so improvement = A - B

        delta_pct = delta / abs(mean_val_a) * 100 if mean_val_a != 0 else 0

        task_deltas[task_name] = {
            "mean_a": float(mean_val_a),
            "mean_b": float(mean_val_b),
            "delta": float(delta),
            "delta_pct": float(delta_pct),
            "unit": metric_info["unit"],
            "direction": direction,
        }

    return task_deltas
