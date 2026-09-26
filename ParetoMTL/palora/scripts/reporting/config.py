"""Dataset definitions, metric specs, reference points, and ray descriptions.

This is the single source of truth for all dataset-specific reporting constants.
"""
import numpy as np


BASELINE_METHODS = ("ls", "uw", "dwa", "imtl", "cagrad", "palora", "pamal")
NONPARETO_BASELINE_METHODS = ("ls", "uw", "dwa", "imtl", "cagrad")
PARETO_BASELINE_METHODS = ("palora", "pamal")


# ---------------------------------------------------------------------------
# Task and metric definitions per dataset
# ---------------------------------------------------------------------------

DATASET_SPECS = {
    "multimnist": {
        "num_tasks": 2,
        "task_names": ["top-left", "bottom-right"],
        "task_short": ["TL", "BR"],
        # Per-task primary metrics (what gets reported in tables)
        "primary_metrics": {
            "top-left": {"key": "acc/top-left", "unit": "%", "scale": 100, "direction": "up", "label": "Accuracy"},
            "bottom-right": {"key": "acc/bottom-right", "unit": "%", "scale": 100, "direction": "up", "label": "Accuracy"},
        },
        # Per-task loss metrics
        "loss_metrics": {
            "top-left": {"key": "loss/top-left", "label": "CE Loss"},
            "bottom-right": {"key": "loss/bottom-right", "label": "CE Loss"},
        },
        # Aggregate metrics
        "aggregate_metrics": {
            "avg_acc": {"key": "avg_acc", "scale": 100, "direction": "up", "label": "Avg Accuracy (%)"},
        },
        # Number of eval rays
        "num_eval_rays": 11,
        # Loss type info
        "loss_type": "CrossEntropy for both tasks",
    },
    "sarcos": {
        "num_tasks": 7,
        "task_names": [f"Task {i}" for i in range(1, 8)],
        "task_short": [f"T{i}" for i in range(1, 8)],
        "primary_metrics": {
            f"Task {i}": {"key": f"error/Task {i}", "unit": "x100", "scale": 100, "direction": "down", "label": "MSE"}
            for i in range(1, 8)
        },
        "loss_metrics": {
            f"Task {i}": {"key": f"error/Task {i}", "label": "MSE"}
            for i in range(1, 8)
        },
        "aggregate_metrics": {
            "avg_error": {"key": "avg_error", "scale": 100, "direction": "down", "label": "Avg Error (x100)"},
        },
        "num_eval_rays": 8,
        "loss_type": "MSE for all 7 tasks",
    },
    "utkface": {
        "num_tasks": 3,
        "task_names": ["age", "gender", "race"],
        "task_short": ["Age", "Gen", "Race"],
        "primary_metrics": {
            "age": {"key": "huber/age", "unit": "x100", "scale": 100, "direction": "down", "label": "Huber Loss"},
            "gender": {"key": "acc/gender", "unit": "%", "scale": 100, "direction": "up", "label": "Accuracy"},
            "race": {"key": "acc/race", "unit": "%", "scale": 100, "direction": "up", "label": "Accuracy"},
        },
        "loss_metrics": {
            "age": {"key": "huber/age", "label": "Huber Loss"},
            "gender": {"key": "loss/gender", "label": "CE Loss"},
            "race": {"key": "loss/race", "label": "CE Loss"},
        },
        "aggregate_metrics": {},
        "num_eval_rays": 7,  # 7 representative rays (was 66 from meshzoo)
        "loss_type": "Huber (age) + CrossEntropy (gender, race)",
    },
}


# ---------------------------------------------------------------------------
# Hypervolume reference points
# ---------------------------------------------------------------------------

HV_CONFIG = {
    "multimnist": {
        # Accuracy space: negate to minimization, ref [0, 0] = 0% accuracy (worst)
        "ref_point": np.array([0.0, 0.0]),
        "space": "accuracy (negated)",
        "objective_keys": ["acc/top-left", "acc/bottom-right"],
        "negate": [True, True],  # negate accuracy for minimization
    },
    "sarcos": {
        # Error space: already minimization
        "ref_point": np.array([1, 5, 1, 1, 10, 10, 1]),
        "space": "error",
        "objective_keys": [f"error/Task {i}" for i in range(1, 8)],
        "negate": [False] * 7,
    },
    "utkface": {
        # Mixed: huber (minimize), acc_gender (maximize→negate), acc_race (maximize→negate)
        # Ref [0.5, 0, 0]: huber=0.5 (worst), acc=0% (worst, after negation)
        "ref_point": np.array([0.5, 0.0, 0.0]),
        "space": "huber / accuracy (negated)",
        "objective_keys": ["huber/age", "acc/gender", "acc/race"],
        "negate": [False, True, True],  # negate accuracies for minimization
    },
}


# ---------------------------------------------------------------------------
# Evaluation ray definitions
# ---------------------------------------------------------------------------

def get_eval_rays(dataset, num_rays=None):
    """Generate evaluation rays for a dataset.

    Returns:
        List of (ray_index, ray_vector, description) tuples.
    """
    if dataset == "multimnist":
        n = num_rays or 11
        rays = []
        for i in range(n):
            p = i / (n - 1)
            ray = [1 - p, p]
            desc = f"[{ray[0]:.1f}, {ray[1]:.1f}]"
            if p == 0:
                desc += " (TL only)"
            elif p == 1:
                desc += " (BR only)"
            elif abs(p - 0.5) < 0.01:
                desc += " (balanced)"
            rays.append((i, ray, desc))
        return rays

    elif dataset == "sarcos":
        rays = []
        for i in range(7):
            ray = [0.0] * 7
            ray[i] = 1.0
            desc = f"e_{i+1} (Task {i+1} only)"
            rays.append((i, ray, desc))
        ray_uniform = [1/7] * 7
        rays.append((7, ray_uniform, "[1/7]*7 (uniform)"))
        return rays

    elif dataset == "utkface":
        # Return the 7 representative rays from WORKFLOW.md
        representative = [
            ([1, 0, 0], "age only"),
            ([0.5, 0.5, 0], "age-gender"),
            ([0, 1, 0], "gender only"),
            ([0.5, 0, 0.5], "age-race"),
            ([0, 0.5, 0.5], "gender-race"),
            ([0, 0, 1], "race only"),
            ([0.4, 0.3, 0.3], "near-center"),
        ]
        rays = []
        for i, (ray, desc) in enumerate(representative):
            rays.append((i, ray, f"{ray} ({desc})"))
        return rays

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# UTKFace representative ray indices (approximate mapping from meshzoo)
# These are the ray indices in the 66-ray grid that are closest to the
# 7 representative rays defined above.
UTKFACE_REPRESENTATIVE_RAY_INDICES = {
    # (approximate — depends on meshzoo.triangle(n) output order)
    # vertex rays: [1,0,0] -> idx 0, [0,1,0] -> idx 10, [0,0,1] -> idx ~65
    # edge midpoints and center vary with subdivision level
    # Use find_closest_rays() to compute at runtime
}


def find_closest_rays_utkface(per_ray_results):
    """Find ray indices closest to the 7 representative UTKFace rays.

    When exactly 7 rays: they ARE the representative rays (indices 0-6).
    Otherwise: uses meshzoo grid to find closest matches.

    Returns:
        Dict mapping representative ray label to (ray_index, actual_ray_vector).
    """
    representative_labels = [
        "age only", "age-gender", "gender only",
        "age-race", "gender-race", "race only", "near-center",
    ]
    representative_vectors = [
        np.array([1, 0, 0]), np.array([0.5, 0.5, 0]), np.array([0, 1, 0]),
        np.array([0.5, 0, 0.5]), np.array([0, 0.5, 0.5]), np.array([0, 0, 1]),
        np.array([0.4, 0.3, 0.3]),
    ]

    num_rays = len([k for k in per_ray_results if k != "population"])

    # Short-circuit: when exactly 7 rays, they are the representative rays
    if num_rays == 7:
        return {
            label: (i, vec.tolist())
            for i, (label, vec) in enumerate(zip(representative_labels, representative_vectors))
        }

    # Fallback: meshzoo grid lookup for legacy 66-ray results
    try:
        import meshzoo
    except ImportError:
        return None

    import math
    discriminant = 9 + 8 * num_rays - 8
    k = int((-3 + math.sqrt(discriminant)) / 2 + 0.5)
    points, _ = meshzoo.triangle(k)
    points = points.T  # shape: (num_points, 3)

    result = {}
    for label, target in zip(representative_labels, representative_vectors):
        dists = np.linalg.norm(points - target, axis=1)
        closest_idx = int(np.argmin(dists))
        actual_ray = points[closest_idx].tolist()
        result[label] = (closest_idx, actual_ray)

    return result


# ---------------------------------------------------------------------------
# Fair comparison config verification
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS = {
    "multimnist": {"epochs": 10, "rank": 1, "lr": 0.001, "batch_size": 256},
    "sarcos": {"epochs": 100, "rank": 4, "lr": 0.0001, "batch_size": 512},
    "utkface": {"epochs": 100, "rank": 4, "lr": 0.001, "batch_size": 128},
}


def verify_config(dataset, config):
    """Verify experiment config matches fair comparison defaults.

    Returns:
        (is_ok, list_of_issues)
    """
    expected = DEFAULT_CONFIGS[dataset]
    issues = []
    for key, expected_val in expected.items():
        actual_val = config.get(key)
        if actual_val is not None and actual_val != expected_val:
            issues.append(f"{key}: got {actual_val}, expected {expected_val}")
    return len(issues) == 0, issues
