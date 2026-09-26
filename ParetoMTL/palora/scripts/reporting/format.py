"""Format results as markdown and LaTeX tables.

Follows the WORKFLOW.md evaluation protocol exactly.
"""
import numpy as np

from .config import DATASET_SPECS, HV_CONFIG, get_eval_rays, find_closest_rays_utkface
from .compute import (
    compute_hv_stats,
    compute_dominance_rate,
    compute_per_task_delta,
    compute_diversity_stats,
)


def _fmt(val, decimals=2):
    """Format a float to given decimal places."""
    return f"{val:.{decimals}f}"


def _fmt_pm(mean, std, decimals=2):
    """Format mean +/- std."""
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def _bold_better(val_a, val_b, direction, decimals=2):
    """Return (str_a, str_b) with the better value bolded (markdown)."""
    sa = _fmt(val_a, decimals)
    sb = _fmt(val_b, decimals)
    if direction == "up":
        if val_a > val_b:
            return f"**{sa}**", sb
        elif val_b > val_a:
            return sa, f"**{sb}**"
    else:
        if val_a < val_b:
            return f"**{sa}**", sb
        elif val_b < val_a:
            return sa, f"**{sb}**"
    return sa, sb


# ---------------------------------------------------------------------------
# Per-Ray Per-Task Table (PRIMARY result)
# ---------------------------------------------------------------------------

def format_per_ray_table_multimnist(all_agg, methods, method_labels=None):
    """Format MultiMNIST per-ray per-task table.

    Args:
        all_agg: {method_name: aggregated_results}
        methods: List of method names to include (order matters).
        method_labels: Optional {method_name: display_label}.

    Returns:
        str: Markdown table.
    """
    if method_labels is None:
        method_labels = {m: m for m in methods}

    spec = DATASET_SPECS["multimnist"]
    rays = get_eval_rays("multimnist")
    ray_headers = [f"[{r[1][0]:.1f},{r[1][1]:.1f}]" for r in rays]

    lines = []
    lines.append(f"**MultiMNIST** (mean over {all_agg[methods[0]]['num_seeds']} seeds)")
    lines.append("")

    # Header
    header = "| Task | Method | " + " | ".join(ray_headers) + " |"
    sep = "|------|--------|" + "|".join(["--------"] * len(rays)) + "|"
    lines.append(header)
    lines.append(sep)

    for task_name in spec["task_names"]:
        task_short = spec["task_short"][spec["task_names"].index(task_name)]
        metric_info = spec["primary_metrics"][task_name]
        key = metric_info["key"]
        scale = metric_info["scale"]

        for i, method in enumerate(methods):
            agg = all_agg[method]
            label = method_labels.get(method, method)
            values = []
            for ray_idx, _, _ in rays:
                if ray_idx in agg["per_ray_mean"] and key in agg["per_ray_mean"][ray_idx]:
                    val = agg["per_ray_mean"][ray_idx][key] * scale
                    values.append(_fmt(val))
                else:
                    values.append("--")

            row = f"| {task_short if i == 0 else ''} | {label} | " + " | ".join(values) + " |"
            lines.append(row)

        # Delta row (if 2 methods)
        if len(methods) == 2:
            m_a, m_b = methods[0], methods[1]
            deltas = []
            for ray_idx, _, _ in rays:
                agg_a = all_agg[m_a]
                agg_b = all_agg[m_b]
                if (ray_idx in agg_a["per_ray_mean"] and key in agg_a["per_ray_mean"][ray_idx]
                        and ray_idx in agg_b["per_ray_mean"] and key in agg_b["per_ray_mean"][ray_idx]):
                    va = agg_a["per_ray_mean"][ray_idx][key] * scale
                    vb = agg_b["per_ray_mean"][ray_idx][key] * scale
                    d = vb - va if metric_info["direction"] == "up" else va - vb
                    deltas.append(f"{d:+.2f}")
                else:
                    deltas.append("--")
            label_b = method_labels.get(methods[1], methods[1])
            row = f"| | Delta ({label_b}) | " + " | ".join(deltas) + " |"
            lines.append(row)

    return "\n".join(lines)


def format_per_ray_table_sarcos(all_agg, methods, method_labels=None):
    """Format SARCOS per-ray per-task table."""
    if method_labels is None:
        method_labels = {m: m for m in methods}

    spec = DATASET_SPECS["sarcos"]
    rays = get_eval_rays("sarcos")
    ray_headers = [r[2].split(" (")[0] for r in rays]  # e.g., "e_1", "e_2", ..., "[1/7]*7"

    lines = []
    lines.append(f"**SARCOS** (mean over {all_agg[methods[0]]['num_seeds']} seeds, errors x100)")
    lines.append("")

    header = "| Task | Method | " + " | ".join(ray_headers) + " |"
    sep = "|------|--------|" + "|".join(["--------"] * len(rays)) + "|"
    lines.append(header)
    lines.append(sep)

    for task_name in spec["task_names"]:
        task_short = spec["task_short"][spec["task_names"].index(task_name)]
        metric_info = spec["primary_metrics"][task_name]
        key = metric_info["key"]
        scale = metric_info["scale"]

        for i, method in enumerate(methods):
            agg = all_agg[method]
            label = method_labels.get(method, method)
            values = []
            for ray_idx, _, _ in rays:
                if ray_idx in agg["per_ray_mean"] and key in agg["per_ray_mean"][ray_idx]:
                    val = agg["per_ray_mean"][ray_idx][key] * scale
                    values.append(_fmt(val))
                else:
                    values.append("--")
            row = f"| {task_short if i == 0 else ''} | {label} | " + " | ".join(values) + " |"
            lines.append(row)

        # Delta row
        if len(methods) == 2:
            m_a, m_b = methods[0], methods[1]
            deltas = []
            for ray_idx, _, _ in rays:
                agg_a = all_agg[m_a]
                agg_b = all_agg[m_b]
                if (ray_idx in agg_a["per_ray_mean"] and key in agg_a["per_ray_mean"][ray_idx]
                        and ray_idx in agg_b["per_ray_mean"] and key in agg_b["per_ray_mean"][ray_idx]):
                    va = agg_a["per_ray_mean"][ray_idx][key] * scale
                    vb = agg_b["per_ray_mean"][ray_idx][key] * scale
                    # For error (lower is better): positive = method reduced error
                    d = va - vb
                    deltas.append(f"{d:+.2f}")
                else:
                    deltas.append("--")
            label_b = method_labels.get(methods[1], methods[1])
            row = f"| | Delta ({label_b}) | " + " | ".join(deltas) + " |"
            lines.append(row)

    return "\n".join(lines)


def format_per_ray_table_utkface(all_agg, methods, method_labels=None):
    """Format UTKFace per-ray per-task table (7 representative rays)."""
    if method_labels is None:
        method_labels = {m: m for m in methods}

    spec = DATASET_SPECS["utkface"]

    # Find representative ray indices from the actual results
    sample_method = methods[0]
    sample_result = all_agg[sample_method]["raw_results"][0]
    per_ray = {int(k): v for k, v in sample_result["per_ray_results"].items() if k != "population"}
    closest = find_closest_rays_utkface(per_ray)

    rep_labels = ["age only", "age-gender", "gender only", "age-race", "gender-race", "race only", "near-center"]

    if closest is None:
        # Fallback: use fixed indices for standard 66-ray grid (meshzoo.triangle(10))
        # These are the approximate indices for the 7 representative rays.
        num_rays = len(per_ray)
        if num_rays == 66:
            fallback = {
                "age only": (0, [1.0, 0.0, 0.0]),
                "age-gender": (5, [0.5, 0.5, 0.0]),
                "gender only": (10, [0.0, 1.0, 0.0]),
                "age-race": (30, [0.5, 0.0, 0.5]),
                "gender-race": (60, [0.0, 0.5, 0.5]),
                "race only": (65, [0.0, 0.0, 1.0]),
                "near-center": (24, [0.4, 0.3, 0.3]),
            }
            closest = fallback
        else:
            # Use evenly-spaced indices
            step = max(1, num_rays // 7)
            indices = [0] + [min(i * step, num_rays - 1) for i in range(1, 6)] + [num_rays - 1]
            closest = {
                label: (idx, [0, 0, 0])
                for label, idx in zip(rep_labels, indices)
            }

    rep_indices = [closest[label][0] for label in rep_labels]
    rep_ray_strs = []
    for label in rep_labels:
        idx, ray_vec = closest[label]
        rep_ray_strs.append(f"[{ray_vec[0]:.1f},{ray_vec[1]:.1f},{ray_vec[2]:.1f}]")

    lines = []
    lines.append(f"**UTKFace** (mean over {all_agg[methods[0]]['num_seeds']} seeds, 7 representative rays)")
    lines.append("")

    header = "| Task | Method | " + " | ".join(rep_ray_strs) + " |"
    sep = "|------|--------|" + "|".join(["--------"] * len(rep_indices)) + "|"
    lines.append(header)
    lines.append(sep)

    for task_name in spec["task_names"]:
        task_short = spec["task_short"][spec["task_names"].index(task_name)]
        metric_info = spec["primary_metrics"][task_name]
        key = metric_info["key"]
        scale = metric_info["scale"]
        unit = metric_info["unit"]

        for i, method in enumerate(methods):
            agg = all_agg[method]
            label = method_labels.get(method, method)
            values = []
            for ray_idx in rep_indices:
                if ray_idx in agg["per_ray_mean"] and key in agg["per_ray_mean"][ray_idx]:
                    val = agg["per_ray_mean"][ray_idx][key] * scale
                    values.append(_fmt(val))
                else:
                    values.append("--")
            task_label = f"{task_short} ({unit})" if i == 0 else ""
            row = f"| {task_label} | {label} | " + " | ".join(values) + " |"
            lines.append(row)

        # Delta row
        if len(methods) == 2:
            m_a, m_b = methods[0], methods[1]
            deltas = []
            for ray_idx in rep_indices:
                agg_a = all_agg[m_a]
                agg_b = all_agg[m_b]
                if (ray_idx in agg_a["per_ray_mean"] and key in agg_a["per_ray_mean"][ray_idx]
                        and ray_idx in agg_b["per_ray_mean"] and key in agg_b["per_ray_mean"][ray_idx]):
                    va = agg_a["per_ray_mean"][ray_idx][key] * scale
                    vb = agg_b["per_ray_mean"][ray_idx][key] * scale
                    if metric_info["direction"] == "up":
                        d = vb - va
                    else:
                        d = va - vb
                    deltas.append(f"{d:+.2f}")
                else:
                    deltas.append("--")
            label_b = method_labels.get(methods[1], methods[1])
            row = f"| | Delta ({label_b}) | " + " | ".join(deltas) + " |"
            lines.append(row)

    return "\n".join(lines)


def format_per_ray_table(dataset, all_agg, methods, method_labels=None):
    """Dispatch to dataset-specific per-ray table formatter."""
    if dataset == "multimnist":
        return format_per_ray_table_multimnist(all_agg, methods, method_labels)
    elif dataset == "sarcos":
        return format_per_ray_table_sarcos(all_agg, methods, method_labels)
    elif dataset == "utkface":
        return format_per_ray_table_utkface(all_agg, methods, method_labels)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Hypervolume Table
# ---------------------------------------------------------------------------

def format_hv_table(dataset, all_agg, methods, method_labels=None, baseline=None):
    """Format hypervolume comparison table.

    Args:
        baseline: Method name to compute % change against.
    """
    if dataset not in HV_CONFIG:
        return f"*{dataset}: No standard HV reference point. Use per-ray dominance instead.*"

    if method_labels is None:
        method_labels = {m: m for m in methods}

    hv_cfg = HV_CONFIG[dataset]
    ref_str = ", ".join(str(x) for x in hv_cfg["ref_point"])

    lines = []
    lines.append(f"**HV** (ref=[{ref_str}], {hv_cfg['space']} space)")
    lines.append("")
    lines.append("| Method | HV (mean +/- std) | CV (%) | vs Baseline |")
    lines.append("|--------|-------------------|--------|-------------|")

    baseline_hv = None
    hv_data = {}

    for method in methods:
        agg = all_agg[method]
        stats = compute_hv_stats(agg["raw_results"], dataset)
        hv_data[method] = stats

        if baseline and method == baseline:
            baseline_hv = stats["mean"] if stats else None

    for method in methods:
        stats = hv_data[method]
        label = method_labels.get(method, method)
        if stats is None:
            lines.append(f"| {label} | N/A | N/A | N/A |")
            continue

        hv_str = _fmt_pm(stats["mean"], stats["std"], 4)
        cv_str = _fmt(stats["cv"] * 100, 2)

        if baseline_hv is not None and baseline_hv > 0:
            delta_pct = (stats["mean"] - baseline_hv) / baseline_hv * 100
            delta_str = f"{delta_pct:+.2f}%"
        else:
            delta_str = "--"

        lines.append(f"| {label} | {hv_str} | {cv_str} | {delta_str} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dominance Table
# ---------------------------------------------------------------------------

def format_dominance_table(dataset, all_agg, method_a, method_b, method_labels=None):
    """Format per-ray Pareto dominance comparison."""
    if method_labels is None:
        method_labels = {method_a: method_a, method_b: method_b}

    label_a = method_labels.get(method_a, method_a)
    label_b = method_labels.get(method_b, method_b)

    dom = compute_dominance_rate(all_agg[method_a], all_agg[method_b], dataset)

    lines = []
    lines.append(f"**Dominance**: {label_a} vs {label_b}")
    lines.append(f"- {label_a} dominates: {dom['a_dominates']}/{dom['total']} ({dom['a_pct']:.1f}%)")
    lines.append(f"- {label_b} dominates: {dom['b_dominates']}/{dom['total']} ({dom['b_pct']:.1f}%)")
    lines.append(f"- Neither: {dom['neither']}/{dom['total']} ({dom['neither_pct']:.1f}%)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stability (CV) Table
# ---------------------------------------------------------------------------

def format_stability_table(dataset, all_agg, methods, method_labels=None, baseline=None):
    """Format CV stability comparison."""
    if dataset not in HV_CONFIG:
        return f"*{dataset}: CV not applicable (no HV).*"

    if method_labels is None:
        method_labels = {m: m for m in methods}

    lines = []
    lines.append("**Stability (CV of HV across seeds)**")
    lines.append("")
    lines.append("| Method | CV (%) | Reduction vs Baseline |")
    lines.append("|--------|--------|-----------------------|")

    baseline_cv = None
    cv_data = {}

    for method in methods:
        stats = compute_hv_stats(all_agg[method]["raw_results"], dataset)
        cv_data[method] = stats
        if baseline and method == baseline:
            baseline_cv = stats["cv"] if stats else None

    for method in methods:
        label = method_labels.get(method, method)
        stats = cv_data[method]
        if stats is None:
            lines.append(f"| {label} | N/A | N/A |")
            continue

        cv_str = _fmt(stats["cv"] * 100, 2)
        if baseline_cv and baseline_cv > 0 and stats["cv"] > 0:
            reduction = baseline_cv / stats["cv"]
            red_str = f"{reduction:.1f}x"
        else:
            red_str = "--"

        lines.append(f"| {label} | {cv_str} | {red_str} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary section (per-task delta)
# ---------------------------------------------------------------------------

def format_diversity_table(dataset, all_agg, methods, method_labels=None):
    """Format Pareto front diversity / population quality table.

    Reports: num_non_dominated (fraction and count), uniformity (2D),
    spearman correlation (2D).
    """
    if method_labels is None:
        method_labels = {m: m for m in methods}

    # Collect stats for all methods
    div_data = {}
    has_uniformity = False
    has_spearman = False
    for method in methods:
        stats = compute_diversity_stats(all_agg[method], dataset)
        div_data[method] = stats
        if stats["uniformity"] is not None:
            has_uniformity = True
        if stats["spearman"] is not None:
            has_spearman = True

    lines = []
    lines.append("**Pareto Front Diversity**")
    lines.append("")

    # Build header
    header_cols = ["Method", "Non-Dominated (mean +/- std)", "ND Fraction"]
    if has_spearman:
        header_cols.append("Spearman (mean +/- std)")
    if has_uniformity:
        header_cols.append("Uniformity (mean +/- std)")

    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["--------"] * len(header_cols)) + "|")

    for method in methods:
        label = method_labels.get(method, method)
        stats = div_data[method]
        cols = [label]

        # Non-dominated count
        if stats["num_non_dominated"] is not None:
            nd = stats["num_non_dominated"]
            cols.append(_fmt_pm(nd["mean"], nd["std"], 1))
            total = stats["num_total_rays"]
            if total > 0:
                frac = nd["mean"] / total * 100
                cols.append(f"{frac:.1f}%")
            else:
                cols.append("--")
        else:
            cols.extend(["--", "--"])

        # Spearman
        if has_spearman:
            sp = stats["spearman"]
            if sp is not None:
                cols.append(_fmt_pm(sp["mean"], sp["std"], 3))
            else:
                cols.append("--")

        # Uniformity
        if has_uniformity:
            uni = stats["uniformity"]
            if uni is not None:
                cols.append(_fmt_pm(uni["mean"], uni["std"], 4))
            else:
                cols.append("--")

        lines.append("| " + " | ".join(cols) + " |")

    return "\n".join(lines)


def format_per_task_summary(dataset, all_agg, method_a, method_b, method_labels=None):
    """Format per-task improvement summary (averaged across rays)."""
    if method_labels is None:
        method_labels = {method_a: method_a, method_b: method_b}

    label_a = method_labels.get(method_a, method_a)
    label_b = method_labels.get(method_b, method_b)

    deltas = compute_per_task_delta(all_agg[method_a], all_agg[method_b], dataset)

    lines = []
    lines.append(f"**Per-Task Improvement** ({label_b} vs {label_a}, averaged across rays)")
    lines.append("")
    lines.append("| Task | Metric | Baseline | Method | Delta | Delta % |")
    lines.append("|------|--------|----------|--------|-------|---------|")

    spec = DATASET_SPECS[dataset]
    for task_name, d in deltas.items():
        metric_label = spec["primary_metrics"][task_name]["label"]
        arrow = "^" if d["direction"] == "up" else "v"
        lines.append(
            f"| {task_name} | {metric_label} ({d['unit']}) {arrow} | "
            f"{_fmt(d['mean_a'])} | {_fmt(d['mean_b'])} | "
            f"{d['delta']:+.2f} | {d['delta_pct']:+.2f}% |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ray description table
# ---------------------------------------------------------------------------

def format_ray_description(dataset):
    """Format a table describing what each evaluation ray represents."""
    spec = DATASET_SPECS[dataset]
    rays = get_eval_rays(dataset)

    lines = []
    lines.append(f"**Evaluation Rays for {dataset}** ({spec['num_tasks']} tasks: {', '.join(spec['task_names'])})")
    lines.append(f"Loss type: {spec['loss_type']}")
    lines.append("")
    lines.append("| Ray # | Preference Vector | Interpretation |")
    lines.append("|-------|-------------------|----------------|")

    for ray_idx, ray_vec, desc in rays:
        vec_str = "[" + ", ".join(f"{v:.2f}" for v in ray_vec) + "]"
        lines.append(f"| {ray_idx} | {vec_str} | {desc} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def format_full_report(dataset, all_agg, methods, method_labels=None, baseline=None):
    """Generate the complete WORKFLOW.md-compliant report for a dataset.

    Args:
        dataset: Dataset name.
        all_agg: {method_name: aggregated_results}
        methods: List of method names (order: baseline first, then others).
        method_labels: {method_name: display_label}
        baseline: Method name to use as baseline for comparisons.

    Returns:
        str: Complete markdown report.
    """
    if method_labels is None:
        method_labels = {m: m for m in methods}

    if baseline is None:
        baseline = methods[0]

    num_seeds = all_agg[methods[0]]["num_seeds"]
    sections = []

    # Title
    sections.append(f"# {dataset.upper()} Results Report")
    sections.append(f"Methods: {', '.join(method_labels.get(m, m) for m in methods)}")
    sections.append(f"Seeds: {num_seeds}")
    sections.append("")

    # Ray description
    sections.append("## 1. Evaluation Ray Description")
    sections.append(format_ray_description(dataset))
    sections.append("")

    # Per-ray per-task table
    sections.append("## 2. Per-Ray Per-Task Results (Primary)")
    sections.append(format_per_ray_table(dataset, all_agg, methods, method_labels))
    sections.append("")

    # Hypervolume
    sections.append("## 3. Hypervolume")
    sections.append(format_hv_table(dataset, all_agg, methods, method_labels, baseline=baseline))
    sections.append("")

    # Dominance
    sections.append("## 4. Per-Ray Pareto Dominance")
    for method in methods:
        if method == baseline:
            continue
        sections.append(format_dominance_table(dataset, all_agg, baseline, method, method_labels))
        sections.append("")

    # Stability
    sections.append("## 5. Stability (CV)")
    sections.append(format_stability_table(dataset, all_agg, methods, method_labels, baseline=baseline))
    sections.append("")

    # Diversity
    sections.append("## 6. Pareto Front Diversity")
    sections.append(format_diversity_table(dataset, all_agg, methods, method_labels))
    sections.append("")

    # Per-task summary
    sections.append("## 7. Per-Task Improvement Summary")
    for method in methods:
        if method == baseline:
            continue
        sections.append(format_per_task_summary(dataset, all_agg, baseline, method, method_labels))
        sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def format_per_ray_latex_multimnist(all_agg, methods, method_labels=None):
    """Generate LaTeX table for MultiMNIST per-ray results."""
    if method_labels is None:
        method_labels = {m: m for m in methods}

    spec = DATASET_SPECS["multimnist"]
    rays = get_eval_rays("multimnist")

    lines = []
    n_cols = len(rays) + 2  # task + method + ray columns
    col_spec = "ll" + "r" * len(rays)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    ray_headers = " & ".join(f"$[{r[1][0]:.1f},{r[1][1]:.1f}]$" for r in rays)
    lines.append(f"Task & Method & {ray_headers} \\\\")
    lines.append("\\midrule")

    for task_name in spec["task_names"]:
        task_short = spec["task_short"][spec["task_names"].index(task_name)]
        metric_info = spec["primary_metrics"][task_name]
        key = metric_info["key"]
        scale = metric_info["scale"]

        for i, method in enumerate(methods):
            agg = all_agg[method]
            label = method_labels.get(method, method)
            values = []
            for ray_idx, _, _ in rays:
                if ray_idx in agg["per_ray_mean"] and key in agg["per_ray_mean"][ray_idx]:
                    val = agg["per_ray_mean"][ray_idx][key] * scale
                    values.append(_fmt(val))
                else:
                    values.append("--")
            task_label = task_short if i == 0 else ""
            row = f"{task_label} & {label} & " + " & ".join(values) + " \\\\"
            lines.append(row)

        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)
