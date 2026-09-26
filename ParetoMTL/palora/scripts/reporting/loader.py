"""Load and aggregate JSON result files across seeds.

Reads results from results/<version>/{dataset}_{method}_{fim_method}..._s{seed}.json.
"""
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import BASELINE_METHODS, DATASET_SPECS, verify_config

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results") / os.environ.get("MADAM_RESULTS_VERSION", "v5")


def normalize_fim_method(fim_method):
    if fim_method is None:
        return "none"
    return str(fim_method)


def _extract_file_id(dataset, path):
    name = path.stem
    rest = name[len(dataset) + 1 :]
    parts = rest.rsplit("_s", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return None


def _embedded_method_id(data):
    method = str(data.get("method", "unknown"))
    fim_method = normalize_fim_method(data.get("fim_method", "none"))
    return f"{method}_{fim_method}"


def _is_allowed_baseline_id(method_id):
    method = str(method_id).split("_", 1)[0]
    return method in BASELINE_METHODS


def _candidate_paths(dataset, method_id, seed, rdir):
    exact = []
    for suffix in ("", "_nobn"):
        exact.append(rdir / f"{dataset}_{method_id}{suffix}_s{seed}.json")
    return [path for path in exact if path.exists()]


def _scan_matching_path(dataset, method_id, seed, rdir):
    matches = []
    for path in sorted(rdir.glob(f"{dataset}_*_s{seed}.json")):
        try:
            file_id = _extract_file_id(dataset, path)
            if file_id == method_id:
                matches.append(path)
                continue
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if _embedded_method_id(data) == method_id:
            matches.append(path)

    if not matches:
        return None

    matches.sort(key=lambda path: (len(path.stem), path.name))
    if len(matches) > 1:
        logger.warning(
            "Multiple result files matched %s/%s/s%d; using %s",
            dataset,
            method_id,
            seed,
            matches[0].name,
        )
    return matches[0]


def find_result_files(dataset, method_id, seeds=(0, 1, 2), results_dir=None):
    """Find JSON result files for a dataset/result-id across seeds.

    Returns:
        Dict[int, Path]: {seed: path} for found files.
    """
    rdir = Path(results_dir) if results_dir else RESULTS_DIR
    found = {}
    for seed in seeds:
        exact = _candidate_paths(dataset, method_id, seed, rdir)
        if exact:
            found[seed] = exact[0]
            continue

        scanned = _scan_matching_path(dataset, method_id, seed, rdir)
        if scanned is not None:
            found[seed] = scanned
    return found


def load_result(path):
    """Load a single JSON result file.

    Returns:
        dict with keys: dataset, method, fim_method, seed, config, per_ray_results
    """
    with open(path) as f:
        data = json.load(f)
    return data


def load_method_results(dataset, method_id, seeds=(0, 1, 2), results_dir=None):
    """Load all seed results for a dataset/result-id.

    Returns:
        list of result dicts (one per seed), sorted by seed.
    """
    files = find_result_files(dataset, method_id, seeds, results_dir)
    results = []
    for seed in sorted(files.keys()):
        data = load_result(files[seed])
        # Verify config
        is_ok, issues = verify_config(dataset, data.get("config", {}))
        if not is_ok:
            logger.warning(
                "Config mismatch for %s/%s/s%d: %s",
                dataset, method_id, seed, "; ".join(issues)
            )
        embedded_id = _embedded_method_id(data)
        file_id = _extract_file_id(dataset, files[seed])
        if method_id == embedded_id and file_id != method_id:
            logger.warning(
                "Loaded legacy-named result for %s/%s/s%d from %s",
                dataset,
                method_id,
                seed,
                files[seed].name,
            )
        results.append(data)
    return results


def get_per_ray_metrics(result):
    """Extract per-ray metrics from a result, excluding 'population'.

    Returns:
        Dict[int, dict]: {ray_index: {metric_key: value}}
    """
    per_ray = result.get("per_ray_results", {})
    out = {}
    for k, v in per_ray.items():
        if k == "population":
            continue
        out[int(k)] = v
    return out


def get_population_metrics(result):
    """Extract population-level metrics (HV, spearman, etc.) if present."""
    return result.get("per_ray_results", {}).get("population", {})


def aggregate_across_seeds(results_list, dataset):
    """Aggregate per-ray metrics across seeds.

    Args:
        results_list: List of result dicts (one per seed).
        dataset: Dataset name for metric spec lookup.

    Returns:
        dict with:
            "per_ray_mean": {ray_idx: {metric: mean_value}}
            "per_ray_std": {ray_idx: {metric: std_value}}
            "per_ray_all": {ray_idx: {metric: [val_seed0, val_seed1, ...]}}
            "population_mean": {metric: mean}
            "population_std": {metric: std}
            "num_seeds": int
    """
    if not results_list:
        return None

    # Collect per-ray values across seeds
    ray_values = defaultdict(lambda: defaultdict(list))
    pop_values = defaultdict(list)

    for result in results_list:
        per_ray = get_per_ray_metrics(result)
        for ray_idx, metrics in per_ray.items():
            for metric_key, value in metrics.items():
                ray_values[ray_idx][metric_key].append(float(value))

        pop = get_population_metrics(result)
        for metric_key, value in pop.items():
            pop_values[metric_key].append(float(value))

    # Compute mean and std
    per_ray_mean = {}
    per_ray_std = {}
    per_ray_all = {}

    for ray_idx in sorted(ray_values.keys()):
        per_ray_mean[ray_idx] = {}
        per_ray_std[ray_idx] = {}
        per_ray_all[ray_idx] = {}
        for metric_key, vals in ray_values[ray_idx].items():
            arr = np.array(vals)
            per_ray_mean[ray_idx][metric_key] = float(np.mean(arr))
            per_ray_std[ray_idx][metric_key] = float(np.std(arr))
            per_ray_all[ray_idx][metric_key] = vals

    pop_mean = {k: float(np.mean(v)) for k, v in pop_values.items()}
    pop_std = {k: float(np.std(v)) for k, v in pop_values.items()}

    return {
        "per_ray_mean": per_ray_mean,
        "per_ray_std": per_ray_std,
        "per_ray_all": per_ray_all,
        "population_mean": pop_mean,
        "population_std": pop_std,
        "num_seeds": len(results_list),
    }


def discover_methods(dataset, results_dir=None):
    """Discover all result ids available for a dataset.

    Scans result files matching {dataset}_*_s*.json.

    Returns:
        List of result-id strings found.
    """
    rdir = Path(results_dir) if results_dir else RESULTS_DIR
    if not rdir.exists():
        return []

    methods = set()
    for f in rdir.glob(f"{dataset}_*_s*.json"):
        file_id = _extract_file_id(dataset, f)
        if file_id is None:
            continue
        if not _is_allowed_baseline_id(file_id):
            continue
        try:
            data = load_result(f)
        except Exception:
            methods.add(file_id)
            continue

        embedded_id = _embedded_method_id(data)
        if not _is_allowed_baseline_id(embedded_id):
            continue
        if file_id.startswith(embedded_id):
            methods.add(file_id)
        else:
            methods.add(file_id)
            methods.add(embedded_id)

    return sorted(methods)


def load_all_methods(dataset, methods=None, seeds=(0, 1, 2), results_dir=None):
    """Load and aggregate results for all methods of a dataset.

    Args:
        dataset: Dataset name.
        methods: List of result ids. If None, auto-discover.
        seeds: Tuple of seed indices.
        results_dir: Override results directory.

    Returns:
        Dict[str, dict]: {method_name: aggregated_results}
    """
    if methods is None:
        methods = discover_methods(dataset, results_dir)

    all_results = {}
    for method in methods:
        raw = load_method_results(dataset, method, seeds, results_dir)
        if raw:
            agg = aggregate_across_seeds(raw, dataset)
            agg["raw_results"] = raw
            all_results[method] = agg
        else:
            logger.warning("No results found for %s/%s", dataset, method)

    return all_results
