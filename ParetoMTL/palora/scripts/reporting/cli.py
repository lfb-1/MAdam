"""CLI entry point for the standardized reporting system.

Usage:
    # Full report for a dataset (auto-discover all methods)
    python -m scripts.reporting multimnist

    # Compare specific result ids
    python -m scripts.reporting multimnist --methods ls_none ls_madam palora_none

    # Set baseline and custom labels
    python -m scripts.reporting multimnist --methods ls_none ls_madam --baseline ls_none \
        --labels "LS" "LS + MAdam"

    # All datasets at once
    python -m scripts.reporting all

    # Save to file
    python -m scripts.reporting multimnist --output reports/multimnist_report.md

    # JSON summary output (machine-readable)
    python -m scripts.reporting multimnist --json

    # Custom results directory
    python -m scripts.reporting multimnist --results-dir results/v_madam_neurips_main
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from .loader import load_all_methods, discover_methods, RESULTS_DIR
from .format import format_full_report
from .compute import compute_hv_stats, compute_dominance_rate, compute_per_task_delta, compute_diversity_stats
from .config import DATASET_SPECS, HV_CONFIG


logger = logging.getLogger(__name__)

ALL_DATASETS = ["multimnist", "sarcos", "utkface"]


def build_json_summary(dataset, all_agg, methods, baseline):
    """Build a machine-readable JSON summary of all results."""
    summary = {
        "dataset": dataset,
        "methods": methods,
        "baseline": baseline,
        "results": {},
    }

    for method in methods:
        agg = all_agg[method]
        method_summary = {
            "num_seeds": agg["num_seeds"],
            "per_ray_mean": {},
            "per_ray_std": {},
        }

        # Per-ray results
        for ray_idx in sorted(agg["per_ray_mean"].keys()):
            method_summary["per_ray_mean"][ray_idx] = agg["per_ray_mean"][ray_idx]
            method_summary["per_ray_std"][ray_idx] = agg["per_ray_std"][ray_idx]

        # HV stats
        hv_stats = compute_hv_stats(agg["raw_results"], dataset)
        if hv_stats:
            method_summary["hypervolume"] = {
                "mean": hv_stats["mean"],
                "std": hv_stats["std"],
                "cv": hv_stats["cv"],
                "per_seed": hv_stats["per_seed"],
                "ref_point": hv_stats["ref_point"],
            }

        # Diversity stats
        div_stats = compute_diversity_stats(agg, dataset)
        method_summary["diversity"] = div_stats

        # Dominance vs baseline
        if method != baseline and baseline in all_agg:
            dom = compute_dominance_rate(all_agg[baseline], all_agg[method], dataset)
            method_summary["dominance_vs_baseline"] = dom

            # Per-task delta
            deltas = compute_per_task_delta(all_agg[baseline], all_agg[method], dataset)
            method_summary["per_task_delta"] = deltas

        summary["results"][method] = method_summary

    return summary


def run_report(dataset, methods=None, method_labels=None, baseline=None,
               output=None, results_dir=None, json_output=False, seeds=(0, 1, 2)):
    """Generate report for a single dataset."""
    all_agg = load_all_methods(dataset, methods, seeds, results_dir)

    if not all_agg:
        print(f"No results found for {dataset}.", file=sys.stderr)
        return False

    # Default methods: all discovered, sorted
    if methods is None:
        methods = sorted(all_agg.keys())
    else:
        # Filter to methods that have results
        methods = [m for m in methods if m in all_agg]

    if not methods:
        print(f"No valid methods with results for {dataset}.", file=sys.stderr)
        return False

    # Default baseline: prefer a *_none variant, else legacy "none", else first method
    if baseline is None:
        none_variants = [m for m in methods if m.endswith("_none")]
        if none_variants:
            baseline = none_variants[0]
        else:
            baseline = "none" if "none" in methods else methods[0]

    # Build labels
    if method_labels is None:
        method_labels = {m: m for m in methods}

    if json_output:
        summary = build_json_summary(dataset, all_agg, methods, baseline)
        text = json.dumps(summary, indent=2, default=str)
    else:
        text = format_full_report(dataset, all_agg, methods, method_labels, baseline)

    if output:
        outpath = Path(output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "w") as f:
            f.write(text)
        print(f"Report saved to {outpath}")
    else:
        print(text)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Standardized PaLoRA result reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset",
        choices=ALL_DATASETS + ["all"],
        help="Dataset to report on, or 'all' for all datasets",
    )
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=None,
        help="Result ids to include, e.g. ls_none or palora_madam (default: auto-discover all)",
    )
    parser.add_argument(
        "--labels", "-l",
        nargs="+",
        default=None,
        help="Display labels for methods (must match --methods count)",
    )
    parser.add_argument(
        "--baseline", "-b",
        default=None,
        help="Method to use as baseline for comparisons (default: 'none')",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--results-dir", "-d",
        default=None,
        help=f"Results directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON instead of markdown",
    )
    parser.add_argument(
        "--seeds", "-s",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Seed indices to include (default: 0 1 2)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Build method labels dict
    method_labels = None
    if args.labels:
        if args.methods is None:
            parser.error("--labels requires --methods")
        if len(args.labels) != len(args.methods):
            parser.error(f"--labels count ({len(args.labels)}) must match --methods count ({len(args.methods)})")
        method_labels = dict(zip(args.methods, args.labels))

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        output = args.output
        if args.dataset == "all" and args.output:
            # For 'all' mode, append dataset name to output path
            p = Path(args.output)
            output = str(p.parent / f"{p.stem}_{dataset}{p.suffix}")

        run_report(
            dataset=dataset,
            methods=args.methods,
            method_labels=method_labels,
            baseline=args.baseline,
            output=output,
            results_dir=args.results_dir,
            json_output=args.json,
            seeds=tuple(args.seeds),
        )

        if args.dataset == "all" and dataset != datasets[-1]:
            print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
