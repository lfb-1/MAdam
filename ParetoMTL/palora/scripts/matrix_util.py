#!/usr/bin/env python3
"""Maintain the MAdam validation matrix CSVs from result JSON files."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reporting.compute import compute_hypervolume_from_results
from scripts.reporting.config import HV_CONFIG


def _normalize_fim_method(fim_method):
    if fim_method is None:
        return "none"
    return str(fim_method)


def _normalize_ablation_name(ablation_name):
    if ablation_name is None:
        return None
    value = str(ablation_name).strip().lower()
    return value or None


def _fallback_build_result_id(
    dataset_name,
    method_name,
    fim_method,
    *,
    ablation_name=None,
    exclude_bn=False,
    fim_eps_damp=1e-6,
    fim_eps_num=1e-12,
    optimizer_name="Adam",
    optimizer_lr=None,
    ray_sampler_name=None,
):
    result_id = f"{dataset_name}_{method_name}_{_normalize_fim_method(fim_method)}"
    ablation_name = _normalize_ablation_name(ablation_name)
    if ablation_name is not None:
        result_id += f"_{ablation_name}"


    if float(fim_eps_damp) != 1e-6:
        exp = -int(round(math.log10(float(fim_eps_damp))))
        result_id += f"_epsdamp{exp}"
    if float(fim_eps_num) != 1e-12:
        exp = -int(round(math.log10(float(fim_eps_num))))
        result_id += f"_epsnum{exp}"

    if str(optimizer_name).upper() not in ("ADAM",):
        lr_val = float(optimizer_lr) if optimizer_lr is not None else 0.0
        exp = int(math.log10(lr_val)) if lr_val > 0 else 0
        mantissa = lr_val / (10**exp) if lr_val > 0 else 0
        if lr_val > 0 and abs(mantissa - 1.0) < 1e-9:
            lr_tag = f"lr1e{exp}"
        else:
            lr_tag = f"lr{lr_val:.0e}".replace("e-0", "e-").replace("e+0", "e")
        result_id += f"_{str(optimizer_name).lower()}_{lr_tag}"

    if ray_sampler_name is not None and str(ray_sampler_name).lower() == "dirichlet":
        result_id += "_dir"

    return result_id


def _load_build_result_id():
    results_io_path = PROJECT_ROOT / "src" / "utils" / "results_io.py"
    spec = importlib.util.spec_from_file_location("_matrix_results_io", results_io_path)
    if spec is None or spec.loader is None:
        return _fallback_build_result_id
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "build_result_id", _fallback_build_result_id)


build_result_id = _load_build_result_id()


DEFAULT_RESULTS_VERSION = os.environ.get("MADAM_RESULTS_VERSION", "v5")

NONPARETO_METHODS = ["ls", "uw", "dwa", "imtl", "cagrad"]
PARETO_METHODS = ["palora", "pamal"]
BASELINE_METHODS = [*NONPARETO_METHODS, *PARETO_METHODS]
FIM_METHODS = ["none", "madam", "diag_madam"]

NONPARETO_METRICS = {
    "multimnist": [
        ("acc_tl(%)", "acc/top-left", 100.0),
        ("acc_br(%)", "acc/bottom-right", 100.0),
        ("avg_acc(%)", "avg_acc", 100.0),
    ],
    "sarcos": [
        *[(f"err_t{i}(x100)", f"error/Task {i}", 100.0) for i in range(1, 8)],
        ("avg_err(x100)", "avg_error", 100.0),
    ],
    "utkface": [
        ("age_huber(x100)", "huber/age", 100.0),
        ("gender_acc(%)", "acc/gender", 100.0),
        ("race_acc(%)", "acc/race", 100.0),
    ],
    "cityscapes": [
        ("mIoU(%)", "test/sem/iou", 100.0),
        ("pix_acc(%)", "test/sem/pix_acc", 100.0),
        ("abs_depth_err", "test/depth/abs_err", 1.0),
        ("rel_depth_err", "test/depth/rel_err", 1.0),
    ],
    "nyuv2": [
        ("mIoU(%)", "test/sem/iou", 100.0),
        ("pix_acc(%)", "test/sem/pix_acc", 100.0),
        ("abs_depth_err", "test/depth/abs_err", 1.0),
        ("rel_depth_err", "test/depth/rel_err", 1.0),
        ("angle_mean(deg)", "test/normal/angle_mean", 1.0),
        ("angle_med(deg)", "test/normal/angle_med", 1.0),
        ("t1(%)", "test/normal/t1", 100.0),
        ("t2(%)", "test/normal/t2", 100.0),
        ("t3(%)", "test/normal/t3", 100.0),
    ],
}

BESTSWEEP_METRICS = {
    "cityscapes": [
        ("best_mIoU(%)", "test/sem/iou", 100.0, "max"),
        ("best_pix_acc(%)", "test/sem/pix_acc", 100.0, "max"),
        ("best_abs_depth_err", "test/depth/abs_err", 1.0, "min"),
        ("best_rel_depth_err", "test/depth/rel_err", 1.0, "min"),
    ],
    "nyuv2": [
        ("best_mIoU(%)", "test/sem/iou", 100.0, "max"),
        ("best_pix_acc(%)", "test/sem/pix_acc", 100.0, "max"),
        ("best_abs_depth_err", "test/depth/abs_err", 1.0, "min"),
        ("best_rel_depth_err", "test/depth/rel_err", 1.0, "min"),
        ("best_angle_mean(deg)", "test/normal/angle_mean", 1.0, "min"),
        ("best_angle_med(deg)", "test/normal/angle_med", 1.0, "min"),
        ("best_t1(%)", "test/normal/t1", 100.0, "max"),
        ("best_t2(%)", "test/normal/t2", 100.0, "max"),
        ("best_t3(%)", "test/normal/t3", 100.0, "max"),
    ],
}

GENERATED_NOTE_PREFIXES = ("submit_error:",)
GENERATED_NOTES = {"state_unknown", "COMPLETED_no_json", "job_id_missing"}


@dataclass(frozen=True)
class ReportFile:
    path: Path
    dataset: str
    kind: str


@dataclass(frozen=True)
class RemoteResult:
    path: str
    result_file: str
    result: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Path to the palora project root.",
    )
    parser.add_argument(
        "--results-version",
        default=DEFAULT_RESULTS_VERSION,
        help="Results version under results/<version>.",
    )
    parser.add_argument("--dataset", help="Restrict to one dataset.")
    parser.add_argument("--method", help="Restrict to one method.")
    parser.add_argument("--fim-method", help="Restrict to one FIM method.")
    parser.add_argument("--seed", type=int, help="Restrict to one seed.")
    parser.add_argument("--limit", type=int, help="Stop after N matching rows.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap", help="Normalize identity fields and backfill completed rows.")
    subparsers.add_parser("refresh", help="Refresh rows from local JSON results.")
    collect = subparsers.add_parser(
        "collect-remote",
        help="Gather JSON results from a remote results directory and fill report CSVs.",
    )
    collect.add_argument("--remote-host", default=os.environ.get("PALORA_REMOTE_HOST"), help="SSH host for remote JSON results.")
    collect.add_argument(
        "--remote-results-dir",
        default=os.environ.get("PALORA_REMOTE_RESULTS_DIR"),
        help="Remote directory containing result JSON files.",
    )
    collect.add_argument("--ssh-bin", default="ssh", help="SSH executable.")
    collect.add_argument("--remote-python", default="python3", help="Remote Python executable.")
    return parser.parse_args()


def format_number(value: float) -> str:
    return f"{float(value):.4f}"


def canonical_run_name(dataset: str, method: str, fim_method: str, seed: int) -> str:
    return f"madam-{dataset}-{method}-{fim_method}-s{seed}"


def canonical_result_file(dataset: str, method: str, fim_method: str, seed: int, results_version: str) -> str:
    result_id = build_result_id(dataset, method, fim_method)
    return f"results/{results_version}/{result_id}_s{seed}.json"


def normalize_result_file(path: str) -> str:
    return str(Path(path.strip()))


def is_runtime_note(note: str) -> bool:
    note = note.strip()
    return note in GENERATED_NOTES or note.startswith(GENERATED_NOTE_PREFIXES)


def semantic_notes(row: dict[str, str]) -> list[str]:
    notes = [note.strip() for note in row.get("notes", "").split(";") if note.strip()]
    return [note for note in notes if not is_runtime_note(note)]


def set_notes(row: dict[str, str], runtime_note: str | None = None) -> None:
    notes = semantic_notes(row)
    if runtime_note:
        notes.append(runtime_note)
    row["notes"] = "; ".join(notes)


def discover_reports(project_root: Path) -> list[ReportFile]:
    reports_dir = project_root / "reports"
    report_files: list[ReportFile] = []
    for path in sorted(reports_dir.glob("*.csv")):
        stem = path.stem
        if stem.endswith("_nonpareto"):
            dataset = stem[: -len("_nonpareto")]
            kind = "nonpareto"
        elif stem.endswith("_pareto"):
            dataset = stem[: -len("_pareto")]
            kind = "pareto"
        elif stem.endswith("_bestsweep"):
            dataset = stem[: -len("_bestsweep")]
            kind = "bestsweep"
        else:
            continue
        report_files.append(ReportFile(path=path, dataset=dataset, kind=kind))
    return report_files


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: (value or "") for key, value in row.items()} for row in reader]
        return list(reader.fieldnames or []), rows


def save_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row_matches(args: argparse.Namespace, dataset: str, row: dict[str, str]) -> bool:
    if args.dataset and dataset != args.dataset:
        return False
    if args.method and row.get("method") != args.method:
        return False
    if args.fim_method and row.get("fim_method") != args.fim_method:
        return False
    if args.seed is not None and str(args.seed) != row.get("seed"):
        return False
    return True


def row_identity(dataset: str, row: dict[str, str]) -> tuple[str, str, int]:
    method = row["method"]
    fim_method = row.get("fim_method") or "none"
    seed = int(row["seed"])
    return method, fim_method, seed


def method_belongs_to_kind(method: str, kind: str) -> bool:
    if kind == "nonpareto":
        return method in NONPARETO_METHODS
    if kind in {"pareto", "bestsweep"}:
        return method in PARETO_METHODS
    return False


def empty_report_row(fieldnames: list[str]) -> dict[str, str]:
    return {fieldname: "" for fieldname in fieldnames}


def reconcile_identity_fields(row: dict[str, str], dataset: str, results_version: str) -> None:
    method, fim_method, seed = row_identity(dataset, row)
    run_name = canonical_run_name(dataset, method, fim_method, seed)
    row["run_id"] = run_name
    row["wandb_run"] = run_name
    if not row.get("result_file", "").strip():
        row["result_file"] = canonical_result_file(dataset, method, fim_method, seed, results_version)


def result_path(project_root: Path, row: dict[str, str]) -> Path:
    result_file = row.get("result_file", "").strip()
    if not result_file:
        raise ValueError("row is missing result_file")
    path = Path(result_file)
    if path.is_absolute():
        return path
    return project_root / path


def load_result(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def extract_single_metrics(result: dict) -> dict[str, float]:
    per_ray = result.get("per_ray_results", {})
    metrics = per_ray.get("0")
    if metrics is None:
        metrics = per_ray.get(0)
    return metrics or {}


def iter_ray_metrics(result: dict) -> list[dict[str, float]]:
    per_ray = result.get("per_ray_results", {})
    rows: list[tuple[int, dict[str, float]]] = []
    for key, metrics in per_ray.items():
        if str(key) == "population":
            continue
        try:
            ray_idx = int(key)
        except (TypeError, ValueError):
            continue
        rows.append((ray_idx, metrics))
    rows.sort(key=lambda item: item[0])
    return [metrics for _, metrics in rows]


def per_ray_for_hv(result: dict) -> dict[int, dict[str, float]]:
    per_ray = result.get("per_ray_results", {})
    out: dict[int, dict[str, float]] = {}
    for key, metrics in per_ray.items():
        if str(key) == "population":
            continue
        try:
            ray_idx = int(key)
        except (TypeError, ValueError):
            continue
        out[ray_idx] = metrics
    return out


def fill_nonpareto(row: dict[str, str], dataset: str, result: dict) -> None:
    metrics = extract_single_metrics(result)
    for column, metric_key, scale in NONPARETO_METRICS[dataset]:
        if metric_key in metrics:
            row[column] = format_number(float(metrics[metric_key]) * scale)


def fill_bestsweep(row: dict[str, str], dataset: str, result: dict) -> None:
    rays = iter_ray_metrics(result)
    for column, metric_key, scale, direction in BESTSWEEP_METRICS[dataset]:
        values = [float(metrics[metric_key]) for metrics in rays if metric_key in metrics]
        if not values:
            continue
        best = max(values) if direction == "max" else min(values)
        row[column] = format_number(best * scale)


def fill_ray_columns(row: dict[str, str], dataset: str, result: dict) -> None:
    if dataset == "multimnist":
        metrics = [
            ("avg_acc", "avg_acc_r{ray}(%)", 100.0),
            ("acc/top-left", "acc_tl_r{ray}(%)", 100.0),
            ("acc/bottom-right", "acc_br_r{ray}(%)", 100.0),
        ]
    elif dataset == "sarcos":
        metrics = [
            ("avg_error", "avg_err_r{ray}(x100)", 100.0),
            *[
                (f"error/Task {task}", f"err_t{task}_r{{ray}}(x100)", 100.0)
                for task in range(1, 8)
            ],
        ]
    else:
        return

    for ray_idx, ray_metrics in enumerate(iter_ray_metrics(result)):
        for metric_key, column_template, scale in metrics:
            column = column_template.format(ray=ray_idx)
            if column in row and metric_key in ray_metrics:
                row[column] = format_number(float(ray_metrics[metric_key]) * scale)


def fill_pareto(row: dict[str, str], dataset: str, result: dict) -> None:
    if "n_eval_rays" in row:
        row["n_eval_rays"] = str(len(iter_ray_metrics(result)))

    fill_ray_columns(row, dataset, result)

    if dataset not in HV_CONFIG:
        return

    row["hv_ref_point"] = json.dumps(HV_CONFIG[dataset]["ref_point"].tolist())
    try:
        hv_value = compute_hypervolume_from_results(per_ray_for_hv(result), dataset)
    except Exception as exc:  # pragma: no cover - defensive for remote env drift
        set_notes(row, f"hv_error:{type(exc).__name__}")
        return

    if hv_value is not None:
        row["hv"] = format_number(hv_value)


def fill_from_result(row: dict[str, str], dataset: str, kind: str, result: dict) -> None:
    if kind == "nonpareto":
        fill_nonpareto(row, dataset, result)
    elif kind == "pareto":
        fill_pareto(row, dataset, result)
    elif kind == "bestsweep":
        fill_bestsweep(row, dataset, result)
    else:  # pragma: no cover - unreachable with current discovery
        raise ValueError(f"Unknown report kind: {kind}")


def result_identity(result: dict) -> tuple[str, str, str, int] | None:
    try:
        dataset = str(result["dataset"])
        method = str(result["method"])
        fim_method = str(_normalize_fim_method(result.get("fim_method", "none")))
        seed = int(result["seed"])
    except (KeyError, TypeError, ValueError):
        return None
    return dataset, method, fim_method, seed


def fetch_remote_results(args: argparse.Namespace) -> list[RemoteResult]:
    if not args.remote_host or not args.remote_results_dir:
        raise ValueError(
            "collect-remote requires --remote-host and --remote-results-dir "
            "or PALORA_REMOTE_HOST and PALORA_REMOTE_RESULTS_DIR."
        )
    pattern = f"{args.dataset}_*.json" if args.dataset else "*.json"
    script = f"""
import json
import sys
from pathlib import Path

root = Path({json.dumps(str(args.remote_results_dir))})
pattern = {json.dumps(pattern)}
results = []
for path in sorted(root.glob(pattern)):
    try:
        with path.open() as handle:
            result = json.load(handle)
    except Exception as exc:
        results.append({{"path": str(path), "error": f"{{type(exc).__name__}}:{{exc}}"}})
        continue
    results.append({{"path": str(path), "result": result}})
json.dump({{"results": results}}, sys.stdout)
"""
    proc = subprocess.run(
        [args.ssh_bin, args.remote_host, f"{args.remote_python} -c {shlex.quote(script)}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        error_text = (proc.stderr or proc.stdout or "remote_collect_failed").strip()
        raise RuntimeError(error_text.splitlines()[-1] if error_text else "remote_collect_failed")

    payload = json.loads(proc.stdout)
    remote_results: list[RemoteResult] = []
    for item in payload.get("results", []):
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        path = str(item.get("path", ""))
        filename = Path(path).name
        if not filename:
            continue
        remote_results.append(
            RemoteResult(
                path=path,
                result_file=f"results/{args.results_version}/{filename}",
                result=result,
            )
        )
    return remote_results


def build_remote_result_index(remote_results: list[RemoteResult]) -> dict[tuple[str, str, str, int], list[RemoteResult]]:
    index: dict[tuple[str, str, str, int], list[RemoteResult]] = {}
    for item in remote_results:
        identity = result_identity(item.result)
        if identity is None:
            continue
        index.setdefault(identity, []).append(item)

    for matches in index.values():
        matches.sort(key=lambda item: (len(Path(item.path).stem), Path(item.path).name))
    return index


def remote_results_for_report(
    dataset: str,
    kind: str,
    remote_index: dict[tuple[str, str, str, int], list[RemoteResult]],
) -> list[tuple[str, str, int, RemoteResult]]:
    items: list[tuple[str, str, int, RemoteResult]] = []
    for (remote_dataset, method, fim_method, seed), matches in remote_index.items():
        if remote_dataset != dataset:
            continue
        if not method_belongs_to_kind(method, kind):
            continue
        items.append((method, fim_method, seed, matches[0]))
    items.sort(key=lambda item: report_sort_key(item[0], item[1], item[2]))
    return items


def report_sort_key(method: str, fim_method: str, seed: int) -> tuple[int, int, int, str, str]:
    method_order = NONPARETO_METHODS + PARETO_METHODS
    method_idx = method_order.index(method) if method in method_order else len(method_order)
    fim_idx = FIM_METHODS.index(fim_method) if fim_method in FIM_METHODS else len(FIM_METHODS)
    return method_idx, fim_idx, seed, method, fim_method


def sort_report_rows(rows: list[dict[str, str]]) -> None:
    def key(row: dict[str, str]) -> tuple[int, int, int, str, str]:
        method = row.get("method", "")
        fim_method = row.get("fim_method", "") or "none"
        try:
            seed = int(row.get("seed", "0"))
        except ValueError:
            seed = 0
        return report_sort_key(method, fim_method, seed)

    rows.sort(key=key)


def append_missing_remote_rows(
    args: argparse.Namespace,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    dataset: str,
    kind: str,
    remote_index: dict[tuple[str, str, str, int], list[RemoteResult]],
    results_version: str,
) -> int:
    existing = {
        (dataset, row.get("method", ""), row.get("fim_method", "") or "none", int(row.get("seed", "0")))
        for row in rows
        if row.get("method") and row.get("seed", "").isdigit()
    }

    added = 0
    for method, fim_method, seed, item in remote_results_for_report(dataset, kind, remote_index):
        if args.method and method != args.method:
            continue
        if args.fim_method and fim_method != args.fim_method:
            continue
        if args.seed is not None and seed != args.seed:
            continue
        identity = (dataset, method, fim_method, seed)
        if identity in existing:
            continue
        row = empty_report_row(fieldnames)
        row["method"] = method
        row["fim_method"] = fim_method
        row["seed"] = str(seed)
        row["result_file"] = item.result_file
        row["status"] = "pending"
        reconcile_identity_fields(row, dataset, results_version)
        rows.append(row)
        existing.add(identity)
        added += 1

    if added:
        sort_report_rows(rows)
    return added


def select_remote_result(
    row: dict[str, str],
    dataset: str,
    remote_index: dict[tuple[str, str, str, int], list[RemoteResult]],
) -> RemoteResult | None:
    method, fim_method, seed = row_identity(dataset, row)
    matches = remote_index.get((dataset, method, fim_method, seed), [])
    if not matches:
        return None

    current_result_file = normalize_result_file(row.get("result_file", ""))
    for item in matches:
        if normalize_result_file(item.result_file) == current_result_file:
            return item
    return matches[0]


def collect_remote_row(
    row: dict[str, str],
    dataset: str,
    kind: str,
    remote_index: dict[tuple[str, str, str, int], list[RemoteResult]],
) -> bool:
    item = select_remote_result(row, dataset, remote_index)
    if item is None:
        return False

    fill_from_result(row, dataset, kind, item.result)
    row["result_file"] = item.result_file
    row["status"] = "done"
    set_notes(row)
    return True


def update_delta_hv(rows: list[dict[str, str]]) -> None:
    baselines: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("fim_method") != "none":
            continue
        if row.get("status") != "done":
            continue
        hv_text = row.get("hv", "").strip()
        if not hv_text:
            continue
        baselines[(row["method"], row["seed"])] = float(hv_text)

    for row in rows:
        if "delta_hv_vs_none(%)" not in row:
            continue
        if row.get("fim_method") == "none":
            row["delta_hv_vs_none(%)"] = ""
            continue
        hv_text = row.get("hv", "").strip()
        baseline = baselines.get((row["method"], row["seed"]))
        if not hv_text or baseline in (None, 0.0):
            row["delta_hv_vs_none(%)"] = ""
            continue
        delta = (float(hv_text) - baseline) / baseline * 100.0
        row["delta_hv_vs_none(%)"] = format_number(delta)


def bootstrap_row(project_root: Path, dataset: str, kind: str, row: dict[str, str]) -> bool:
    result_json = try_load_row_result(project_root, row)
    if result_json is None:
        return False

    fill_from_result(row, dataset, kind, result_json)
    row["status"] = "done"
    set_notes(row)
    return True


def try_load_row_result(project_root: Path, row: dict[str, str]) -> dict | None:
    try:
        path = result_path(project_root, row)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        return load_result(path)
    except Exception:
        return None


def refresh_row(project_root: Path, dataset: str, kind: str, row: dict[str, str]) -> bool:
    result_json = try_load_row_result(project_root, row)
    if result_json is not None:
        fill_from_result(row, dataset, kind, result_json)
        row["status"] = "done"
        set_notes(row)
        return True

    if row.get("status") == "blocked":
        return False
    return False


def process_reports(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    total_matches = 0
    total_updates = 0
    total_completed = 0
    total_missing = 0
    remote_results: list[RemoteResult] = []
    remote_index: dict[tuple[str, str, str, int], list[RemoteResult]] = {}

    if args.command == "collect-remote":
        try:
            remote_results = fetch_remote_results(args)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "command": args.command,
                        "remote_host": args.remote_host,
                        "remote_results_dir": args.remote_results_dir,
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
            return 1
        remote_index = build_remote_result_index(remote_results)

    for report in discover_reports(project_root):
        fieldnames, rows = load_rows(report.path)
        changed = False

        if args.command == "collect-remote" and (not args.dataset or report.dataset == args.dataset):
            added = append_missing_remote_rows(
                args,
                fieldnames,
                rows,
                report.dataset,
                report.kind,
                remote_index,
                args.results_version,
            )
            if added:
                changed = True

        for row in rows:
            if not row_matches(args, report.dataset, row):
                continue
            if args.limit is not None and total_matches >= args.limit:
                break
            total_matches += 1

            reconcile_identity_fields(row, report.dataset, args.results_version)

            if args.command == "bootstrap":
                if bootstrap_row(project_root, report.dataset, report.kind, row):
                    total_completed += 1
                    changed = True
            elif args.command == "refresh":
                before = dict(row)
                completed = refresh_row(project_root, report.dataset, report.kind, row)
                if completed:
                    total_completed += 1
                if row != before:
                    changed = True
            elif args.command == "collect-remote":
                before = dict(row)
                completed = collect_remote_row(row, report.dataset, report.kind, remote_index)
                if completed:
                    total_completed += 1
                else:
                    total_missing += 1
                if row != before:
                    changed = True

        if report.kind == "pareto":
            before_rows = [dict(row) for row in rows]
            update_delta_hv(rows)
            if rows != before_rows:
                changed = True

        if changed:
            save_rows(report.path, fieldnames, rows)
            total_updates += 1

        if args.limit is not None and total_matches >= args.limit:
            break

    print(
        json.dumps(
            {
                "command": args.command,
                "matched_rows": total_matches,
                "updated_files": total_updates,
                "completed_rows": total_completed,
                "missing_results": total_missing,
                "remote_results": len(remote_results),
                "remote_identities": len(remote_index),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    return process_reports(args)


if __name__ == "__main__":
    raise SystemExit(main())
