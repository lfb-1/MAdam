import json
import hashlib
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np


RESULTS_VERSION_ENV = "PINNACLE_RESULTS_VERSION"
DEFAULT_RESULTS_VERSION = "v1"

ERROR_COLUMNS = [
    "epoch",
    "mae",
    "mse",
    "mxe",
    "l1re",
    "l2re",
    "crmse",
    "frmse_low",
    "frmse_mid",
    "frmse_high",
]


def _project_root():
    return Path(__file__).resolve().parents[2]


def _sanitize_token(value):
    text = str(value or "none").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "none"


def _stable_hash(payload):
    text = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value


def _json_safe(value):
    value = _finite_or_none(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _load_table(path):
    path = Path(path)
    if not path.exists():
        return None
    data = np.loadtxt(path)
    if data.size == 0:
        return None
    return np.atleast_2d(data)


def _load_final_and_best_errors(save_path):
    data = _load_table(Path(save_path) / "errors.txt")
    if data is None:
        return {}, {}

    final = {}
    best = {}
    for idx, name in enumerate(ERROR_COLUMNS):
        if idx >= data.shape[1]:
            continue
        values = data[:, idx]
        final[name] = _finite_or_none(values[-1])
        if name == "epoch":
            best[name] = _finite_or_none(values[-1])
            continue
        valid = values[np.isfinite(values)]
        best[name] = _finite_or_none(np.min(valid)) if valid.size else None
    return final, best


def _loss_component_names(model):
    config = getattr(getattr(model, "pde", None), "loss_config", None)
    if not config:
        return []
    return [entry.get("name", f"component_{idx}") for idx, entry in enumerate(config)]


def _load_final_losses(model, save_path):
    data = _load_table(Path(save_path) / "loss.txt")
    if data is None:
        return {}

    row = data[-1]
    pde = getattr(model, "pde", None)
    num_loss = int(getattr(pde, "num_loss", 0) or 0)
    names = _loss_component_names(model)
    if num_loss <= 0:
        return {"step": _finite_or_none(row[0]) if row.size else None}

    train = row[1 : 1 + num_loss]
    test = row[1 + num_loss : 1 + 2 * num_loss]
    weights = row[1 + 2 * num_loss : 1 + 3 * num_loss]
    loss = {
        "step": _finite_or_none(row[0]),
        "train_total": _finite_or_none(np.sum(train)),
        "test_total": _finite_or_none(np.sum(test)),
        "train_components": {},
        "test_components": {},
        "weights": {},
    }
    for idx in range(num_loss):
        name = names[idx] if idx < len(names) else f"component_{idx}"
        if idx < len(train):
            loss["train_components"][name] = _finite_or_none(train[idx])
        if idx < len(test):
            loss["test_components"][name] = _finite_or_none(test[idx])
        if idx < len(weights):
            loss["weights"][name] = _finite_or_none(weights[idx])
    return loss


def _extract_runtime_seconds(save_path):
    log_path = Path(save_path) / "log.txt"
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("'train'"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    return float(parts[2])
                except ValueError:
                    return None
    return None


def build_setup_identity(pde_name, baseline_label, components, benchmark_config):
    identity_keys = [
        "optimizer",
        "preconditioner",
        "preconditioner_stochastic",
        "preconditioner_shared_only",
        "preconditioner_exclude_batchnorm",
        "loss_balancing",
        "sampling",
        "loss_augmentation",
        "architecture",
        "learning_rate",
        "iterations",
        "display_every",
        "plot_every",
        "hidden_layers",
        "loss_weight",
        "pde_num_loss",
        "pde_num_pde",
    ]
    config = {key: benchmark_config.get(key) for key in identity_keys if key in benchmark_config}
    return {
        "dataset": "pinnacle",
        "pde": str(pde_name),
        "method": str(baseline_label),
        "components": components,
        "config": config,
    }


def build_result_id(pde_name, baseline_label, components, seed, setup_id=None):
    parts = [
        "pinnacle",
        _sanitize_token(pde_name),
        _sanitize_token(baseline_label),
        _sanitize_token(components.get("preconditioner", "none")),
        _sanitize_token(components.get("loss_balancing", "none")),
        _sanitize_token(components.get("sampling", "none")),
        _sanitize_token(components.get("loss_augmentation", "none")),
        _sanitize_token(components.get("architecture", "tanh")),
    ]
    if setup_id is not None:
        parts.append(_sanitize_token(setup_id))
    parts.append(f"s{int(seed)}")
    return "_".join(parts)


def _trainer_metrics(model, status, error, runtime_seconds):
    train_state = getattr(model, "train_state", None)
    if train_state is None:
        return {
            "status": status,
            "error": str(error) if error is not None else None,
            "runtime_seconds": runtime_seconds,
        }
    loss_train = getattr(train_state, "loss_train", None)
    loss_test = getattr(train_state, "loss_test", None)
    return {
        "status": status,
        "error": str(error) if error is not None else None,
        "runtime_seconds": runtime_seconds,
        "final_step": _finite_or_none(getattr(train_state, "step", None)),
        "final_epoch": _finite_or_none(getattr(train_state, "epoch", None)),
        "final_loss_train_total": _finite_or_none(np.sum(loss_train)) if loss_train is not None else None,
        "final_loss_test_total": _finite_or_none(np.sum(loss_test)) if loss_test is not None else None,
        "best_step": _finite_or_none(getattr(train_state, "best_step", None)),
        "best_loss_train_total": _finite_or_none(getattr(train_state, "best_loss_train", None)),
        "best_loss_test_total": _finite_or_none(getattr(train_state, "best_loss_test", None)),
    }


def save_run_result(model, save_path, status="success", error=None):
    """Save one comparison-ready JSON artifact for a PINNacle run."""
    save_path = Path(save_path)
    benchmark_context = dict(getattr(model, "_madam_benchmark_context", {}) or {})
    run_context = dict(getattr(model, "_madam_run_context", {}) or {})
    components = dict(benchmark_context.get("components", {}) or {})
    pde = getattr(model, "pde", None)
    pde_name = type(pde).__name__ if pde is not None else "unknown"
    seed = run_context.get("run_seed", benchmark_context.get("command_seed", -1))
    baseline_label = benchmark_context.get("baseline_label", components.get("optimizer", "unknown"))
    benchmark_config = dict(benchmark_context.get("wandb_config", {}) or {})

    final_metrics, best_metrics = _load_final_and_best_errors(save_path)
    final_loss = _load_final_losses(model, save_path)
    runtime_seconds = _extract_runtime_seconds(save_path)
    setup_identity = build_setup_identity(pde_name, baseline_label, components, benchmark_config)
    setup_id = f"h{_stable_hash(setup_identity)}"
    result_id = build_result_id(pde_name, baseline_label, components, int(seed), setup_id=setup_id)

    result = {
        "dataset": "pinnacle",
        "pde": pde_name,
        "method": str(baseline_label),
        "preconditioner": str(components.get("preconditioner", "none")),
        "seed": int(seed),
        "status": status,
        "error": str(error) if error is not None else None,
        "setup_id": setup_id,
        "setup_identity": setup_identity,
        "result_id": result_id,
        "run_path": str(save_path),
        "logging": {
            "wandb_project": benchmark_context.get("wandb_project"),
            "wandb_entity": benchmark_context.get("wandb_entity"),
            "wandb_group": benchmark_context.get("wandb_group"),
            "wandb_job_type": benchmark_context.get("wandb_job_type"),
            "wandb_tags": benchmark_context.get("wandb_tags", []),
            "wandb_run_name": run_context.get("wandb_run_name"),
            "wandb_run_id": run_context.get("wandb_run_id"),
            "slurm_job_id": benchmark_context.get("slurm_job_id"),
            "slurm_job_name": benchmark_context.get("slurm_job_name"),
        },
        "config": {
            "components": components,
            "benchmark": benchmark_config,
            "run": run_context,
            "num_loss": int(getattr(pde, "num_loss", 0) or 0),
            "num_pde": int(getattr(pde, "num_pde", 0) or 0),
            "input_dim": int(getattr(pde, "input_dim", 0) or 0),
            "output_dim": int(getattr(pde, "output_dim", 0) or 0),
        },
        "metrics": {
            "final": final_metrics,
            "best": best_metrics,
            "loss": final_loss,
            "runtime_seconds": runtime_seconds,
            "trainer": _trainer_metrics(model, status, error, runtime_seconds),
        },
    }

    result = _json_safe(result)
    save_path.mkdir(parents=True, exist_ok=True)
    local_path = save_path / "result.json"
    with local_path.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    results_version = os.environ.get(RESULTS_VERSION_ENV, DEFAULT_RESULTS_VERSION)
    outdir = _project_root() / "results" / results_version
    outdir.mkdir(parents=True, exist_ok=True)
    canonical_path = outdir / f"{result_id}.json"
    shutil.copyfile(local_path, canonical_path)

    print(f"Saved PINNacle result JSON to: {canonical_path}")
    return str(canonical_path)
