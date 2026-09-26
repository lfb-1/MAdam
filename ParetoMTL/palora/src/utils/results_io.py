import json
import logging
import math
import os


def normalize_fim_method(fim_method):
    if fim_method is None:
        return "none"
    return str(fim_method)


def normalize_ablation_name(ablation_name):
    if ablation_name is None:
        return None
    value = str(ablation_name).strip().lower()
    return value or None


def coerce_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off", ""):
            return False
    return bool(value)


def resolve_fim_filter_flag(value, fim_method):
    resolved = coerce_optional_bool(value)
    if resolved is None:
        return normalize_fim_method(fim_method) != "none"
    return resolved

def resolve_exclude_bn_flag(value):
    resolved = coerce_optional_bool(value)
    if resolved is None:
        return False
    return resolved

def build_result_id(
    dataset_name,
    method_name,
    fim_method,
    *,
    ablation_name=None,
    exclude_bn=False,
    fim_eps_damp=1e-8,
    fim_eps_num=1e-12,
    optimizer_name="Adam",
    optimizer_lr=None,
    ray_sampler_name=None,
):
    result_id = f"{dataset_name}_{method_name}_{normalize_fim_method(fim_method)}"
    ablation_name = normalize_ablation_name(ablation_name)
    if ablation_name is not None:
        result_id += f"_{ablation_name}"


    if float(fim_eps_damp) != 1e-8:
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


def save_test_results(test_metrics, config, dataset_name, sigma=None, *, trainer=None):
    """Save per-ray test results as structured JSON.

    Args:
        test_metrics: Dict from trainer.predict(). For PFL methods (EnsembleTrainer),
            this is {ray_idx: {metric_name: value}}. For baseline methods (BaseTrainer),
            this is a flat {metric_name: value}.
        config: Hydra DictConfig with experiment configuration.
        dataset_name: One of 'multimnist', 'sarcos', 'utkface'.
    """
    seed = config.seed
    fim_method = normalize_fim_method(
        getattr(config.method, "fim_method", "none") if hasattr(config, "method") else "none"
    )
    exclude_bn = resolve_exclude_bn_flag(getattr(config.method, 'exclude_bn_from_fim', None)) if hasattr(config, 'method') else False
    exclude_head = resolve_fim_filter_flag(getattr(config.method, 'exclude_head_from_fim', None), fim_method) if hasattr(config, 'method') else False
    method_name = "unknown"
    if hasattr(config, "method") and hasattr(config.method, "name"):
        method_name = str(config.method.name)
    fim_eps_damp = float(getattr(config.method, 'fim_eps_damp', 1e-8)) if hasattr(config, 'method') else 1e-8
    fim_eps_num = float(getattr(config.method, 'fim_eps_num', 1e-12)) if hasattr(config, 'method') else 1e-12
    ablation_name = normalize_ablation_name(
        getattr(config.method, "fim_ablation_name", None) if hasattr(config, "method") else None
    )

    results = {
        "dataset": dataset_name,
        "method": method_name,
        "fim_method": str(fim_method),
        "ablation": ablation_name,
        "seed": int(seed),
        "config": {
            "epochs": int(config.training.epochs),
            "lr": float(config.optimizer.lr),
            "rank": int(getattr(config.method, 'rank', -1)) if hasattr(config, 'method') else -1,
            "batch_size": int(config.data.batch_size),
            "fim_ema_decay": float(getattr(config.method, 'fim_ema_decay', 0)) if hasattr(config, 'method') else 0,
            "validate_every_n": int(getattr(config, 'validate_every_n', -1)),
            "exclude_bn_from_fim": bool(exclude_bn),
            "exclude_head_from_fim": bool(exclude_head),
            "fim_eps_num": fim_eps_num,
            "fim_eps_damp": fim_eps_damp,
            "fim_start_fraction": float(getattr(config.method, 'fim_start_fraction', 0.0))
            if hasattr(config, 'method') else 0.0,
            "fim_end_fraction": float(getattr(config.method, 'fim_end_fraction', 1.0))
            if hasattr(config, 'method') and getattr(config.method, 'fim_end_fraction', None) is not None else 1.0,
            "fim_freeze_after_fraction": (
                float(getattr(config.method, 'fim_freeze_after_fraction'))
                if hasattr(config, 'method')
                and getattr(config.method, 'fim_freeze_after_fraction', None) is not None
                else None
            ),
            "fim_reduce_denom": (
                str(getattr(config.method, 'fim_reduce_denom'))
                if hasattr(config, 'method')
                and getattr(config.method, 'fim_reduce_denom', None) is not None
                else None
            ),
            "fim_normalize_denom_mean": bool(getattr(config.method, 'fim_normalize_denom_mean', False))
            if hasattr(config, 'method') else False,
        },
        "per_ray_results": {},
    }

    if trainer is not None:
        diagnostics = getattr(trainer.method, "get_diagnostics", None)
        if callable(diagnostics):
            from omegaconf import OmegaConf

            results["method_config"] = OmegaConf.to_container(config.method, resolve=True)
            results["method_diagnostics"] = diagnostics()

    if sigma is not None:
        results["sigma"] = {str(i): float(v) for i, v in sigma.items()}

    if test_metrics is None:
        logging.warning("test_metrics is None — no test results to save.")
    elif isinstance(test_metrics, dict):
        # Check if this is per-ray (EnsembleTrainer) or flat (BaseTrainer)
        first_value = next(iter(test_metrics.values()), None)
        if isinstance(first_value, dict):
            # Per-ray results: {ray_idx: {metric: value}}
            for ray_idx, metrics in test_metrics.items():
                results["per_ray_results"][str(ray_idx)] = {
                    k: round(float(v), 6) for k, v in metrics.items()
                }
        else:
            # Flat results from BaseTrainer: {metric: value}
            results["per_ray_results"]["0"] = {
                k: round(float(v), 6) for k, v in test_metrics.items()
            }

    # Use Hydra's original cwd so results go to project root, not Hydra output dir
    try:
        from hydra.utils import get_original_cwd

        project_root = get_original_cwd()
    except ValueError:
        # get_original_cwd() fails outside Hydra context (e.g., testing, standalone use)
        logging.warning("Hydra not initialized; using os.getcwd() as project root.")
        project_root = os.getcwd()
    except ImportError:
        logging.warning("hydra is unavailable; using os.getcwd() as project root.")
        project_root = os.getcwd()
    results_version = os.environ.get("MADAM_RESULTS_VERSION", "v5")
    outdir = os.path.join(project_root, "results", results_version)
    os.makedirs(outdir, exist_ok=True)
    opt_type = 'Adam'
    if hasattr(config, 'optimizer'):
        if hasattr(config.optimizer, 'type'):
            opt_type = str(config.optimizer.type)
        elif hasattr(config.optimizer, 'name'):
            opt_type = str(config.optimizer.name)
    ray_sampler_name = None
    if hasattr(config, 'ray_sampler') and hasattr(config.ray_sampler, 'name'):
        ray_sampler_name = str(config.ray_sampler.name)
    result_id = build_result_id(
        dataset_name,
        method_name,
        fim_method,
        ablation_name=ablation_name,
        exclude_bn=exclude_bn,
        fim_eps_damp=fim_eps_damp,
        fim_eps_num=fim_eps_num,
        optimizer_name=opt_type,
        optimizer_lr=float(config.optimizer.lr) if hasattr(config, "optimizer") else None,
        ray_sampler_name=ray_sampler_name,
    )
    outpath = os.path.join(outdir, f"{result_id}_s{seed}.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary to stdout
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {dataset_name} | method={method_name} | fim_method={fim_method} | seed={seed}")
    print("=" * 70)
    if test_metrics is not None and isinstance(test_metrics, dict):
        first_value = next(iter(test_metrics.values()), None)
        if isinstance(first_value, dict):
            for ray_idx in sorted(test_metrics.keys(), key=lambda x: (isinstance(x, str), x)):
                metrics = test_metrics[ray_idx]
                task_metrics = {k: round(float(v), 4) for k, v in metrics.items()
                                if 'Task' in k or 'error' in k or 'acc' in k or 'loss' in k}
                print(f"  Ray {ray_idx}: {task_metrics}")
        else:
            task_metrics = {k: round(float(v), 4) for k, v in test_metrics.items()
                            if 'Task' in k or 'error' in k or 'acc' in k or 'loss' in k}
            print(f"  {task_metrics}")
    print(f"\nSaved to: {outpath}")
    print("=" * 70 + "\n")

    return outpath
