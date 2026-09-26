import json
from pathlib import Path

import numpy as np
import pandas as pd


JSON_COLUMNS = [
    'pde', 'iter', 'success_rate', 'run_time', 'run_time_std',
    'train_loss', 'train_loss_std',
    'mae', 'mae_std', 'mse', 'mse_std', 'mxe', 'mxe_std',
    'l1rel', 'l1rel_std', 'l2rel', 'l2rel_std', 'crmse', 'crmse_std',
    'frmse_low', 'frmse_low_std', 'frmse_mid', 'frmse_mid_std',
    'frmse_high', 'frmse_high_std',
]

LEGACY_COLUMNS = [
    'pde', 'iter', 'success_rate', 'run_time', 'run_time_std',
    'train_loss', 'train_loss_std',
    'mse', 'mse_std', 'mxe', 'mxe_std', 'l2rel', 'l2rel_std',
    'crmse', 'crmse_std', 'frmse_low', 'frmse_low_std',
    'frmse_mid', 'frmse_mid_std', 'frmse_high', 'frmse_high_std',
]


def _process(func, path, repeat):
    data = []
    try:
        for i in range(repeat):
            data.append(func(np.loadtxt(path.format(i))))
        data = np.array(data)[~np.isnan(data)]  # exclude nan
        return np.mean(data), np.std(data)
    except ValueError as e:  # should use method below
        if len(data) != 0:
            print(e)
            return np.nan, np.nan

    try:
        for i in range(repeat):
            data.append(func(open(path.format(i)).readlines()))
        data = np.array(data)[~np.isnan(data)]
        return np.mean(data), np.std(data)
    except Exception as e:
        print(e)
        return np.nan, np.nan

def extract_time(lines):
    # example: 'train' took 253.845810 s
    for line in lines:
        line = line.strip()
        if line.startswith("'train'"):
            return float(line.split(' ')[2])
    print("\033[33mWarning:Could not find training time.\033[0m")  # yellow
    return np.nan

def extract_name(path):
    # example: PDE Class Name: PoissonND
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("PDE Class Name:"):
                return line.split(' ')[3]
    print("\033[33mWarning:Could not find PDE Class Name.\033[0m")  # yellow
    return ""

def extract_success(lines):
    # example: Epoch 20000: saving model to runs/08.10-05.59.14-LBFGS_MainExp/0-0/20000.pt ...
    flags = [False, False]
    for line in lines:
        line = line.strip()
        if line.startswith("Epoch 20000:"):
            flags[0]=True
        elif line.startswith("'train'"):
            flags[1]=True
    return 1 if flags[0] and flags[1] else 0


def _load_result_json(path, task_idx, repeat_idx):
    json_path = Path(path) / f"{task_idx}-{repeat_idx}" / "result.json"
    if not json_path.exists():
        return None
    with json_path.open() as handle:
        return json.load(handle)


def _finite_values(values):
    clean = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            clean.append(value)
    return clean


def _mean_std(values):
    clean = _finite_values(values)
    if not clean:
        return np.nan, np.nan
    data = np.asarray(clean, dtype=float)
    return float(np.mean(data)), float(np.std(data))


def _metric_mean_std(results, key):
    return _mean_std([r.get("metrics", {}).get("final", {}).get(key) for r in results])


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _summarize_from_json(path, tasknum, repeat, iters):
    columns = JSON_COLUMNS
    rows = []
    summaries = [None] * tasknum
    found_any = False

    for i in range(tasknum):
        run_results = []
        for j in range(repeat):
            result = _load_result_json(path, i, j)
            if result is not None:
                found_any = True
                run_results.append(result)

        if not run_results:
            rows.append(None)
            continue

        first = run_results[0]
        success_rate, _ = _mean_std([1.0 if r.get("status") == "success" else 0.0 for r in run_results])
        runtime_mean, runtime_std = _mean_std([r.get("metrics", {}).get("runtime_seconds") for r in run_results])
        train_loss_mean, train_loss_std = _mean_std(
            [r.get("metrics", {}).get("loss", {}).get("train_total") for r in run_results]
        )
        mae_mean, mae_std = _metric_mean_std(run_results, "mae")
        mse_mean, mse_std = _metric_mean_std(run_results, "mse")
        mxe_mean, mxe_std = _metric_mean_std(run_results, "mxe")
        l1rel_mean, l1rel_std = _metric_mean_std(run_results, "l1re")
        l2rel_mean, l2rel_std = _metric_mean_std(run_results, "l2re")
        crmse_mean, crmse_std = _metric_mean_std(run_results, "crmse")
        flow_mean, flow_std = _metric_mean_std(run_results, "frmse_low")
        fmid_mean, fmid_std = _metric_mean_std(run_results, "frmse_mid")
        fhigh_mean, fhigh_std = _metric_mean_std(run_results, "frmse_high")

        row = [
            first.get("pde", ""), iters[i], success_rate, runtime_mean, runtime_std,
            train_loss_mean, train_loss_std,
            mae_mean, mae_std, mse_mean, mse_std, mxe_mean, mxe_std,
            l1rel_mean, l1rel_std, l2rel_mean, l2rel_std, crmse_mean, crmse_std,
            flow_mean, flow_std, fmid_mean, fmid_std, fhigh_mean, fhigh_std,
        ]
        rows.append(row)
        summaries[i] = {
            "pde": first.get("pde", ""),
            "iter": iters[i],
            "runs": run_results,
            "aggregate": dict(zip(columns, row)),
        }

    if not found_any:
        return None

    return columns, rows, summaries


def summary(path, tasknum, repeat, iters):
    json_summary = _summarize_from_json(path, tasknum, repeat, iters)
    if json_summary is not None:
        columns, result, summaries = json_summary
        # Fill any missing task rows with the legacy scraper so partial reruns still summarize.
        for i, row in enumerate(result):
            if row is not None:
                continue
            result[i] = _legacy_summary_row(path, i, repeat, iters[i], include_extra_metrics=True)
            summaries[i] = {
                "pde": result[i][0],
                "iter": iters[i],
                "runs": [],
                "aggregate": dict(zip(columns, result[i])),
            }
        df = pd.DataFrame(result, columns=columns)
        df.to_csv(f'{path}/result.csv')
        with open(f'{path}/summary.json', 'w') as handle:
            json.dump(_json_safe({"tasks": summaries}), handle, indent=2, allow_nan=False)
            handle.write("\n")
        return

    columns = LEGACY_COLUMNS
    result = []
    for i in range(tasknum):
        result.append(_legacy_summary_row(path, i, repeat, iters[i]))

    # save csv
    df = pd.DataFrame(result, columns=columns)
    df.to_csv(f'{path}/result.csv')


def _legacy_summary_row(path, i, repeat, iters, include_extra_metrics=False):
    name = extract_name('{}/{}-0/log.txt'.format(path, i))
    try:
        success_mean, success_std = _process(extract_success, '{}/{}-{{}}/log.txt'.format(path, i), repeat)
        run_time_mean, run_time_std = _process(extract_time, '{}/{}-{{}}/log.txt'.format(path, i), repeat)
        train_loss_mean, train_loss_std = _process(lambda data: data[-1, 1], '{}/{}-{{}}/loss.txt'.format(path, i), repeat)
    except (FileNotFoundError, IOError):
        success_mean = np.nan
        run_time_mean = run_time_std = np.nan
        train_loss_mean = train_loss_std = np.nan
    try:
        mae_mean, mae_std = _process(lambda data: data[-1, 1], '{}/{}-{{}}/errors.txt'.format(path, i), repeat)
        mse_mean, mse_std = _process(lambda data: data[-1, 2], '{}/{}-{{}}/errors.txt'.format(path, i), repeat)
        mxe_mean, mxe_std = _process(lambda data: data[-1, 3], '{}/{}-{{}}/errors.txt'.format(path, i), repeat)
        l1rel_mean, l1rel_std = _process(lambda data: data[-1, 4], '{}/{}-{{}}/errors.txt'.format(path, i), repeat)
        l2rel_mean, l2rel_std = _process(lambda data: data[-1, 5], '{}/{}-{{}}/errors.txt'.format(path, i), repeat)
        crmse_mean, crmse_std = _process(lambda data: data[-1, 6], '{}/{}-{{}}/errors.txt'.format(path, i), repeat)
        flow_mean, flow_std = _process(lambda data: data[-1, 7], '{}/{}-{{}}/errors.txt'.format(path, i), repeat)
        fmid_mean, fmid_std = _process(lambda data: data[-1, 8], '{}/{}-{{}}/errors.txt'.format(path, i), repeat)
        fhigh_mean, fhigh_std = _process(lambda data: data[-1, 9], '{}/{}-{{}}/errors.txt'.format(path, i), repeat)
    except Exception:
        mae_mean = mae_std = np.nan
        mse_mean = mse_std = mxe_mean = mxe_std = np.nan
        l1rel_mean = l1rel_std = np.nan
        l2rel_mean = l2rel_std = crmse_mean = crmse_std = np.nan
        flow_mean = flow_std = fmid_mean = fmid_std = np.nan
        fhigh_mean = fhigh_std = np.nan
    if include_extra_metrics:
        return [name, iters, success_mean, run_time_mean, run_time_std, train_loss_mean, train_loss_std, \
                mae_mean, mae_std, mse_mean, mse_std, mxe_mean, mxe_std, \
                l1rel_mean, l1rel_std, l2rel_mean, l2rel_std, crmse_mean, crmse_std, \
                flow_mean, flow_std, fmid_mean, fmid_std, fhigh_mean, fhigh_std]
    return [name, iters, success_mean, run_time_mean, run_time_std, train_loss_mean, train_loss_std, mse_mean, mse_std, \
            mxe_mean, mxe_std, l2rel_mean, l2rel_std, crmse_mean, crmse_std, \
            flow_mean, flow_std, fmid_mean, fmid_std, fhigh_mean, fhigh_std]
