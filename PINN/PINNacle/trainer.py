import dill
import json
import multiprocessing
import os
import shutil
import sys
import time

dill.settings['recurse'] = True


def _finalize_wandb_success(model):
    try:
        import wandb
    except ImportError:
        return

    run = getattr(wandb, "run", None)
    if run is None:
        return

    run.summary["trainer/status"] = "stopped" if getattr(model, "stop_training", False) else "success"
    wandb.finish()


def _finalize_wandb_failure(exc):
    try:
        import wandb
    except ImportError:
        return

    run = getattr(wandb, "run", None)
    if run is None:
        return

    run.summary["trainer/status"] = "failed"
    run.summary["trainer/error"] = str(exc)
    wandb.finish(exit_code=1)


def _attach_run_context(model, exp_name, task_index, repeat_index, seed, save_path, device):
    benchmark_context = dict(getattr(model, "_madam_benchmark_context", {}) or {})
    pde_name = type(getattr(model, "pde", object())).__name__
    baseline = benchmark_context.get("baseline_label", "unknown")
    preconditioner = benchmark_context.get("components", {}).get("preconditioner", "none")
    run_name = f"{pde_name}-{baseline}-{preconditioner}-s{seed}"
    model._madam_run_context = {
        "exp_name": exp_name,
        "task_index": task_index,
        "repeat_index": repeat_index,
        "run_seed": seed,
        "save_path": save_path,
        "device": device,
        "wandb_run_name": run_name,
        "wandb_run_id": f"auto-{run_name}",
    }


def _run_task(data, train_args, save_path, seed, device, exp_name, task_index, repeat_index):
    import deepxde as dde
    from src.utils.results_io import save_run_result

    get_model, _ = dill.loads(data)
    model = get_model()
    _attach_run_context(model, exp_name, task_index, repeat_index, seed, save_path, device)

    try:
        model.train(**train_args, model_save_path=save_path)
        save_run_result(model, save_path, status="success")
        _finalize_wandb_success(model)
    except Exception as exc:
        save_run_result(model, save_path, status="failed", error=exc)
        _finalize_wandb_failure(exc)
        raise


class HookedStdout:
    original_stdout = sys.stdout

    def __init__(self, filename, stdout=None) -> None:
        if stdout is None:
            self.stdout = self.original_stdout
        else:
            self.stdout = stdout
        self.file = open(filename, 'w')

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def train_process(data, save_path, device, seed):
    hooked = HookedStdout(f"{save_path}/log.txt")
    sys.stdout = hooked
    sys.stderr = HookedStdout(f"{save_path}/logerr.txt", sys.stderr)

    import torch
    import deepxde as dde
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    dde.config.set_default_float('float32')
    dde.config.set_random_seed(seed)

    _, train_args = dill.loads(data)
    _run_task(
        data,
        train_args,
        save_path,
        seed,
        device,
        os.path.basename(os.path.dirname(save_path)),
        int(os.path.basename(save_path).split("-")[0]),
        int(os.path.basename(save_path).split("-")[1]),
    )


class Trainer:

    def __init__(self, exp_name, device) -> None:
        self.exp_name = exp_name
        self.device = device.split(",")
        self.repeat = 1
        self.seed = None
        self.tasks = []

    def set_repeat(self, repeat):
        if int(repeat) != 1:
            raise ValueError(
                "PINNacle W&B-aligned runs require repeat=1. "
                "Launch separate runs with explicit seeds instead."
            )
        self.repeat = repeat

    def add_task(self, get_model, train_args):
        data = dill.dumps((get_model, train_args))
        self.tasks.append((data, train_args))

    def setup(self, filename, seed):
        if seed is None:
            raise ValueError(
                "PINNacle W&B-aligned runs require an explicit seed. "
                "Pass --seed and launch separate runs for each seed."
            )
        self.seed = int(seed)
        os.makedirs(f"runs/{self.exp_name}", exist_ok=True)
        shutil.copy(filename, f"runs/{self.exp_name}/script.py.bak")
        json.dump({"seed": seed, "task": self.tasks}, open(f"runs/{self.exp_name}/config.json", 'w'), indent=4, default=lambda _: "...")

    def train_all(self):
        if len(self.device) > 1:
            return self.train_all_parallel()

        # no multi-processing when only one device is available
        import torch
        import deepxde as dde

        if self.device[0] != 'cpu':
            device = "cuda:" + self.device[0]
            torch.cuda.set_device(device)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
        dde.config.set_default_float('float32')

        for j in range(self.repeat):
            for i, (data, _) in enumerate(self.tasks):
                seed = self.seed
                save_path = f"runs/{self.exp_name}/{i}-{j}"
                os.makedirs(save_path, exist_ok=True)

                hooked = HookedStdout(f"{save_path}/log.txt")
                sys.stdout = hooked
                sys.stderr = HookedStdout(f"{save_path}/logerr.txt", sys.stderr)
                dde.config.set_random_seed(seed)

                print(f"***** Begin #{i}-{j} *****")
                _, train_args = dill.loads(data)
                _run_task(data, train_args, save_path, seed, device, self.exp_name, i, j)
                print(f"*****  End #{i}-{j}  *****")

    def train_all_parallel(self):
        # maintain a pool of processes
        # do not start all processes at the same time
        # keep the number of processes equal to the number of devices
        # if a process is done, start a new one on the same device

        multiprocessing.set_start_method('spawn')
        processes = [None] * len(self.device)
        for j in range(self.repeat):
            for i, (data, _) in enumerate(self.tasks):
                # find a free device
                for k, p in enumerate(processes):
                    if p is None:
                        device = "cuda:" + self.device[k]
                        seed = self.seed
                        save_path = f"runs/{self.exp_name}/{i}-{j}"
                        os.makedirs(save_path)

                        print(f"***** Start #{i}-{j} *****")
                        p = multiprocessing.Process(target=train_process, args=(data, save_path, device, seed), daemon=True)
                        p.start()
                        processes[k] = p
                        break
                else:
                    raise RuntimeError("No free device")

                # wait for a process to finish
                while True:
                    for k, p in enumerate(processes):
                        if p is None:
                            break
                        if not p.is_alive():
                            p.join()
                            if p.exitcode != 0:
                                raise RuntimeError(
                                    f"Parallel PINNacle worker on {self.device[k]} failed with exit code {p.exitcode}."
                                )
                            # free device
                            processes[k] = None
                            break
                    else:
                        time.sleep(5)
                        continue
                    break

        for p in processes:
            if p is not None:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError(
                        f"Parallel PINNacle worker failed with exit code {p.exitcode}."
                    )

    def summary(self):
        from src.utils import summary
        summary.summary(f"runs/{self.exp_name}", len(self.tasks), self.repeat, list(map(lambda t:t[1]['iterations'], self.tasks)))
