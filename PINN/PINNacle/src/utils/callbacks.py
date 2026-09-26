import logging
import os
import time
import torch
import numpy as np
import scipy
import itertools

from deepxde.geometry import Hypercube, Interval
from deepxde.callbacks import Callback
from deepxde.utils.internal import list_to_str
from src.utils import plot

logger = logging.getLogger(__name__)


def _require_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is required for PINNacle benchmark runs. "
            "Install wandb in the active environment before training."
        ) from exc
    return wandb


def _active_wandb_run():
    try:
        import wandb
    except ImportError:
        return None
    return getattr(wandb, "run", None)


def _coerce_metric_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (bool, int, float)):
        value = float(value)
        if np.isfinite(value):
            return value
    return None


def _sanitize_metrics(payload):
    clean = {}
    for key, value in payload.items():
        scalar = _coerce_metric_value(value)
        if scalar is not None:
            clean[key] = scalar
    return clean


def _wandb_log(payload, step=None):
    run = _active_wandb_run()
    if run is None:
        return

    clean = _sanitize_metrics(payload)
    if not clean:
        return

    if step is None:
        run.log(clean)
    else:
        run.log(clean, step=int(step))


def _wandb_summary_update(payload):
    run = _active_wandb_run()
    if run is None:
        return

    clean = _sanitize_metrics(payload)
    for key, value in clean.items():
        run.summary[key] = value


def _loss_component_names(model):
    config = getattr(getattr(model, "pde", None), "loss_config", None)
    if not config:
        return []
    return [entry.get("name", f"component_{idx}") for idx, entry in enumerate(config)]


class WandbCallback(Callback):

    def __init__(self):
        super(WandbCallback, self).__init__()
        self.run = None
        self.start_time = None
        self.last_history_len = 0

    def _build_run_name(self, benchmark_context, run_context):
        pde_name = type(self.model.pde).__name__
        baseline = benchmark_context.get("baseline_label", "unknown")
        preconditioner = benchmark_context.get("components", {}).get("preconditioner", "none")
        seed = run_context.get("run_seed", benchmark_context.get("command_seed", "na"))
        return f"{pde_name}-{baseline}-{preconditioner}-s{seed}"

    def _build_config(self, benchmark_context, run_context):
        config = dict(benchmark_context.get("wandb_config", {}))
        config.update(
            {
                "pde_name": type(self.model.pde).__name__,
                "exp_name": run_context.get("exp_name"),
                "task_index": run_context.get("task_index"),
                "repeat_index": run_context.get("repeat_index"),
                "run_seed": run_context.get("run_seed"),
                "device": run_context.get("device"),
                "save_path": run_context.get("save_path"),
            }
        )
        return config

    def _log_latest_history(self, force=False):
        history = self.model.losshistory
        if len(history.steps) == 0:
            return
        if not force and len(history.steps) == self.last_history_len:
            return

        index = len(history.steps) - 1
        step = int(history.steps[index])
        loss_names = _loss_component_names(self.model)
        loss_train = np.asarray(history.loss_train[index]).reshape(-1)
        loss_test = np.asarray(history.loss_test[index]).reshape(-1)

        payload = {
            "trainer/step": step,
            "trainer/epoch": int(self.model.train_state.epoch),
            "loss/weighted/train_total": float(np.sum(loss_train)),
            "loss/weighted/test_total": float(np.sum(loss_test)),
        }

        for idx, value in enumerate(loss_train):
            name = loss_names[idx] if idx < len(loss_names) else f"component_{idx}"
            payload[f"loss/weighted/train/{name}"] = value
        for idx, value in enumerate(loss_test):
            name = loss_names[idx] if idx < len(loss_names) else f"component_{idx}"
            payload[f"loss/weighted/test/{name}"] = value

        optimizer = getattr(self.model, "opt", None)
        param_groups = getattr(optimizer, "param_groups", None)
        if param_groups:
            payload["optimizer/lr"] = param_groups[0].get("lr")

        _wandb_log(payload, step=step)
        self.last_history_len = len(history.steps)

    def on_train_begin(self):
        wandb = _require_wandb()
        benchmark_context = dict(getattr(self.model, "_madam_benchmark_context", {}) or {})
        run_context = dict(getattr(self.model, "_madam_run_context", {}) or {})
        run_name = run_context.get("wandb_run_name") or self._build_run_name(benchmark_context, run_context)
        pde_name = type(self.model.pde).__name__
        group = benchmark_context.get("wandb_group") or pde_name
        tags = list(benchmark_context.get("wandb_tags", []))
        tags.extend(
            [
                pde_name,
                benchmark_context.get("baseline_label", "unknown"),
                benchmark_context.get("components", {}).get("preconditioner", "none"),
            ]
        )
        slurm_job_id = benchmark_context.get("slurm_job_id")
        if slurm_job_id is not None:
            tags.append(f"slurm:{slurm_job_id}")
        tags = list(dict.fromkeys([tag for tag in tags if tag]))

        self.run = wandb.init(
            project=benchmark_context.get("wandb_project", "MAdam-PINNacle"),
            entity=benchmark_context.get("wandb_entity"),
            group=group,
            job_type=benchmark_context.get("wandb_job_type", "benchmark"),
            name=run_name,
            id=run_context.get("wandb_run_id", f"auto-{run_name}"),
            tags=tags,
            config=self._build_config(benchmark_context, run_context),
        )
        if self.run is None:
            raise RuntimeError("wandb.init() returned no run for the current PINNacle job.")

        wandb.define_metric("trainer/step")
        wandb.define_metric("*", step_metric="trainer/step")

        self.start_time = time.time()
        self.last_history_len = 0
        self._log_latest_history(force=True)

    def on_epoch_end(self):
        self._log_latest_history()

    def on_train_end(self):
        self._log_latest_history(force=True)
        if self.run is None:
            return

        payload = {
            "trainer/runtime_seconds": time.time() - self.start_time if self.start_time is not None else None,
            "trainer/best_step": self.model.train_state.best_step,
            "loss/weighted/best_train_total": self.model.train_state.best_loss_train,
            "loss/weighted/best_test_total": self.model.train_state.best_loss_test,
        }
        _wandb_summary_update(payload)


class PlotCallback(Callback):

    def __init__(self, log_every=None, verbose=False, fast=False):
        super(PlotCallback, self).__init__()

        self.log_every = log_every
        self.verbose = verbose
        self.fast = fast
        self.valid_epoch = 0

    def plot(self, save_path):
        train_state = self.model.train_state
        plot.plot_state(self.model.pde, train_state, save_path, is_best=False, fast=self.fast)

    def on_train_begin(self):
        self.base_save_path = self.model.model_save_path + "/"
        if not os.path.exists(self.base_save_path):
            os.mkdir(self.base_save_path)

    def on_epoch_end(self):
        self.valid_epoch += 1
        if self.log_every is None or self.valid_epoch % self.log_every != 0:
            return
        if self.verbose:
            print("Plotting at epoch {} ...".format(self.valid_epoch))

        save_path = self.base_save_path + str(self.valid_epoch) + '/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.plot(save_path)

    def on_train_end(self):
        if self.verbose:
            print("Plotting at train end ...")
        self.plot(self.base_save_path)


class LossCallback(Callback):

    def __init__(self, verbose=False):
        super(LossCallback, self).__init__()
        self.log_every = None
        self.verbose = verbose
        self.valid_epoch = 0
        self.loss_weights = []

    def on_train_begin(self):
        self.log_every = self.model.display_every
        if self.model.losshistory.loss_weights is not None:
            self.loss_weights.append(self.model.losshistory.loss_weights)
        else:
            self.loss_weights.append(np.ones(self.model.pde.num_loss))
        self._log_latest_losses()

    def on_epoch_end(self):
        self.valid_epoch += 1
        if self.valid_epoch % self.log_every != 0:
            return

        if self.model.losshistory.loss_weights is not None:
            self.loss_weights.append(self.model.losshistory.loss_weights.copy())
        else:
            self.loss_weights.append(np.ones(self.model.pde.num_loss))

        if self.verbose:
            loss_weight = self.loss_weights[-1]
            loss_train = self.model.losshistory.loss_train[-1] / loss_weight
            loss_test = self.model.losshistory.loss_test[-1] / loss_weight
            print('Unweighted Loss: {}  {} Weights: {}'.format(
                list_to_str(loss_train),
                list_to_str(loss_test),
                list_to_str(loss_weight),
            ))
        self._log_latest_losses()

    def on_train_end(self):
        save_path = self.model.model_save_path + "/"
        loss_history = self.model.losshistory
        loss_weights = np.array(self.loss_weights)
        loss = np.hstack((
            np.array(loss_history.steps)[:, None],
            np.array(loss_history.loss_train) / loss_weights,
            np.array(loss_history.loss_test) / loss_weights,
            loss_weights,
        ))
        np.savetxt(save_path + "loss.txt", loss, header="step, loss_train, loss_test, loss_weight")
        plot.plot_loss_history(self.model.pde, loss_history, save_path)
        plot.plot_loss_history(self.model.pde, loss_history, save_path, loss_weights=loss_weights)

    def _log_latest_losses(self):
        if not self.loss_weights:
            return
        if self.model.train_state.loss_train is None or self.model.train_state.loss_test is None:
            return

        loss_names = _loss_component_names(self.model)
        loss_weight = np.asarray(self.loss_weights[-1], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            loss_train = np.asarray(self.model.train_state.loss_train, dtype=float) / loss_weight
            loss_test = np.asarray(self.model.train_state.loss_test, dtype=float) / loss_weight

        step = int(self.model.train_state.step)
        payload = {
            "trainer/step": step,
            "loss/unweighted/train_total": np.sum(loss_train),
            "loss/unweighted/test_total": np.sum(loss_test),
        }

        for idx, value in enumerate(loss_weight):
            name = loss_names[idx] if idx < len(loss_names) else f"component_{idx}"
            payload[f"loss/weight/{name}"] = value
        for idx, value in enumerate(loss_train):
            name = loss_names[idx] if idx < len(loss_names) else f"component_{idx}"
            payload[f"loss/unweighted/train/{name}"] = value
        for idx, value in enumerate(loss_test):
            name = loss_names[idx] if idx < len(loss_names) else f"component_{idx}"
            payload[f"loss/unweighted/test/{name}"] = value

        _wandb_log(payload, step=step)


class TesterCallback(Callback):

    def __init__(self, log_every=100, verbose=True, fRMSE_param={'enable':True, 'iLow':5, 'iHigh':13, 'calc_every':2000}):
        super(TesterCallback, self).__init__()

        self.log_every = log_every
        self.verbose = verbose
        self.fRMSE = fRMSE_param.get('enable', True)
        if self.fRMSE:
            self.fRMSE_l = fRMSE_param.get('iLow', 5)
            self.fRMSE_h = fRMSE_param.get('iHigh', 13)
            self.fRMSE_every = fRMSE_param.get('calc_every', 2000)

        self.indexes = []
        self.maes = []    # Mean Average Error
        self.mses = []    # Mean Square Error
        self.mxes = []    # Maximum Error
        self.l1res = []   # L1 Relative Error
        self.l2res = []   # L2 Relative Error
        self.crmses = []  # CSV_Loss
        self.frmses = []  # Mean Square Error in Fourier Space

        self.epochs_since_last_resample = 0
        self.valid_epoch = 0
        self.disable = False

    def on_train_begin(self):
        self.save_path = self.model.model_save_path + "/"
        pde = self.model.pde

        # Load / Generate Test Data
        if pde.ref_sol is not None: # sample points from geometry
            sample_points = 2500 if pde.input_dim == 2 else 20000
            if getattr(self.model.data.geom, "uniform_points", None) is None:
                logger.warning(f"Method \'Uniform Points\' not found for class {type(self.model.data.geom)}, \
                                 Use random points for testing ...")
                sample_func = self.model.data.geom.random_points
            else:
                sample_func = self.model.data.geom.uniform_points
            
            self.test_x = sample_func(sample_points, boundary=False)
            self.test_y = pde.ref_sol(self.test_x)
        elif pde.ref_data is not None:
            nan_mask = np.isnan(pde.ref_data).any(axis=1)
            self.test_x = pde.ref_data[~nan_mask, :pde.input_dim]
            self.test_y = pde.ref_data[~nan_mask, pde.input_dim:]
        else:
            self.disable = True
            logger.info("No reference solution or data provided, skipping TesterCallback")
            return

        self.solution_l1 = np.abs(self.test_y).mean()
        self.solution_l2 = np.sqrt((self.test_y**2).mean())

        if self.fRMSE:
            self.frmse_init()

    def on_epoch_end(self):
        self.epochs_since_last_resample += 1
        self.valid_epoch += 1
        if self.disable or self.log_every is None or self.epochs_since_last_resample < self.log_every:
            return
        self.epochs_since_last_resample = 0

        with torch.no_grad():
            y = self.model.predict(self.test_x)

        mse = ((y - self.test_y)**2).mean()
        mae = np.abs(y - self.test_y).mean()
        mxe = np.max(np.abs(y - self.test_y))
        l1re = mae / self.solution_l1
        l2re = np.sqrt(mse) / self.solution_l2
        crmse = np.abs((y - self.test_y).mean())
        if self.fRMSE and self.valid_epoch % self.fRMSE_every == 0:
            frmse = self.frmse_calc(y)
        else:
            frmse = (np.nan, np.nan, np.nan)

        self.indexes.append(self.valid_epoch)
        self.mses.append(mse)
        self.maes.append(mae)
        self.mxes.append(mxe)
        self.l1res.append(l1re)
        self.l2res.append(l2re)
        self.crmses.append(crmse)
        self.frmses.append(frmse)

        if self.verbose:
            if np.isnan(frmse[0]):
                print('Validation: epoch {} MSE {:.5f} MAE {:.5f} MXE {:.5f} L1RE {:.5f} L2RE {:.5f} CRMSE {:.5f}'.\
                       format(self.valid_epoch, mse, mae, mxe, l1re, l2re, crmse))
            else:
                print('Validation: epoch {} MSE {:.5f} MAE {:.5f} MXE {:.5f} L1RE {:.5f} L2RE {:.5f} CRMSE {:.5f} FRMSE ({:.5f}, {:.5f}, {:.5f})'.\
                       format(self.valid_epoch, mse, mae, mxe, l1re, l2re, crmse, frmse[0], frmse[1], frmse[2]))
        self._log_latest_metrics(mse, mae, mxe, l1re, l2re, crmse, frmse)

    def on_train_end(self):
        if self.disable:
            return

        self._log_summary()

        self.indexes = np.array(self.indexes)
        self.frmses = np.array(self.frmses)
        np.savetxt(
            self.save_path + 'errors.txt',
            np.array([self.indexes, self.maes, self.mses, self.mxes, self.l1res, self.l2res, self.crmses,\
                      self.frmses[:, 0], self.frmses[:, 1], self.frmses[:, 2]]).T,
            header="epochs, maes, mses, mxes, l1res, l2res, crmses, frmses(low, mid, high)"
        )

        plot.plot_lines([self.indexes, self.maes], xlabel="epochs", labels=['maes'], path=self.save_path + "maes.png", title="mean average error")
        plot.plot_lines([self.indexes, self.mses], xlabel="epochs", labels=['mses'], path=self.save_path + "mses.png", title="mean square error")
        plot.plot_lines([self.indexes, self.mxes], xlabel="epochs", labels=['mxes'], path=self.save_path + "mxes.png", title="maximum error")
        plot.plot_lines([self.indexes, self.l1res, self.l2res],
                        xlabel="epochs",
                        labels=['l1re', 'l2re'],
                        path=self.save_path + "relerr.png",
                        title="relative error")
        X = ~np.isnan(self.frmses).any(axis=1)
        plot.plot_lines([self.indexes[X], self.frmses[X, 0], self.frmses[X, 1], self.frmses[X, 2]], 
                        xlabel="epochs", 
                        labels=['low freq', 'mid freq', 'high freq'], 
                        path=self.save_path + "frmses.png", 
                        title="mean square error in fourier space")

        self.indexes = []
        self.maes = []   
        self.mses = []   
        self.mxes = []   
        self.l1res = []  
        self.l2res = []  
        self.crmses = [] 
        self.frmses = [] 

        self.epochs_since_last_resample = 0
        self.valid_epoch = 0

    def _log_latest_metrics(self, mse, mae, mxe, l1re, l2re, crmse, frmse):
        step = int(self.model.train_state.step)
        payload = {
            "trainer/step": step,
            "validation/mse": mse,
            "validation/mae": mae,
            "validation/mxe": mxe,
            "validation/l1re": l1re,
            "validation/l2re": l2re,
            "validation/crmse": crmse,
            "validation/frmse_low": frmse[0],
            "validation/frmse_mid": frmse[1],
            "validation/frmse_high": frmse[2],
        }
        _wandb_log(payload, step=step)

    def _log_summary(self):
        if not self.indexes:
            return

        frmses = np.asarray(self.frmses)
        summary = {
            "validation/final_mse": self.mses[-1],
            "validation/final_mae": self.maes[-1],
            "validation/final_mxe": self.mxes[-1],
            "validation/final_l1re": self.l1res[-1],
            "validation/final_l2re": self.l2res[-1],
            "validation/final_crmse": self.crmses[-1],
            "validation/best_mse": np.nanmin(self.mses),
            "validation/best_mae": np.nanmin(self.maes),
            "validation/best_mxe": np.nanmin(self.mxes),
            "validation/best_l1re": np.nanmin(self.l1res),
            "validation/best_l2re": np.nanmin(self.l2res),
            "validation/best_crmse": np.nanmin(self.crmses),
            "validation/final_frmse_low": frmses[-1, 0],
            "validation/final_frmse_mid": frmses[-1, 1],
            "validation/final_frmse_high": frmses[-1, 2],
            "validation/best_frmse_low": np.nanmin(frmses[:, 0]),
            "validation/best_frmse_mid": np.nanmin(frmses[:, 1]),
            "validation/best_frmse_high": np.nanmin(frmses[:, 2]),
        }
        _wandb_summary_update(summary)
    
    def frmse_init(self):
        pde = self.model.pde
        if not isinstance(pde.geom, Hypercube) and not isinstance(pde.geom, Interval):
            logger.warning(f"Fourier transform errors are enabled only in Interval / Hypercube and their combination with Time domains. \
                           Type {type(pde.geom).__name__} is not a valid geometry and fRMSE has been disabled")
            self.fRMSE=False
            return
        if pde.input_dim > 3:
            logger.warning(f"For high dimensional PDEs like {type(pde).__name__} with dim {pde.input_dim} is slow to calculate fRMSE. \
                           fRMSE has been disabled")
            self.fRMSE=False
            return 

        # prepare calculation
        self.test_x_delaunay = scipy.spatial.Delaunay(self.test_x)
        ptn = 3e4 # generate about 3e4 uniform sampling points in the domain
        for i in range(pde.input_dim):
            ptn /= pde.bbox[i * 2 + 1] - pde.bbox[i * 2]
        ptn = ptn ** (1 / pde.input_dim)
        xlist = [np.linspace(pde.bbox[i * 2], pde.bbox[i * 2 + 1], int(np.ceil((pde.bbox[i*2+1] - pde.bbox[i*2]) * ptn)) + 1, endpoint=False)[1:] \
                 for i in range(pde.input_dim)]
        self.sample_x = np.stack(np.meshgrid(*xlist), axis=-1)
    
    def frmse_calc(self, y):
        pde = self.model.pde
        res = scipy.interpolate.LinearNDInterpolator(self.test_x_delaunay, y - self.test_y)(self.sample_x.reshape((-1, pde.input_dim)))
        resn = scipy.interpolate.NearestNDInterpolator(self.test_x, y - self.test_y)(self.sample_x.reshape((-1, pde.input_dim)))
        res[np.isnan(res)] = resn[np.isnan(res)]
        err = np.fft.rfftn(res, axes=tuple(range(res.ndim-1))) # transform except the last dim (pde.output_dim)
        err = np.mean(np.abs(err) ** 2 / res.size, axis=-1) # take average through the last dim

        if pde.input_dim == 1:
            err_low = err[:self.fRMSE_l].mean()
            err_mid = err[self.fRMSE_l:self.fRMSE_h].mean()
            err_high = err[self.fRMSE_h:].mean()
        else:
            err_low, err_mid, err_high = 0.0, 0.0, 0.0
            err_low_cnt, err_mid_cnt, err_high_cnt = 0, 0, 0
            for ids in itertools.product(*[range((k+1)//2) for k in err.shape[:-1]]):
                freq2 = sum(i ** 2 for i in ids)
                ilow = min(int(np.sqrt(max(0, self.fRMSE_l**2 - freq2))), err.shape[-1])
                ihigh = min(int(np.sqrt(max(0, self.fRMSE_h**2 - freq2))), err.shape[-1])

                err_low += err[(*ids, slice(None, ilow, None))].sum()
                err_mid += err[(*ids, slice(ilow, ihigh, None))].sum()
                err_high += err[(*ids, slice(ihigh, None, None))].sum()

                err_low_cnt += ilow 
                err_mid_cnt += ihigh - ilow
                err_high_cnt += err.shape[-1] - ihigh
            
            err_low /= err_low_cnt # calculate mean square error
            err_mid /= err_mid_cnt
            err_high /= err_high_cnt

        return err_low, err_mid, err_high
