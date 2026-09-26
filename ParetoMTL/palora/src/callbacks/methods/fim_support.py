import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn

from src.utils.results_io import resolve_exclude_bn_flag, resolve_fim_filter_flag

if TYPE_CHECKING:
    from src.trainer.base_trainer import BaseTrainer


REMOVED_FIM_KWARGS = {
    "fim_adaptive_eps",
    "fim_q_annealing",
    "fim_fix2_only",
    "fim_curvature_lag_steps",
    "fim_shuffle_curvature",
    "fim_shuffle_seed",
    "fim_kappa_correction",
    "fim_diagnostic",
    "fim_diagnostic_dataset",
    "fim_eps",
    "fim_regression_task_ids",
    "fim_warmup_steps",
}


FULL_MADAM_METHODS = {"madam"}
DIAGONAL_MADAM_METHODS = {"diag_madam"}
SUPPORTED_FIM_METHODS = {"none", *FULL_MADAM_METHODS, *DIAGONAL_MADAM_METHODS}


BATCHNORM_LIKE_CLASS_NAMES = {
    "ConditionalBatchNorm2d",
    "SubspaceBatchNorm2d",
}

FIM_TASK_HEAD_NAME_MARKERS = (
    "decoders.",
    "pred_task",
    "segmentation_decoder",
    "depth_decoder",
)


def _ensure_madam_core() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "madam_core" / "src"
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return
    raise ImportError("Could not locate madam_core/src")


_ensure_madam_core()

from madam_core.madam import MAdamPreconditioner
from madam_core.diag_madam import DiagonalMAdamPreconditioner


def raise_on_removed_fim_kwargs(kwargs, owner: str) -> None:
    bad = sorted(key for key in kwargs if key in REMOVED_FIM_KWARGS)
    if bad:
        removed = ", ".join(f"'{key}'" for key in bad)
        raise ValueError(
            f"{owner} no longer supports {removed}. "
            "Remove these legacy FIM overrides."
        )
    if kwargs:
        unknown = ", ".join(f"'{key}'" for key in sorted(kwargs))
        raise ValueError(f"{owner} received unsupported FIM override(s): {unknown}.")


class FIMSupportMixin:
    """Shared MAdam support for methods that expose a current task ray."""

    @staticmethod
    def _pop_bool_kwarg(kwargs: dict, key: str, default: bool) -> bool:
        if key not in kwargs:
            return default
        value = kwargs.pop(key)
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "y", "on")
        return bool(value)

    @staticmethod
    def _pop_float_kwarg(kwargs: dict, key: str, default: float) -> float:
        if key not in kwargs:
            return float(default)
        value = kwargs.pop(key)
        return float(value)

    def init_fim_support(
        self,
        *,
        fim_method: str = "none",
        fim_ema_decay: float = 0.999,
        fim_eps_num: float = 1e-12,
        fim_eps_damp: float = 1e-6,
        fim_clip_grad_norm: float = 0.0,
        fim_bias_correction: bool = False,
        fim_stochastic_pairs: bool = True,
        fim_gradient_balance: bool = False,
        fim_gradient_balance_eps: float = 1e-3,
        exclude_bn_from_fim: Optional[bool] = None,
        exclude_head_from_fim: Optional[bool] = None,
        fim_head_pattern: str = "decoders",
        log_prefix: str = "FIM",
        extra_kwargs=None,
    ) -> None:
        extra_kwargs = dict(extra_kwargs or {})
        if "fim_head_pattern" in extra_kwargs:
            fim_head_pattern = str(extra_kwargs.pop("fim_head_pattern"))
        fim_gradient_balance = self._pop_bool_kwarg(
            extra_kwargs, "fim_gradient_balance", fim_gradient_balance
        )
        fim_gradient_balance_eps = self._pop_float_kwarg(
            extra_kwargs, "fim_gradient_balance_eps", fim_gradient_balance_eps
        )
        raise_on_removed_fim_kwargs(extra_kwargs, log_prefix)

        if fim_method is None:
            fim_method = "none"

        self.fim_method = fim_method
        self.fim_ema_decay = fim_ema_decay
        self.fim_eps_num = fim_eps_num
        self.fim_eps_damp = fim_eps_damp
        self.fim_clip_grad_norm = fim_clip_grad_norm
        self.fim_bias_correction = fim_bias_correction
        self.fim_stochastic_pairs = fim_stochastic_pairs
        self.fim_gradient_balance = fim_gradient_balance
        self.fim_gradient_balance_eps = fim_gradient_balance_eps
        self.exclude_bn_from_fim = resolve_exclude_bn_flag(exclude_bn_from_fim)
        self.exclude_head_from_fim = resolve_fim_filter_flag(exclude_head_from_fim, fim_method)
        self.fim_head_pattern = fim_head_pattern
        self.fim_log_prefix = log_prefix

        self._madam: Optional[object] = None
        self._diag_madam: Optional[object] = None
        self._trainer: Optional["BaseTrainer"] = None
        self._fim_active_this_step = False
        self._step_curvature_tensors: Optional[List[torch.Tensor]] = None

        if fim_method != "none":
            if fim_method not in SUPPORTED_FIM_METHODS:
                raise ValueError(
                    "Unrecognized fim_method="
                    f"'{fim_method}'. Supported: 'none', 'madam', 'diag_madam'."
                )
            logging.info(
                "[%s] FIM preconditioning enabled: method=%s, ema_decay=%s, "
                "eps_num=%s, eps_damp=%s, bias_correction=%s, "
                "stochastic_pairs=%s, gradient_balance=%s, "
                "gradient_balance_eps=%s, "
                "exclude_bn=%s, exclude_head=%s, head_pattern=%s",
                self.fim_log_prefix,
                fim_method,
                fim_ema_decay,
                fim_eps_num,
                fim_eps_damp,
                fim_bias_correction,
                fim_stochastic_pairs,
                self.fim_gradient_balance,
                self.fim_gradient_balance_eps,
                self.exclude_bn_from_fim,
                self.exclude_head_from_fim,
                fim_head_pattern,
            )

    def connect(self, trainer: "BaseTrainer"):
        super().connect(trainer)
        self._trainer = trainer

    def _is_batchnorm_module(self, module: nn.Module) -> bool:
        return isinstance(module, nn.modules.batchnorm._BatchNorm) or (
            module.__class__.__name__ in BATCHNORM_LIKE_CLASS_NAMES
        )

    def _is_bn_param(self, name: str) -> bool:
        parts = name.split(".")
        for part in parts:
            if part.startswith("bn") or part == "norm":
                return True
        if "downsample" in parts:
            idx = parts.index("downsample")
            if idx + 1 < len(parts) and parts[idx + 1] == "1":
                return True
        return False

    def _is_head_param(self, name: str) -> bool:
        pattern = getattr(self, "fim_head_pattern", "")
        return bool(pattern) and pattern.lower() in name.lower()

    def _batchnorm_param_ids(self, trainer: "BaseTrainer") -> tuple[set[int], bool]:
        param_ids: set[int] = set()
        found_batchnorm_module = False

        for module in trainer.model.modules():
            if not self._is_batchnorm_module(module):
                continue

            found_batchnorm_module = True
            for param in module.parameters(recurse=True):
                if param.requires_grad:
                    param_ids.add(id(param))

        return param_ids, found_batchnorm_module

    def _task_specific_param_ids(
        self, trainer: "BaseTrainer"
    ) -> Optional[set[int]]:
        if not hasattr(trainer.model, "task_specific_parameters"):
            return None

        params = trainer.model.task_specific_parameters()
        return {
            id(param)
            for param in params
            if isinstance(param, nn.Parameter) and param.requires_grad
        }

    def _shared_param_ids(self, trainer: "BaseTrainer") -> set[int]:
        if not hasattr(trainer.model, "shared_parameters"):
            raise ValueError(
                f"{self.fim_log_prefix} requires model.shared_parameters() for "
                "shared-only FIM construction."
            )

        params = trainer.model.shared_parameters()
        param_ids = {
            id(param)
            for param in params
            if isinstance(param, nn.Parameter) and param.requires_grad
        }
        if not param_ids:
            raise ValueError(
                f"{self.fim_log_prefix} found no trainable shared parameters. "
                "FIM is defined only on the shared parameter block."
            )
        return param_ids

    def _validate_shared_fim_partition(
        self,
        trainer: "BaseTrainer",
        shared_param_ids: set[int],
        task_param_ids: Optional[set[int]],
    ) -> None:
        task_param_ids = task_param_ids or set()
        named_params = [
            (name, param)
            for name, param in trainer.model.named_parameters()
            if param.requires_grad
        ]
        overlap_ids = shared_param_ids & task_param_ids
        if overlap_ids:
            overlap_names = [
                name
                for name, param in named_params
                if id(param) in overlap_ids
            ]
            raise ValueError(
                f"{self.fim_log_prefix} model declares parameters as both shared "
                "and task-specific: "
                + ", ".join(overlap_names[:20])
            )

        unclassified = [
            name
            for name, param in named_params
            if id(param) not in shared_param_ids and id(param) not in task_param_ids
        ]
        if unclassified:
            raise ValueError(
                f"{self.fim_log_prefix} model has trainable parameters outside "
                "the shared/task-specific partition. FIM would be ambiguous for: "
                + ", ".join(unclassified[:20])
            )

    def _loss_source_for_fim(self, trainer: "BaseTrainer"):
        fim_losses = getattr(trainer, "fim_losses", None)
        return trainer.losses if fim_losses is None else fim_losses

    def _collect_losses(self, loss_source, preserve_rays: bool = False):
        if isinstance(loss_source, list):
            if not loss_source:
                return []

            first = loss_source[0]
            if isinstance(first, dict):
                if preserve_rays:
                    keys = list(first.keys())
                    return [[loss_dict[key] for key in keys] for loss_dict in loss_source]
                num_rays = len(loss_source)
                avg = {}
                for loss_dict in loss_source:
                    for key, value in loss_dict.items():
                        avg[key] = avg.get(key, 0) + value / num_rays
                return list(avg.values())

            if isinstance(first, (list, tuple)):
                if preserve_rays:
                    return [list(ray_losses) for ray_losses in loss_source]
                num_rays = len(loss_source)
                num_tasks = len(first)
                return [
                    sum(ray_losses[idx] for ray_losses in loss_source) / num_rays
                    for idx in range(num_tasks)
                ]

            return list(loss_source)

        if isinstance(loss_source, dict):
            return list(loss_source.values())

        if isinstance(loss_source, (tuple, list)):
            return list(loss_source)

        return [loss_source]

    def _get_lam(self, trainer: "BaseTrainer") -> Optional[torch.Tensor]:
        if hasattr(trainer, "rays") and trainer.rays is not None and len(trainer.rays) > 1:
            return torch.stack([ray.detach() for ray in trainer.rays])

        ray = getattr(trainer, "ray", None)
        if ray is None:
            return None

        lam = ray.detach() if isinstance(ray, torch.Tensor) else torch.as_tensor(ray, device=trainer.device)
        if lam.ndim == 1:
            lam = lam.unsqueeze(0)
        return lam

    def _named_params_for_fim(self, trainer: "BaseTrainer"):
        bn_param_ids: set[int] = set()
        found_batchnorm_module = False
        if self.exclude_bn_from_fim:
            bn_param_ids, found_batchnorm_module = self._batchnorm_param_ids(trainer)

        shared_param_ids = self._shared_param_ids(trainer)
        task_param_ids = self._task_specific_param_ids(trainer)
        self._validate_shared_fim_partition(trainer, shared_param_ids, task_param_ids)

        named = []
        for name, param in trainer.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) not in shared_param_ids:
                continue
            if self.exclude_bn_from_fim:
                if found_batchnorm_module:
                    if id(param) in bn_param_ids:
                        continue
                elif self._is_bn_param(name):
                    continue
            named.append((name, param))
        if not named:
            raise ValueError(
                f"{self.fim_log_prefix} selected zero FIM parameters after "
                "shared-only and BatchNorm filters."
            )
        names = [name for name, _ in named]
        params = [param for _, param in named]
        head_like_names = [
            name
            for name in names
            if any(marker in name for marker in FIM_TASK_HEAD_NAME_MARKERS)
        ]
        if head_like_names:
            raise ValueError(
                f"{self.fim_log_prefix} selected task-head-like parameters for "
                "FIM, which is only defined on shared parameters: "
                + ", ".join(head_like_names[:20])
            )
        logging.info(
            "[%s] FIM parameter set: selected_shared_tensors=%d, "
            "selected_shared_scalars=%d, task_specific_tensors=%d",
            self.fim_log_prefix,
            len(params),
            sum(param.numel() for param in params),
            len(task_param_ids or set()),
        )
        return names, params

    def _ensure_madam(self, trainer: "BaseTrainer"):
        if self._madam is None:
            _, params = self._named_params_for_fim(trainer)
            self._madam = MAdamPreconditioner(
                params,
                num_tasks=self.num_tasks,
                ema_decay=self.fim_ema_decay,
                eps_num=self.fim_eps_num,
                eps_damp=self.fim_eps_damp,
                stochastic=self.fim_stochastic_pairs,
                gradient_balance=self.fim_gradient_balance,
                gradient_balance_eps=self.fim_gradient_balance_eps,
            )
        return self._madam

    def _ensure_diag_madam(self, trainer: "BaseTrainer"):
        if self._diag_madam is None:
            _, params = self._named_params_for_fim(trainer)

            self._diag_madam = DiagonalMAdamPreconditioner(
                params,
                num_tasks=self.num_tasks,
                ema_decay=self.fim_ema_decay,
                eps_num=self.fim_eps_num,
                eps_damp=self.fim_eps_damp,
                stochastic=self.fim_stochastic_pairs,
                gradient_balance=self.fim_gradient_balance,
                gradient_balance_eps=self.fim_gradient_balance_eps,
                bias_correct=self.fim_bias_correction,
            )
        return self._diag_madam

    def _preconditioner_for_current_method(self):
        return self._madam if self.fim_method in FULL_MADAM_METHODS else self._diag_madam

    def _begin_fim_step(self, trainer: "BaseTrainer") -> bool:
        self._step_curvature_tensors = None
        return self.fim_method != "none"

    def _resolve_curvature_tensors(self, trainer: "BaseTrainer", precond) -> List[torch.Tensor]:
        if self._step_curvature_tensors is not None:
            return self._step_curvature_tensors

        values = [
            curvature.to(param.device, param.dtype).detach().clone()
            for param, curvature in zip(precond.params, precond.get_curvature_per_param())
        ]
        self._step_curvature_tensors = values
        return values

    def _build_custom_denom_tensors(
        self,
        trainer: "BaseTrainer",
        precond,
        curvature_tensors: List[torch.Tensor],
    ) -> Optional[List[torch.Tensor]]:
        return None

    @staticmethod
    def _inject_preconditioned_gradients(precond, denom_tensors: List[torch.Tensor]) -> None:
        lam = getattr(precond, "_cached_lam", None)
        batch_rays = int(lam.shape[0]) if lam is not None else 1
        for param, denom in zip(precond.params, denom_tensors):
            if param.grad is None:
                continue
            param.grad = (param.grad / batch_rays) / denom.to(param.device, param.dtype)

    @staticmethod
    def _uses_external_curvature_optimizer(trainer: "BaseTrainer") -> bool:
        optimizer = getattr(trainer, "optimizer", None)
        return callable(getattr(optimizer, "load_external_curvature", None))

    def _load_curvature_into_optimizer(
        self,
        trainer: "BaseTrainer",
        curvature_tensors: Optional[List[torch.Tensor]] = None,
    ) -> None:
        optimizer = getattr(trainer, "optimizer", None)
        loader = getattr(optimizer, "load_external_curvature", None)
        if not callable(loader):
            raise RuntimeError(
                "This FIM mode requires an optimizer exposing load_external_curvature()."
            )

        precond = self._preconditioner_for_current_method()
        if precond is None or precond.fim is None:
            return

        if curvature_tensors is None:
            curvature_tensors = self._resolve_curvature_tensors(trainer, precond)

        pairs = []
        for param, curvature in zip(precond.params, curvature_tensors):
            pairs.append((param, curvature.to(param.device, param.dtype)))
        loader(pairs, bias_corrected=getattr(precond, "bias_correct", False))

    def _clip_preconditioned_grads(self, trainer: "BaseTrainer") -> None:
        if self.fim_clip_grad_norm <= 0:
            return
        params = [param for param in trainer.model.parameters() if param.grad is not None]
        torch.nn.utils.clip_grad_norm_(params, self.fim_clip_grad_norm)

    def on_before_backward(self, trainer: "BaseTrainer", *args, **kwargs):
        if self.fim_method == "none":
            self._fim_active_this_step = False
            self._step_curvature_tensors = None
            return

        self._fim_active_this_step = bool(self._begin_fim_step(trainer))
        if not self._fim_active_this_step:
            return

        lam = self._get_lam(trainer)
        if lam is None:
            self._fim_active_this_step = False
            return

        loss_source = self._loss_source_for_fim(trainer)
        preserve_rays = self.fim_method in FULL_MADAM_METHODS and lam.shape[0] > 1
        task_losses = self._collect_losses(loss_source, preserve_rays=preserve_rays)
        if not task_losses:
            self._fim_active_this_step = False
            return

        if self.fim_method in FULL_MADAM_METHODS:
            self._ensure_madam(trainer).update(task_losses, lam)
        else:
            self._ensure_diag_madam(trainer).update(task_losses, lam)

    def on_after_backward(self, trainer: "BaseTrainer", *args, **kwargs):
        if self.fim_method == "none" or not self._fim_active_this_step:
            return

        precond = self._preconditioner_for_current_method()
        if precond is None or precond.fim is None:
            return

        curvature_tensors = self._resolve_curvature_tensors(trainer, precond)

        if self._uses_external_curvature_optimizer(trainer):
            self._clip_preconditioned_grads(trainer)
            return

        denom_tensors = self._build_custom_denom_tensors(trainer, precond, curvature_tensors)
        if denom_tensors is None:
            precond.precondition()
        else:
            self._inject_preconditioned_gradients(precond, denom_tensors)

        self._clip_preconditioned_grads(trainer)

    def on_before_optimizer_step(self, trainer: "BaseTrainer", *args, **kwargs):
        if (
            self.fim_method == "none"
            or not self._fim_active_this_step
            or not self._uses_external_curvature_optimizer(trainer)
        ):
            return

        precond = self._preconditioner_for_current_method()
        curvature_tensors = None
        if precond is not None and precond.fim is not None:
            curvature_tensors = self._resolve_curvature_tensors(trainer, precond)
        self._load_curvature_into_optimizer(trainer, curvature_tensors=curvature_tensors)
