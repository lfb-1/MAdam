"""Composition support used only by the adaptive task-weighting adapters.

Keep these options out of the existing methods' estimator and training paths.
"""

import torch

from .fim_support import FIMSupportMixin


class AdaptiveFIMSupportMixin(FIMSupportMixin):
    def init_fim_support(
        self, *, fim_lambda_policy="aggregation_raw",
        fim_preserve_shared_grad_norm=False, fim_solver_geometry="raw", **kwargs
    ):
        if fim_lambda_policy not in {"aggregation_raw", "aggregation_simplex", "uniform"}:
            raise ValueError("Unsupported fim_lambda_policy: " + str(fim_lambda_policy))
        if fim_solver_geometry not in {"raw", "lagged_preconditioned"}:
            raise ValueError("Unsupported fim_solver_geometry: " + str(fim_solver_geometry))
        super().init_fim_support(**kwargs)
        self.fim_lambda_policy = fim_lambda_policy
        self.fim_preserve_shared_grad_norm = bool(fim_preserve_shared_grad_norm)
        self.fim_solver_geometry = fim_solver_geometry
        self._fim_lam = None
        self._fim_lambda_updates = 0
        self._lambda_diagnostics = {}
        self._composition_diagnostics = {
            "precondition_steps": 0, "solver_geometry_steps": 0,
            "shared_clip_calls": 0, "shared_clip_hits": 0,
        }
        self._norm_before_precondition = None
        self._grads_unscaled = False

    def on_before_backward(self, trainer, *args, **kwargs):
        self._grads_unscaled = False
        super().on_before_backward(trainer, *args, **kwargs)

    def _unscale_gradients(self, trainer):
        if not self._grads_unscaled and getattr(trainer.scaler, "is_enabled", lambda: False)():
            trainer.scaler.unscale_(trainer.optimizer)
            self._grads_unscaled = True

    def set_fim_preference_from_aggregation(self, aggregation_weights):
        weights = aggregation_weights.detach()
        if weights.ndim != 1 or weights.numel() != self.num_tasks:
            raise ValueError(f"Aggregation weights must have shape [{self.num_tasks}].")
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("Aggregation weights must be finite and nonnegative.")
        weight_sum = weights.sum()
        if not torch.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("Aggregation weights must have positive finite sum.")
        if self.fim_lambda_policy == "aggregation_raw":
            lam = weights.clone()
        elif self.fim_lambda_policy == "aggregation_simplex":
            lam = weights / weight_sum
        else:
            lam = torch.full_like(weights, 1.0 / self.num_tasks)
        self._fim_lam = lam
        self._fim_lambda_updates += 1
        total = float(weight_sum.cpu())
        d = self._lambda_diagnostics
        d.update(
            aggregation_weight_sum_last=total,
            aggregation_weight_sum_min=min(d.get("aggregation_weight_sum_min", total), total),
            aggregation_weight_sum_max=max(d.get("aggregation_weight_sum_max", total), total),
            aggregation_weight_max_last=float(weights.max().cpu()),
            fim_lambda_sum_last=float(lam.sum().cpu()),
            fim_lambda_max_last=float(lam.max().cpu()),
        )
        return lam

    def _get_lam(self, trainer):
        if self._fim_lam is not None:
            return self._fim_lam.unsqueeze(0)
        return super()._get_lam(trainer)

    def get_fim_lambda_diagnostics(self):
        return {
            "fim_lambda_policy": self.fim_lambda_policy,
            "fim_lambda_updates": self._fim_lambda_updates,
            **self._lambda_diagnostics,
        }

    def transform_task_gradients_for_solver(self, gradients, shared_parameters):
        if self.fim_solver_geometry == "raw" or self.fim_method == "none":
            return gradients
        precond = self._preconditioner_for_current_method()
        if precond is None or precond.fim is None:
            return gradients
        if [id(p) for p in shared_parameters] != [id(p) for p in precond.params]:
            raise RuntimeError("Shared-parameter order does not match the lagged MAdam denominator.")
        denom = torch.cat([
            d.to(gradients.device, gradients.dtype).reshape(-1)
            for d in precond.get_denominator_per_param()
        ])
        if denom.numel() != gradients.shape[1]:
            raise RuntimeError("Solver gradient width does not match MAdam denominator width.")
        if not torch.isfinite(denom).all() or torch.any(denom <= 0):
            raise RuntimeError("Lagged MAdam denominator is invalid.")
        d = self._composition_diagnostics
        d["solver_geometry_steps"] += 1
        d.update(
            solver_denom_min_last=float(denom.min().cpu()),
            solver_denom_mean_last=float(denom.mean().cpu()),
            solver_denom_max_last=float(denom.max().cpu()),
        )
        return gradients / denom.unsqueeze(0)

    def get_fim_composition_diagnostics(self):
        d = self._composition_diagnostics
        return {
            "fim_preserve_shared_grad_norm": self.fim_preserve_shared_grad_norm,
            "fim_solver_geometry": self.fim_solver_geometry,
            **d,
            "shared_clip_rate": d["shared_clip_hits"] / max(d["shared_clip_calls"], 1),
        }

    def clip_shared_gradients(self, trainer, max_norm):
        if max_norm <= 0:
            return
        self._unscale_gradients(trainer)
        norm = torch.nn.utils.clip_grad_norm_(list(trainer.model.shared_parameters()), max_norm)
        value = float(norm.detach().cpu())
        d = self._composition_diagnostics
        d["shared_clip_calls"] += 1
        d["shared_clip_hits"] += int(value > max_norm)
        d["shared_grad_norm_before_clip_last"] = value

    @staticmethod
    def _gradient_norm(params):
        values = [p.grad.detach().float().square().sum() for p in params if p.grad is not None]
        return torch.stack(values).sum().sqrt() if values else torch.tensor(0.0)

    def on_after_backward(self, trainer, *args, **kwargs):
        precond = self._preconditioner_for_current_method()
        active = (self.fim_method != "none" and self._fim_active_this_step
                  and precond is not None and precond.fim is not None
                  and not self._uses_external_curvature_optimizer(trainer))
        if active:
            self._unscale_gradients(trainer)
        self._norm_before_precondition = self._gradient_norm(precond.params) if active else None
        try:
            super().on_after_backward(trainer, *args, **kwargs)
        finally:
            self._norm_before_precondition = None

    def _clip_preconditioned_grads(self, trainer):
        # The base hook calls this after preconditioning, before optional FIM clipping.
        raw = self._norm_before_precondition
        if raw is not None:
            params = self._preconditioner_for_current_method().params
            after = self._gradient_norm(params)
            raw_value, after_value = float(raw.cpu()), float(after.cpu())
            d = self._composition_diagnostics
            d["precondition_steps"] += 1
            d["shared_grad_norm_before_precondition_last"] = raw_value
            d["shared_grad_norm_after_precondition_last"] = after_value
            if raw_value > 0 and after_value > 0:
                ratio = after_value / raw_value
                d["shared_grad_norm_ratio_min"] = min(d.get("shared_grad_norm_ratio_min", ratio), ratio)
                d["shared_grad_norm_ratio_max"] = max(d.get("shared_grad_norm_ratio_max", ratio), ratio)
                if self.fim_preserve_shared_grad_norm:
                    for param in params:
                        if param.grad is not None:
                            param.grad.mul_((raw / after).to(param.device, param.dtype))
            d["shared_grad_norm_after_composition_last"] = float(self._gradient_norm(params).cpu())
        super()._clip_preconditioned_grads(trainer)
