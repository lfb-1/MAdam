import math
from typing import Iterable, Sequence, Tuple

import torch
from torch.optim.optimizer import Optimizer


class AdamWithScalarCurvaturePreconditioning(Optimizer):
    """Fix-1-only Adam ablation.

    This optimizer keeps Adam's scalarized second-moment stream ``v_t`` on the
    raw gradient, then applies that curvature externally before the first-moment
    EMA. The effective order is therefore normalize-then-average:

    ``v_t <- EMA[g_t^2]``
    ``g_hat_t = g_t / sqrt(v_t)``
    ``m_t <- EMA[g_hat_t]``
    ``theta <- theta - eta * m_t``
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _ensure_state(self, param: torch.nn.Parameter) -> dict:
        state = self.state[param]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError(
                        "AdamWithScalarCurvaturePreconditioning does not support sparse gradients"
                    )

                grad = param.grad
                state = self._ensure_state(param)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(param, alpha=group["weight_decay"])

                # Keep Adam's scalarized curvature estimate on the raw gradient.
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction2 = 1 - beta2**step
                denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add_(group["eps"])

                # Fix 1: normalize before updating the first-moment EMA.
                preconditioned_grad = grad / denom
                exp_avg.mul_(beta1).add_(preconditioned_grad, alpha=1 - beta1)

                bias_correction1 = 1 - beta1**step
                step_size = group["lr"] / bias_correction1
                param.add_(exp_avg, alpha=-step_size)

        return loss


class AdamWithExternalCurvature(Optimizer):
    """Adam variant for the Fix-2-only ablation.

    ``load_external_curvature`` replaces Adam's second-moment estimate
    ``v_t`` with an externally-computed per-parameter curvature tensor on
    the subsequent ``step``. This is the theory-driven Fix-2-only ablation:
    keep Adam's first-moment history, but swap the denominator source.

    ``load_external_denom`` is kept as a compatibility path for any older
    callers that still provide a fully formed denominator directly.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._curvature_generation = 0
        self._has_fresh_curvature = False

    def _ensure_state(self, param: torch.nn.Parameter) -> dict:
        state = self.state[param]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state['external_curvature'] = None
            state['curvature_generation'] = -1
            state['curvature_bias_corrected'] = False
            state['external_denom'] = None
            state['denom_generation'] = -1
        return state

    def load_external_curvature(
        self,
        pairs: Sequence[Tuple[torch.nn.Parameter, torch.Tensor]],
        *,
        bias_corrected: bool = False,
    ) -> None:
        """Register per-parameter second moments to use on the next step."""
        self._curvature_generation += 1
        self._has_fresh_curvature = True
        current_gen = self._curvature_generation
        for param, curvature in pairs:
            if curvature is None:
                continue
            state = self._ensure_state(param)
            state['external_curvature'] = curvature.detach().clone()
            state['curvature_generation'] = current_gen
            state['curvature_bias_corrected'] = bias_corrected

    def load_external_denom(self, pairs: Sequence[Tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
        """Register per-parameter denominators to use on the next step."""
        self._curvature_generation += 1
        self._has_fresh_curvature = True
        current_gen = self._curvature_generation
        for param, curvature in pairs:
            if curvature is None:
                continue
            state = self._ensure_state(param)
            state['external_denom'] = curvature.detach().clone()
            state['denom_generation'] = current_gen

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamWithExternalCurvature does not support sparse gradients")
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self._ensure_state(p)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                state['step'] += 1
                state_steps.append(state['step'])

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]
                state = self.state[param]

                if group['weight_decay'] != 0:
                    grad = grad.add(param, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                use_external = (
                    self._has_fresh_curvature
                    and (
                        (
                            state.get('curvature_generation', -1) == self._curvature_generation
                            and state.get('external_curvature') is not None
                        )
                        or (
                            state.get('denom_generation', -1) == self._curvature_generation
                            and state.get('external_denom') is not None
                        )
                    )
                )

                if use_external:
                    external_denom = state.get('external_denom')
                    external_curvature = state.get('external_curvature')
                    if (
                        state.get('denom_generation', -1) == self._curvature_generation
                        and external_denom is not None
                    ):
                        denom = external_denom.to(param.device, param.dtype).clone()
                    else:
                        curvature = external_curvature.to(param.device, param.dtype).clone()
                        curvature = torch.clamp(curvature, min=0.0)
                        if not state.get('curvature_bias_corrected', False):
                            bias_correction2 = 1 - beta2 ** step
                            curvature = curvature.div(max(bias_correction2, 1e-16))
                        denom = curvature.sqrt().add_(group['eps'])
                else:
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction2 = 1 - beta2 ** step
                    denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add_(group['eps'])

                bias_correction1 = 1 - beta1 ** step
                step_size = group['lr'] / bias_correction1
                param.addcdiv_(exp_avg, denom, value=-step_size)

        self._has_fresh_curvature = False
        return loss


class AdamWithExternalCurvaturePreconditioning(Optimizer):
    """External-curvature normalize-then-average without residual Adam variance.

    This is the clean external-curvature counterpart of
    ``AdamWithScalarCurvaturePreconditioning``:

    ``curvature_t <- external source``
    ``g_hat_t = g_t / sqrt(curvature_t)``
    ``m_t <- EMA[g_hat_t]``
    ``theta <- theta - eta * m_t``

    It isolates the combination of:
    1. an external curvature source (e.g. MAdam), and
    2. storing momentum in preconditioned coordinates,

    while removing Adam's residual second-moment adaptation on the
    preconditioned gradient.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._curvature_generation = 0
        self._has_fresh_curvature = False

    def _ensure_state(self, param: torch.nn.Parameter) -> dict:
        state = self.state[param]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["external_curvature"] = None
            state["curvature_generation"] = -1
            state["curvature_bias_corrected"] = False
        return state

    def load_external_curvature(
        self,
        pairs: Sequence[Tuple[torch.nn.Parameter, torch.Tensor]],
        *,
        bias_corrected: bool = False,
    ) -> None:
        """Register per-parameter curvature tensors for the next step."""
        self._curvature_generation += 1
        self._has_fresh_curvature = True
        current_gen = self._curvature_generation
        for param, curvature in pairs:
            if curvature is None:
                continue
            state = self._ensure_state(param)
            state["external_curvature"] = curvature.detach().clone()
            state["curvature_generation"] = current_gen
            state["curvature_bias_corrected"] = bias_corrected

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError(
                        "AdamWithExternalCurvaturePreconditioning does not support sparse gradients"
                    )

                grad = param.grad
                state = self._ensure_state(param)
                exp_avg = state["exp_avg"]

                state["step"] += 1
                step = state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(param, alpha=group["weight_decay"])

                use_external = (
                    self._has_fresh_curvature
                    and state.get("curvature_generation", -1) == self._curvature_generation
                    and state.get("external_curvature") is not None
                )
                if not use_external:
                    raise RuntimeError(
                        "AdamWithExternalCurvaturePreconditioning requires fresh external curvature "
                        "before each optimizer.step()."
                    )

                curvature = state["external_curvature"].to(param.device, param.dtype).clone()
                curvature = torch.clamp(curvature, min=0.0)
                if not state.get("curvature_bias_corrected", False):
                    bias_correction2 = 1 - beta2**step
                    curvature = curvature.div(max(bias_correction2, 1e-16))
                denom = curvature.sqrt().add_(group["eps"])

                preconditioned_grad = grad / denom
                exp_avg.mul_(beta1).add_(preconditioned_grad, alpha=1 - beta1)

                bias_correction1 = 1 - beta1**step
                step_size = group["lr"] / bias_correction1
                param.add_(exp_avg, alpha=-step_size)

        self._has_fresh_curvature = False
        return loss


class ExternalAdam(Optimizer):
    """Adam with external application of Adam's own scalarized curvature.

    This optimizer isolates the placement contribution for fixed-weight MTL:
    1. update a raw-gradient second-moment buffer ``v_raw``;
    2. precondition the current gradient by ``sqrt(v_raw)`` before EMA;
    3. run standard Adam on the preconditioned gradient.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _ensure_state(self, param: torch.nn.Parameter) -> dict:
        state = self.state[param]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["raw_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("ExternalAdam does not support sparse gradients")

                state = self._ensure_state(param)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                raw_exp_avg_sq = state["raw_exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(param, alpha=group["weight_decay"])

                raw_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                raw_bias_correction2 = 1 - beta2 ** step
                raw_denom = raw_exp_avg_sq.sqrt().div(math.sqrt(raw_bias_correction2)).add_(group["eps"])
                preconditioned_grad = grad / raw_denom

                exp_avg.mul_(beta1).add_(preconditioned_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    preconditioned_grad,
                    preconditioned_grad,
                    value=1 - beta2,
                )

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
