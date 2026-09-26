import torch
from torch.optim import Optimizer

from .madam_optimizer import MAdam, DiagMAdam


class Adam_LBFGS(Optimizer):
    
    def __init__(
        self, 
        params, 
        switch_epoch=10000, 
        adam_param={'lr': 1e-3, 'betas': (0.9, 0.999)}, 
        lbfgs_param={'lr': 1, 'max_iter': 20},
        preconditioner=None,
        num_tasks=None,
        ema_decay=0.999,
        stochastic=False,
        preconditioned_params=None,
    ):
        self.params = list(params)
        self.switch_epoch = switch_epoch

        if preconditioner is None:
            self.adam = torch.optim.Adam(self.params, **adam_param)
        else:
            if num_tasks is None:
                raise ValueError("num_tasks is required when Adam_LBFGS uses a preconditioned Adam stage.")
            if preconditioner == "madam":
                adam_cls = MAdam
            elif preconditioner == "diag_madam":
                adam_cls = DiagMAdam
            else:
                raise ValueError(f"Unsupported Adam_LBFGS preconditioner: {preconditioner!r}")
            self.adam = adam_cls(
                self.params,
                num_tasks=num_tasks,
                ema_decay=ema_decay,
                stochastic=stochastic,
                preconditioned_params=preconditioned_params,
                **adam_param,
            )
        self.lbfgs = torch.optim.LBFGS(self.params, **lbfgs_param)

        super().__init__(self.params, defaults={})

        self.state['current_step'] = 0
    
    def step(self, closure=None):
        self.state['current_step'] += 1

        if self.state['current_step'] < self.switch_epoch:
            return self.adam.step(closure)

        loss = self.lbfgs.step(closure)
        if self.state['current_step'] == self.switch_epoch:
            print(f"Switch to LBFGS at epoch {self.switch_epoch}")
        return loss
