import argparse
import time
import os
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm
import deepxde as dde
from src.model.laaf import DNN_GAAF, DNN_LAAF
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK, Adam_LBFGS, MAdam, DiagMAdam
from src.pde.burgers import Burgers1D, Burgers2D
from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.pde.heat import Heat2D_VaryingCoef, Heat2D_Multiscale, Heat2D_ComplexGeometry, Heat2D_LongTime, HeatND
from src.pde.ns import NS2D_LidDriven, NS2D_BackStep, NS2D_LongTime
from src.pde.poisson import Poisson2D_Classic, PoissonBoltzmann2D, Poisson3D_ComplexGeometry, Poisson2D_ManyArea, PoissonND
from src.pde.wave import Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime
from src.pde.inverse import PoissonInv, HeatInv
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback, WandbCallback
from src.utils.rar import rar_wrapper

# It is recommended not to modify this example file.
# Please copy it as benchmark_xxx.py and make changes according to your own ideas.
pde_list = \
    [Burgers1D, Burgers2D] + \
    [Poisson2D_Classic, PoissonBoltzmann2D, Poisson3D_ComplexGeometry, Poisson2D_ManyArea] + \
    [Heat2D_VaryingCoef, Heat2D_Multiscale, Heat2D_ComplexGeometry, Heat2D_LongTime] + \
    [NS2D_LidDriven, NS2D_BackStep, NS2D_LongTime] + \
    [Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime] + \
    [KuramotoSivashinskyEquation, GrayScottEquation] + \
    [PoissonND, HeatND]

# pde_list += \
#     [(Burgers2D, {"datapath": "ref/burgers2d_1.dat", "icpath": ("ref/burgers2d_init_u_1.dat", "ref/burgers2d_init_v_1.dat")})] + \
#     [(Burgers2D, {"datapath": "ref/burgers2d_2.dat", "icpath": ("ref/burgers2d_init_u_2.dat", "ref/burgers2d_init_v_2.dat")})] + \
#     [(Burgers2D, {"datapath": "ref/burgers2d_3.dat", "icpath": ("ref/burgers2d_init_u_3.dat", "ref/burgers2d_init_v_3.dat")})] + \
#     [(Burgers2D, {"datapath": "ref/burgers2d_4.dat", "icpath": ("ref/burgers2d_init_u_4.dat", "ref/burgers2d_init_v_4.dat")})] + \
#     [(Poisson2D_Classic, {"scale": 2})] + \
#     [(Poisson2D_Classic, {"scale": 4})] + \
#     [(Poisson2D_Classic, {"scale": 8})] + \
#     [(Poisson2D_Classic, {"scale": 16})] + \
#     [(Heat2D_Multiscale, {"init_coef": (5 * np.pi, np.pi)})] + \
#     [(Heat2D_Multiscale, {"init_coef": (10 * np.pi, np.pi)})] + \
#     [(Heat2D_Multiscale, {"init_coef": (40 * np.pi, np.pi)})] + \
#     [(NS2D_LidDriven, {"datapath": "ref/lid_driven_a2.dat", "a": 2})] + \
#     [(NS2D_LidDriven, {"datapath": "ref/lid_driven_a8.dat", "a": 8})] + \
#     [(NS2D_LidDriven, {"datapath": "ref/lid_driven_a16.dat", "a": 16})] + \
#     [(NS2D_LidDriven, {"datapath": "ref/lid_driven_a32.dat", "a": 32})] + \
#     [(Wave1D, {"a": 2})] + \
#     [(Wave1D, {"a": 6})] + \
#     [(Wave1D, {"a": 8})] + \
#     [(Wave1D, {"a": 10})] + \
#     [(HeatND, {"dim": 4})] + \
#     [(HeatND, {"dim": 6})] + \
#     [(HeatND, {"dim": 8})] + \
#     [(HeatND, {"dim": 10})]

DEFAULT_COMPONENTS = {
    "optimizer": "adam",
    "preconditioner": "none",
    "loss_balancing": "none",
    "sampling": "none",
    "loss_augmentation": "none",
    "architecture": "tanh",
}

def filter_pdes(name):
    pde_name_map = {cls.__name__ if not isinstance(cls, tuple) else cls[0].__name__: cls for cls in pde_list}
    if name and name in pde_name_map:
        return [pde_name_map[name]]
    if name and name != "all":
        matches = [v for k, v in pde_name_map.items() if name.lower() in k.lower()]
        return matches if matches else pde_list
    return pde_list


def parse_wandb_tags(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    return [tag.strip() for tag in str(value).split(",") if tag.strip()]


def resolve_components(command_args):
    components = DEFAULT_COMPONENTS.copy()
    components["optimizer"] = command_args.optimizer
    components["preconditioner"] = command_args.preconditioner
    components["loss_balancing"] = command_args.loss_balancing
    components["sampling"] = command_args.sampling
    components["loss_augmentation"] = command_args.loss_augmentation
    components["architecture"] = command_args.architecture

    if components["optimizer"] == "multiadam" and components["preconditioner"] != "none":
        raise ValueError("MultiAdam is a comparison optimizer and cannot be combined with MAdam preconditioning.")

    return argparse.Namespace(**components)


def build_network(pde, command_args, components):
    hidden_layers = parse_hidden_layers(command_args)
    if components.architecture == "laaf":
        return DNN_LAAF(len(hidden_layers) - 1, hidden_layers[0], pde.input_dim, pde.output_dim).float()
    if components.architecture == "gaaf":
        return DNN_GAAF(len(hidden_layers) - 1, hidden_layers[0], pde.input_dim, pde.output_dim).float()
    return dde.nn.FNN([pde.input_dim] + hidden_layers + [pde.output_dim], "tanh", "Glorot normal").float()


def is_shared_parameter(net, name):
    if hasattr(net, "linears"):
        linears = getattr(net, "linears")
        if len(linears) > 1 and name.startswith(f"linears.{len(linears) - 1}."):
            return False
        return True

    if hasattr(net, "layers"):
        layers = getattr(net, "layers")

        if isinstance(layers, torch.nn.Sequential):
            last_name = next(reversed(layers._modules))
            if name.startswith(f"layers.{last_name}."):
                return False
            return True

        if isinstance(layers, torch.nn.ModuleList):
            last_idx = len(layers) - 1
            if name.startswith(f"layers.{last_idx}."):
                return False
            for idx, layer in enumerate(layers[:-1]):
                if name.startswith(f"layers.{idx}.") and isinstance(layer, torch.nn.ModuleList):
                    return False
            return True

    return True


def select_preconditioned_parameters(net, command_args):
    named_params = list(net.named_parameters())
    batchnorm_param_names = set()
    for module_name, module in net.named_modules():
        if isinstance(module, _BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            for param_name, _ in module.named_parameters(recurse=False):
                batchnorm_param_names.add(f"{prefix}{param_name}")

    selected = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if command_args.preconditioner_exclude_batchnorm and name in batchnorm_param_names:
            continue
        if command_args.preconditioner_shared_only and not is_shared_parameter(net, name):
            continue
        selected.append(param)

    if not selected:
        selected = [param for _, param in named_params if param.requires_grad]

    return selected


def build_optimizer(net, pde, command_args, components, loss_weights):
    params = net.parameters()
    preconditioned_params = select_preconditioned_parameters(net, command_args)

    if components.optimizer == "multiadam":
        optimizer = MultiAdam(params, lr=1e-3, betas=(0.99, 0.99), loss_group_idx=[pde.num_pde])
    elif components.optimizer == "lbfgs":
        optimizer = Adam_LBFGS(
            params,
            switch_epoch=5000,
            adam_param={"lr": command_args.lr},
            preconditioner=(None if components.preconditioner == "none" else components.preconditioner),
            num_tasks=(None if components.preconditioner == "none" else pde.num_loss),
            stochastic=command_args.preconditioner_stochastic,
            preconditioned_params=preconditioned_params,
        )
    elif components.preconditioner == "madam":
        optimizer = MAdam(
            params,
            lr=command_args.lr,
            num_tasks=pde.num_loss,
            stochastic=command_args.preconditioner_stochastic,
            preconditioned_params=preconditioned_params,
        )
    elif components.preconditioner == "diag_madam":
        optimizer = DiagMAdam(
            params,
            lr=command_args.lr,
            num_tasks=pde.num_loss,
            stochastic=command_args.preconditioner_stochastic,
            preconditioned_params=preconditioned_params,
        )
    else:
        optimizer = torch.optim.Adam(params, command_args.lr)

    if components.loss_balancing == "lra":
        optimizer = LR_Adaptor(optimizer, loss_weights, pde.num_pde)
    elif components.loss_balancing == "ntk":
        optimizer = LR_Adaptor_NTK(optimizer, loss_weights, pde)

    return optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINNBench trainer')
    parser.add_argument('--name', type=str, default="benchmark")
    parser.add_argument('--case', type=str, default=None)
    parser.add_argument('--device', type=str, default="0")  # set to "cpu" enables cpu training 
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--hidden-layers', type=str, default="100*5")
    parser.add_argument('--loss-weight', type=str, default="")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--iter', type=int, default=20000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--plot-every', type=int, default=2000)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default=DEFAULT_COMPONENTS["optimizer"], choices=["adam", "lbfgs", "multiadam"])
    parser.add_argument('--preconditioner', type=str, default=DEFAULT_COMPONENTS["preconditioner"], choices=["none", "madam", "diag_madam"])
    parser.add_argument('--preconditioner-stochastic', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--preconditioner-shared-only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--preconditioner-exclude-batchnorm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--loss-balancing', type=str, default=DEFAULT_COMPONENTS["loss_balancing"], choices=["none", "lra", "ntk"])
    parser.add_argument('--sampling', type=str, default=DEFAULT_COMPONENTS["sampling"], choices=["none", "rar"])
    parser.add_argument('--loss-augmentation', type=str, default=DEFAULT_COMPONENTS["loss_augmentation"], choices=["none", "gepinn"])
    parser.add_argument('--architecture', type=str, default=DEFAULT_COMPONENTS["architecture"], choices=["tanh", "laaf", "gaaf"])
    parser.add_argument('--wandb-project', type=str, default=os.environ.get("WANDB_PROJECT", "MAdam-PINNacle"))
    parser.add_argument('--wandb-entity', type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument('--wandb-group', type=str, default=os.environ.get("WANDB_RUN_GROUP"))
    parser.add_argument('--wandb-job-type', type=str, default=os.environ.get("WANDB_JOB_TYPE", "benchmark"))
    parser.add_argument('--wandb-tags', type=str, default=os.environ.get("WANDB_TAGS", ""))

    command_args = parser.parse_args()
    components = resolve_components(command_args)

    seed = command_args.seed
    if seed is None:
        raise ValueError(
            "PINNacle W&B-aligned runs require an explicit --seed. "
            "Launch separate runs for each seed you want to evaluate."
        )
    if int(command_args.repeat) != 1:
        raise ValueError(
            "PINNacle W&B-aligned runs require --repeat 1. "
            "Use explicit seed sweeps instead of repeat-based reruns."
        )
    dde.config.set_random_seed(seed)
    date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
    trainer = Trainer(f"{date_str}-{command_args.name}", command_args.device)
    filtered_list = filter_pdes(command_args.case if command_args.case is not None else command_args.name)
    wandb_tags = parse_wandb_tags(command_args.wandb_tags)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_job_name = os.environ.get("SLURM_JOB_NAME")

    for pde_config in filtered_list:

        def get_model_dde(pde_config=pde_config):
            if isinstance(pde_config, tuple):
                pde = pde_config[0](**pde_config[1])
            else:
                pde = pde_config()
            
            # pde.training_points()
            if components.loss_augmentation == "gepinn":
                pde.use_gepinn()

            net = build_network(pde, command_args, components)

            loss_weights = parse_loss_weight(command_args)
            if loss_weights is None:
                loss_weights = np.ones(pde.num_loss)
            else:
                loss_weights = np.array(loss_weights)

            opt = build_optimizer(net, pde, command_args, components, loss_weights)

            model = pde.create_model(net)
            model.compile(opt, loss_weights=loss_weights)
            if components.sampling == "rar":
                model.train = rar_wrapper(pde, model, {"interval": 1000, "count": 1})

            model._madam_benchmark_context = {
                "requested_name": command_args.name,
                "case_filter": command_args.case if command_args.case is not None else command_args.name,
                "command_seed": command_args.seed,
                "baseline_label": (
                    command_args.method
                    if command_args.method is not None
                    else components.optimizer
                ),
                "wandb_project": command_args.wandb_project,
                "wandb_entity": command_args.wandb_entity,
                "wandb_group": command_args.wandb_group,
                "wandb_job_type": command_args.wandb_job_type,
                "wandb_tags": wandb_tags,
                "slurm_job_id": slurm_job_id,
                "slurm_job_name": slurm_job_name,
                "components": vars(components).copy(),
                "wandb_config": {
                    "benchmark_name": command_args.name,
                    "case_filter": command_args.case if command_args.case is not None else command_args.name,
                    "command_seed": command_args.seed,
                    "slurm_job_id": slurm_job_id,
                    "slurm_job_name": slurm_job_name,
                    "baseline_label": (
                        command_args.method
                        if command_args.method is not None
                        else components.optimizer
                    ),
                    "optimizer": components.optimizer,
                    "preconditioner": components.preconditioner,
                    "preconditioner_stochastic": command_args.preconditioner_stochastic,
                    "preconditioner_shared_only": command_args.preconditioner_shared_only,
                    "preconditioner_exclude_batchnorm": command_args.preconditioner_exclude_batchnorm,
                    "loss_balancing": components.loss_balancing,
                    "sampling": components.sampling,
                    "loss_augmentation": components.loss_augmentation,
                    "architecture": components.architecture,
                    "learning_rate": command_args.lr,
                    "iterations": command_args.iter,
                    "display_every": command_args.log_every,
                    "plot_every": command_args.plot_every,
                    "repeat": command_args.repeat,
                    "hidden_layers": command_args.hidden_layers,
                    "loss_weight": command_args.loss_weight,
                    "pde_num_loss": pde.num_loss,
                    "pde_num_pde": pde.num_pde,
                },
            }
            # the trainer calls model.train(**train_args)
            return model

        def get_model_others():
            model = None
            # create a model object which support .train() method, and param @model_save_path is required
            # create the object based on command_args and return it to be trained
            # schedule the task using trainer.add_task(get_model_other, {training args})
            return model

        trainer.add_task(
            get_model_dde, {
                "iterations": command_args.iter,
                "display_every": command_args.log_every,
                "callbacks": [
                    WandbCallback(),
                    TesterCallback(log_every=command_args.log_every),
                    PlotCallback(log_every=command_args.plot_every, fast=True),
                    LossCallback(verbose=True),
                ]
            }
        )

    trainer.setup(__file__, seed)
    trainer.set_repeat(command_args.repeat)
    trainer.train_all()
    trainer.summary()
