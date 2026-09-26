from .algo_callback import AlgoCallback, ParetoFrontApproximationAlgoCallback

from .cagrad import CAGrad
from .dwa import DynamicWeightAverage
from .excessmtl import ExcessMTL
from .fairgrad import FairGrad
from .nashmtl import NashMTL
from .imtl import IMTLG
from .ls import LinearScalarization
from .palora import PaLoRA, PaLoRA_GB, PaLoRA_LB, PaLoRAFull
from .pamal import PaMaL, PaMaL_GB, PaMaL_LB
from .uw import UncertaintyWeighting


BASELINE_METHODS = ("ls", "uw", "dwa", "imtl", "cagrad", "nashmtl", "fairgrad", "excessmtl", "palora", "pamal")
NONPARETO_BASELINE_METHODS = ("ls", "uw", "dwa", "imtl", "cagrad", "nashmtl", "fairgrad", "excessmtl")
PARETO_BASELINE_METHODS = ("palora", "pamal")

METHODS = dict(
    cagrad=CAGrad,
    nashmtl=NashMTL,
    fairgrad=FairGrad,
    excessmtl=ExcessMTL,
    dwa=DynamicWeightAverage,
    imtl=IMTLG,
    ls=LinearScalarization,
    uw=UncertaintyWeighting,
    pamal=PaMaL,
    palora=PaLoRA,
)

PFL_METHODS = dict(
    pamal=PaMaL,
    palora=PaLoRA,
)


def get_method(name, num_tasks, **kwargs) -> AlgoCallback:
    if name not in METHODS:
        allowed = ", ".join(BASELINE_METHODS)
        raise ValueError(f"Unknown method {name}. Supported baselines: {allowed}.")

    if name not in PARETO_BASELINE_METHODS:
        return METHODS[name](num_tasks, **kwargs)

    if name == "palora":
        inner_method = kwargs.get("inner_method")
        if inner_method == "ls":
            return PaLoRA(num_tasks, **kwargs)
        elif inner_method == "gb":
            return PaLoRA_GB(num_tasks, **kwargs)
        elif inner_method == "lb":
            return PaLoRA_LB(num_tasks, **kwargs)
        elif inner_method == "full":
            return PaLoRAFull(num_tasks, **kwargs)
        else:
            raise ValueError(f"Unknown inner method {inner_method}")

    if name == "pamal":
        inner_method = kwargs.get("inner_method")
        if inner_method == "ls":
            return PaMaL(num_tasks, **kwargs)
        elif inner_method in ("gb", "gradnorm"):
            return PaMaL_GB(num_tasks, **kwargs)
        elif inner_method == "lb":
            return PaMaL_LB(num_tasks, **kwargs)
        else:
            raise ValueError(f"Unknown inner method {inner_method}")

    raise ValueError(f"Unknown method {name}")


__all__ = [
    "METHODS",
    "PFL_METHODS",
    "BASELINE_METHODS",
    "NONPARETO_BASELINE_METHODS",
    "PARETO_BASELINE_METHODS",
    "AlgoCallback",
    "ParetoFrontApproximationAlgoCallback",
    "CAGrad",
    "NashMTL",
    "FairGrad",
    "ExcessMTL",
    "DynamicWeightAverage",
    "IMTLG",
    "LinearScalarization",
    "UncertaintyWeighting",
    "PaMaL",
    "PaMaL_LB",
    "PaMaL_GB",
    "PaLoRA",
    "PaLoRA_LB",
    "PaLoRA_GB",
    "PaLoRAFull",
    "get_method",
]
