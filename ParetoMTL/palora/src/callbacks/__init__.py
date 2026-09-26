from .methods import METHODS
from .metrics.cityscapes_metric_cb import CityscapesMetricCallback
from .metrics.mtl_metric_callback import (
    BinaryClassificationMetricCallback,
    ClassificationMultiTaskMetricCallback,
    MultiTaskMetricCallback,
    UTKFaceMultiTaskMetricCallback,
)
from .metrics.nyu_metric_cb import NYUMetricCallback
from .utils.console_logger_callback import ConsoleLoggerCallback
from .utils.count_params_callback import CountParametersCallback
from .utils.pareto_front_visualizer import ParetoFrontVisualizerCallback
from .utils.save_model import SaveModelCallback
from .utils.scheduler_callback import SchedulerCallback
from .utils.timer_callback import TimerCallback
from .utils.tqdm_callback import TqdmCallback


def get_default_callbacks(logging_frequency=3):
    return [
        ConsoleLoggerCallback(logging_frequency=logging_frequency),
        TimerCallback(),
        CountParametersCallback(),
    ]


def get_verbose_callbacks():
    return [TqdmCallback(), TimerCallback(), CountParametersCallback()]


__all__ = [
    "SaveModelCallback",
    "CityscapesMetricCallback",
    "CountParametersCallback",
    "ConsoleLoggerCallback",
    "MultiTaskMetricCallback",
    "UTKFaceMultiTaskMetricCallback",
    "ClassificationMultiTaskMetricCallback",
    "BinaryClassificationMetricCallback",
    "ParetoFrontVisualizerCallback",
    "SchedulerCallback",
    "TimerCallback",
    "TqdmCallback",
    "NYUMetricCallback",
    "METHODS",
    "get_default_callbacks",
    "get_verbose_callbacks",
]
