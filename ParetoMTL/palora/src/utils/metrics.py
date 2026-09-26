from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor
from torchmetrics import JaccardIndex


class MaskedL1Metric(torchmetrics.MeanAbsoluteError):
    def __init__(self, ignore_index, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            target = target.unsqueeze(1)
        mask = target != self.ignore_index
        return super().update(torch.masked_select(preds, mask), torch.masked_select(target, mask))


class CrossEntropyLossMetric(torchmetrics.MeanMetric):
    def __init__(self, ignore_index=-100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index

    def update(self, preds, target) -> None:
        value = F.cross_entropy(input=preds, target=target, ignore_index=self.ignore_index)
        return super().update(value)


class HuberLossMetric(torchmetrics.MeanMetric):
    def update(self, preds, target) -> None:
        value = F.huber_loss(input=preds.squeeze(), target=target, delta=1)
        return super().update(value)


class DummyMetric(torchmetrics.MaxMetric):
    def update(self, value, *args, **kwargs) -> None:
        return super().update(123)


class ModifiedJaccardIndex(JaccardIndex):
    def __init__(self):
        super().__init__(compute_on_step=False, num_classes=19, ignore_index=250)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        mask = target != self.ignore_index
        preds = preds.argmax(1)
        return super().update(torch.masked_select(preds, mask), torch.masked_select(target, mask))


def compute_hv(num_tasks: int, task_names: List[str], results: Dict[str, float]):
    import numpy as np
    from pymoo.indicators.hv import HV

    assert num_tasks == 2, "only 2 tasks are supported for now"

    def get_metric(result: dict, id):
        return [v for k, v in result.items() if task_names[id] in k and "loss" not in k][0]

    def get_all_metrics(result, num_tasks):
        return [get_metric(result, id) for id in range(num_tasks)]

    results = [get_all_metrics(a, 2) for a in results.values()]

    ref_point = np.array([0, 0])
    A = -np.array(results)
    ind = HV(ref_point=ref_point)
    return ind(A)
