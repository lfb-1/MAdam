# adapted from https://github.com/lorenmt/auto-lambda/blob/main/create_dataset.py
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------
# Define VGG-16 (for CIFAR-100 experiments)
# --------------------------------------------------------------------------------
class ConditionalBatchNorm2d(nn.Module):
    fast_eval_mode = None

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_tasks = num_classes
        self.bn_list = nn.ModuleList()

        for i in range(num_classes):
            self.bn_list.append(nn.BatchNorm2d(num_features))

    def forward(self, x: torch.Tensor):
        if self.fast_eval_mode is not None:
            assert self.fast_eval_mode in range(self.num_tasks)
            return self.bn_list[self.fast_eval_mode](x)
        batch_size = len(x) // self.num_tasks
        x = x.split(batch_size, dim=0)
        out = [self.bn_list[task_id](xx) for task_id, xx in enumerate(x)]
        out = torch.cat(out, dim=0)
        return out


class MTLVGG16(nn.Module):
    fast_eval_mode = None

    def set_fast_eval_mode(self, mode):
        assert mode in range(self.num_tasks)
        self.fast_eval_mode = mode
        for m in self.modules():
            if isinstance(m, ConditionalBatchNorm2d):
                m.fast_eval_mode = mode

    def __init__(self, task_names):
        super(MTLVGG16, self).__init__()
        filter = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
        ]
        self.num_tasks = len(task_names)

        # define VGG-16 block
        network_layers = []
        channel_in = 3
        for ch in filter:
            if ch == "M":
                network_layers += [nn.MaxPool2d(2, 2)]
            else:
                network_layers += [
                    nn.Conv2d(channel_in, ch, kernel_size=3, padding=1),
                    ConditionalBatchNorm2d(ch, self.num_tasks),
                    nn.ReLU(inplace=True),
                ]
                channel_in = ch

        self.encoder = nn.Sequential(*network_layers)

        # define classifiers here
        self.decoders = nn.ModuleDict()
        for t in task_names:
            self.decoders.update({t: nn.Sequential(nn.Linear(filter[-1], 5))})

    def forward(self, x, return_embedding=False):
        batch_size = len(x) // self.num_tasks

        x = self.encoder(x)

        embedding = F.adaptive_avg_pool2d(x, 1)
        if self.fast_eval_mode is not None:
            assert self.fast_eval_mode in range(self.num_tasks)
            k = list(self.decoders.keys())[self.fast_eval_mode]
            pred = {k: self.decoders[k](embedding.view(embedding.shape[0], -1))}
            return pred, embedding
        embedding = embedding.split(batch_size, dim=0)
        pred = {k: self.decoders[k](xx.view(xx.shape[0], -1)) for k, xx in zip(self.decoders.keys(), embedding)}
        if return_embedding:
            return pred, embedding
        return pred

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.encoder.parameters()

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.decoders.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.encoder[-2].parameters()
