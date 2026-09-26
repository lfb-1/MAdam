import logging

import torch
import wandb

from src.callbacks.callback import Callback
from src.trainer.base_trainer import BaseTrainer

# mnist encoder: MultiLeNetR(in_channels=1)
# mnist decoder: MultiLeNetO()
# MNIST_COUNT = 24390 # (uses STL model)
MNIST_COUNT = 27450


MULTIMNIST3_COUNT = 30510

# census encoder: SimpleMLP(in_features=468, hidden_features=[256], remove_last_activation=False)
# census encoder: SimpleMLP(in_features=256, hidden_features=[2])
# CENSUS_COUNT = 120578 # (use?s STL model)
CENSUS_COUNT = 121092

# celeba encoder: ResNetEncoder(block=BasicBlock, num_blocks=[2, 2, 2, 2])
# celeba decoder: FaceAttributeDecoder()
# CELEBA_COUNT = 11172930 # (uses STL model)
CELEBA_COUNT = 11332752

# FILTER = [64, 128, 256, 512, 512]
# cityscapes encoder: SegNetSplitEncoder(in_channels=3)
# cityscapes decoder: SegNetSegmentationDecoder(filters=FILTER)
# WARNING: the two decoders are not the same,
# - the first one is for the segmentation task (37383 parameters),
# - the second one is for the depth task (36993 parameters)
# The difference is small, for convenience we use the segmentation decoder.
# CITYSCAPES_COUNT = 24980679 # (uses STL model)
CITYSCAPES_COUNT = 25017672

# For Segnet
NYU_COUNT = 25055185


PARAMETER_COUNTS = {
    "multimnist": MNIST_COUNT,
    "multimnist3": MULTIMNIST3_COUNT,
    "census_age_education": CENSUS_COUNT,
    "celeba": CELEBA_COUNT,
    "cityscapes": CITYSCAPES_COUNT,
    "cityscapes2": CITYSCAPES_COUNT,
    "Cityscapes2": CITYSCAPES_COUNT,
    "UTKFace": 11185224,
    "NYUv2": NYU_COUNT,
    "sarcos": 139015,
}


def count_parameters_from_optimizer(optimizer: torch.optim.Optimizer):
    def single_param_group_parameter_count(param_group):
        return sum(p.numel() for p in param_group["params"] if p.requires_grad)

    return sum(single_param_group_parameter_count(param_group) for param_group in optimizer.param_groups)


class CountParametersCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_before_fit(self, trainer: BaseTrainer, *args, **kwargs):
        # we count the parameters from the optimizer because the method might have some parameters (e.g., UW)
        num_parameters = count_parameters_from_optimizer(trainer.optimizer)
        if trainer.benchmark.name not in PARAMETER_COUNTS:
            print(sum(p.numel() for p in trainer.model.parameters()))
        stl_parameters = PARAMETER_COUNTS[trainer.benchmark.name]
        ratio = num_parameters / stl_parameters
        ratio_increase = ratio - 1

        logging.info(f"Number of trainable parameters: {num_parameters}")
        logging.info(f"This model has {ratio_increase*100:.2f}% more trainable parameters than the vanilla MTL model")
        if wandb.run is not None:
            wandb.config.update(
                {
                    "num_parameters": num_parameters,
                    "ratio_increase": ratio_increase,
                    "ratio": ratio,
                },
                allow_val_change=True,
            )
