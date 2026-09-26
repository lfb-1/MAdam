from .lenet import MultiLeNetO, MultiLeNetR
from .mlp import MultiTaskMLP, SimpleMLP, MultiTaskMLPDifferentNumClasses
from .resnet import ResNetEncoder, UtkFaceResnet, CelebAResnet
from .segnet_cityscapes import SegNet, SegNetMtan
from .vgg16 import MTLVGG16

__all__ = [
    "MultiLeNetO",
    "MultiLeNetR",
    "MultiTaskMLP",
    "SimpleMLP",
    "MultiTaskMLPDifferentNumClasses",
    "ResNetEncoder",
    "UtkFaceResnet",
    "CelebAResnet",
    "SegNet",
    "SegNetMtan",
    "MTLVGG16",
]
