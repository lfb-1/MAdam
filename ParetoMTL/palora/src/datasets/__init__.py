from src.datasets.base_data_module import BaseDataModule
from src.datasets.celeba import CelebaDataModule
from src.datasets.census import (
    CensusDataModule,
    LinCensusDataModule,
    MaCensusDataModule,
    TestCensusDataModule,
)
from src.datasets.multimnist import MultiMnistDataModule
from src.datasets.multimnist3digits import MultiMnistThreeDataModule
from src.datasets.utkface import UTKFaceDataModule
from src.datasets.cityscapes2 import Cityscapes2SplitDataModule, Cityscapes2DataModule
from src.datasets.nyuv2 import NYUv2DataModule

__all__ = [
    "BaseDataModule",
    "CelebaDataModule",
    "CensusDataModule",
    "LinCensusDataModule",
    "MaCensusDataModule",
    "TestCensusDataModule",
    "MultiMnistDataModule",
    "MultiMnistThreeDataModule",
    "MultiFashionDataModule",
    "UTKFaceDataModule",
    "Cityscapes2SplitDataModule",
    "Cityscapes2DataModule",
    "NYUv2DataModule",
]
