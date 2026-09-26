import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
from torchvision import transforms

from src.datasets.base_data_module import BaseDataModule
from src.datasets.utils.data_download import _download, unzip
from src.datasets.utils.enums import TaskCategoriesEnum
from src.utils.variables_and_paths import DATA_DIR


class MultiTaskTensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, *tensors: torch.Tensor, task_names) -> None:
        super().__init__(*tensors)
        self.task_names = task_names

    def __getitem__(self, index):
        X, y = super().__getitem__(index)
        # y = tuple(yy for yy in y)
        y = dict(zip(self.task_names, y))
        return X, y


def get_data(dataroot, task_names):
    dataroot = os.path.join(DATA_DIR, "sarcos")
    path = Path(dataroot)

    # https://github.com/Kaixhin/SARCOS/blob/master/data.py
    assert path.exists(), f"Data not found at {path}. Please download the data first."
    assert os.path.exists(
        os.path.join(path, "sarcos_inv.mat")
    ), f"Data not found at {path}. Please download the data first."
    train_data = loadmat(os.path.join(path, "sarcos_inv.mat"))["sarcos_inv"].astype(np.float32)
    val_data, train_data = train_data[:4448], train_data[4448:]
    test_data = loadmat(os.path.join(path, "sarcos_inv_test.mat"))["sarcos_inv_test"].astype(np.float32)

    X_train, Y_train = train_data[:, :21], train_data[:, 21:]
    X_val, Y_val = val_data[:, :21], val_data[:, 21:]
    X_test, Y_test = test_data[:, :21], test_data[:, 21:]

    quant = np.quantile(Y_train, q=0.9, axis=0)
    Y_train /= quant
    Y_val /= quant
    Y_test /= quant
    logging.info(f"training examples: {len(X_train)}, validation {len(X_val)}, test {len(X_test)}")

    return (
        MultiTaskTensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train), task_names=task_names),
        MultiTaskTensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val), task_names=task_names),
        MultiTaskTensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test), task_names=task_names),
    )


class SarcosDataModule(BaseDataModule):

    num_features: int = 21

    def __init__(self, *args, **kwargs):
        kwargs["root"] = os.path.expanduser("~/data/sarcos")
        super().__init__(*args, **kwargs)
        self.root = os.path.expanduser("~/data/sarcos")

    @property
    def input_dims(self):
        return [self.num_features]

    @property
    def name(self) -> str:
        return "sarcos"

    @property
    def task_names(self) -> List[str]:
        return [f"Task {i+1}" for i in range(7)]

    @property
    def num_tasks(self):
        return 7

    @property
    def task_categories(self):
        return [TaskCategoriesEnum.REGRESSION] * 7

    def prepare_data(self, *args, **kwargs):
        """Prepares the data for the Census dataset."""
        self.train, self.valid, self.test = get_data(self.root, self.task_names)
