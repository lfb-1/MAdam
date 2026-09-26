# source: https://github.com/lorenmt/auto-lambda/blob/main/create_dataset.py

import fnmatch
import logging
import os
import random
from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from PIL import Image
from torch.utils.data import Subset
from src.utils.variables_and_paths import DATA_DIR

from src.datasets.base_data_module import BaseDataModule

ANTIALIAS = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DataTransform(object):
    def __init__(self, scales, crop_size, is_disparity=False):
        self.scales = scales
        self.crop_size = crop_size
        self.is_disparity = is_disparity

    def __call__(self, data_dict):
        if type(self.scales) == tuple:
            # Continuous range of scales
            sc = np.random.uniform(*self.scales)

        elif type(self.scales) == list:
            # Fixed range of scales
            sc = random.sample(self.scales, 1)[0]

        raw_h, raw_w = data_dict["im"].shape[-2:]
        resized_size = [int(raw_h * sc), int(raw_w * sc)]
        i, j, h, w = 0, 0, 0, 0  # initialise cropping coordinates
        flip_prop = random.random()

        for task in data_dict:
            if len(data_dict[task].shape) == 2:  # make sure single-channel labels are in the same size [H, W, 1]
                data_dict[task] = data_dict[task].unsqueeze(0)

            # Resize based on randomly sampled scale
            if task in ["im", "noise"]:
                data_dict[task] = transforms_f.resize(
                    data_dict[task], resized_size, Image.BILINEAR, antialias=ANTIALIAS
                )
            elif task in ["normal", "depth", "sem", "part_seg", "disp"]:
                data_dict[task] = transforms_f.resize(
                    data_dict[task], resized_size, Image.NEAREST, antialias=ANTIALIAS
                )

            # Add padding if crop size is smaller than the resized size
            if self.crop_size[0] > resized_size[0] or self.crop_size[1] > resized_size[1]:
                right_pad, bottom_pad = max(self.crop_size[1] - resized_size[1], 0), max(
                    self.crop_size[0] - resized_size[0], 0
                )
                if task in ["im"]:
                    data_dict[task] = transforms_f.pad(
                        data_dict[task],
                        padding=(0, 0, right_pad, bottom_pad),
                        padding_mode="reflect",
                    )
                elif task in ["sem", "part_seg", "disp"]:
                    data_dict[task] = transforms_f.pad(
                        data_dict[task],
                        padding=(0, 0, right_pad, bottom_pad),
                        fill=-1,
                        padding_mode="constant",
                    )  # -1 will be ignored in loss
                elif task in ["normal", "depth", "noise"]:
                    data_dict[task] = transforms_f.pad(
                        data_dict[task],
                        padding=(0, 0, right_pad, bottom_pad),
                        fill=0,
                        padding_mode="constant",
                    )  # 0 will be ignored in loss

            # Random Cropping
            if i + j + h + w == 0:  # only run once
                i, j, h, w = transforms.RandomCrop.get_params(data_dict[task], output_size=self.crop_size)
            data_dict[task] = transforms_f.crop(data_dict[task], i, j, h, w)

            # Random Flip
            if flip_prop > 0.5:
                data_dict[task] = torch.flip(data_dict[task], dims=[2])
                if task == "normal":
                    data_dict[task][0, :, :] = -data_dict[task][0, :, :]

            # Final Check:
            if task == "depth":
                data_dict[task] = data_dict[task] / sc

            if task == "disp":  # disparity is inverse depth
                data_dict[task] = data_dict[task] * sc

            if task in ["sem", "part_seg"]:
                data_dict[task] = data_dict[task].squeeze(0)
        return data_dict


class NYUv2(data.Dataset):
    """
    NYUv2 dataset, 3 tasks + 1 generated useless task
    Included tasks:
        1. Semantic Segmentation,
        2. Depth prediction,
        3. Surface Normal prediction,
        4. Noise prediction [to test auxiliary learning, purely conflict gradients]
    """

    def __init__(self, root, train=True, augmentation=False, use_noise=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.use_noise = use_noise

        # read the data file
        if train:
            self.data_path = root + "/train"
        else:
            self.data_path = root + "/val"

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + "/image"), "*.npy"))
        self.noise = torch.rand(self.data_len, 1, 288, 384)

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + "/image/{:d}.npy".format(index)), -1, 0)).float()
        semantic = torch.from_numpy(np.load(self.data_path + "/label/{:d}.npy".format(index))).long()
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + "/depth/{:d}.npy".format(index)), -1, 0)).float()
        normal = torch.from_numpy(
            np.moveaxis(np.load(self.data_path + "/normal/{:d}.npy".format(index)), -1, 0)
        ).float()

        data_dict = {"im": image, "sem": semantic, "depth": depth, "normal": normal}

        if self.use_noise:
            noise = self.noise[index].float()
            data_dict["noise"] = noise

        # apply data augmentation if required
        if self.augmentation:
            # print(f"{self.train} performs augmentation")
            data_dict = DataTransform(crop_size=[288, 384], scales=[1.0, 1.2, 1.5])(data_dict)

        im = 2.0 * data_dict.pop("im") - 1.0  # normalised to [-1, 1]
        return im, data_dict

    def __len__(self):
        return self.data_len

    @staticmethod
    def __download__(root):
        import zipfile
        from pathlib import Path

        import gdown

        if not os.path.exists(root):
            Path(root).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(Path(root, "train/").as_posix()):
            print("Downloading NYUv2 dataset: train split")
            id = "13p3SoEgLwlArC5BR9PiI3Y-1bMrk5CHA"
            gdown.download(id=id, output=Path(root, "train.zip").as_posix(), quiet=False)

            print("Extracting NYUv2 dataset: train split")
            with zipfile.ZipFile(Path(root, "train.zip").as_posix(), "r") as zip_ref:
                zip_ref.extractall(Path(root, "train/").as_posix())
        else:
            print("Train split already downloaded and extracted.")

        if not os.path.exists(Path(root, "val/").as_posix()):
            print("Downloading NYUv2 dataset: val split")
            id = "1Knhb8MOYTYxQyABjW1_0H81Mpg22tkKh"
            gdown.download(id=id, output=Path(root, "val.zip").as_posix(), quiet=False)

            print("Extracting NYUv2 dataset: val split")
            with zipfile.ZipFile(Path(root, "val.zip").as_posix(), "r") as zip_ref:
                zip_ref.extractall(Path(root, "val/").as_posix())
        else:
            print("Val split already downloaded and extracted.")

        print("Finished downloading and extracting NYUv2 dataset.")


class NYUv2DataModule(BaseDataModule):
    num_tasks = 3
    task_names = ["sem", "depth", "normal"]
    name = "NYUv2"
    input_dims = [3, 288, 384]

    def __init__(
        self,
        root: str = DATA_DIR,
        batch_size: int = 512,
        valid_batch_size: int | None = None,
        num_workers: int = 30,
        shuffle: bool = True,
        pin_memory: bool = True,
    ):
        if not root.endswith("nyu"):
            root = os.path.join(root, "nyu")

        super().__init__(root, batch_size, valid_batch_size, num_workers, shuffle, pin_memory)

    def prepare_data(self) -> None:
        dataset = NYUv2(root=self.root, train=True, augmentation=True)

        # we deepcopy the dataset instead of using a direct subset so that the two datasets do not share the same
        # underlying object. Then we can turn off the augmentation for the validation set only.
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(795))
        self.train, self.valid = Subset(deepcopy(dataset), indices[:700]), Subset(deepcopy(dataset), indices[700:])
        self.valid.dataset.augmentation = False

        self.test = NYUv2(root=self.root, train=False)

        print(f"train length: {len(self.train)}")
        print(f"val length: {len(self.valid)}")
        print(f"test length: {len(self.test)}")

        assert self.train.dataset.augmentation == True
        assert self.valid.dataset.augmentation == False
        assert self.test.augmentation == False

    @staticmethod
    def __download__(root):
        NYUv2.__download__(root)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        dl = torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker,
            drop_last=True,  # to resolve an error when batch_size=1
            # generator=g,
        )
        logging.info(f"TRAIN_DL: batch_size={self.batch_size}, num_workers={self.num_workers}")
        return dl
