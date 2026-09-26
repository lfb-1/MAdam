import logging
import os
from pathlib import Path
from typing import Any, List, Tuple

import torchvision.transforms as transforms
from src.datasets.base_data_module import BaseDataModule
from src.datasets.utils.enums import TaskCategoriesEnum
from torchvision.datasets import CelebA
from torchvision.datasets.utils import download_file_from_google_drive

from torch.utils.data import Subset


class CelebaDataModule(BaseDataModule):
    """The data module for Celeb-A. Inherits from BaseDataModule. The dataset consists of 10,177 people, 202,599 images and 40 binary classification tasks.

    See more at https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
    """

    _celeba_task_names = [
        "0-5_o_Clock_Shadow",
        "1-Arched_Eyebrows",
        "2-Attractive",
        "3-Bags_Under_Eyes",
        "4-Bald",
        "5-Bangs",
        "6-Big_Lips",
        "7-Big_Nose",
        "8-Black_Hair",
        "9-Blond_Hair",
        "10-Blurry",
        "11-Brown_Hair",
        "12-Bushy_Eyebrows",
        "13-Chubby",
        "14-Double_Chin",
        "15-Eyeglasses",
        "16-Goatee",
        "17-Gray_Hair",
        "18-Heavy_Makeup",
        "19-High_Cheekbones",
        "20-Male",
        "21-Mouth_Slightly_Open",
        "22-Mustache",
        "23-Narrow_Eyes",
        "24-No_Beard",
        "25-Oval_Face",
        "26-Pale_Skin",
        "27-Pointy_Nose",
        "28-Receding_Hairline",
        "29-Rosy_Cheeks",
        "30-Sideburns",
        "31-Smiling",
        "32-Straight_Hair",
        "33-Wavy_Hair",
        "34-Wearing_Earrings",
        "35-Wearing_Hat",
        "36-Wearing_Lipstick",
        "37-Wearing_Necklace",
        "38-Wearing_Necktie",
        "39-Young",
    ]

    def __init__(
        self,
        root=None,
        num_tasks: int = 40,
        transform=transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()]),
        *args,
        **kwargs,
    ):
        """Inits the data module for the celeba dataset. Inherits from BaseDataModule. See more at the parent class.

        Args:
            transform (callable, optional): transform to be applied to the dataset. Defaults to transforms.ToTensor().
        """
        print(f"kwargs: {kwargs}")
        self.task_ids = kwargs.get("task_ids", None)
        print(self.task_ids)
        if root is None:
            root = Path.joinpath(Path.home(), "data")
        self.transform = transform
        self.num_tasks = num_tasks
        super().__init__(root=root, *args, **kwargs)

    @property
    def num_tasks(self):
        return self._num_tasks

    @num_tasks.setter
    def num_tasks(self, value):
        self._num_tasks = value
        if getattr(self, "train", None) is not None:
            self.train.num_tasks = value

        if getattr(self, "valid", None) is not None:
            self.valid.num_tasks = value

        if getattr(self, "test", None) is not None:
            self.test.num_tasks = value

    @property
    def name(self):
        return "celeba"

    @property
    def task_categories(self):
        return [TaskCategoriesEnum.BINARY_CLASSIFICATION] * self.num_tasks

    @property
    def task_names(self) -> List[str]:
        return self._celeba_task_names

    def prepare_data(self, *args, **kwargs):
        """Prepares the data for Celeb-A dataset. The dataset comes with predefined train/val/test splits."""
        kwargs = dict(root=self.root, num_tasks=self.num_tasks)
        self.train = TensorCeleba(split="train", **kwargs)
        self.valid = TensorCeleba(split="valid", **kwargs)
        self.test = TensorCeleba(split="test", **kwargs)

        # FOR DEBUGGING
        # self.train = Subset(self.train, list(range(1000)))
        # self.valid = Subset(self.valid, list(range(100)))
        # self.train = CustomCelebA(
        #     root=self.root,
        #     split="train",
        #     download=True,
        #     transform=self.transform,
        #     task_ids=self.task_ids,
        #     num_tasks=self.num_tasks,
        # )

        # self.valid = CustomCelebA(
        #     root=self.root,
        #     split="valid",
        #     download=True,
        #     transform=self.transform,
        #     task_ids=self.task_ids,
        #     num_tasks=self.num_tasks,
        # )

        # self.test = CustomCelebA(
        #     root=self.root,
        #     split="test",
        #     download=True,
        #     transform=self.transform,
        #     task_ids=self.task_ids,
        #     num_tasks=self.num_tasks,
        # )


import torch
from torch.utils.data import Dataset, TensorDataset


class TensorCeleba(Dataset):
    def __init__(self, root, num_tasks, split) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.num_tasks = num_tasks

        self.inputs: torch.Tensor = torch.load(Path.joinpath(Path(root), "celeba", f"{split}_inputs.pt"))
        self.targets = torch.load(Path.joinpath(Path(root), "celeba", f"{split}_targets.pt"))

    def __getitem__(self, index) -> Any:
        return self.inputs[index], dict(
            zip(
                CelebaDataModule._celeba_task_names[: self.num_tasks],
                self.targets[index][: self.num_tasks],
            )
        )

    def __len__(self):
        return len(self.targets)


class CustomCelebA(CelebA):
    """Wrapper class for CelebA. Essentially it replaces the google drive downloading from the default location to
    my personal drive. This is due to the quotas imposed by Google drive resulting in inability to download from the
    default location."""

    def __init__(self, *args, **kwargs) -> None:
        self.task_ids = kwargs.pop("task_ids", None)
        self.num_tasks = kwargs.pop("num_tasks", 40)
        super().__init__(*args, **kwargs)
        if self.task_ids is not None:
            logging.info(f"using task ids: {self.task_ids}")
            logging.info([a for i, a in enumerate(self.attr_names) if i in self.task_ids])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, target = super().__getitem__(index)
        target = tuple(tt for i, tt in enumerate(target) if i < self.num_tasks)

        if self.task_ids is not None:
            target = tuple(tt for i, tt in enumerate(target) if i in self.task_ids)

        return X, target

    def download(self) -> None:
        pass
        # self.__class__.download(self.root)

    def _check_integrity(self) -> bool:
        return True

    @staticmethod
    def _download(root):
        """Downloads the dataset from my personal Google drive.

        Args:
            root (str): where to store the downloaded files.
        """
        path = Path.joinpath(Path().resolve(), Path(root), "data")
        path = Path.joinpath(path, "celeba")

        if not path.exists():
            print(f"creating dirs for {str(path)}")
            path.mkdir(parents=True, exist_ok=True)

        base_url = "https://drive.google.com/uc?id="
        ids = {
            "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS": "identity_celebA.txt",
            "1BcfJ3AN4NK_E7to3-ZRO_zSCHMUa3EhQ": "list_attr_celeba.txt",
            "1dhtLE_2v2ELQ0ItUQpU-EoIWyJ91t4OI": "list_bbox_celeba.txt",
            "1yVG9OPmOv8jaAbfy4NZdKmM290K026xm": "list_landmarks_align_celeba.txt",
            "13r-ohk_4QfQHA4eQjdeGAJVhrmQD3ubr": "list_landmarks_celeba.txt",
            "1hzynUBVUQWgpEAZXsp3JKflwlEv0p5Ys": "list_eval_partition.txt",
            "19yhnQFuJgnlPQXsYT3TarPLv67agtvF1": "img_align_celeba.zip",
        }

        for file_id, name in ids.items():
            file_output = Path.joinpath(path, name)

            if not os.path.exists(file_output):
                file_url = "%s%s" % (base_url, file_id)
                download_file_from_google_drive(file_id=file_id, root=path, filename=name)
