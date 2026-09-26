import dataclasses
import logging
import math

import meshzoo
import torch


@dataclasses.dataclass
class Sampler:
    num_tasks: int
    num: int

    def sample(self, t: float) -> torch.Tensor:
        raise NotImplementedError


@dataclasses.dataclass
class AnnealingSampler(Sampler):
    mul: float
    epsilon: float

    def __post_init__(self):
        if self.num_tasks == 3:
            num = (-1 + math.sqrt(-1 + 4 * 2 * self.num)) / 2
            self.num = int(num)
            actual_num_points = int((self.num + 1) * (self.num + 2) / 2)
            logging.info(f"Adjusted number of points to {self.num}. We will sample {actual_num_points} points.")
            self.num = actual_num_points

    def get_evenly_spaced_points_2d(self, num_points: int) -> torch.Tensor:
        points = torch.linspace(self.epsilon, 1 - self.epsilon, num_points)
        return torch.stack([points, points.flip(dims=(0,))]).T

    def get_evenly_spaced_points_3d(self, num_points: int) -> torch.Tensor:
        points, _ = meshzoo.triangle(num_points)
        return torch.tensor(points.T)

    def sample(self, t: float) -> torch.Tensor:
        assert self.num > 1, "Number of points should be greater than 1."
        if self.num_tasks == 2:
            rays = self.get_evenly_spaced_points_2d(self.num)
        elif self.num_tasks == 3:
            rays = self.get_evenly_spaced_points_3d(self.num)
        else:
            raise NotImplementedError

        rays = rays ** (self.mul * t)
        rays = rays / rays.sum(dim=1, keepdim=True)
        return rays.squeeze()


@dataclasses.dataclass
class FixedSampler(Sampler):
    mul: float
    epsilon: float

    def __post_init__(self):
        if self.num_tasks == 3:
            num = (-1 + math.sqrt(-1 + 4 * 2 * self.num)) / 2
            self.num = int(num)
            actual_num_points = int((self.num + 1) * (self.num + 2) / 2)
            logging.info(f"Adjusted number of points to {self.num}. We will sample {actual_num_points} points.")
            self.num = actual_num_points

    def get_evenly_spaced_points_2d(self, num_points: int) -> torch.Tensor:
        points = torch.linspace(self.epsilon, 1 - self.epsilon, num_points)
        return torch.stack([points, points.flip(dims=(0,))]).T

    def get_evenly_spaced_points_3d(self, num_points: int) -> torch.Tensor:
        points, _ = meshzoo.triangle(num_points)
        return torch.tensor(points.T)

    def sample(self, t: float) -> torch.Tensor:
        assert self.num > 1, "Number of points should be greater than 1."
        if self.num_tasks == 2:
            rays = self.get_evenly_spaced_points_2d(self.num)
        elif self.num_tasks == 3:
            rays = self.get_evenly_spaced_points_3d(self.num)
        else:
            raise NotImplementedError

        rays = rays / rays.sum(dim=1, keepdim=True)
        return rays.squeeze()


@dataclasses.dataclass
class DirichletSampler(Sampler):
    p: float

    def sample(self, t: float = None) -> torch.Tensor:
        if self.num == 1:
            return torch.distributions.dirichlet.Dirichlet(torch.ones(self.num_tasks) * self.p).sample()
        return torch.distributions.dirichlet.Dirichlet(torch.ones(self.num_tasks) * self.p).sample((self.num,))


@dataclasses.dataclass
class AnnealingDirichletSampler(Sampler):
    p: float

    def sample(self, t: float = None) -> torch.Tensor:
        p = self.p * (1 - t) + 0.001
        if self.num == 1:
            return torch.distributions.dirichlet.Dirichlet(torch.ones(self.num_tasks) * p).sample()
        return torch.distributions.dirichlet.Dirichlet(torch.ones(self.num_tasks) * p).sample((self.num,))
