from torch import Tensor 


import torch 
import math 
from mdgp.utils import spherical_antiharmonic, spherical_harmonic, sphere_uniform_grid, rotate
from dataclasses import dataclass, field
from torch.utils.data import TensorDataset



def smooth_target_function(x): 
    return spherical_harmonic(x, m=2, n=3)


def singular_target_function(x):
    return spherical_antiharmonic(x, m=1, n=2) + spherical_antiharmonic(rotate(x, roll=math.pi / 2), m=1, n=1)


def get_target_function(name='smooth'):
    if name == 'smooth':
        return smooth_target_function
    if name == 'singular':
        return singular_target_function
    raise ValueError(f"Unknown target function: {name}. Must be one of ['smooth', 'singular']")


def get_data(func, n, noise_std=0.01) -> tuple[Tensor, Tensor]:
    x = sphere_uniform_grid(n=n)
    f = func(x)
    y = f + torch.randn_like(f) * noise_std
    return x, y


class SphereDataset: 

    def __init__(self, name: str, num_train: int, num_test: int, noise_std: float = 0.01):
        assert name in ['smooth', 'singular']
        self.name = name
        target_function = get_target_function(name=name)
        self.train_x, self.train_y = get_data(target_function, n=num_train, noise_std=noise_std)
        self.test_x, self.test_y = get_data(target_function, n=num_test, noise_std=noise_std)

    @property
    def train_dataset(self) -> TensorDataset:
        return TensorDataset(self.train_x, self.train_y)

    @property 
    def test_dataset(self) -> TensorDataset:
        return TensorDataset(self.test_x, self.test_y)


@dataclass
class DataArguments: 
    dataset_name: str = field(default='smooth', metadata={'help': 'Name of the target function. Must be one of ["smooth", "singular"]'})
    num_train: int = field(default=400, metadata={'help': 'Number of training points'})
    num_test: int = field(default=2000, metadata={'help': 'Number of test points'})

    @property
    def dataset(self) -> SphereDataset:
        return SphereDataset(name=self.dataset_name, num_train=self.num_train, num_test=self.num_test)
