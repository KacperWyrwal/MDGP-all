from torch import Tensor 


import torch 
import math 
from mdgp.utils import spherical_antiharmonic, spherical_harmonic, sphere_uniform_grid, rotate
from dataclasses import dataclass, field


__all__ = [
    'DataArguments',
    'get_target_function', 
    'get_data', 
    'smooth_target_function', 
    'singular_target_function', 
]


@dataclass
class DataArguments: 
    target_name: str = field(default='smooth', metadata={'help': 'Name of the target function. Must be one of ["smooth", "singular"]'})
    num_train: int = field(default=400, metadata={'help': 'Number of training points'})
    num_val: int = field(default=500, metadata={'help': 'Number of validation points'})
    num_test: int = field(default=2000, metadata={'help': 'Number of test points'})
    noise_std: float = field(default=0.01, metadata={'help': 'Standard deviation of the noise'})


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


def _get_data(n, target_fnc, noise_std=0.01):
    x = sphere_uniform_grid(n=n)
    f = target_fnc(x)
    y = f + torch.randn_like(f) * noise_std
    return x, y


def get_data(data_args: DataArguments) -> tuple[Tensor, Tensor]:
    target_fnc = get_target_function(name=data_args.target_name)
    train_inputs, train_targets = _get_data(n=data_args.num_train, target_fnc=target_fnc, noise_std=data_args.noise_std)
    val_inputs, val_targets = _get_data(n=data_args.num_val, target_fnc=target_fnc, noise_std=data_args.noise_std)
    test_inputs, test_targets = _get_data(n=data_args.num_test, target_fnc=target_fnc, noise_std=data_args.noise_std)
    return train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets
