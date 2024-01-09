from pymanopt.manifolds.manifold import Manifold 


import torch 
from mdgp.bo_experiment.data.target_functions import Ackley, Levy, StyblinskiTang, ProductOfSines, DGPSample, PermutedSphericalHarmonic
from mdgp.bo_experiment.utils import manifold_class_from_name, ExcludeFromNameMixin
from mdgp.utils import sphere_uniform_grid
from dataclasses import dataclass, field


def target_function_class_from_name(name): 
    if name == "ackley": 
        return Ackley
    if name == "levy": 
        return Levy
    if name == "styblinski_tang":
        return StyblinskiTang
    if name == "product_of_sines":
        return ProductOfSines
    if name == 'dgp_sample': 
        return DGPSample
    if name == 'perm_sph_harm':
        return PermutedSphericalHarmonic
    raise ValueError(f"Unknown target function {name}")


@dataclass
class DataArguments(ExcludeFromNameMixin): 
    target_function_name: str = field(default="ackley", metadata={"help": "Target function to optimize"})
    num_initial_data: int = field(default=5, metadata={"help": "Number of initial data points"})
    initial_data_method: str = field(default="random", metadata={"help": "Method to generate initial data"})
    manifold_name: str = field(default="hypersphere", metadata={"help": "Manifold to optimize on"})
    manifold_dim: int = field(default=2, metadata={"help": "Dimension of the manifold"})

    def __post_init__(self): 
        super().__post_init__()
        self.exclude_from_name(["manifold_name", "manifold_dim"])

    @property
    def target_function(self): 
        return target_function_class_from_name(self.target_function_name)(self.manifold)
    
    @property
    def manifold(self) -> Manifold: 
        return manifold_class_from_name(self.manifold_name)(self.manifold_dim + 1)


def get_initial_data(data_args: DataArguments): 
    if data_args.initial_data_method == "random": 
        return torch.tensor([data_args.manifold.random_point() for _ in range(data_args.num_initial_data)])
    if data_args.initial_data_method == "grid" and data_args.manifold_name == "hypersphere" and data_args.manifold_dim == 2: 
        return sphere_uniform_grid(data_args.num_initial_data)
        
    raise ValueError(f"Unknown initial data method {data_args.initial_data_method}")
