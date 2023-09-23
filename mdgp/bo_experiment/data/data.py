from geometric_kernels.spaces import DiscreteSpectrumSpace 


import torch 
from mdgp.bo_experiment.data.target_functions import Ackley, Levy, StyblinskiTang, ProductOfSines
from geometric_kernels.spaces import Hypersphere
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
    raise ValueError(f"Unknown target function {name}")


def space_class_from_name(name):
    if name == "hypersphere": 
        return Hypersphere
    raise ValueError(f"Unknown space {name}")


@dataclass
class DataArguments: 
    target_function_name: str = field(default="ackley", metadata={"help": "Target function to optimize"})
    num_initial_data: int = field(default=10, metadata={"help": "Number of initial data points"})
    initial_data_method: str = field(default="random", metadata={"help": "Method to generate initial data"})
    space_name: str = field(default="hypersphere", metadata={"help": "Manifold to optimize on"})
    space_dim: int = field(default=2, metadata={"help": "Dimension of the manifold"})

    @property
    def target_function(self): 
        return target_function_class_from_name(self.target_function_name)(self.space)
    
    @property
    def space(self) -> DiscreteSpectrumSpace: 
        return space_class_from_name(self.space_name)(self.space_dim)


def get_initial_data(args: DataArguments): 
    if args.initial_data_method == "random": 
        x = torch.tensor([args.space.random() for _ in range(args.num_initial_data)])
        return args.target_function(x)
    raise ValueError(f"Unknown initial data method {args.initial_data_method}")
