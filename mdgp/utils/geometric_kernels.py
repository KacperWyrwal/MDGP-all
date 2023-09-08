import torch 
from geometric_kernels.spaces import Space, Hypersphere 
from geoopt import Manifold, Sphere


def extrinsic_dimension(space: Space) -> int: 
    if isinstance(space, Hypersphere): 
        return space.dim + 1
    raise NotImplementedError    


def space_to_manifold(space: Space) -> Manifold: 
    if isinstance(space, Hypersphere): 
        return Sphere(torch.eye(space.dim + 1))
    raise NotImplementedError
