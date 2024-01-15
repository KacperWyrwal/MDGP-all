import torch 
from torch import Tensor
from geometric_kernels.spaces import Space, Hypersphere, DiscreteSpectrumSpace, Euclidean
from mdgp.utils import sphere_uniform_grid, sphere_kmeans_centers
from scipy.cluster.vq import kmeans2


def initialize_grid(space: DiscreteSpectrumSpace, n: int) -> Tensor: 
    if not isinstance(space, Hypersphere) and not space.dim == 2: 
        raise NotImplementedError("Grid initialization is currently only implemented for the sphere")
    return sphere_uniform_grid(n)


def euclidean_kmeans(x: Tensor, k: int) -> Tensor: 
    return torch.tensor(kmeans2(x, k)[0])


def initialize_kmeans(x: Tensor, n: int, space: Space | None = None) -> Tensor: 
    # Currently we only have specialised kmeans for the sphere 
    # TODO we could implement a naive clustering on manifolds by projecting centers at each step
    if isinstance(space, Hypersphere) and space.dim == 2: 
        return sphere_kmeans_centers(x, n)
    # If no space given, default to Euclidean
    if space is None or isinstance(space, Euclidean): 
        return euclidean_kmeans(x, n)
    # Otherwise, naively project centers into the space
    return space.projection(euclidean_kmeans(x, n))
