from typing import Tuple
from torch import Tensor

import math 
import torch 
from scipy.special import sph_harm
from coclust.clustering import SphericalKmeans


def sphere_kmeans_centers(x, k=5): 
    return torch.from_numpy(SphericalKmeans(n_clusters=k).fit(x.detach().numpy()).centers.toarray())


def sph_to_cart(theta: Tensor, phi: Tensor) -> Tuple[Tensor, Tensor, Tensor]: 
    x = theta.cos() * phi.sin()
    y = theta.sin() * phi.sin()
    z = phi.cos()
    return x, y, z


def cart_to_sph(x: Tensor) -> Tuple[Tensor, Tensor]:
    theta = torch.atan2(x[..., 1], x[..., 0])
    phi = torch.acos(x[..., 2])
    return theta, phi


def sphere_meshgrid(num_theta: int, num_phi: int, eps=0.0) -> Tuple[Tensor, Tensor, Tensor]:
    theta = torch.linspace(0, 2 * math.pi, num_theta)
    phi = torch.linspace(eps, math.pi - eps, num_phi)
    theta, phi = torch.meshgrid(theta, phi, indexing='xy')
    return torch.stack(sph_to_cart(theta, phi), dim=-1)


def sphere_uniform_grid(n: int) -> Tensor:
    indices = torch.arange(0, n) + 0.5
    phi = torch.acos(1 - 2 * indices / n)
    const = (1 + 5 ** 0.5)  # golden ratio makes for a fairly uniform distribution
    theta = math.pi * const * indices
    return torch.stack(sph_to_cart(theta, phi), dim=-1)


def spherical_harmonic(x: Tensor, m: int, n: int):
    theta, phi = cart_to_sph(x)
    return sph_harm(m, n, theta, phi).real.to(torch.get_default_dtype())


def spherical_antiharmonic(x: Tensor, m: int, n: int) -> Tensor:
    theta, phi = cart_to_sph(x)
    return sph_harm(m, n, phi, theta).real.to(torch.get_default_dtype())


def sphere_random_uniform(*sample_shape):
    sample = torch.randn(*sample_shape)
    return sample / sample.norm(dim=-1, keepdim=True)


def rotation_matrix(roll=0.0, pitch=0.0, yaw=0.0) -> Tensor:
    cos_roll, sin_roll = math.cos(roll), math.sin(roll)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

    rotation_matrix = torch.tensor([
        [cos_pitch*cos_yaw, -cos_roll*sin_yaw + sin_roll*sin_pitch*cos_yaw, sin_roll*sin_yaw + cos_roll*sin_pitch*cos_yaw],
        [cos_pitch*sin_yaw, cos_roll*cos_yaw + sin_roll*sin_pitch*sin_yaw, -sin_roll*cos_yaw + cos_roll*sin_pitch*sin_yaw],
        [-sin_pitch, sin_roll*cos_pitch, cos_roll*cos_pitch]
    ])
    
    return rotation_matrix


def rotate(x: Tensor, roll=0.0, pitch=0.0, yaw=0.0) -> Tensor: 
    return torch.einsum('nm, ...m -> ...n', rotation_matrix(roll=roll, pitch=pitch, yaw=yaw), x)


def spherical_distance(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the spherical distance between two points on a unit sphere given their Cartesian coordinates.

    :param a: PyTorch tensor, shape [..., 3]
    :param b: PyTorch tensor, shape [..., 3]
    :return: spherical distance between points in a and b, shape [...]
    """
    # Calculate the dot product along the last dimension
    dot_product = torch.einsum('...m, ...m -> ...', a, b)

    # Ensure the dot_product values are in the range [-1, 1] to avoid numerical issues
    dot_product = torch.clamp(dot_product, -1, 1)

    # Calculate the angle between the vectors
    angle = torch.acos(dot_product)

    return angle
