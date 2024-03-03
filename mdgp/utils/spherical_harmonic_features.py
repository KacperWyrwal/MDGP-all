from torch import Tensor 

import torch 
from math import comb 
from linear_operator.operators import DiagLinearOperator
from spherical_harmonics import SphericalHarmonics


def num_harmonics_single(ell: int, d: int) -> int:
    r"""
    Number of spherical harmonics of degree ell on S^{d - 1}.
    """
    if ell == 0:
        return 1
    if d == 3:
        return 2 * ell + 1
    else:
        return (2 * ell + d - 2) * comb(ell + d - 3, ell - 1) // ell


def num_harmonics(ell: Tensor | int, d: int) -> Tensor:
    """
    Vectorized version of num_harmonics_single
    """
    if isinstance(ell, int):
        return num_harmonics_single(ell, d)
    return ell.apply_(lambda e: num_harmonics_single(ell=e, d=d))


def total_num_harmonics(max_ell: int, d: int) -> int:
    """
    Total number of spherical harmonics on S^{d-1} with degree <= max_ell
    """
    return int(sum(num_harmonics(ell=torch.arange(max_ell + 1), d=d)))


def eigenvalue_laplacian(ell: Tensor, d: int) -> Tensor:
    """
    Eigenvalue of the Laplace-Beltrami operator for a spherical harmonic of degree ell on S_{d-1}
    ell: [...]
    d: []
    return: [...]
    """
    return ell * (ell + d - 2)


def unnormalized_matern_spectral_density(n: Tensor, d: int, kappa: Tensor | float, nu: Tensor | float) -> Tensor | float: 
    """
    compute (unnormalized) spectral density of the matern kernel on S_{d-1}
    n: [N]
    d: []
    kappa: [O, 1, 1]
    nu: [O, 1, 1]
    return: [O, 1, N]
    """
    # Squared exponential kernel 
    if torch.all(nu.isinf()):
        exponent = -kappa ** 2 / 2 * eigenvalue_laplacian(ell=n, d=d) # [O, N, 1]
        return torch.exp(exponent)
    # Matern kernel
    else:
        base = (
            2.0 * nu / kappa**2 + # [O, 1, 1]
            eigenvalue_laplacian(ell=n, d=d).unsqueeze(-1) # [N, 1]
        ) # [O, N, 1]
        exponent = -nu - (d - 1) / 2.0 # [O, 1, 1]
        return base ** exponent # [O, N, 1]


def matern_spectral_density_normalizer(d: int, max_ell: int, kappa: Tensor | float, nu: Tensor | float) -> Tensor:
    """
    Normalizing constant for the spectral density of the Matern kernel on S^{d-1}. 
    Depends on kappa and nu. Also depends on max_ell, as truncation of the infinite 
    sum from Karhunen-Loeve decomposition. 
    """
    n = torch.arange(max_ell + 1)
    spectral_values = unnormalized_matern_spectral_density(n=n, d=d, kappa=kappa, nu=nu) # [O, max_ell + 1, 1]
    num_harmonics_per_level = num_harmonics(torch.arange(max_ell + 1), d=d).type(spectral_values.dtype) # [max_ell + 1]
    normalizer = spectral_values.mT @ num_harmonics_per_level # [O, 1, max_ell + 1] @ [max_ell + 1] -> [O, 1]
    return normalizer.unsqueeze(-2) # [O, 1, 1]


def matern_spectral_density(n: Tensor, d: int, kappa: Tensor, nu: Tensor, max_ell: int, sigma: float = 1.0) -> Tensor:
    """
    Spectral density of the Matern kernel on S^{d-1}
    """
    return (
        unnormalized_matern_spectral_density(n=n, d=d, kappa=kappa, nu=nu) / # [O, N, 1]
        matern_spectral_density_normalizer(d=d, max_ell=max_ell, kappa=kappa, nu=nu) * # [O, 1, 1]
        (sigma ** 2)[..., *(None,) * (kappa.ndim - 1)] # [O, 1, 1] NOTE the reason for this seemingly overcomplicated broadcasting is that sigma can be a scalar if O is empty
    ) # [O, N, 1] / [O, 1, 1] * [O, 1, 1] -> [O, N, 1]


def matern_ahat(ell: Tensor, d: int, max_ell: int, kappa: Tensor | float, nu: Tensor | float, 
                m: int | None = None, sigma: Tensor | float = 1.0) -> float:
    """
    :math: `\hat{a} = \rho(\ell)` where :math: `\rho` is the spectral density on S^{d-1}
    """
    return matern_spectral_density(n=ell, d=d, kappa=kappa, nu=nu, max_ell=max_ell, sigma=sigma) # [O, N, 1]


def matern_repeated_ahat(max_ell: int, d: int, kappa: Tensor | float, nu: Tensor | float, sigma: Tensor | float = 1.0) -> Tensor:
    """
    Returns a tensor of repeated ahat values for each ell. 
    """
    ells = torch.arange(max_ell + 1) # [max_ell + 1]
    ahat = matern_ahat(ell=ells, d=d, max_ell=max_ell, kappa=kappa, nu=nu, sigma=sigma) # [O, max_ell + 1, 1]
    repeats = num_harmonics(ell=ells, d=d) # [max_ell + 1]
    return torch.repeat_interleave(ahat, repeats=repeats, dim=-2) # [O, num_harmonics, 1]


def matern_Kuu(max_ell: int, d: int, kappa: float, nu: float, sigma: float = 1.0) -> Tensor | DiagLinearOperator: 
    """
    Returns the covariance matrix, which is a diagonal matrix with entries 
    equal to inv_ahat of the corresponding ell. 
    """
    return DiagLinearOperator(
        1 / matern_repeated_ahat(max_ell, d, kappa, nu, sigma=sigma).squeeze(-1)
    ) # [O, num_harmonics, num_harmonics]


def spherical_harmonics(x: Tensor, max_ell: int, d: int) -> Tensor: 
    # Make sure that x is at least 2d and flatten it
    x = torch.atleast_2d(x)
    batch_shape, n = x.shape[:-2], x.shape[-2]
    x = x.flatten(0, -2)

    # Get spherical harmonics callable
    f = SphericalHarmonics(dimension=d, degrees=max_ell + 1) # [... * O, N, num_harmonics]

    # Evaluate x and reintroduce batch dimensions
    return f(x).reshape(*batch_shape, n, total_num_harmonics(max_ell, d)) # [..., O, N, num_harmonics]


def matern_Kux(x: Tensor, max_ell: int, d: int) -> Tensor: 
    return spherical_harmonics(x, max_ell=max_ell, d=d).mT # [..., O, num_harmonics, N]


def matern_LT_Phi(x: Tensor, max_ell: int, d: int, kappa: float, nu: float, sigma: float = 1.0) -> Tensor: 
    """
    Returns the feature vector of spherical harmonics evaluated at x. 
    x: [..., O, N, D]
    max_ell: []
    d: []
    kappa: [O, 1, 1]
    nu: [O, 1, 1]
    sigma: [O, 1, 1]
    """
    Kux = matern_Kux(x, max_ell=max_ell, d=d) # [..., O, num_harmonics, N]
    ahat_sqrt = matern_repeated_ahat(max_ell=max_ell, d=d, kappa=kappa, nu=nu, sigma=sigma).sqrt() # [O, num_harmonics, 1]
    return Kux * ahat_sqrt # [..., O, num_harmonics, N]


def num_spherical_harmonics_to_degree(num_spherical_harmonics: int, dimension: int) -> tuple[int, int]:
    """
    Returns the minimum degree for which there are at least
    `num_eigenfunctions` in the collection.
    """
    n, degree = 0, 0  # n: number of harmonics, d: degree (or level)
    while n < num_spherical_harmonics:
        n += num_harmonics(d=dimension, ell=degree)
        degree += 1

    if n > num_spherical_harmonics:
        print(
            "The number of spherical harmonics requested does not lead to complete "
            "levels of spherical harmonics. We have thus increased the number to "
            f"{n}, which includes all spherical harmonics up to degree {degree} (incl.)"
        )
    return degree - 1, n
