from torch import Tensor 


import torch 
from math import comb 
from spherical_harmonics import SphericalHarmonics
from gpytorch.kernels import Kernel, ScaleKernel
from lab import on_device



def num_harmonics_single(ell: int, d: int) -> int:
    r"""
    Number of spherical harmonics of degree ell on S^d.
    """
    if ell == 0:
        return 1
    if d == 2:
        return 2 * ell + 1
    else:
        return (2 * ell + d - 1) * comb(ell + d - 2, ell - 1) // ell


def num_harmonics(ell: Tensor, d: int) -> Tensor:
    """
    Number of spherical harmonics of degree ell on S^d.
    """
    return ell.apply_(lambda e: num_harmonics_single(ell=e, d=d)).int()


def total_num_harmonics(max_ell: int, d: int) -> int:
    """
    Total number of spherical harmonics on S^d with degree < max_ell
    """
    return num_harmonics(ell=torch.arange(max_ell), d=d).sum().item()


def eigenvalue_laplacian(ell: Tensor, d: int) -> Tensor:
    """
    Eigenvalue of the Laplace-Beltrami operator for a spherical harmonic of degree ell on S_{d}
    ell: [...]
    d: []
    return: [...]
    """
    return ell * (ell + d - 1)


def unnormalized_matern_spectral_density(n: Tensor, d: int, kappa: Tensor, nu: Tensor) -> Tensor: 
    """
    compute (unnormalized) spectral density of the matern kernel on S_{d}
    n: [N]
    d: []
    kappa: [O, 1, 1]
    nu: [O, 1, 1]
    return: [O, 1, N]
    """
    # Squared exponential kernel 
    if nu.isinf().all():
        exponent = -kappa ** 2 / 2 * eigenvalue_laplacian(ell=n, d=d).unsqueeze(-1) # [O, N, 1]
        return torch.exp(exponent)
    # Matern kernel
    else:
        base = (
            2.0 * nu / kappa**2 + # [O, 1, 1]
            eigenvalue_laplacian(ell=n, d=d).unsqueeze(-1) # [N, 1]
        ) # [O, N, 1]
        exponent = -nu - d / 2.0 # [O, 1, 1]
        return base ** exponent # [O, N, 1]


def matern_spectral_density_normalizer(d: int, max_ell: int, kappa: Tensor, nu: Tensor) -> Tensor:
    """
    Normalizing constant for the spectral density of the Matern kernel on S^d. 
    Depends on kappa and nu. Also depends on max_ell, as truncation of the infinite 
    sum from Karhunen-Loeve decomposition. 
    """
    n = torch.arange(max_ell)
    spectral_values = unnormalized_matern_spectral_density(n=n, d=d, kappa=kappa, nu=nu) # [O, max_ell + 1, 1]
    num_harmonics_per_level = num_harmonics(torch.arange(max_ell), d=d).type(spectral_values.dtype) # [max_ell + 1]
    normalizer = spectral_values.mT @ num_harmonics_per_level # [O, 1, max_ell + 1] @ [max_ell + 1] -> [O, 1]
    return normalizer.unsqueeze(-2) # [O, 1, 1]


def matern_spectral_density(n: Tensor, d: int, kappa: Tensor, nu: Tensor, max_ell: int, sigma: float = 1.0) -> Tensor:
    """
    Spectral density of the Matern kernel on S^{d-1}
    """
    return (
        unnormalized_matern_spectral_density(n=n, d=d, kappa=kappa, nu=nu) / # [O, N, 1]
        matern_spectral_density_normalizer(d=d, max_ell=max_ell, kappa=kappa, nu=nu) * # [O, 1, 1]
        (sigma ** 2)[..., *(None,) * (kappa.ndim - 1)] # [O, 1, 1]
    ) # [O, N, 1] / [O, 1, 1] * [O, 1, 1] -> [O, N, 1]


def matern_ahat(ell: Tensor, d: int, max_ell: int, kappa: Tensor | float, nu: Tensor | float, 
                m: int | None = None, sigma: Tensor | float = 1.0) -> float:
    """
    :math: `\hat{a} = \rho(\ell)` where :math: `\rho` is the spectral density on S^{d-1}
    """
    return matern_spectral_density(n=ell, d=d, kappa=kappa, nu=nu, max_ell=max_ell, sigma=sigma) # [O, N, 1]


def matern_repeated_ahat(max_ell: int, max_ell_prior: int, d: int, kappa: Tensor | float, nu: Tensor | float, sigma: Tensor | float = 1.0) -> Tensor:
    """
    Returns a tensor of repeated ahat values for each ell. 
    """
    ells = torch.arange(max_ell) # [max_ell + 1]
    ahat = matern_ahat(ell=ells, d=d, max_ell=max_ell_prior, kappa=kappa, nu=nu, sigma=sigma) # [O, max_ell + 1, 1]
    repeats = num_harmonics(ell=ells, d=d) # [max_ell + 1]
    return torch.repeat_interleave(ahat, repeats=repeats, dim=-2) # [O, num_harmonics, 1]


def matern_Kuu(max_ell: int, d: int, kappa: float, nu: float, sigma: float = 1.0) -> Tensor: 
    """
    Returns the covariance matrix, which is a diagonal matrix with entries 
    equal to inv_ahat of the corresponding ell. 
    """
    return torch.diag(1 / matern_repeated_ahat(max_ell, d, kappa, nu, sigma=sigma).squeeze(-1)) # [O, num_harmonics, num_harmonics]


def spherical_harmonics(x: Tensor, max_ell: int, d: int) -> Tensor: 
    # Flatten -> evaluate -> unflatten
    x = torch.atleast_2d(x)
    batch_shape, n = x.shape[:-2], x.shape[-2]
    x = x.flatten(0, -2)

    # SphericalHarmonics works with S^{d-1}, while we work with S^d as in GeometricKernels. 
    # Also SphericalHarmonics uses levels up to `degrees` (exclusive); hence, the +1. 
    with on_device(x.device):
        x = SphericalHarmonics(dimension=d + 1, degrees=max_ell)(x) # [... * O, N, num_harmonics]

    return x.reshape(*batch_shape, n, total_num_harmonics(max_ell, d)) # [..., O, N, num_harmonics]


def matern_Kux(x: Tensor, max_ell: int, d: int) -> Tensor: 
    return spherical_harmonics(x, max_ell=max_ell, d=d).mT # [..., O, num_harmonics, N]


def num_spherical_harmonics_to_num_levels(num_spherical_harmonics: int, dimension: int) -> tuple[int, int]:
    """
    :return: (level, least_upper_bound)
    """
    least_upper_bound, level = 0, 0
    while least_upper_bound < num_spherical_harmonics:
        least_upper_bound += num_harmonics_single(d=dimension, ell=level)
        level += 1

    if least_upper_bound > num_spherical_harmonics:
        print(
            "The number of spherical harmonics requested does not lead to complete "
            "levels of spherical harmonics. We have thus increased the number to "
            f"{least_upper_bound}, which includes all spherical harmonics up to level {level} (exclusive)"
        )
    return level, least_upper_bound


def matern_LT_Phi(x: Tensor, max_ell: int, max_ell_prior: int, d: int, kappa: float, nu: float, sigma: float = 1.0) -> Tensor: 
    Kux = matern_Kux(x, max_ell=max_ell, d=d) # [..., O, num_harmonics, N]
    ahat_sqrt = matern_repeated_ahat(max_ell=max_ell, max_ell_prior=max_ell_prior, d=d, kappa=kappa, nu=nu, sigma=sigma).sqrt() # [O, num_harmonics, 1]
    return Kux * ahat_sqrt # [..., O, num_harmonics, N]


def matern_LT_Phi_from_kernel(x: Tensor, covar_module: Kernel, num_levels: int, num_levels_prior: int) -> Tensor: 
    if isinstance(covar_module, ScaleKernel):
        sigma = covar_module.outputscale.sqrt()
        base_kernel = covar_module.base_kernel
    else:
        sigma = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        base_kernel = covar_module
    kappa = base_kernel.lengthscale
    nu = base_kernel.nu 

    # TODO Obtaining dimension in this way seems a bit error-prone
    d = base_kernel.space.dimension
    max_ell = num_levels
    max_ell_prior = num_levels_prior
    
    return matern_LT_Phi(x, max_ell=max_ell, max_ell_prior=max_ell_prior, d=d, kappa=kappa, nu=nu, sigma=sigma)
