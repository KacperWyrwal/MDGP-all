from torch import Tensor 
from gpytorch.means import Mean
from gpytorch.kernels import ScaleKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from mdgp.kernels import GeometricMaternKernel


import torch 
from gpytorch import Module, settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.memoize import cached, clear_cache_hook
from linear_operator.operators import DiagLinearOperator
from functools import cached_property
# TODO Maybe just move the functions from spherical_harmonic_features.py into this file?
from mdgp.utils.spherical_harmonic_features import num_spherical_harmonics_to_degree, matern_Kuu, matern_LT_Phi


class SphericalHarmonicFeaturesVariationalStrategy(Module):
    def __init__(
        self,
        model: ApproximateGP,
        variational_distribution: CholeskyVariationalDistribution,
        dimension: int, 
        num_spherical_harmonics: int, 
        jitter_val: float | None = None,
    ):
        super().__init__()
        self._jitter_val = jitter_val

        # model, set via object.__setattr__ to avoid treatment as a module, parameter, or buffer
        object.__setattr__(self, "_model", model)

        # Variational distribution
        self._variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))

        # spherical harmonics 
        self.dimension = dimension 
        self.degree, self.num_spherical_harmonics = num_spherical_harmonics_to_degree(num_spherical_harmonics, dimension)

    @property
    def jitter_val(self) -> float:
        if self._jitter_val is None:
            return settings.variational_cholesky_jitter.value()
        return self._jitter_val

    @property
    def model(self) -> ApproximateGP:
        return self._model
    
    @property
    def covar_module(self) -> ScaleKernel | GeometricMaternKernel:
        return self.model.covar_module
    
    @cached_property
    def base_kernel(self) -> GeometricMaternKernel:
        if isinstance(self.covar_module, ScaleKernel):
            return self.covar_module.base_kernel
        else:
            return self.covar_module
    
    @property
    def mean_module(self) -> Mean:
        return self.model.mean_module

    @property
    def kappa(self) -> Tensor:
        return self.base_kernel.lengthscale
    
    @property
    def nu(self) -> Tensor | float:
        return self.base_kernel.nu
    
    @property 
    def outputscale(self) -> Tensor:
        return self.covar_module.outputscale if hasattr(self.covar_module, "outputscale") else torch.tensor(1.0)

    @property 
    def sigma(self) -> Tensor: 
        return self.outputscale.sqrt()

    def _clear_cache(self) -> None:
        clear_cache_hook(self)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        covariance_matrix = DiagLinearOperator(torch.ones(self.num_spherical_harmonics))
        mean = torch.zeros(self.num_spherical_harmonics)
        return MultivariateNormal(mean=mean, covariance_matrix=covariance_matrix)
    
    @property 
    @cached(name="cholesky_factor_prior_memo")
    def cholesky_factor_prior(self) -> DiagLinearOperator:
        Kuu = matern_Kuu(max_ell=self.degree, d=self.dimension, kappa=self.kappa, nu=self.nu, sigma=self.sigma)
        return Kuu.cholesky() # Kuu is a DiagLinearOperator, so .cholesky() is equivalent to .sqrt() 
        
    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self) -> MultivariateNormal:
        return self._variational_distribution()

    def forward(self, x: Tensor, **kwargs) -> MultivariateNormal:
        # inducing-inducing prior
        pu = self.prior_distribution
        invL_muu, invL_Kuu_invLt = pu.mean, pu.lazy_covariance_matrix

        # input-input prior
        px = self.model.forward(x)
        mux, Kxx = px.mean, px.lazy_covariance_matrix

        # inducing-inducing variational
        qu = self.variational_distribution
        invL_m, invL_S_invLt = qu.mean, qu.lazy_covariance_matrix

        # inducing-input prior  
        LT_Phi = matern_LT_Phi(x, max_ell=self.degree, d=self.dimension, kappa=self.kappa, nu=self.nu, sigma=self.sigma) # [..., O, num_harmonics, N]
        
        # Update the mean and covariance matrix
        updated_mean = (
            mux + 
            torch.einsum('...ij,...i->...j', LT_Phi, invL_m - invL_muu) # [..., O, num_harmonics, N] @ [O, num_harmonics] -> [..., O, N]
        ) # [..., O, N] + [..., O, N] -> [..., O, N]
        updated_covariance_matrix = (
            Kxx + # [..., O, N, N]
            LT_Phi.mT @ ( # [..., O, N, num_harmonics]
                invL_S_invLt - invL_Kuu_invLt) @ # [O, num_harmonics, num_harmonics]
            LT_Phi # [..., O, num_harmonics, N]
        )

        return MultivariateNormal(mean=updated_mean, covariance_matrix=updated_covariance_matrix)

    def kl_divergence(self) -> Tensor:
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence

    def __call__(self, x: Tensor, prior: bool = False, **kwargs) -> MultivariateNormal:
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x, **kwargs)

        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()

        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            prior_dist = self.prior_distribution
            self._variational_distribution.initialize_variational_distribution(prior_dist)
            self.variational_params_initialized.fill_(1)

        return super().__call__(x, **kwargs)
