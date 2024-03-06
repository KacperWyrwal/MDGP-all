import torch 
import gpytorch 
from torch import nn 
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.utils.memoize import cached
from linear_operator.operators import DiagLinearOperator
from mdgp.variational.spherical_harmonic_features.utils import matern_LT_Phi_from_kernel


class SphericalHarmonicFeaturesVariationalStrategy(nn.Module):
    def __init__(self, model: ApproximateGP, variational_distribution: CholeskyVariationalDistribution, num_levels: int, jitter_val: float | None = None):
        super().__init__()
        object.__setattr__(self, "_model", model)
        self._variational_distribution = variational_distribution
        self.jitter_val = jitter_val or gpytorch.settings.cholesky_jitter.value()
        self.num_levels = num_levels

    @property
    def model(self) -> ApproximateGP:
        return self._model
    
    @property
    def variational_distribution(self) -> MultivariateNormal:
        return self._variational_distribution()
    
    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLinearOperator(ones))
        return res

    def forward(self, x) -> MultivariateNormal:
        # prior at x 
        px = self.model.forward(x)
        mu_x, Kxx = px.mean, px.lazy_covariance_matrix
        Kxx = Kxx.add_jitter(self.jitter_val)

        # whitened prior at u
        Linv_pu = self.prior_distribution
        Linv_mu, Linv_Kuu_LTinv = Linv_pu.mean, Linv_pu.lazy_covariance_matrix

        # whitened variational posterior at u
        Linv_qu = self.variational_distribution
        Linv_m, Linv_S_LTinv = Linv_qu.mean, Linv_qu.lazy_covariance_matrix

        # unwhitening + projection and vice-versa
        LT_Phiux = matern_LT_Phi_from_kernel(x, self.model.covar_module, num_levels=self.num_levels, num_levels_prior=self.model.max_ell_prior)
        Phixu_L = LT_Phiux.mT

        # posterior at x 
        qx_sigma = Kxx + Phixu_L @ (Linv_S_LTinv - Linv_Kuu_LTinv) @ LT_Phiux
        qx_sigma = qx_sigma.add_jitter(self.jitter_val)
        qx_mu = mu_x + Phixu_L @ (Linv_m - Linv_mu)
        return MultivariateNormal(qx_mu, qx_sigma)
    
    def kl_divergence(self):
        with gpytorch.settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence
    
    def __call__(self, x, prior: bool = False, **kwargs) -> MultivariateNormal:
        if prior:
            return self.model.forward(x)
        return super().__call__(x, **kwargs)
