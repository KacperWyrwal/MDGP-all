from torch import Tensor 
from geometric_kernels.spaces import Space


import gpytorch
from mdgp.kernels import GeometricMaternKernel


class GeometricExactGP(gpytorch.models.ExactGP): 
    """
    The simplest possible GP model with a GeometricMaternKernel taking in a space parameter.

    TODO: We might need to do some reshaping in the forward function depending on the BO algo. 
    """
    def __init__(
        self, 
        train_x: Tensor, 
        train_y: Tensor, 
        space: Space, 
        nu: float = 2.5, 
        trainable_nu: bool = True, 
        num_eigenfunctions: int = 20, 
        normalize: bool = True, 
        lengthscale: float = 1.0, 
        matern_gabo=True
    ) -> None:
        base_kernel = GeometricMaternKernel(
            space=space, 
            nu=nu, 
            trainable_nu=trainable_nu,
            num_eigenfunctions=num_eigenfunctions,
            normalize=normalize,
            lengthscale=lengthscale,
        )
        # Settings used in the Robotics paper TODO Maybe extract this into a separate class 
        if matern_gabo: 
            noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
                initial_value=noise_prior_mode,
            )
            covar_module = gpytorch.kernels.ScaleKernel(
                base_kernel, 
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
            )
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x, **kwargs): 
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
