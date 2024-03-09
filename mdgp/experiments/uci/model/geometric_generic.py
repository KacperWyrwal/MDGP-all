from torch import Tensor 


import torch 
from gpytorch import settings
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from geometric_kernels.spaces import Hypersphere
from mdgp.experiments.uci.data.datasets import UCIDataset
from mdgp.utils.sphere import sphere_kmeans_centers
from mdgp.kernels import GeometricMaternKernel
from mdgp.variational import SphericalHarmonicFeaturesVariationalStrategy
from mdgp.variational.spherical_harmonic_features.utils import total_num_harmonics, num_spherical_harmonics_to_num_levels
from mdgp.samplers import sample_elementwise


# Settings from the paper  
LIKELIHOOD_VARIANCE = 0.01
LENGTHSCALE = 2.0
INNER_LAYER_VARIANCE = 1e-5
OUTPUT_LAYER_VARIANCE = 1.0 # This is a (reasonable) guess
NUM_INDUCING_POINTS = 100
NUM_HARMONICS_KERNEL = 625 # Minimum number of harmonics to represent a kernel. Equivalent to 25 levels on S^2


def get_hidden_dims(dataset: UCIDataset) -> int:
    return dataset.dimension + 1


def get_inducing_points(dataset: UCIDataset, num_inducing_points: int) -> Tensor:
    """
    Initialize inducing points using kmeans. (from paper)
    """
    return sphere_kmeans_centers(dataset.train_x, num_inducing_points)


class SHFDeepGPLayer(DeepGPLayer):
    def __init__(
        self, 
        d, 
        max_ell_prior: int | None = None, 
        max_ell: int | None = None, 
        kappa: float | None = None, 
        nu=2.5, # This should be torch.inf, but doesn't work yet 
        sigma=1.0, 
        jitter_val: float | None = None, 
        optimize_nu: bool = True, 
        output_dims: int | None = None
    ):
        # Defaults matching the DSVI paper 
        max_ell = max_ell or num_spherical_harmonics_to_num_levels(NUM_HARMONICS_KERNEL, d)[0]
        max_ell_prior = max_ell_prior or num_spherical_harmonics_to_num_levels(NUM_HARMONICS_KERNEL, d)[0]
        kappa = kappa or LENGTHSCALE

        m = total_num_harmonics(max_ell, d)
        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=m, batch_shape=batch_shape)
        variational_strategy = SphericalHarmonicFeaturesVariationalStrategy(self, variational_distribution, num_levels=max_ell, jitter_val=jitter_val)
        super().__init__(variational_strategy, d + 1, output_dims)
        self.batch_shape = batch_shape 

        # constants 
        self.jitter_val = jitter_val or settings.cholesky_jitter.value(torch.get_default_dtype())
        self.max_ell = max_ell
        self.max_ell_prior = max_ell_prior
        self.d = d

        # modules 
        base_kernel = GeometricMaternKernel(
            space=Hypersphere(d),
            lengthscale=kappa, 
            nu=nu, 
            trainable_nu=optimize_nu, 
            num_eigenfunctions=max_ell_prior,
            batch_shape=batch_shape,
        )
        base_kernel.lengthscale = kappa
        self.covar_module = ScaleKernel(base_kernel, batch_shape=batch_shape)
        self.covar_module.outputscale = sigma ** 2
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)  
    

class SHFDeepGP(DeepGP):
    def __init__(
        self, 
        dimension: int, 
        num_layers: int | None = None, 
        num_levels_ker: int | None = None, 
        num_levels_var: int | None = None, 
        lengthscale: float | None = None, 
        nu: float = 2.5, 
        outputscale: float | None = None,
        jitter_val: float | None = None, 
        optimize_nu: bool = True, 
        noise: float | None = None, 
    ) -> None:
        # Defaults matching the DSVI paper
        outputscale = outputscale or OUTPUT_LAYER_VARIANCE
        noise = noise or LIKELIHOOD_VARIANCE


        super().__init__()
        self.space = Hypersphere(dimension)
        self.layers = torch.nn.ModuleList(
            [
                SHFDeepGPLayer(
                    max_ell=num_levels_var,
                    max_ell_prior=num_levels_ker,
                    d=dimension, 
                    kappa=lengthscale,
                    nu=nu, 
                    sigma=INNER_LAYER_VARIANCE ** 0.5, 
                    jitter_val=jitter_val,
                    optimize_nu=optimize_nu,
                    output_dims=dimension + 1, # Embedding dimension of S^d is d + 1
                ) for _ in range(num_layers - 1)
            ] + 
            [
                SHFDeepGPLayer(
                    max_ell=num_levels_var,
                    max_ell_prior=num_levels_ker,
                    d=dimension, 
                    kappa=lengthscale,
                    nu=nu, 
                    sigma=outputscale ** 0.5, 
                    jitter_val=jitter_val,
                    optimize_nu=optimize_nu,
                    output_dims=None, # For UCI datasets output is always 1-dimensional
                )
            ]
        )
        self.likelihood = GaussianLikelihood()
        self.likelihood.noise = noise

    def forward(self, x: Tensor, are_samples: bool = False) -> Tensor:
        for layer in self.layers[:-1]:
            ambient = sample_elementwise(layer(x, are_samples=are_samples))
            tangent = self.space.to_tangent(ambient, x)
            x = self.space.metric.exp(tangent, x)
            are_samples = True
        return self.layers[-1](x, are_samples=are_samples)
