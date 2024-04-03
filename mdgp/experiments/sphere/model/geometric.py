from torch import Tensor 

import torch 
import gpytorch 
from gpytorch.distributions import MultivariateNormal
from geometric_kernels.spaces import Hypersphere
from mdgp.kernels import GeometricMaternKernel
from mdgp.samplers import RFFSampler, VISampler, PosteriorSampler, sample_elementwise
from mdgp.models.initializers import sphere_kmeans_centers
from mdgp.experiments.sphere.data import SphereDataset
from mdgp.experiments.sphere.model.euclidean import EuclideanDeepGPLayer


NU = 2.5
OPTIMIZE_NU = True 
NUM_EIGENFUNCTIONS = 25
OUTPUT_LAYER_VARIANCE = 1.0
LENGTHSCALE = 1.0
LIKELIHOOD_VARIANCE = 1.0


def get_inducing_points(dataset: SphereDataset, num_inducing: int) -> Tensor:
    """
    Initialize inducing points using kmeans. (from paper)
    """
    return sphere_kmeans_centers(x=dataset.train_x, k=num_inducing)


class DeepGPLayer(gpytorch.models.deep_gps.DeepGPLayer):
    def __init__(self, 
            output_dims: int | None, 
            mean_module, 
            covar_module, 
            variational_strategy, 
        ) -> None:

        super().__init__(variational_strategy, input_dims=3, output_dims=output_dims)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)  
    
    def sample_elementwise(self, inputs, are_samples=False, **kwargs):
        return sample_elementwise(self.__call__(inputs, are_samples=are_samples, **kwargs))

    def sample_pathwise(self, inputs, are_samples=False, **kwargs):
        raise NotImplementedError
    
    def __call__(self, inputs, are_samples=False, sample=False, mean=False, resample_weights=True, **kwargs):
        if mean: 
            with gpytorch.settings.num_likelihood_samples(1):
                return super().__call__(inputs, are_samples=are_samples, **kwargs).mean[0]
        if sample is False: 
            return super().__call__(inputs, are_samples=are_samples, **kwargs)
        if sample == 'elementwise':
            return self.sample_elementwise(inputs, are_samples=are_samples, **kwargs)
        if sample == 'pathwise':
            return self.sample_pathwise(inputs, are_samples=are_samples, resample_weights=resample_weights, **kwargs)
        raise ValueError(f"Expected either 'sample' in ['elementwise', 'pathwise', False] or 'mean' == True. Got {sample=} and {mean=}")


def get_output_dims(gvf: str | None, hidden: bool) -> int | None:
    if hidden is False:
        return None
    if gvf == 'projected' or gvf is None:
        return 3
    if gvf == 'frame':
        return 2
    raise ValueError(f"Unknown Gaussian Vector Field type {gvf}. Must be one of [None, 'projected', 'frame']")


import geoopt 
from gpytorch.models.approximate_gp import ApproximateGP
from gpytorch.variational._variational_distribution import _VariationalDistribution
from gpytorch.module import Module 
from gpytorch.variational.variational_strategy import _ensure_updated_strategy_flag_set


class SphereVariationalStrategy(gpytorch.variational.VariationalStrategy):
    def __init__(
        self,
        model: ApproximateGP,
        inducing_points: Tensor,
        variational_distribution: _VariationalDistribution,
        learn_inducing_locations: bool = True,
        jitter_val: float | None = None,
    ):
        super(Module, self).__init__()

        self._jitter_val = jitter_val

        # Model
        object.__setattr__(self, "model", model)

        # Inducing points
        inducing_points = inducing_points.clone()
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)
        if learn_inducing_locations:
            self.register_parameter(name="inducing_points", 
                                    parameter=geoopt.ManifoldParameter(inducing_points), 
                                    manifold=geoopt.manifolds.Sphere(torch.eye(3)))
        else:
            self.register_buffer("inducing_points", inducing_points)

        # Variational distribution
        self._variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.has_fantasy_strategy = True




class GeometricDeepGPLayer(DeepGPLayer):

    def __init__(
        self, 
        inducing_points: Tensor,     
        gvf: str | None = None,
        hidden: bool = False, 
        outputscale_prior_mean: float = 1.0,
        jitter_val: float | None = None,
        learn_inducing_locations: bool = False,
    ) -> None: 
        inducing_points = inducing_points.clone()
        assert not (outputscale_prior_mean != OUTPUT_LAYER_VARIANCE and hidden is False), "Cannot have non-default outputscale prior for output layers"
        assert (gvf is None) or (hidden is True), "Cannot have both Gaussian Vector Field and be an output layer"
        
        num_inducing_points = inducing_points.size(0)
        output_dims = get_output_dims(gvf=gvf, hidden=hidden)
        batch_shape = torch.Size([output_dims]) if output_dims is not None else torch.Size([])

        # Use zero mean for hidden layers, since linear mean is equivalent to zero mean + exponential map
        if hidden is True:
            if gvf is not None:
                mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
            else:
                mean_module = gpytorch.means.LinearMean(input_size=3, batch_shape=batch_shape, bias=False)
        else:
            mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        
        # Covariance function
        base_kernel = GeometricMaternKernel(
            space=Hypersphere(dim=2), nu=NU, trainable_nu=OPTIMIZE_NU, num_eigenfunctions=NUM_EIGENFUNCTIONS, 
            batch_shape=batch_shape
        )
        covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=base_kernel, 
            batch_shape=batch_shape, 
            outputscale_prior=gpytorch.priors.GammaPrior(concentration=1.0, rate=1 / outputscale_prior_mean),
        )

        # Initialize variational strategy
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing_points, batch_shape=batch_shape
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points=inducing_points, variational_distribution=variational_distribution, 
            learn_inducing_locations=False
        )

        super().__init__(
            mean_module=mean_module, covar_module=covar_module, variational_strategy=variational_strategy, 
            output_dims=output_dims
        )
        
        rff_sampler = RFFSampler(covar_module=covar_module, mean_module=mean_module, feature_map=base_kernel.feature_map)
        vi_sampler = VISampler(variational_distribution=self.variational_strategy._variational_distribution)
        self.sampler = PosteriorSampler(rff_sampler=rff_sampler, vi_sampler=vi_sampler, inducing_points=inducing_points, 
                                        whitened_variational_strategy=True, inv_jitter=jitter_val)
        
        # Initialize module values 
        covar_module.base_kernel.lengthscale = LENGTHSCALE
        covar_module.outputscale = outputscale_prior_mean


    # TODO Move this to superclass 
    def sample_pathwise(self, inputs, are_samples=False, resample_weights=True):
        # Clear cache if training, since otherwise we risk "trying to backward through the graph a second time" errors 
        if self.training: 
            self.variational_strategy._clear_cache()
        # Maybe initialize variational distribution (Taken from gpytorch.variational._VariationalStrategy.__call__)
        if not self.variational_strategy.variational_params_initialized.item():
            prior_dist = self.variational_strategy.prior_distribution
            self.variational_strategy._variational_distribution.initialize_variational_distribution(prior_dist)
            self.variational_strategy.variational_params_initialized.fill_(1)

        # Take sample 
        if are_samples: 
            sample_shape = torch.Size([])
            sample = self.sampler(inputs.unsqueeze(-3), sample_shape=sample_shape, resample=resample_weights) # [S, O, N]
        else: 
            sample_shape = torch.Size([gpytorch.settings.num_likelihood_samples.value()])
            sample = self.sampler(inputs, sample_shape=sample_shape, resample=resample_weights) # [S, O, N]
        return sample.mT


class FullyGeometricDeepGP(gpytorch.models.deep_gps.DeepGP):
    def __init__(
        self, 
        dataset: SphereDataset,
        num_inducing_points: int, 
        outputscale_prior_mean: float, 
        gvf: str | None,
        num_layers: int, 
        jitter_val: float | None = None, 
    ) -> None:

        super().__init__()
        self.space = Hypersphere(dim=2)
        inducing_points = get_inducing_points(dataset, num_inducing_points)
        self.layers = torch.nn.ModuleList(
            [
                GeometricDeepGPLayer(
                    inducing_points=inducing_points,
                    gvf=gvf,
                    hidden=True,
                    outputscale_prior_mean=outputscale_prior_mean,
                    jitter_val=jitter_val,
                ) for _ in range(num_layers - 1)
            ] + 
            [
                GeometricDeepGPLayer(
                    inducing_points=inducing_points,
                    gvf=None,
                    hidden=False,
                    outputscale_prior_mean=OUTPUT_LAYER_VARIANCE,
                    jitter_val=jitter_val,
                )
            ]
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = LIKELIHOOD_VARIANCE

    def forward(self, x: Tensor, are_samples: bool = False, sample: str = 'elementwise', mean: bool = False) -> Tensor:
        for layer in self.layers[:-1]:
            ambient = layer(x, are_samples=are_samples, sample=sample, mean=mean)
            tangent = self.space.to_tangent(ambient, x)
            x = self.space.metric.exp(tangent, x)
            are_samples = not mean 
        return self.layers[-1](x, are_samples=are_samples)
    

class InputGeometricDeepGP(gpytorch.models.deep_gps.DeepGP):
    def __init__(
        self, 
        dataset: SphereDataset,
        num_inducing_points: int, 
        outputscale_prior_mean: float, 
        gvf: str | None,
        num_layers: int, 
        jitter_val: float | None = None, 
    ) -> None:

        super().__init__()
        self.space = Hypersphere(dim=2)
        inducing_points = get_inducing_points(dataset, num_inducing_points)
        if num_layers == 1:
            self.layers = torch.nn.ModuleList(
                [
                    GeometricDeepGPLayer(
                        inducing_points=inducing_points,
                        gvf=None,
                        hidden=False,
                        outputscale_prior_mean=OUTPUT_LAYER_VARIANCE,
                        jitter_val=jitter_val,
                    )
                ]
            )
        else:
            self.layers = torch.nn.ModuleList(
                [
                    GeometricDeepGPLayer(
                        inducing_points=inducing_points, 
                        gvf=None,
                        hidden=True,
                        outputscale_prior_mean=outputscale_prior_mean,
                        jitter_val=jitter_val,
                    ) 
                ] + 
                [
                    EuclideanDeepGPLayer(
                        inducing_points=inducing_points,
                        hidden=True,
                        outputscale_prior_mean=outputscale_prior_mean,
                    ) for _ in range(num_layers - 2)
                ] + 
                [
                    EuclideanDeepGPLayer(
                        inducing_points=inducing_points,
                        hidden=False,
                        outputscale_prior_mean=OUTPUT_LAYER_VARIANCE,
                    )
                ]
            )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = LIKELIHOOD_VARIANCE

    def forward(self, x: Tensor, are_samples: bool = False, sample: str = 'elementwise', mean: bool = False) -> Tensor:
        for layer in self.layers[:-1]:
            x = layer(x, are_samples=are_samples, sample=sample, mean=mean)
            are_samples = not mean 
        return self.layers[-1](x, are_samples=are_samples)
