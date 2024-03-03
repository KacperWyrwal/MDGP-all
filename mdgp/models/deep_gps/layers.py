from torch import Tensor 
from gpytorch.priors import Prior, GammaPrior


import torch 
import gpytorch 
from gpytorch.distributions import MultivariateNormal
from geometric_kernels.spaces import Space, Euclidean
from mdgp.kernels import GeometricMaternKernel
from mdgp.samplers import RFFSampler, VISampler, PosteriorSampler, sample_elementwise
from mdgp.projectors import Exp, ToTangent
from mdgp.variational.variational_strategy_factory import VariationalStrategyFactory


class DeepGPLayer(gpytorch.models.deep_gps.DeepGPLayer):
    def __init__(self, 
            input_dims: int, 
            output_dims: int | None, 
            mean_module, 
            covar_module, 
            variational_strategy, 
        ) -> None:

        super().__init__(variational_strategy, input_dims, output_dims)
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
            

def get_mean(mean: str, input_dims: int, batch_shape: torch.Size):
    if mean == 'zero':
        mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
    elif mean == 'constant':
        mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
    elif mean == 'linear': 
        mean_module = gpytorch.means.LinearMean(input_dims, batch_shape=batch_shape)
    else:
        raise ValueError(f"Unknown mean type {mean}. Must be one of ['zero', 'constant', 'linear'].")
    return mean_module


class GeometricDeepGPLayer(DeepGPLayer):

    def __init__(self, 
        space: Space, 
        input_dims: int, 
        variational_strategy_factory: VariationalStrategyFactory,
        output_dims: int | None = None,
        mean: str = 'zero', 
        nu: float = 2.5, 
        optimize_nu: bool = False,
        outputscale_prior: Prior | None = None,
        sampler_inv_jitter: float | None = None,
        num_eigenfunctions: int | None = None,
        num_random_phases: int | None = None,
    ) -> None: 
        batch_shape = torch.Size([output_dims]) if output_dims is not None else torch.Size([])

        # Initialize mean. We use zero mean for hidden layers and a constant mean for the output layer. 
        mean_module = get_mean(mean=mean, input_dims=input_dims, batch_shape=batch_shape)
        base_kernel = GeometricMaternKernel(
            space=space, nu=nu, trainable_nu=optimize_nu, num_eigenfunctions=num_eigenfunctions, 
            num_random_phases=num_random_phases, batch_shape=batch_shape
        )
        covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=base_kernel, batch_shape=batch_shape, outputscale_prior=outputscale_prior,
        )
        if outputscale_prior is not None: 
            covar_module.initialize(outputscale=outputscale_prior.mean)

        # Initialize variational strategy
        variational_strategy = variational_strategy_factory(model=self, space=space, batch_shape=batch_shape)

        super().__init__(mean_module=mean_module, covar_module=covar_module, 
                         variational_strategy=variational_strategy, 
                         input_dims=input_dims, output_dims=output_dims)

        # Set up posterior sampler. VISampler needs the VariationalDistribution object for that the changing parameters are tracked properly
        # TODO Need to update samplers with the new variational strategy setup.
        # rff_sampler = RFFSampler(covar_module=covar_module, mean_module=mean_module, feature_map=base_kernel.feature_map)
        # vi_sampler = VISampler(variational_distribution=self.variational_strategy._variational_distribution)
        # self.sampler = PosteriorSampler(rff_sampler=rff_sampler, vi_sampler=vi_sampler, inducing_points=inducing_points, whitened_variational_strategy=whitened_variational_strategy, inv_jitter=sampler_inv_jitter)

    # TODO Move this to superclass 
    def sample_pathwise(self, inputs, are_samples=False, resample_weights=True):
        pass # TODO Need to update samplers with the new variational strategy setup.
        # # Clear cache if training, since otherwise we risk "trying to backward through the graph a second time" errors 
        # if self.training: 
        #     self.variational_strategy._clear_cache()
        # # Maybe initialize variational distribution (Taken from gpytorch.variational._VariationalStrategy.__call__)
        # if not self.variational_strategy.variational_params_initialized.item():
        #     prior_dist = self.variational_strategy.prior_distribution
        #     self.variational_strategy._variational_distribution.initialize_variational_distribution(prior_dist)
        #     self.variational_strategy.variational_params_initialized.fill_(1)

        # # Take sample 
        # if are_samples: 
        #     sample_shape = torch.Size([])
        #     sample = self.sampler(inputs.unsqueeze(-3), sample_shape=sample_shape, resample=resample_weights) # [S, O, N]
        # else: 
        #     sample_shape = torch.Size([gpytorch.settings.num_likelihood_samples.value()])
        #     sample = self.sampler(inputs, sample_shape=sample_shape, resample=resample_weights) # [S, O, N]
        # return sample.mT


class EuclideanDeepGPLayer(GeometricDeepGPLayer):
    def __init__(
        self, 
        input_dims: int, 
        variational_strategy_factory: VariationalStrategyFactory,
        output_dims: int | None = None,
        mean: str = 'zero', 
        nu: float = 2.5, 
        outputscale_prior=None,
        sampler_inv_jitter: float | None = None,
        num_eigenfunctions: int | None = None,
        num_random_phases: int | None = None,
    ) -> None: 
        super().__init__(
            space=Euclidean(input_dims), 
            input_dims=input_dims, 
            variational_strategy_factory=variational_strategy_factory,
            output_dims=output_dims,
            mean=mean,
            nu=nu,
            optimize_nu=False,
            outputscale_prior=outputscale_prior,
            sampler_inv_jitter=sampler_inv_jitter,
            num_eigenfunctions=num_eigenfunctions,
            num_random_phases=num_random_phases,
        )


class ManifoldToManifoldDeepGPLayer(torch.nn.Module): 
    def __init__(
            self, 
            gp, 
            space, 
            to_tangent: str = 'project',
        ) -> None: 
        assert to_tangent in {'frame', 'project'}
        super().__init__()
        self.gp = gp 

        if to_tangent == 'frame': 
            raise NotImplementedError("Frame implementation is currently being migrated")
        elif to_tangent == 'project': 
            self.to_tangent = ToTangent(space=space)
        self.tangent_to_manifold = Exp(space=space)

    def forward(self, x, are_samples=False, return_hidden=False, mean=False, sample='elementwise', resample_weights=True): 
        coeff = self.gp(x, mean=mean, sample=sample, are_samples=are_samples, resample_weights=resample_weights)
        u = self.to_tangent(x, coeff)
        y = self.tangent_to_manifold(x, u)
        if return_hidden: 
            return {'coefficients': coeff, 'tangent': u, 'manifold': y}
        return y
