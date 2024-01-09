import torch 
import gpytorch 
from torch import Tensor 
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import UnwhitenedVariationalStrategy, VariationalStrategy
from mdgp.kernels import GeometricMaternKernel
from mdgp.samplers import RFFSampler, VISampler, PosteriorSampler, sample_elementwise
from mdgp.projectors import Exp, ToTangent
from geometric_kernels.spaces import Euclidean


class DeepGPLayer(gpytorch.models.deep_gps.DeepGPLayer):
    def __init__(
            self, 
            mean_module, 
            covar_module, 
            inducing_points, 
            output_dims: int | None = None, 
            learn_inducing_locations=False, 
            whitened_variational_strategy=True
        ) -> None:
        batch_shape = torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        num_inducing_points, input_dims = inducing_points.shape

        # Variational Parameters
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing_points,
            batch_shape=batch_shape
        )
        variational_strategy_class = VariationalStrategy if whitened_variational_strategy else UnwhitenedVariationalStrategy
        variational_strategy = variational_strategy_class(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations
        )

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
            

def get_mean_from_str(mean: str, input_dims: int, batch_shape: torch.Size):
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
    def __init__(
        self, 
        space, 
        inducing_points: torch.Tensor,
        output_dims: int | None = None,
        num_eigenfunctions: int | None = None,
        num_random_phases: int | None = None,
        nu: float = 2.5, 
        optimize_nu: bool = False,
        learn_inducing_locations: bool = False,
        whitened_variational_strategy=False, 
        sampler_inv_jitter=10e-8,
        outputscale_prior=None,
        mean: str = 'zero', 
    ) -> None: 
        batch_shape = torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        input_dims = inducing_points.size(-1)

        # Initialize mean. Usually, we use zero mean for the hidden layers and a constant mean for the output layer. 
        mean_module = get_mean_from_str(mean=mean, input_dims=input_dims, batch_shape=batch_shape)

        # Initialize kernel
        base_kernel = GeometricMaternKernel(
            space=space, 
            nu=nu, 
            num_eigenfunctions=num_eigenfunctions, 
            num_random_phases=num_random_phases, 
            batch_shape=batch_shape, 
            trainable_nu=optimize_nu,
        )
        covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=base_kernel,
            batch_shape=batch_shape,
            outputscale_prior=outputscale_prior,
        )
        # Outputscale prior is a useful parameter for deep models. Lower values are usually recommended for deeper models.
        if outputscale_prior is not None: 
            covar_module.initialize(outputscale=outputscale_prior.mean)

        super().__init__(mean_module=mean_module, covar_module=covar_module, inducing_points=inducing_points, output_dims=output_dims, learn_inducing_locations=learn_inducing_locations, whitened_variational_strategy=whitened_variational_strategy)

        # Set up posterior sampler. VISampler needs the VariationalDistribution object for that the changing parameters are tracked properly
        rff_sampler = RFFSampler(covar_module=covar_module, mean_module=mean_module, feature_map=base_kernel.feature_map)
        vi_sampler = VISampler(variational_distribution=self.variational_strategy._variational_distribution)
        self.sampler = PosteriorSampler(rff_sampler=rff_sampler, vi_sampler=vi_sampler, inducing_points=inducing_points, whitened_variational_strategy=whitened_variational_strategy, inv_jitter=sampler_inv_jitter)

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


class EuclideanDeepGPLayer(DeepGPLayer):
    def __init__(
        self, 
        inducing_points: torch.Tensor,
        output_dims: int | None = None,
        num_eigenfunctions: int | None = None,
        num_random_phases: int | None = None,
        nu: float = 2.5, 
        optimize_nu: bool = False,
        learn_inducing_locations: bool = False,
        whitened_variational_strategy=False, 
        sampler_inv_jitter=10e-8,
        outputscale_prior=None,
        mean: str = 'zero', 
    ) -> None: 
        # TODO Maybe instead pass in an input_dim and a number of inducing points and then randomly generate inducing points. 
        # Initialize Euclidean space to match the dimension of the inducing points. 
        dim = inducing_points.size(-1)
        space = Euclidean(dim)

        return super().__init__(
            space=space, 
            inducing_points=inducing_points, 
            output_dims=output_dims, 
            num_eigenfunctions=num_eigenfunctions, 
            num_random_phases=num_random_phases, 
            nu=nu, 
            optimize_nu=optimize_nu, 
            learn_inducing_locations=learn_inducing_locations, 
            whitened_variational_strategy=whitened_variational_strategy, 
            sampler_inv_jitter=sampler_inv_jitter, 
            outputscale_prior=outputscale_prior, 
            mean=mean, 
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
