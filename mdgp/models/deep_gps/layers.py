import torch 
import gpytorch 
from torch import Tensor 
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import UnwhitenedVariationalStrategy, VariationalStrategy
from mdgp.kernels import GeometricMaternKernel
from mdgp.samplers import RFFSampler, VISampler, PosteriorSampler, sample_naive
from mdgp.models.projectors import ProjectToTangentExtrinsic, ProjectToTangentIntrinsic, ExponentialMap, Retraction


class DeepGPLayer(gpytorch.models.deep_gps.DeepGPLayer):
    def __init__(self, mean_module, covar_module, inducing_points, output_dims, learn_inducing_locations=False, whitened_variational_strategy=True):
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
    
    def sample_naive(self, inputs, are_samples=False, **kwargs):
        return sample_naive(self.__call__(inputs, are_samples=are_samples, **kwargs))

    def sample_pathwise(self, inputs, are_samples=False, **kwargs):
        raise NotImplementedError
    
    def __call__(self, inputs, are_samples=False, sample=False, mean=False, resample_weights=True, **kwargs):
        if mean: 
            with gpytorch.settings.num_likelihood_samples(1):
                return super().__call__(inputs, are_samples=are_samples, **kwargs).mean[0]
        if sample is False: 
            return super().__call__(inputs, are_samples=are_samples, **kwargs)
        if sample == 'naive':
            return self.sample_naive(inputs, are_samples=are_samples, **kwargs)
        if sample == 'pathwise':
            return self.sample_pathwise(inputs, are_samples=are_samples, resample_weights=resample_weights, **kwargs)
        raise NotImplementedError(f"Expected sample argument to be either 'naive', 'pathwise', or False. Got {sample}")
            

class GeometricDeepGPLayer(DeepGPLayer):
    def __init__(
        self, 
        space, 
        num_eigenfunctions: int,
        output_dims: int,
        inducing_points: torch.Tensor,
        nu: float = 2.5, 
        optimize_nu: bool = False,
        feature_map: str = 'deterministic', 
        learn_inducing_locations: bool = False,
        whitened_variational_strategy=False, 
        sampler_inv_jitter=10e-8,
        outputscale_prior=None,
        zero_mean=True, 
    ) -> None: 
        batch_shape = torch.Size([output_dims]) if output_dims is not None else torch.Size([])

        # Initialize mean and kernel modules 
        if zero_mean:
            mean_module = gpytorch.means.ZeroMean(
                batch_shape=batch_shape,
            )
        else: 
            mean_module = gpytorch.means.ConstantMean(
                batch_shape=batch_shape,
            )
        base_kernel = GeometricMaternKernel(
            space=space, nu=nu, num_eigenfunctions=num_eigenfunctions, batch_shape=batch_shape, trainable_nu=optimize_nu
        )
        covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=base_kernel,
            batch_shape=batch_shape,
            outputscale_prior=outputscale_prior,
        )
        if outputscale_prior is not None: 
            covar_module.initialize(outputscale=outputscale_prior.mean)

        super().__init__(mean_module=mean_module, covar_module=covar_module, inducing_points=inducing_points, output_dims=output_dims, learn_inducing_locations=learn_inducing_locations, whitened_variational_strategy=whitened_variational_strategy)

        # Set up posterior sampler. VISampler needs the VariationalDistribution object for that the changing parameters are tracked properly
        rff_sampler = RFFSampler(covar_module=covar_module, mean_module=mean_module, feature_map=feature_map)
        vi_sampler = VISampler(variational_distribution=self.variational_strategy._variational_distribution)
        self.sampler = PosteriorSampler(rff_sampler=rff_sampler, vi_sampler=vi_sampler, inducing_points=inducing_points, whitened_variational_strategy=whitened_variational_strategy, inv_jitter=sampler_inv_jitter)

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
            inducing_points, 
            output_dims, 
            mean_type='constant', 
            learn_inducing_locations=False, 
            nu=2.5,
            constant_prior=None, 
            whitened_variational_strategy=True,
            outputscale_prior=None,
        ) -> None:
        batch_shape = torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        input_dims = inducing_points.size(-1)

        # Mean 
        if mean_type == 'constant':
            mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape, constant_prior=constant_prior)
        else:
            mean_module = gpytorch.means.LinearMean(input_dims)

        # Covariance 
        base_kernel = gpytorch.kernels.MaternKernel(nu=nu, batch_shape=batch_shape, ard_num_dims=input_dims)
        covar_module = gpytorch.kernels.ScaleKernel(base_kernel=base_kernel, batch_shape=batch_shape, ard_num_dims=None, outputscale_prior=outputscale_prior)
        if outputscale_prior is not None: 
            covar_module.initialize(outputscale=outputscale_prior.mean)

        super().__init__(mean_module=mean_module, covar_module=covar_module, inducing_points=inducing_points, output_dims=output_dims, learn_inducing_locations=learn_inducing_locations, whitened_variational_strategy=whitened_variational_strategy)


class ManifoldToManifoldDeepGPLayer(torch.nn.Module): 
    def __init__(self, gp, space, project_to_tangent: str = 'intrinsic', tangent_to_manifold: str = 'exp', get_normal_vector='nn'): 
        assert project_to_tangent in {'intrinsic', 'extrinsic'}
        assert tangent_to_manifold in {'exp', 'retr'}
        super().__init__()
        self.gp = gp 

        if project_to_tangent == 'intrinsic': 
            self.project_to_tangent = ProjectToTangentIntrinsic(space=space, get_normal_vector=get_normal_vector)
        if project_to_tangent == 'extrinsic': 
            self.project_to_tangent = ProjectToTangentExtrinsic(space=space)

        if tangent_to_manifold == 'exp': 
            self.tangent_to_manifold = ExponentialMap(space=space)
        if tangent_to_manifold == 'retr': 
            self.tangent_to_manifold = Retraction(space=space)

    def forward(self, x, are_samples=False, return_hidden=False, mean=False, sample='naive', resample_weights=True): 
        coeff = self.gp(x, mean=mean, sample=sample, are_samples=are_samples, resample_weights=resample_weights)
        u = self.project_to_tangent(x=x, coeff=coeff)
        y = self.tangent_to_manifold(x=x, u=u)
        if return_hidden: 
            return {'coefficients': coeff, 'tangent': u, 'manifold': y}
        return y
