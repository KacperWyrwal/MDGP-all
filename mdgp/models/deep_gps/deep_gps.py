import torch 
import gpytorch 
from torch import Tensor
from geometric_kernels.spaces import Space
from mdgp.models.deep_gps import GeometricDeepGPLayer, EuclideanDeepGPLayer, ManifoldToManifoldDeepGPLayer
from mdgp.utils import extrinsic_dimension


class ManifoldDeepGP(gpytorch.models.deep_gps.DeepGP): 

    def __init__(self, hidden_gps, output_gp, space, project_to_tangent='instrinsic', tangent_to_manifold='exp', parametrised_frame=False):
        if parametrised_frame is True: 
            get_normal_vector = 'nn'
        elif parametrised_frame is False:
            get_normal_vector = None
        else:
            get_normal_vector = parametrised_frame
        super().__init__()
        self.hidden_layers = torch.nn.ModuleList([
            ManifoldToManifoldDeepGPLayer(gp=gp, space=space, project_to_tangent=project_to_tangent, tangent_to_manifold=tangent_to_manifold, get_normal_vector=get_normal_vector)
            for gp in hidden_gps
        ])
        self.output_layer = output_gp
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward_return_hidden(self, x: Tensor, are_samples: bool = False, sample_hidden: str = 'naive', sample_output=False, mean=False):
        hidden_factors = []
        for hidden_layer in self.hidden_layers: 
            hidden_dict = hidden_layer(x=x, are_samples=are_samples, sample=sample_hidden, mean=mean, return_hidden=True)
            hidden_factors.append(hidden_dict)
            x = hidden_dict['manifold']
            are_samples = False if mean else True 
        y = self.output_layer(x, are_samples=are_samples, sample=sample_output, mean=mean)
        return hidden_factors, y 

    def forward(self, x: Tensor, are_samples: bool = False, sample_hidden: str = 'naive', sample_output=False, mean=False, resample_weights: bool = True):
        for hidden_layer in self.hidden_layers: 
            x = hidden_layer(x, are_samples=are_samples, sample=sample_hidden, mean=mean, resample_weights=resample_weights)
            are_samples = False if mean else True 
        return self.output_layer(x, are_samples=are_samples, sample=sample_output, mean=mean, resample_weights=resample_weights)
    

class GeometricManifoldDeepGP(ManifoldDeepGP):
    def __init__(
        self,
        space, 
        num_hidden: int,
        inducing_points, # [N, 3]
        output_dims=None, 
        num_eigenfunctions: int = 20, 
        nu: float = 2.5, 
        feature_map = 'deterministic',
        learn_inducing_locations: bool = False, 
        project_to_tangent: str = 'intrinsic',
        tangent_to_manifold: str = 'exp',
        optimize_nu: bool = False,
        whitened_variational_strategy=True, 
        sampler_inv_jitter=10e-8,
        outputscale_prior=None,
        parametrised_frame=False, 
        ) -> None:

        # Dimension of the manifold is the last dimension of the inducing points
        if project_to_tangent == 'intrinsic': 
            hidden_output_dims = space.dim 
        elif project_to_tangent == 'extrinsic': 
            hidden_output_dims = extrinsic_dimension(space)
        else: 
            raise NotImplementedError(f"Expected project_to_tangent either 'intrinsic' or 'extrinsic'. Got {project_to_tangent}.")


        hidden_gps = [
            GeometricDeepGPLayer(
                space=space,
                num_eigenfunctions=num_eigenfunctions,
                output_dims=hidden_output_dims,
                inducing_points=inducing_points,
                nu=nu, 
                feature_map=feature_map,
                learn_inducing_locations=learn_inducing_locations,
                optimize_nu=optimize_nu, 
                whitened_variational_strategy=whitened_variational_strategy,
                sampler_inv_jitter=sampler_inv_jitter, 
                outputscale_prior=outputscale_prior,
                zero_mean=True, 
            )
            for _ in range(num_hidden)
        ]

        output_gp = GeometricDeepGPLayer(
            space=space,
            num_eigenfunctions=num_eigenfunctions,
            output_dims=output_dims,
            inducing_points=inducing_points,
            nu=nu, 
            feature_map=feature_map,
            learn_inducing_locations=learn_inducing_locations,
            optimize_nu=optimize_nu, 
            whitened_variational_strategy=whitened_variational_strategy,
            sampler_inv_jitter=sampler_inv_jitter,
            zero_mean=False, 
        )

        super().__init__(hidden_gps=hidden_gps, output_gp=output_gp, project_to_tangent=project_to_tangent, tangent_to_manifold=tangent_to_manifold, space=space, parametrised_frame=parametrised_frame)


class EuclideanManifoldDeepGP(ManifoldDeepGP):

    def __init__(
        self,
        space: Space,
        num_hidden: int,
        inducing_points,
        output_dims=None, 
        nu: float = 2.5, 
        learn_inducing_locations: bool = False, 
        project_to_tangent='intrinsic', 
        tangent_to_manifold='exp',
        outputscale_prior=None,
        parametrised_frame=False,
        ) -> None:
        if project_to_tangent == 'intrinsic': 
            hidden_output_dims = space.dim 
        elif project_to_tangent == 'extrinsic': 
            hidden_output_dims = extrinsic_dimension(space)
        else: 
            raise NotImplementedError(f"Expected project_to_tangent either 'intrinsic' or 'extrinsic'. Got {project_to_tangent}.")

        hidden_gps = [
            EuclideanDeepGPLayer(
                output_dims=hidden_output_dims,
                inducing_points=inducing_points,
                nu=nu, 
                learn_inducing_locations=learn_inducing_locations,
                mean_type='constant',
                outputscale_prior=outputscale_prior,
            )
            for _ in range(num_hidden)
        ]

        output_gp = EuclideanDeepGPLayer(
            output_dims=output_dims,
            inducing_points=inducing_points,
            nu=nu, 
            learn_inducing_locations=learn_inducing_locations,
            mean_type='constant',
        )
        super().__init__(hidden_gps=hidden_gps, output_gp=output_gp, project_to_tangent=project_to_tangent, tangent_to_manifold=tangent_to_manifold, space=space, parametrised_frame=parametrised_frame)


class EuclideanDeepGP(gpytorch.models.deep_gps.DeepGP):

    def __init__(
        self,
        num_hidden: int,
        inducing_points,
        output_dims = None, 
        nu: float = 2.5, 
        learn_inducing_locations: bool = False, 
        outputscale_prior=None,
        ) -> None:
        super().__init__()
        # Dimension of the manifold is the last dimension of the inducing points
        hidden_output_dims = inducing_points.shape[-1]

        self.hidden_gp_layers = torch.nn.ModuleList([
            EuclideanDeepGPLayer(
                output_dims=hidden_output_dims,
                inducing_points=inducing_points,
                nu=nu, 
                learn_inducing_locations=learn_inducing_locations,
                mean_type='linear',
                outputscale_prior=outputscale_prior,
            )
            for _ in range(num_hidden)
        ])
        self.output_gp_layer = EuclideanDeepGPLayer(
            output_dims=output_dims,
            inducing_points=inducing_points,
            nu=nu, 
            learn_inducing_locations=learn_inducing_locations,
            mean_type='constant',
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
    def forward(self, inputs: Tensor, are_samples: bool = False, sample_hidden: str = 'naive', sample_output=False, mean=False, **kwargs):
        for gp_layer in self.hidden_gp_layers:
            inputs = gp_layer(inputs, are_samples=are_samples, sample=sample_hidden, mean=mean)
            are_samples = False if mean else True
        return self.output_gp_layer(inputs, are_samples=are_samples, sample=sample_output, mean=mean)
    