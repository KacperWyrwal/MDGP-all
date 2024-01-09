import torch 
import gpytorch 
from torch import Tensor
from geometric_kernels.spaces import Space
from mdgp.models.deep_gps import GeometricDeepGPLayer, EuclideanDeepGPLayer, ManifoldToManifoldDeepGPLayer
from mdgp.utils import extrinsic_dimension


class DeepGP(gpytorch.models.deep_gps.DeepGP): 

    def __init__(
            self, 
            hidden_gps, 
            output_gp, 
            space, 
            to_tangent: str = 'frame', # TODO maybe change into an enum or some immutable map 
        ) -> None:
        # TODO propagate the change and merging of parameterised frame and rotated frame into to_tangent 
        super().__init__()
        self.hidden_layers = torch.nn.ModuleList([
            ManifoldToManifoldDeepGPLayer(gp=gp, space=space, to_tangent=to_tangent)
            for gp in hidden_gps
        ])
        self.output_layer = output_gp
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward_return_hidden(
            self, 
            x: Tensor, 
            are_samples: bool = False, 
            sample_hidden: str = 'elementwise', 
            sample_output: bool = False, 
            mean: bool = False, 
            resample_weights=True,
        ):
        hidden_outputs = [] 
        for hidden_layer in self.hidden_layers: 
            hidden_dict = hidden_layer(x=x, are_samples=are_samples, sample=sample_hidden, mean=mean, return_hidden=True, resample_weights=resample_weights)
            hidden_outputs.append(hidden_dict)
            x = hidden_dict['manifold']
            are_samples = False if mean else True 
        y = self.output_layer(x, are_samples=are_samples, sample=sample_output, mean=mean, resample_weights=resample_weights)
        return hidden_outputs, y 

    def forward(
            self, 
            x: Tensor, 
            are_samples: bool = False, 
            sample_hidden: str = 'elementwise', 
            sample_output: bool = False, 
            mean: bool = False, 
            resample_weights: bool = True,
        ):
        for hidden_layer in self.hidden_layers: 
            x = hidden_layer(x, are_samples=are_samples, sample=sample_hidden, mean=mean, resample_weights=resample_weights)
            are_samples = False if mean else True 
        return self.output_layer(x, are_samples=are_samples, sample=sample_output, mean=mean, resample_weights=resample_weights)
    

class ResidualGeometricDeepGP(DeepGP):
    def __init__(
        self,
        space, 
        num_hidden: int,
        inducing_points, # [N, 3]
        output_dims=None, 
        num_eigenfunctions: int = 20, 
        nu: float = 2.5, 
        learn_inducing_locations: bool = False, 
        to_tangent: str = 'project',
        optimize_nu: bool = False,
        whitened_variational_strategy=True, 
        sampler_inv_jitter=10e-8,
        outputscale_prior=None,
        ) -> None:
        # Dimension of the manifold is the last dimension of the inducing points
        if to_tangent == 'frame': 
            hidden_output_dims = space.dim 
        elif to_tangent == 'project': 
            hidden_output_dims = extrinsic_dimension(space)
        else: 
            raise NotImplementedError(f"Expected to_tangent either 'frame' or 'project'. Got {to_tangent}.")


        hidden_gps = [
            GeometricDeepGPLayer(
                space=space,
                num_eigenfunctions=num_eigenfunctions,
                output_dims=hidden_output_dims,
                inducing_points=inducing_points,
                nu=nu, 
                learn_inducing_locations=learn_inducing_locations,
                optimize_nu=optimize_nu, 
                whitened_variational_strategy=whitened_variational_strategy,
                sampler_inv_jitter=sampler_inv_jitter, 
                outputscale_prior=outputscale_prior,
                mean='zero', 
            )
            for _ in range(num_hidden)
        ]

        output_gp = GeometricDeepGPLayer(
            space=space,
            num_eigenfunctions=num_eigenfunctions,
            output_dims=output_dims,
            inducing_points=inducing_points,
            nu=nu, 
            learn_inducing_locations=learn_inducing_locations,
            optimize_nu=optimize_nu, 
            whitened_variational_strategy=whitened_variational_strategy,
            sampler_inv_jitter=sampler_inv_jitter,
            mean='constant', 
        )

        super().__init__(hidden_gps=hidden_gps, output_gp=output_gp, to_tangent=to_tangent, space=space)


class ResidualEuclideanDeepGP(DeepGP):

    def __init__(
        self,
        space: Space,
        num_hidden: int,
        inducing_points,
        output_dims=None, 
        nu: float = 2.5, 
        learn_inducing_locations: bool = False, 
        to_tangent='project', 
        outputscale_prior=None,
        parametrised_frame=False,
        ) -> None:
        # output dim is the dimension of the inducing points 
        hidden_output_dims = inducing_points.shape[-1]

        hidden_gps = [
            EuclideanDeepGPLayer(
                output_dims=hidden_output_dims,
                inducing_points=inducing_points,
                nu=nu, 
                learn_inducing_locations=learn_inducing_locations,
                mean='constant',
                outputscale_prior=outputscale_prior,
            )
            for _ in range(num_hidden)
        ]

        # We probably should learn inducing points, since we cannot just uniformly cover the Euclidean space 
        output_gp = EuclideanDeepGPLayer(
            output_dims=output_dims,
            inducing_points=inducing_points,
            nu=nu, 
            learn_inducing_locations=True,
            mean='constant',
        )
        super().__init__(hidden_gps=hidden_gps, output_gp=output_gp, to_tangent=to_tangent, space=space, parametrised_frame=parametrised_frame)


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
    

class GeometricHeadDeepGP(gpytorch.models.deep_gps.DeepGP):

    def __init__(
            self,
            space, 
            num_hidden: int,
            inducing_points, # [N, 3]
            hidden_output_dims: int = None, 
            output_dims: int | None = None, 
            num_eigenfunctions: int = 20, 
            nu: float = 2.5, 
            optimize_nu: bool = False,
            learn_inducing_locations: bool = False, 
            whitened_variational_strategy: bool = True, 
            sampler_inv_jitter: float = 10e-8,
            outputscale_prior=None,
        ) -> None:
        super().__init__()
        if hidden_output_dims is None:
            hidden_output_dims = inducing_points.shape[-1]
        
        self.hidden_gp_layers = torch.nn.ModuleList([
            GeometricDeepGPLayer(
                space=space,
                num_eigenfunctions=num_eigenfunctions,
                output_dims=hidden_output_dims,
                inducing_points=inducing_points,
                nu=nu, 
                learn_inducing_locations=learn_inducing_locations,
                optimize_nu=optimize_nu, 
                whitened_variational_strategy=whitened_variational_strategy,
                sampler_inv_jitter=sampler_inv_jitter, 
                outputscale_prior=outputscale_prior,
                mean='linear', 
            ), 
            *[
                EuclideanDeepGPLayer(
                    output_dims=hidden_output_dims,
                    inducing_points=torch.randn(inducing_points.shape[0], hidden_output_dims),
                    nu=nu, 
                    learn_inducing_locations=True,
                    mean='linear',
                    outputscale_prior=outputscale_prior,
                )
                for _ in range(num_hidden - 1)
            ],
        ])
        self.output_gp_layer = EuclideanDeepGPLayer(
            output_dims=output_dims,
            inducing_points=torch.randn(inducing_points.shape[0], hidden_output_dims),
            nu=nu, 
            learn_inducing_locations=True,
            mean='constant',
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
    def forward(self, inputs: Tensor, are_samples: bool = False, sample_hidden: str = 'elementwise', sample_output=False, mean=False, **kwargs):
        for gp_layer in self.hidden_gp_layers:
            inputs = gp_layer(inputs, are_samples=are_samples, sample=sample_hidden, mean=mean)
            are_samples = False if mean else True
        return self.output_gp_layer(inputs, are_samples=are_samples, sample=sample_output, mean=mean)
    