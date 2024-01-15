import torch 
import gpytorch 
from torch import Tensor
from geometric_kernels.spaces import Space, Euclidean
from mdgp.models.deep_gps import GeometricDeepGPLayer, EuclideanDeepGPLayer, ManifoldToManifoldDeepGPLayer
from mdgp.utils import extrinsic_dimension
from mdgp.models.initializers import initialize_grid, initialize_kmeans


class DeepGP(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, hidden_layers, output_layer) -> None: 
        super().__init__()
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    def forward(self, 
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


class ResidualDeepGP(DeepGP): 

    def __init__(
            self, 
            hidden_gps, 
            output_layer, 
            space, 
            to_tangent: str = 'frame', # TODO maybe change into an enum or some immutable map 
        ) -> None:
        # TODO propagate the change and merging of parameterised frame and rotated frame into to_tangent 
        hidden_layers = torch.nn.ModuleList([
            ManifoldToManifoldDeepGPLayer(gp=gp, space=space, to_tangent=to_tangent)
            for gp in hidden_gps
        ])
        super().__init__(hidden_layers=hidden_layers, output_layer=output_layer)

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


def get_output_dim(to_tangent: str, space: Space) -> int: 
    if to_tangent == 'frame': 
        return space.dim 
    elif to_tangent == 'project': 
        return extrinsic_dimension(space)
    else: 
        raise NotImplementedError(f"Expected to_tangent either 'frame' or 'project'. Got {to_tangent}.")


class ResidualGeometricDeepGP(ResidualDeepGP):
    def __init__(
        self,
        space, 
        num_hidden: int,
        num_inducing: int, # [N, 3]
        input_points: Tensor,
        output_dims=None, 
        num_eigenfunctions: int = 20, 
        nu: float = 2.5, 
        learn_inducing_locations: bool = False, 
        to_tangent: str = 'project',
        optimize_nu: bool = False,
        whitened_variational_strategy: bool = True, 
        sampler_inv_jitter=10e-8,
        outputscale_prior=None,
        ) -> None:
        hidden_output_dims = get_output_dim(to_tangent, space)
        inducing_points = initialize_kmeans(input_points, space=space, n=num_inducing)
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

        output_layer = GeometricDeepGPLayer(
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

        super().__init__(hidden_gps=hidden_gps, output_layer=output_layer, to_tangent=to_tangent, space=space)


class ResidualEuclideanDeepGP(ResidualDeepGP):

    def __init__(
        self,
        space: Space,
        num_hidden: int,
        num_inducing: int,
        input_points: Tensor,
        output_dims=None, 
        nu: float = 2.5, 
        learn_inducing_locations: bool = False, 
        to_tangent='project', 
        outputscale_prior=None,
        ) -> None:
        hidden_output_dims = get_output_dim(to_tangent, space)
        inducing_points = initialize_kmeans(input_points, space=space, n=num_inducing)

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

        # We learn inducing points, since we cannot just uniformly cover the Euclidean space 
        output_layer = EuclideanDeepGPLayer(
            output_dims=output_dims,
            inducing_points=inducing_points,
            nu=nu, 
            learn_inducing_locations=True,
            mean='constant',
        )
        super().__init__(hidden_gps=hidden_gps, output_layer=output_layer, to_tangent=to_tangent, space=space)


class EuclideanDeepGP(DeepGP):

    def __init__(
        self,
        num_hidden: int,
        num_inducing: int,
        input_points: Tensor,
        hidden_output_dims: int | None = None, 
        output_dims = None, 
        nu: float = 2.5, 
        learn_inducing_locations: bool = False, 
        outputscale_prior=None,
        whitened_variational_strategy: bool = True, 
        sampler_inv_jitter: float = 10e-8,
    ) -> None:
        if hidden_output_dims is None:
            hidden_output_dims = input_points.shape[-1]
        inducing_points = initialize_kmeans(input_points, space=Euclidean(dim=hidden_output_dims), n=num_inducing)

        hidden_layers = torch.nn.ModuleList([
            EuclideanDeepGPLayer(
                output_dims=hidden_output_dims,
                inducing_points=inducing_points,
                nu=nu, 
                learn_inducing_locations=learn_inducing_locations,
                mean='linear',
                outputscale_prior=outputscale_prior,
                whitened_variational_strategy=whitened_variational_strategy,
                sampler_inv_jitter=sampler_inv_jitter, 
            )
            for _ in range(num_hidden)
        ])

        output_layer = EuclideanDeepGPLayer(
            output_dims=output_dims,
            inducing_points=inducing_points,
            nu=nu, 
            learn_inducing_locations=learn_inducing_locations,
            mean='constant',
            whitened_variational_strategy=whitened_variational_strategy,
            sampler_inv_jitter=sampler_inv_jitter, 
        )
        super().__init__(hidden_layers=hidden_layers, output_layer=output_layer)
    

class GeometricHeadDeepGP(DeepGP):

    def __init__(
        self,
        space, 
        num_hidden: int,
        num_inducing: int, 
        input_points: Tensor,
        hidden_output_dims: int | None = None, 
        output_dims: int | None = None, 
        num_eigenfunctions: int = 20, 
        nu: float = 2.5, 
        optimize_nu: bool = False,
        learn_inducing_locations: bool = False, 
        whitened_variational_strategy: bool = True, 
        sampler_inv_jitter: float = 10e-8,
        outputscale_prior=None,
    ) -> None:
        if hidden_output_dims is None:
            hidden_output_dims = input_points.shape[-1]
        geometric_inducing_points = initialize_kmeans(input_points, space=space, n=num_inducing)
        euclidean_inducing_points = initialize_kmeans(input_points, space=Euclidean(dim=hidden_output_dims), n=num_inducing)

        
        hidden_layers = torch.nn.ModuleList([
            GeometricDeepGPLayer(
                space=space,
                num_eigenfunctions=num_eigenfunctions,
                output_dims=hidden_output_dims,
                inducing_points=geometric_inducing_points,
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
                    inducing_points=euclidean_inducing_points,
                    nu=nu, 
                    learn_inducing_locations=True,
                    mean='linear',
                    outputscale_prior=outputscale_prior,
                    whitened_variational_strategy=whitened_variational_strategy,
                    sampler_inv_jitter=sampler_inv_jitter,
                )
                for _ in range(num_hidden - 1)
            ],
        ])
        output_layer = EuclideanDeepGPLayer(
            output_dims=output_dims,
            inducing_points=euclidean_inducing_points,
            nu=nu, 
            learn_inducing_locations=True,
            mean='constant',
            whitened_variational_strategy=whitened_variational_strategy,
            sampler_inv_jitter=sampler_inv_jitter,
        )
        super().__init__(hidden_layers=hidden_layers, output_layer=output_layer)
    