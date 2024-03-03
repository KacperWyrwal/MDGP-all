from torch import Tensor 
from gpytorch.priors import Prior 


import torch 
import gpytorch 
from geometric_kernels.spaces import Space
from mdgp.models.deep_gps import GeometricDeepGPLayer, EuclideanDeepGPLayer, ManifoldToManifoldDeepGPLayer
from mdgp.utils import extrinsic_dimension
from mdgp.variational.variational_strategy_factory import VariationalStrategyFactory


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

    def __init__(self, 
            hidden_gps, 
            output_layer, 
            space, 
            to_tangent: str = 'project', # TODO maybe change into an enum
        ) -> None:
        # TODO propagate the change and merging of parameterised frame and rotated frame into to_tangent 
        hidden_layers = torch.nn.ModuleList([
            ManifoldToManifoldDeepGPLayer(gp=gp, space=space, to_tangent=to_tangent)
            for gp in hidden_gps
        ])
        super().__init__(hidden_layers=hidden_layers, output_layer=output_layer)

    def forward_return_hidden(self, 
            x: Tensor, 
            are_samples: bool = False, 
            sample_hidden: str = 'elementwise', 
            sample_output: bool = False, 
            mean: bool = False, 
            resample_weights=True,
        ) -> tuple[list[dict[str, Tensor]], Tensor]:
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
    def __init__(self,
        space, 
        num_hidden: int,
        variational_strategy_factory: VariationalStrategyFactory,
        output_dims: int | None = None, 
        to_tangent: str = 'project',
        nu: float = 2.5, 
        optimize_nu: bool = False,
        outputscale_prior: Prior | None = None,
        sampler_inv_jitter: float | None = None,
        num_eigenfunctions: int | None = None, 
        num_random_phases: int | None = None,
        ) -> None:
        input_dims = extrinsic_dimension(space)
        hidden_dims = get_output_dim(to_tangent, space)
        hidden_gps = [
            GeometricDeepGPLayer(
                space=space,
                input_dims=input_dims,
                variational_strategy_factory=variational_strategy_factory,
                output_dims=hidden_dims,
                mean='zero',
                nu=nu,
                optimize_nu=optimize_nu,
                outputscale_prior=outputscale_prior,
                sampler_inv_jitter=sampler_inv_jitter,
                num_eigenfunctions=num_eigenfunctions,
                num_random_phases=num_random_phases,
            )
            for _ in range(num_hidden)
        ]

        output_layer = GeometricDeepGPLayer(
            space=space,
            input_dims=input_dims,
            variational_strategy_factory=variational_strategy_factory,
            output_dims=output_dims,
            mean='constant',
            nu=nu,
            optimize_nu=optimize_nu,
            outputscale_prior=None,
            sampler_inv_jitter=sampler_inv_jitter,
            num_eigenfunctions=num_eigenfunctions,
            num_random_phases=num_random_phases,
        )

        super().__init__(hidden_gps=hidden_gps, output_layer=output_layer, to_tangent=to_tangent, space=space)


class ResidualEuclideanDeepGP(ResidualDeepGP):

    def __init__(self,
        space: Space,
        num_hidden: int,
        variational_strategy_factory: VariationalStrategyFactory,
        to_tangent: str = 'project',
        output_dims = None, 
        nu: float = 2.5, 
        outputscale_prior=None,
        sampler_inv_jitter: float | None = None,
        num_eigenfunctions: int | None = None,
        num_random_phases: int | None = None,
        ) -> None:
        input_dims = extrinsic_dimension(space)
        hidden_dims = get_output_dim(to_tangent, space)

        hidden_gps = [
            EuclideanDeepGPLayer(
                input_dims=input_dims,
                variational_strategy_factory=variational_strategy_factory,
                output_dims=hidden_dims,
                mean='zero',
                nu=nu,
                outputscale_prior=outputscale_prior,
                sampler_inv_jitter=sampler_inv_jitter,
                num_eigenfunctions=num_eigenfunctions,
                num_random_phases=num_random_phases,
            )
            for _ in range(num_hidden)
        ]

        # We learn inducing points, since we cannot just uniformly cover the Euclidean space 
        output_layer = EuclideanDeepGPLayer(
            input_dims=input_dims,
            variational_strategy_factory=variational_strategy_factory,
            output_dims=output_dims,
            mean='constant',
            nu=nu,
            outputscale_prior=outputscale_prior,
            sampler_inv_jitter=sampler_inv_jitter,
            num_eigenfunctions=num_eigenfunctions,
            num_random_phases=num_random_phases,
        )
        super().__init__(hidden_gps=hidden_gps, output_layer=output_layer, to_tangent=to_tangent, space=space)


class EuclideanDeepGP(DeepGP):

    def __init__(self,
        space: Space,
        num_hidden: int,
        variational_strategy_factory: VariationalStrategyFactory,
        output_dims = None, 
        nu: float = 2.5, 
        outputscale_prior=None,
        sampler_inv_jitter: float | None = None,
        num_eigenfunctions: int | None = None,
        num_random_phases: int | None = None,
    ) -> None:
        input_dims = extrinsic_dimension(space)
        hidden_dims = input_dims
        hidden_layers = torch.nn.ModuleList([
            EuclideanDeepGPLayer(
                input_dims=input_dims,
                variational_strategy_factory=variational_strategy_factory,
                output_dims=hidden_dims,
                mean='linear',
                nu=nu,
                outputscale_prior=outputscale_prior,
                sampler_inv_jitter=sampler_inv_jitter,
                num_eigenfunctions=num_eigenfunctions,
                num_random_phases=num_random_phases,
            )
            for _ in range(num_hidden)
        ])

        output_layer = EuclideanDeepGPLayer(
            input_dims=input_dims,
            variational_strategy_factory=variational_strategy_factory,
            output_dims=output_dims,
            mean='constant',
            nu=nu,
            outputscale_prior=outputscale_prior,
            sampler_inv_jitter=sampler_inv_jitter,
            num_eigenfunctions=num_eigenfunctions,
            num_random_phases=num_random_phases,
        )
        super().__init__(hidden_layers=hidden_layers, output_layer=output_layer)
    

class GeometricHeadDeepGP(DeepGP):

    def __init__(
        self,
        space, 
        num_hidden: int,
        variational_strategy_factory: VariationalStrategyFactory,
        output_dims: int | None = None, 
        nu: float = 2.5, 
        optimize_nu: bool = False,
        outputscale_prior=None,
        sampler_inv_jitter: float = 10e-8,
        num_eigenfunctions: int | None = None, 
        num_random_phases: int | None = None,
    ) -> None:
        input_dims = extrinsic_dimension(space)
        hidden_dims = input_dims
        
        hidden_layers = torch.nn.ModuleList([
            GeometricDeepGPLayer(
                space=space,
                input_dims=input_dims,
                variational_strategy_factory=variational_strategy_factory,
                output_dims=hidden_dims,
                mean='linear',
                nu=nu,
                optimize_nu=optimize_nu,
                outputscale_prior=outputscale_prior,
                sampler_inv_jitter=sampler_inv_jitter,
                num_eigenfunctions=num_eigenfunctions,
                num_random_phases=num_random_phases,
            ), 
            *[
                EuclideanDeepGPLayer(
                    input_dims=input_dims,
                    variational_strategy_factory=variational_strategy_factory,
                    output_dims=hidden_dims,
                    mean='linear',
                    nu=nu,
                    outputscale_prior=outputscale_prior,
                    sampler_inv_jitter=sampler_inv_jitter,
                    num_eigenfunctions=num_eigenfunctions,
                    num_random_phases=num_random_phases,
                )
                for _ in range(num_hidden - 1)
            ],
        ])
        output_layer = EuclideanDeepGPLayer(
            input_dims=input_dims,
            variational_strategy_factory=variational_strategy_factory,
            output_dims=output_dims,
            mean='constant',
            nu=nu,
            outputscale_prior=outputscale_prior,
            sampler_inv_jitter=sampler_inv_jitter,
            num_eigenfunctions=num_eigenfunctions,
            num_random_phases=num_random_phases,
        )
        super().__init__(hidden_layers=hidden_layers, output_layer=output_layer)
    