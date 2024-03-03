from torch import Tensor 
from gpytorch.models import ApproximateGP
from geometric_kernels.spaces import Space
from gpytorch.variational import VariationalStrategy
from mdgp.variational.spherical_harmonic_features_variational_strategy import SphericalHarmonicFeaturesVariationalStrategy


import torch 
from gpytorch.variational import CholeskyVariationalDistribution
from geometric_kernels.spaces import Hypersphere
from mdgp.utils.spherical_harmonic_features import num_spherical_harmonics_to_degree
from mdgp.variational.inducing_points import initialize_kmeans



class VariationalStrategyFactory:
    def __init__(self, name: str, num_inducing: int, inputs: Tensor | None = None, 
        learn_inducing_locations: bool = False,
    ) -> None: 
        self.name = name 
        self.learn_inducing_locations = learn_inducing_locations
        self.num_inducing_variables = num_inducing

        if name == 'points': 
            assert inputs is not None, 'Must provide inputs to use points variational strategy.'
        self.inputs = inputs 

    @staticmethod
    def make_variational_strategy(
        name: str, model: ApproximateGP, space: Space, num_inducing_variables: int, inputs: Tensor, 
        learn_inducing_locations: bool = False, batch_shape: torch.Size = torch.Size([]),
    ) -> VariationalStrategy | SphericalHarmonicFeaturesVariationalStrategy: 
        if name == 'points': 
            inducing_points = initialize_kmeans(x=inputs, n=num_inducing_variables, space=space)
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=num_inducing_variables,
                batch_shape=batch_shape,
            )
            variational_strategy = VariationalStrategy(
                model=model, inducing_points=inducing_points, 
                variational_distribution=variational_distribution, 
                learn_inducing_locations=learn_inducing_locations,
            )
        elif name == 'harmonics': 
            assert isinstance(space, Hypersphere), f'Harmonic features only implemented for hyperspheres, not {space}.'
            dimension = space.dim + 1
            num_spherical_harmonics = num_spherical_harmonics_to_degree(
                num_spherical_harmonics=num_inducing_variables, dimension=dimension
            )[1]
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=num_spherical_harmonics,
                batch_shape=batch_shape,
            )
            variational_strategy = SphericalHarmonicFeaturesVariationalStrategy(
                model=model, 
                variational_distribution=variational_distribution, 
                dimension=dimension,
                num_spherical_harmonics=num_spherical_harmonics,
            )
        else: 
            raise ValueError(f'Variational strategy {name} not recognized. Must be one of ["points", "harmonics"].')
        
        return variational_strategy
    
    def __call__(self, model: ApproximateGP, space: Space, batch_shape: torch.Size = torch.Size([])) -> VariationalStrategy:
        return self.make_variational_strategy(
            name=self.name, model=model, space=space, num_inducing_variables=self.num_inducing_variables, 
            inputs=self.inputs, learn_inducing_locations=self.learn_inducing_locations, batch_shape=batch_shape,
        )
