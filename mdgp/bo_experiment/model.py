# Type hints
from torch import Tensor

# Imports 
import torch 
from dataclasses import dataclass, field
from mdgp.bo_experiment.utils import space_class_from_name
from mdgp.models.deep_gps import GeometricManifoldDeepGP
from mdgp.models.exact_gps import GeometricManifoldExactGP
from gpytorch.priors import GammaPrior
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO, ExactMarginalLogLikelihood


@dataclass
class ModelArguments:
    space_name: str = field(default='hypersphere', metadata={'help': 'The space where the data lives'})
    space_dim: int = field(default=2, metadata={'help': 'The dimension of the space where the data lives'})
    model_name: str = field(default='deep', metadata={'help': 'Name of the model. Must be one of ["geometric_manifold", "euclidean_manifold", "euclidean"]'})
    num_hidden: int = field(default=1, metadata={'help': 'Number of hidden layers'})
    num_eigenfunctions: int = field(default=20, metadata={'help': 'Number of eigenfunctions to use'})
    learn_inducing_locations: bool = field(default=False, metadata={'help': 'Whether to learn the inducing locations'})
    optimize_nu: bool = field(default=True, metadata={'help': 'Whether to optimize the smoothness parameter'})
    nu: float = field(default=2.5, metadata={'help': 'Smoothness parameter'})
    tangent_to_manifold: str = field(default='exp', metadata={'help': 'Name of the function to map from the tangent space to the manifold. Must be one of ["exp", "log"]'})
    project_to_tangent: str = field(default='intrinsic', metadata={'help': 'Name of the function to map from the manifold to the tangent space. Must be one of ["intrinsic", "extrinsic"]'})
    parametrised_frame: bool = field(default=False, metadata={'help': 'Whether to use a parametrised frame'})
    rotated_frame: bool = field(default=False, metadata={'help': 'Whether to use a rotated frame'})
    outputscale_mean: float = field(default=1.0, metadata={'help': 'Mean of the outputscale'})

    def dict_factory(self, x):
        return {k: v for k, v in x 
                if self.__dataclass_fields__[k].metadata.get('exclude_from_asdict', False) is False}

    @property 
    def space(self):
        return space_class_from_name(self.space_name)(dim=self.space_dim)
    
    @property
    def outputscale_prior(self):
        return GammaPrior(concentration=1.0, rate=1 / self.outputscale_mean)
    
    @property 
    def optimizer_factory(self): 
        def get_optimizer(model, lr): 
            return torch.optim.Adam(model.parameters(), lr=lr)
        return get_optimizer
    
    @property
    def mll_factory(self): 
        if self.model_name == 'exact': 
            def get_mll(model, y: Tensor | None = None): 
                return ExactMarginalLogLikelihood(model.likelihood, model)
            return get_mll
        if self.model_name == 'deep' and self.num_hidden == 0:
            def get_mll(model, y: Tensor):
                return VariationalELBO(model=model, likelihood=model.likelihood, num_data=y.numel())
            return get_mll
        if self.model_name == 'deep' and self.num_hidden > 0:
            def get_mll(model, y: Tensor):
                return DeepApproximateMLL(
                    VariationalELBO(model=model, likelihood=model.likelihood, num_data=y.numel())
                )
            return get_mll
        raise ValueError(f"Unknown model name {self.model_name}")


def create_model(inducing_points, model_args: ModelArguments, 
                 train_x: Tensor | None = None, train_y: Tensor | None = None):
    if model_args.model_name == 'deep':
        return GeometricManifoldDeepGP(
            inducing_points=inducing_points, 
            space=model_args.space, 
            outputscale_prior=model_args.outputscale_prior, 
            num_hidden=model_args.num_hidden, 
            num_eigenfunctions=model_args.num_eigenfunctions, 
            learn_inducing_locations=model_args.learn_inducing_locations, 
            optimize_nu=model_args.optimize_nu, 
            nu=model_args.nu,
            project_to_tangent=model_args.project_to_tangent, 
            tangent_to_manifold=model_args.tangent_to_manifold,
        )
    if model_args.model_name == 'exact': 
        return GeometricManifoldExactGP(
            train_x=train_x,
            train_y=train_y,
            space=model_args.space,
            nu=model_args.nu,
            trainable_nu=model_args.optimize_nu,
            num_eigenfunctions=model_args.num_eigenfunctions,
        )
    raise ValueError(f"Unknown model name: {model_args.model_name}. Must be one of ['deep', 'exact']")
