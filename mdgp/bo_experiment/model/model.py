# Type hints
from torch import Tensor

# Imports 
import torch 
import warnings
from dataclasses import dataclass, field
from mdgp.bo_experiment.utils import space_class_from_name
from mdgp.models.deep_gps import GeometricManifoldDeepGP
from mdgp.models.exact_gps import GeometricManifoldExactGP
from gpytorch.priors import GammaPrior
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO, ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, AnalyticAcquisitionFunction
from mdgp.bo_experiment.model.acquisition import DeepAnalyticAcquisitionFunction
from mdgp.bo_experiment.model.botorch import BotorchGP
from mdgp.bo_experiment.utils import ExcludeFromNameMixin


@dataclass
class ModelArguments(ExcludeFromNameMixin):
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
    acqf_name: str = field(default="log_expected_improvement", metadata={"help": "Name of the acquisition function to use"})
    acqf_maximize: bool = field(default=False, metadata={"help": "Whether to maximize or minimize the target function"})
    posterior_sample_method: str = field(default="pathwise", metadata={"help": "Name of the method to sample from the posterior. Must be one of ['pathwise', 'naive']"})


    def __post_init__(self):
        super().__post_init__()
        if self.model_name == 'deep' and self.num_hidden: 
            warnings.warn("A deep model with no hidden layers is specificied. Are you sure this is what you want?")

    @property 
    def space(self):
        return space_class_from_name(self.space_name)(dim=self.space_dim)
    
    @property
    def outputscale_prior(self):
        return GammaPrior(concentration=1.0, rate=1 / self.outputscale_mean)
    
    @property 
    def optimizer_factory(self): 
        if self.model_name == 'exact':
            maximize = False 
        elif self.model_name == 'deep':
            maximize = True 
        else:
            raise ValueError(f"Unknown model name {self.model_name}")
        def get_optimizer(model, lr): 
            return torch.optim.Adam(model.parameters(), lr=lr, maximize=maximize)
        return get_optimizer
    
    @property
    def mll_factory(self): 
        if self.model_name == 'exact': 
            def get_mll(model, y: Tensor | None = None): 
                return ExactMarginalLogLikelihood(model.likelihood, model)
            return get_mll
        if self.model_name == 'deep':
            def get_mll(model, y: Tensor):
                return DeepApproximateMLL(
                    VariationalELBO(model=model, likelihood=model.likelihood, num_data=y.numel())
                )
            return get_mll
        raise ValueError(f"Unknown model name {self.model_name}")
    
    @property 
    def acqf_factory(self): 
        if self.model_name == 'exact':
            def get_acqf(model, best_f, posterior_transform=None) -> AnalyticAcquisitionFunction: 
                return acqf_class_from_name(self.acqf_name)(
                    model=model, best_f=best_f, 
                    maximize=self.acqf_maximize, posterior_transform=posterior_transform
                )
            return get_acqf 
        if self.model_name == 'deep':
            def get_acqf(model, best_f, posterior_transform=None) -> AnalyticAcquisitionFunction: 
                return DeepAnalyticAcquisitionFunction(acqf_class_from_name(self.acqf_name)(
                    model=model, best_f=best_f, 
                    maximize=self.acqf_maximize, posterior_transform=posterior_transform
                ))
            return get_acqf
    

def acqf_class_from_name(name):
    if name == "log_expected_improvement":
        return LogExpectedImprovement
    if name == "expected_improvement":
        return ExpectedImprovement
    raise ValueError(f"Unknown acquisition function {name}")


def create_model(model_args: ModelArguments, inducing_points: Tensor | None = None,
                 train_x: Tensor | None = None, train_y: Tensor | None = None):
    if model_args.model_name == 'deep':
        return BotorchGP(
            GeometricManifoldDeepGP(
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
            ), 
            posterior_sample_method=model_args.posterior_sample_method,
        )
    if model_args.model_name == 'exact': 
        return BotorchGP(
            GeometricManifoldExactGP(
                train_x=train_x,
                train_y=train_y,
                space=model_args.space,
                nu=model_args.nu,
                trainable_nu=model_args.optimize_nu,
                num_eigenfunctions=model_args.num_eigenfunctions,
            ), 
            posterior_sample_method=model_args.posterior_sample_method
        )
    raise ValueError(f"Unknown model name: {model_args.model_name}. Must be one of ['deep', 'exact']")
