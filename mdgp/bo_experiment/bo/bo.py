from pymanopt.manifolds.manifold import Manifold
from pymanopt.optimizers.optimizer import Optimizer
from torch import Tensor 


import torch 
from botorch.optim import gen_batch_initial_conditions, optimize_acqf
from pymanopt.manifolds import Sphere 
from pymanopt import optimizers
from dataclasses import dataclass, field 
from mdgp.bo_experiment.bo.manifold_botorch import ManifoldRandomPointGenerator, gen_candidates_manifold
from mdgp.bo_experiment.utils import ExcludeFromNameMixin


@dataclass 
class BOArguments(ExcludeFromNameMixin): 
    manifold_name: str = field(default="sphere", metadata={"help": "Name of the manifold to optimize on"})
    manifold_dim: int = field(default=2, metadata={"help": "Dimension of the manifold"})
    num_iter: int = field(default=200, metadata={"help": "Number of iterations"})
    num_restarts: int = field(default=5, metadata={"help": "Number of restarts"})
    raw_samples: int = field(default=100, metadata={"help": "Number of raw samples"})
    q: int = field(default=1, metadata={"help": "Number of candidates to generate at each iteration"})
    optimizer_name: str = field(default="steepest_descent", metadata={"help": "Name of the optimizer to use"})
    optimizer_max_iterations: int = field(default=100, metadata={"help": "Maximum number of iterations for the optimizer"})
    optimizer_verbosity: int = field(default=0, metadata={"help": "Verbosity of the optimizer"})
    switch_to_deep_iter: int | None = field(default=None, metadata={"help": "Switch to deep model after this iteration"})

    def __post_init__(self):
        super().__post_init__()
        self.exclude_from_name(["manifold_name", "manifold_dim"])

    @property
    def manifold(self) -> Manifold: 
        # Pymanopt manifolds are intialized with the dimension of the embedding space, not the intrinsic dimension
        return manifold_class_from_name(self.manifold_name)(self.manifold_dim + 1) 
    
    @property
    def optimizer(self) -> Optimizer:
        return optimizer_class_from_name(self.optimizer_name)(
            max_iterations=self.optimizer_max_iterations, 
            verbosity=self.optimizer_verbosity
        )


def manifold_class_from_name(name):
    if name == "sphere":
        return Sphere
    raise ValueError(f"Unknown manifold {name}")


def optimizer_class_from_name(name):
    if name == "steepest_descent":
        return optimizers.SteepestDescent
    raise ValueError(f"Unknown optimizer {name}")


class ManifoldGenCandidates: 
    def __init__(self, bo_args: BOArguments): 
        self.bo_args = bo_args

    def __call__(self, initial_conditions: Tensor, acquisition_function): 
        # FIXME actually it seems like we don't need to covert to numpy here
        # initial_conditions = initial_conditions.detach().numpy()  
        return gen_candidates_manifold(
            initial_conditions=initial_conditions, 
            acquisition_function=acquisition_function, 
            manifold=self.bo_args.manifold,
            optimizer=self.bo_args.optimizer, 
            pre_processing_manifold=None, 
            post_processing_manifold=None,
            approx_hessian=False, 
        )


def optimize_acqf_manifold(acq_function, bo_args: BOArguments):
    # 1. Get initial conditions 
    batch_initial_conditions = gen_batch_initial_conditions(
        acq_function=acq_function, 
        bounds=torch.tensor([1.]), # This is only used for its shape  
        q=bo_args.q, 
        num_restarts=bo_args.num_restarts, 
        raw_samples=bo_args.raw_samples, 
        generator=ManifoldRandomPointGenerator(bo_args.manifold),
    )
    assert not batch_initial_conditions.isnan().any(), "Generated nan batch initial conditions"
    batch_initial_conditions = batch_initial_conditions.squeeze(-2)
    # 2. Optimize acquisition function from initial conditions
    best_candidate = optimize_acqf(
        acq_function=acq_function, 
        q=bo_args.q, 
        bounds=torch.randn(2, batch_initial_conditions.shape[-1]), # This is just a placeholder, not actually used internally
        num_restarts=bo_args.num_restarts, 
        raw_samples=bo_args.raw_samples, 
        gen_candidates=ManifoldGenCandidates(bo_args=bo_args), 
        batch_initial_conditions=batch_initial_conditions,  
    )
    return best_candidate