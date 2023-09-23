from pymanopt.manifolds.manifold import Manifold
from pymanopt.optimizers.optimizer import Optimizer
from torch import Tensor 

import torch 
import pymanopt 


class ManifoldRandomPointGenerator: 
    def __init__(self, manifold: Manifold): 
        self.manifold = manifold

    def __call__(self, n, q, seed=None): 
        assert q == 1, "Only q=1 is supported"
        if seed is not None: 
            torch.manual_seed(seed)
        res = torch.tensor([self.manifold.random_point() for _ in range(n * q)]).reshape(n, q, -1)
        return res 


def gen_candidates_manifold(
        initial_conditions: Tensor,
        acquisition_function,
        manifold,
        optimizer: Optimizer,
        pre_processing_manifold=None,
        post_processing_manifold=None,
        approx_hessian: bool = False,
):
    @pymanopt.function.pytorch(manifold)
    def cost(point: Tensor):
        assert point.dtype == torch.float64, "Optimization should be done with float64"
        with torch.set_grad_enabled(point.requires_grad):
            loss = -acquisition_function(point.unsqueeze(0)).sum()
            return loss

    # Instantiate the problem on the manifold
    problem = pymanopt.Problem(manifold=manifold, cost=cost)

    # Solve problem on the manifold for each of the initial conditions
    nb_initial_conditions = initial_conditions.shape[0]
    candidates = torch.zeros(nb_initial_conditions, *initial_conditions.shape[1:])

    # TODO this does not handle the case where q!=1
    for i in range(nb_initial_conditions):
        opt_x = optimizer.run(problem, initial_point=initial_conditions[i]).point
        candidates[i] = torch.tensor(opt_x[None])
    batch_acquisition = acquisition_function(candidates.unsqueeze(-2))
    return candidates, batch_acquisition
