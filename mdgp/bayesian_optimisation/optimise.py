import torch 
import pymanopt 
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim.optimize import optimize_acqf
from numpy import asarray 


def get_initial_data(manifold, num_initial_data, target_function, method='random'): 
    if method == 'random':
        x = torch.tensor([manifold.random_point() for _ in range(num_initial_data)])
    elif method == 'uniform':
        if isinstance(manifold, pymanopt.manifolds.Sphere) and manifold.dim == 3:
            from mdgp.utils import sphere_uniform_grid
            x = sphere_uniform_grid(num_initial_data)
        else: 
            raise NotImplementedError
    y = torch.zeros(num_initial_data)
    for i in range(num_initial_data):
        y[i] = target_function(x[i])
    y = y.unsqueeze(-1)

    assert x.requires_grad == False
    assert y.requires_grad == False
    return x, y 


class ManifoldRandomPointGenerator: 
    def __init__(self, manifold): 
        self.manifold = manifold

    def __call__(self, n, q, seed=None): 
        assert q == 1, "Only q=1 is supported"
        if seed is not None: 
            torch.manual_seed(seed)
        res = torch.tensor([self.manifold.random_point() for _ in range(n * q)]).reshape(n, q, -1)
        return res 


def optimize_acqf_manifold(manifold, acq_function, gen_candidates, num_restarts=5, raw_samples=100, q=None): 
    # 1. Get initial conditions 
    batch_initial_conditions = gen_batch_initial_conditions(
        acq_function=acq_function, 
        bounds=torch.tensor([1.]), # This is only used for its shape  
        q=q, 
        num_restarts=num_restarts, 
        raw_samples=raw_samples, 
        generator=ManifoldRandomPointGenerator(manifold),
    )
    assert not batch_initial_conditions.isnan().any(), "Generated nan batch initial conditions"
    batch_initial_conditions = batch_initial_conditions.squeeze(-2)
    # 2. Optimize acquisition function from initial conditions
    best_candidate = optimize_acqf(
        acq_function=acq_function, 
        q=q, 
        bounds=torch.randn(2, batch_initial_conditions.shape[-1]), # This is just a placeholder, not actually used internally
        num_restarts=num_restarts, 
        raw_samples=raw_samples, 
        gen_candidates=gen_candidates, 
        batch_initial_conditions=batch_initial_conditions,  
    )
    return best_candidate


def gen_candidates_manifold(
        initial_conditions,
        acquisition_function,
        manifold,
        optimizer,
        pre_processing_manifold=None,
        post_processing_manifold=None,
        approx_hessian: bool = False,
):
    @pymanopt.function.pytorch(manifold)
    def cost(point):
        assert point.dtype == torch.float64
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


class ManifoldGenCandidates: 
    def __init__(self, manifold, pre_processing_manifold=None, post_processing_manifold=None, approx_hessian=False): 
        self.manifold = manifold
        self.pre_processing_manifold = pre_processing_manifold
        self.post_processing_manifold = post_processing_manifold
        self.approx_hessian = approx_hessian

    def gen_candidates_factory(self, optimizer): 
        def _gen_candidates(initial_conditions, acquisition_function): 
            initial_conditions = asarray(initial_conditions)
            return gen_candidates_manifold(
                initial_conditions=initial_conditions, 
                acquisition_function=acquisition_function, 
                manifold=self.manifold, 
                optimizer=optimizer, 
                pre_processing_manifold=self.pre_processing_manifold, 
                post_processing_manifold=self.post_processing_manifold, 
                approx_hessian=self.approx_hessian, 
                )
        return _gen_candidates