from torch import Tensor 


import torch 
import gpytorch 
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from scipy.cluster.vq import kmeans2, ClusterError

from mdgp.experiments.sphere.data import SphereDataset



# Settings from the paper  
LIKELIHOOD_VARIANCE = 1.0
LENGTHSCALE = 1.0
INNER_LAYER_VARIANCE = 1e-5
OUTPUT_LAYER_VARIANCE = 1.0 # This is a (reasonable) guess
MAX_HIDDEN_DIMS = 30


def empty_cluster_safe_kmeans(x: Tensor, k: int, num_retries: int = 1000) -> Tensor:
    """
    Initialize inducing points using kmeans. (from paper)
    """
    for _ in range(num_retries):
        try:
            return torch.from_numpy(kmeans2(x, k, missing='raise')[0]).to(x.device, x.dtype)
        except ClusterError:
            continue 
    
    return torch.from_numpy(kmeans2(x, k, missing='warn')[0]).to(x.device, x.dtype)


def get_inducing_points(dataset: SphereDataset, num_inducing: int) -> Tensor:
    """
    Initialize inducing points using kmeans. (from paper)
    """
    return empty_cluster_safe_kmeans(dataset.train_x, k=num_inducing)


class EuclideanDeepGPLayer(DeepGPLayer):
    def __init__(self, inducing_points, outputscale_prior_mean: float, hidden: bool = False, learn_inducing_locations: bool = True):
        inducing_points = inducing_points.clone()
        input_dims = inducing_points.size(-1)
        output_dims = 3 if hidden else None
        batch_shape = torch.Size([output_dims]) if output_dims is not None else torch.Size([])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0), 
            batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy, input_dims, output_dims)

        base_kernel = RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims)
        self.covar_module = ScaleKernel(
            base_kernel, 
            batch_shape=batch_shape, 
            ard_num_dims=input_dims,
            outputscale_prior=gpytorch.priors.GammaPrior(concentration=1.0, rate=1 / outputscale_prior_mean),
        )
        if hidden:
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape, bias=False)
        else:
            self.mean_module = ConstantMean(batch_shape=batch_shape)

        # Initialize module parameters
        self.covar_module.base_kernel.lengthscale = LENGTHSCALE
        self.covar_module.outputscale = outputscale_prior_mean

    def forward(self, x, *args, **kwargs):
        covar = self.covar_module(x)
        mean = self.mean_module(x)
        return MultivariateNormal(mean, covar)
    

class EuclideanDeepGP(DeepGP):
    def __init__(self, dataset: SphereDataset, outputscale_prior_mean: float, num_layers: int, 
                 num_inducing_points: int = None, learn_inducing_locations: bool = True):
        super().__init__()
        inducing_points = get_inducing_points(dataset, num_inducing_points)

        self.layers = torch.nn.ModuleList(
            [EuclideanDeepGPLayer(
                inducing_points, 
                outputscale_prior_mean=outputscale_prior_mean, 
                hidden=True,
                learn_inducing_locations=learn_inducing_locations) 
                for _ in range(num_layers - 1)] + 
            [EuclideanDeepGPLayer(inducing_points, outputscale_prior_mean=outputscale_prior_mean, 
                                  learn_inducing_locations=learn_inducing_locations, hidden=False)]
        )
        self.likelihood = GaussianLikelihood()
        self.likelihood.noise = LIKELIHOOD_VARIANCE

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 
