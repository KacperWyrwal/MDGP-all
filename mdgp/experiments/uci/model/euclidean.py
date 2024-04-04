from torch import Tensor 


import torch 
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from mdgp.experiments.uci.data.datasets import UCIDataset
from scipy.cluster.vq import kmeans2, ClusterError


# Settings from the paper  
LIKELIHOOD_VARIANCE = 0.01
LENGTHSCALE = 2.0
INNER_LAYER_VARIANCE = 1e-5
OUTPUT_LAYER_VARIANCE = 1.0 # This is a (reasonable) guess
MAX_HIDDEN_DIMS = 30


# From Spherical Harmonic Features
dimension_to_num_inducing_points = {
    4: 336,
    6: 294,
    8: 210,
    13: 119, 
}


def get_hidden_dims(dataset: UCIDataset) -> int:
    return min(MAX_HIDDEN_DIMS, dataset.dimension)


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


def get_inducing_points(dataset: UCIDataset, num_inducing_points: int) -> Tensor:
    """
    Initialize inducing points using kmeans. (from paper)
    """
    return empty_cluster_safe_kmeans(dataset.train_x, num_inducing_points)


class EuclideanDeepGPLayer(DeepGPLayer):
    def __init__(self, inducing_points, output_dims, hidden: bool = False):
        input_dims = inducing_points.size(-1)
        batch_shape = torch.Size([output_dims]) if output_dims is not None else torch.Size([])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0), 
            batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy, input_dims, output_dims)

        base_kernel = RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims)
        base_kernel.lengthscale = LENGTHSCALE
        # Use ard_num_dims=input_dims adds a lengthscale for each input dimension 
        # "we choose the RBF kernel with a lengthscale for each dimension" (from paper)
        self.covar_module = ScaleKernel(base_kernel, batch_shape=batch_shape, ard_num_dims=input_dims)
        if hidden:
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)
            self.covar_module.outputscale = INNER_LAYER_VARIANCE
        else:
            self.mean_module = ConstantMean(batch_shape=batch_shape)
            self.covar_module.outputscale = OUTPUT_LAYER_VARIANCE

    def forward(self, x):
        covar = self.covar_module(x)
        mean = self.mean_module(x)
        return MultivariateNormal(mean, covar)
    

class EuclideanDeepGP(DeepGP):
    def __init__(self, dataset: UCIDataset, num_layers: int, num_inducing_points: int = None):
        super().__init__()
        if num_inducing_points is None:
            num_inducing_points = dimension_to_num_inducing_points[dataset.dimension]
        num_hidden_dims = get_hidden_dims(dataset)
        inducing_points = get_inducing_points(dataset, num_inducing_points)

        self.layers = torch.nn.ModuleList(
            [EuclideanDeepGPLayer(inducing_points, num_hidden_dims, hidden=True) for _ in range(num_layers - 1)] + 
            [EuclideanDeepGPLayer(inducing_points, dataset.num_outputs, hidden=False)]
        )
        self.likelihood = MultitaskGaussianLikelihood(dataset.num_outputs)
        self.likelihood.noise = LIKELIHOOD_VARIANCE

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 
