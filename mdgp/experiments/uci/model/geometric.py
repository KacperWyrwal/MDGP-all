from torch import Tensor 


import torch 
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from mdgp.experiments.uci.data.datasets import UCIDataset
from mdgp.utils.sphere import sphere_kmeans_centers


# Settings from the paper  
LIKELIHOOD_VARIANCE = 0.01
LENGTHSCALE = 2.0
INNER_LAYER_VARIANCE = 1e-5
OUTPUT_LAYER_VARIANCE = 1.0 # This is a (reasonable) guess
NUM_INDUCING_POINTS = 100
MAX_HIDDEN_DIMS = 30


def get_hidden_dims(dataset: UCIDataset) -> int:
    return dataset.dimension + 1


def get_inducing_points(dataset: UCIDataset, num_inducing_points: int) -> Tensor:
    """
    Initialize inducing points using kmeans. (from paper)
    """
    return sphere_kmeans_centers(dataset.train_x, num_inducing_points)


class GeometricDeepGPLayer(DeepGPLayer):
    def __init__(self, num_inducing_points, output_dims, hidden: bool = False):
        input_dims = inducing_points.size(-1)
        batch_shape = torch.Size([output_dims]) if output_dims is not None else torch.Size([])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing_points, 
            batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy, input_dims, output_dims)

        base_kernel = RBFKernel(batch_shape=batch_shape)
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
    

class GeometricDeepGP(DeepGP):
    def __init__(self, dataset: UCIDataset, num_layers: int, num_inducing_points: int = NUM_INDUCING_POINTS):
        super().__init__()
        num_hidden_dims = get_hidden_dims(dataset)
        inducing_points = get_inducing_points(dataset, num_inducing_points)

        self.layers = torch.nn.ModuleList(
            [GeometricDeepGPLayer(inducing_points, num_hidden_dims, hidden=True) for _ in range(num_layers - 1)] + 
            [GeometricDeepGPLayer(inducing_points, None, hidden=False)]
        )
        self.likelihood = GaussianLikelihood()
        self.likelihood.noise = LIKELIHOOD_VARIANCE

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 
