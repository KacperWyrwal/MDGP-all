from dataclasses import dataclass, field
from gpytorch.models.deep_gps import DeepGP
from mdgp.experiments.uci.model.euclidean import EuclideanDeepGP
from mdgp.experiments.uci.model.geometric import SHFDeepGP

# Settings from the paper 
NUM_INDUCING_POINTS = 100


@dataclass
class ModelArguments:
    model_name: str = field(default='euclidean', metadata={'help': 'Name of the model. Must be one of ["geometric_manifold", "euclidean_manifold", "euclidean"]'})
    num_layers: int = field(default=1, metadata={'help': 'Number of layers in the model'})
    num_inducing: int = field(default=NUM_INDUCING_POINTS, metadata={'help': 'Number of inducing points'})

    def get_model(self, dataset) -> DeepGP:
        if self.model_name == 'euclidean':
            return EuclideanDeepGP(dataset, num_layers=self.num_layers, num_inducing_points=self.num_inducing)
        else:
            return 
            raise ValueError(f"Unknown model: {self.model_name}")
        