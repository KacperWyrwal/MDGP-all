# Imports 
from dataclasses import dataclass, field
from mdgp.experiments.sphere.model.euclidean import EuclideanDeepGP
from mdgp.experiments.sphere.data import SphereDataset
from mdgp.experiments.sphere.model.geometric import FullyGeometricDeepGP, InputGeometricDeepGP


@dataclass
class ModelArguments:
    model_name: str = field(
        default='fully_geometric', 
        metadata={'help': 'Name of the model. Must be one of ["fully_geometric", "euclidean", "input_geometric"]'}
    )
    num_layers: int = field(default=1, metadata={'help': 'Number of layers'})
    num_inducing: int = field(default=60, metadata={'help': 'Number of inducing points. Should be one of [60, 300]'})
    gvf: str = field(default='projected', metadata={'help': 'Method of mapping from the manifold to the tangent space. Must be one of ["projected", "frame"]'})
    outputscale_mean: float = field(default=0.01, metadata={'help': 'Mean of the outputscale'})
    learn_inducing_locations: bool = field(default=True, metadata={'help': 'Whether to learn inducing locations'})

    def __post_init__(self):
        assert self.model_name in ["fully_geometric", "euclidean", "input_geometric"]
        assert self.gvf in ["projected", "frame", None]
        assert self.num_inducing in [60, 300]
    
    def get_model(self, dataset: SphereDataset):
        if self.model_name == 'fully_geometric':
            return FullyGeometricDeepGP(
                dataset=dataset,
                num_inducing_points=self.num_inducing,
                num_layers=self.num_layers,
                outputscale_prior_mean=self.outputscale_mean,
                gvf=self.gvf,
            )
        if self.model_name == 'input_geometric':
            return InputGeometricDeepGP(
                dataset=dataset, 
                outputscale_prior_mean=self.outputscale_mean,
                num_layers=self.num_layers,
                gvf=self.gvf,
                num_inducing_points=self.num_inducing,
            )
        if self.model_name == 'euclidean': 
            return EuclideanDeepGP(
                dataset=dataset,
                outputscale_prior_mean=self.outputscale_mean,
                num_inducing_points=self.num_inducing,
                num_layers=self.num_layers, 
                learn_inducing_locations=self.learn_inducing_locations,
            )
