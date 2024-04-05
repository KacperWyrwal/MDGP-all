from dataclasses import dataclass, field
from gpytorch.models.deep_gps import DeepGP
from mdgp.experiments.uci.model.euclidean import EuclideanDeepGP
from mdgp.experiments.uci.model.geometric import SHFDeepGP
from mdgp.experiments.uci.model.projectors import Projector, IdentityProjector, SphereProjector


@dataclass
class ModelArguments:
    model_name: str = field(default='euclidean', metadata={'help': 'Name of the model. Must be one of ["geometric_manifold", "euclidean_manifold", "euclidean"]'})
    num_layers: int = field(default=1, metadata={'help': 'Number of layers in the model'})
    num_inducing: int | None = field(default=None, metadata={'help': 'Number of inducing points. None to use the same value as the number of spherical harmonics in the Spherical Harmonic Features model.'})
    learn_inducing: bool = field(default=True, metadata={'help': 'Whether to learn the inducing point locations in the Euclidean model.'})

    def get_model(self, dataset) -> DeepGP:
        if self.model_name == 'euclidean':
            return EuclideanDeepGP(dataset, num_layers=self.num_layers, num_inducing_points=self.num_inducing, learn_inducing_locations=self.learn_inducing)
        elif self.model_name == 'geometric':
            return SHFDeepGP(dataset, num_layers=self.num_layers)
        raise ValueError(f"Unknown model name: {self.model_name}")

    def get_projector(self) -> Projector:
        if self.model_name == 'euclidean':
            return IdentityProjector()
        elif self.model_name == 'geometric':
            return SphereProjector()
        raise ValueError(f"Unknown model name: {self.model_name}")
        