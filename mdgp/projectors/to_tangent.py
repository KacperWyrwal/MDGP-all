from geometric_kernels.spaces import Space
from torch import Tensor
from torch.nn import Module


class ToTangent(Module): 
    """
    Linearly projects vectors in the ambient space into the tangent space.

    Args:
        space (Space): The ambient space in which the vectors reside.
    """

    def __init__(self, space: Space) -> None: 
        super().__init__()
        self.space = space

    def forward(self, base_points: Tensor, ambient_vectors: Tensor) -> Tensor: 
        return self.space.to_tangent(ambient_vectors, base_points)
