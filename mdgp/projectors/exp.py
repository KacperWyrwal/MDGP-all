from geometric_kernels.spaces import Space
from torch import Tensor
from torch.nn import Module


class Exp(Module):
    """
    Projects vectors in the tangent space into the manifold with the exponential map.

    Args:
        space (Space): The manifold space.

    Attributes:
        space (Space): The manifold space.

    """

    def __init__(self, space: Space) -> None: 
        super().__init__()
        self.space = space 

    def forward(self, base_points: Tensor, tangent_vectors: Tensor) -> Tensor: 
        """
        Forward pass of the Exp module.

        Args:
            base_points (Tensor): The base points on the manifold.
            tangent_vectors (Tensor): The tangent vectors in the tangent space.

        Returns:
            Tensor: The projected vectors in the manifold.

        """
        return self.space.metric.exp(tangent_vectors, base_points)
