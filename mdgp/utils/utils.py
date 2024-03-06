from torch import Tensor 


import torch 
from geometric_kernels.spaces import Space, Euclidean, Hypersphere
from linear_operator import LinearOperator


def extrinsic_dimension(space: Space) -> int: 
    if isinstance(space, Euclidean):
        return space.dim
    if isinstance(space, Hypersphere):
        return space.dim + 1
    raise NotImplementedError(f"Extrinsic dimension not implemented for {space.__class__.__name__}")


def test_psd(S: Tensor | LinearOperator, tol: float = 1e-8) -> None:
    if isinstance(S, LinearOperator):
        S = S.to_dense()
    assert torch.allclose(S, S.mT), "K should be symmetric."

    eigs = torch.linalg.eigvalsh(S)
    assert (eigs > -tol).all(), "K should be positive definite."
