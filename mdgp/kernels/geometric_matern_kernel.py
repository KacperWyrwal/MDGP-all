import torch 
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import DiscreteSpectrumSpace
from mdgp.kernels.geometric_kernels_wrappers import GPytorchGeometricKernel


class GeometricMaternKernel(GPytorchGeometricKernel):

    def __init__(self, space: DiscreteSpectrumSpace, nu: float = 2.5, num_eigenfunctions: int = 10, batch_shape=torch.Size([]), optimize_nu=False, **kwargs):
        geometric_kernel = MaternKarhunenLoeveKernel(space=space, num_eigenfunctions=num_eigenfunctions)
        super().__init__(geometric_kernel=geometric_kernel, nu=nu, optimize_nu=optimize_nu, batch_shape=batch_shape, **kwargs)
