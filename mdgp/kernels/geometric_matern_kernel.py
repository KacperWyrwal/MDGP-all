from geometric_kernels.frontends.pytorch.gpytorch import GPytorchGeometricKernel
from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel


class GeometricMaternKernel(GPytorchGeometricKernel): 
    def __init__(self, space, lengthscale=1.0, nu=2.5, trainable_nu=True, num_eigenfunctions=35, normalize=True, **kwargs): 
        geometric_kernel = MaternKarhunenLoeveKernel(
            space=space, 
            num_eigenfunctions=num_eigenfunctions, 
            normalize=normalize, 
        )
        super().__init__(geometric_kernel, lengthscale=lengthscale, nu=nu, trainable_nu=trainable_nu, **kwargs)
        