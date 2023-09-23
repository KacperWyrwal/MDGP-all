from torch import Tensor 
from geometric_kernels.spaces import Space


import gpytorch
from mdgp.kernels import GeometricMaternKernel

class GeometricManifoldExactGP(gpytorch.models.ExactGP): 
    """
    The simplest possible GP model with a GeometricMaternKernel taking in a space parameter.

    TODO: We might need to do some reshaping in the forward function depending on the BO algo. 
    """
    def __init__(self, train_x: Tensor, train_y: Tensor, space: Space, nu: float = 2.5, 
                 trainable_nu: bool = True, num_eigenfunctions: int = 20, 
                 normalize: bool = True, lengthscale: float = 1.):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = GeometricMaternKernel(
            space=space, 
            nu=nu, 
            trainable_nu=trainable_nu,
            num_eigenfunctions=num_eigenfunctions,
            normalize=normalize,
            lengthscale=lengthscale
        )
