from geometric_kernels.frontends.pytorch.gpytorch import GPytorchGeometricKernel
from geometric_kernels.kernels.matern_kernel import MaternGeometricKernel
from geometric_kernels.spaces import Space, Euclidean
from torch import Generator
from warnings import warn 
from gpytorch.kernels import MaternKernel, RBFKernel
from geometric_kernels.types import FeatureMap
from geometric_kernels.kernels.matern_kernel import default_feature_map



class BaseMaternKernel:
    def __init__(self, space, num: int | None = None, seed: None | int = None): 
        # Set RNG randomly or with a seed
        key = Generator() 
        if seed is not None:
            key.manual_seed(seed)        
        self.seed = seed 
        self.key = key 
        self.feature_map = default_feature_map(space=space, num=num)


class _GeometricMaternKernel(BaseMaternKernel, GPytorchGeometricKernel): 
    def __init__(
        self, 
        space: Space, 
        lengthscale: float = 1.0,
        nu: float = 2.5,
        trainable_nu: bool = True,
        seed: int | None = None,
        num_random_phases: int | None = None, 
        num_eigenfunctions: int | None = None, 
        **kwargs, 
    ) -> None: 
        num = num_eigenfunctions or num_random_phases
        if num_random_phases is not None and num_eigenfunctions is not None:
            warn(f"If both {num_random_phases=} and {num_eigenfunctions=} are passed, only num_eigenfunctions will be used.")

        BaseMaternKernel.__init__(self, space=space, seed=seed, num=num)
        base_kernel = MaternGeometricKernel(
            space=space, 
            num=num, 
            normalize=kwargs.pop('normalize', True), 
            return_feature_map=False, 
            key=self.key, 
        )
        GPytorchGeometricKernel.__init__(self, base_kernel, lengthscale=lengthscale, nu=nu, 
                                         trainable_nu=trainable_nu, **kwargs)


from math import prod 
import torch 


class _EuclideanMaternKernel(BaseMaternKernel, MaternKernel):
    def __init__(
        self, 
        space: Euclidean, 
        lengthscale: float = 1.0,
        nu: float = 2.5,
        trainable_nu: bool = False,
        seed: int | None = None,
        **kwargs, 
    ) -> None: 
        assert trainable_nu is False, "Trainable nu is not yet supported for Euclidean spaces."
        BaseMaternKernel.__init__(self, space=space, seed=seed)
        MaternKernel.__init__(self, nu=nu, **kwargs)
        self.initialize(lengthscale=lengthscale) 
        self.space = space 

    @property
    def geometric_kernel_params(self):
        return {
            'lengthscale': self.lengthscale, 
            'nu': self.nu, 
        }
    

class _EuclideanRBFKernel(BaseMaternKernel, RBFKernel):
    def __init__(
        self, 
        space: Euclidean, 
        lengthscale: float = 1.0,
        seed: int | None = None,
        **kwargs, 
    ) -> None: 
        BaseMaternKernel.__init__(self, space=space, seed=seed)
        RBFKernel.__init__(self, **kwargs)
        self.initialize(lengthscale=lengthscale) 
        self.space = space 

    @property
    def geometric_kernel_params(self):
        return {
            'lengthscale': self.lengthscale, 
            'nu': torch.inf, 
        }


class GeometricMaternKernel: 
    def __new__(
        cls, 
        space: Space, 
        lengthscale: float = 1.0,
        nu: float = 2.5,
        trainable_nu: bool = True,
        seed: int | None = None,
        num_random_phases: int | None = None,
        num_eigenfunctions: int | None = None,
        **kwargs,
    ) -> _GeometricMaternKernel | _EuclideanMaternKernel: 
        if isinstance(space, Euclidean): 
            if nu == torch.inf: 
                return _EuclideanRBFKernel(space=space, lengthscale=lengthscale, seed=seed, **kwargs)
            else:
                return _EuclideanMaternKernel(space=space, lengthscale=lengthscale, nu=nu, trainable_nu=trainable_nu, seed=seed, **kwargs)
        return _GeometricMaternKernel(
            space=space, lengthscale=lengthscale, nu=nu, trainable_nu=trainable_nu, seed=seed, 
            num_random_phases=num_random_phases, num_eigenfunctions=num_eigenfunctions, **kwargs)

"""
class GeometricMaternKernel(GPytorchGeometricKernel): 

    def __init__(
        self, 
        space: Space, 
        lengthscale: float = 1.0,
        nu: float = 2.5,
        trainable_nu: bool = True,
        seed: int | None = None,
        num_random_phases: int | None = None, 
        num_eigenfunctions: int | None = None, 
        **kwargs, 
    ) -> None: 
        
        # Set RNG randomly or with a seed
        key = Generator()
        if seed is not None:
            key.manual_seed(seed)

        # If space is Euclidean the Matern kernel implementation from GPyTorch is most efficient.
        # Although it cannot handle trainable nu.
        if isinstance(space, Euclidean):
            assert trainable_nu is False, "Trainable nu is not supported for Euclidean spaces because GPyTorch MaternKernel is used."
            base_kernel = MaternKernel(nu=nu)
            base_kernel.initialize(lengthscale=lengthscale)
            feature_map = default_feature_map(space=space)
        # If space is not Euclidean, we use MaternGeometricKernel from GeometricKernels. 
        else: 
            if num_random_phases is not None and num_eigenfunctions is not None:
                warn("Only one of num_random_phases and num_eigenfunctions will be used. You have passed both. Make sure this is intended.")
            base_kernel, feature_map = MaternGeometricKernel(
                space=space, 
                num=num_random_phases or num_eigenfunctions, 
                normalize=True, 
                return_feature_map=True, 
                key=key, 
            )
        
        super().__init__(base_kernel, lengthscale=lengthscale, nu=nu, trainable_nu=trainable_nu, **kwargs)
        self._space = space 
        self._seed = seed 
        self._key = key 
        self._num_random_phases = num_random_phases
        self._num_eigenfunctions = num_eigenfunctions
        self._feature_map = feature_map

    @property
    def space(self) -> Space:
        return self._space

    @property
    def feature_map(self):
        return self._feature_map

    @property
    def seed(self) -> int | None:
        return self._seed
    
    @property 
    def key(self) -> Generator:
        return self._key
    
    @property
    def num_eigenfunctions(self) -> int:
        return self._num_eigenfunctions
    
    @property
    def num_random_phases(self) -> int:
        return self._num_random_phases
"""