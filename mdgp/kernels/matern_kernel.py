from geometric_kernels.frontends.pytorch.gpytorch import GPytorchGeometricKernel
from geometric_kernels.kernels.matern_kernel import MaternGeometricKernel
from geometric_kernels.spaces import Space, Euclidean
from torch import Generator
from warnings import warn 
from gpytorch.kernels import MaternKernel
from geometric_kernels.types import FeatureMap
from geometric_kernels.kernels.matern_kernel import default_feature_map


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
    