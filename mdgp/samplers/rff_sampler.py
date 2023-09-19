import torch 
import gpytorch 
import math 
import lab as B 
from geometric_kernels.kernels.feature_maps import deterministic_feature_map_compact
from geometric_kernels.spaces import DiscreteSpectrumSpace


class RFFSampler(torch.nn.Module):

    def __init__(self, covar_module, mean_module, feature_map='deterministic') -> None: 
        super().__init__()

        # Is covar_module as ScaleKernel? 
        assert isinstance(covar_module, gpytorch.kernels.ScaleKernel), "RFFSampler only implemented for ScaleKernel."
        self.covar_module = covar_module
        self.base_kernel = covar_module.base_kernel
        self.geometric_kernel = self.base_kernel.geometric_kernel
        self.space = self.geometric_kernel.space

        # Pick feature map to make sampler 
        assert feature_map == 'deterministic', "Only deterministic feature map implemented."
        assert isinstance(self.space, DiscreteSpectrumSpace), "Deterministic feature map only implemented for DiscreteSpectrumSpace."
        self.feature_map = deterministic_feature_map_compact(space=self.space, kernel=self.geometric_kernel)        
        self.mean_module = mean_module
        self._weights = None
        self._num_features = sum(self.geometric_kernel.eigenfunctions.dim_of_eigenspaces)

    @property
    def num_features(self): 
        return self._num_features

    # TODO: Change the weights shape to go by the broadcasted batch shapes of the inputs and the kernel
    def weights(self, num_samples, inputs=None, resample=True) -> torch.Tensor:
        broadcasted_batch_shape = torch.broadcast_shapes(self.covar_module.batch_shape, inputs.shape[:-2]) if inputs is not None else self.covar_module.batch_shape
        if resample: 
            self._weights = torch.randn(*broadcasted_batch_shape, self.num_features, num_samples) # [M, O]
        else: 
            assert self._weights.shape[-1] == num_samples, f"Sample shape mismatch. Resample or use sample_shape with product {self._weights.shape[-1]}."
        return self._weights

    def compute_features(self, inputs, normalize=True):
        params = self.base_kernel.geometric_kernel_params
        key = B.global_random_state(B.dtype(inputs)) # TODO Change to only pytorch 
        features = self.feature_map(inputs, params, key=key, normalize=normalize)[0]
        return features * self.base_kernel.batch_shape_scaling_factor.sqrt()

    def sample(self, inputs, weights, sample_shape: torch.Size = torch.Size([]), normalize=True): 
        """
        :param inputs: [..., D]
        """
        features = self.compute_features(inputs=inputs, normalize=normalize) # [..., batch_shape, N, num_eigenfunctions]
        res = torch.einsum('...ne, ...es -> s...n', features, weights)
        res = self.covar_module.outputscale.sqrt().unsqueeze(-1) * res 
        res = res + self.mean_module(inputs)
        return res.view(*sample_shape, *res.shape[1:])
    
    def forward(self, inputs: torch.Tensor, sample_shape: torch.Size = torch.Size([]), resample_weights=True, normalize=True) -> torch.Tensor: 
        """
        :return: A sample from the model. [S, O, N]
        """ 
        weights = self.weights(num_samples=math.prod(sample_shape), inputs=inputs, resample=resample_weights)
        return self.sample(inputs=inputs, weights=weights, sample_shape=sample_shape, normalize=normalize)
