import torch 
import gpytorch 
import math 
from geometric_kernels.types import FeatureMap
from mdgp.kernels import GeometricMaternKernel
from geometric_kernels.spaces import Space 


class RFFSampler(torch.nn.Module):

    def __init__(self, covar_module, mean_module, feature_map: FeatureMap) -> None: 
        super().__init__()

        # Is covar_module as ScaleKernel? 
        assert isinstance(covar_module, gpytorch.kernels.ScaleKernel), "RFFSampler only implemented for ScaleKernel."

        self.covar_module = covar_module
        self.base_kernel: GeometricMaternKernel = covar_module.base_kernel
        self.feature_map = feature_map
        self.mean_module = mean_module

        self._weights = None
        # Dynamically learn the number of features returned by the feature map 
        # FIXME This is a hack. Would be better if the feature map was a class with this property.
        self._num_features = self.compute_features(torch.tensor(self.base_kernel.space.random_point()[None])).shape[-1]

    @property
    def num_features(self): 
        return self._num_features

    def weights(self, num_samples, inputs=None, resample=True) -> torch.Tensor:
        # Broadcast the batch shapes of the inputs and the kernel if possible. 
        if inputs is not None: 
            batch_shape = torch.broadcast_shapes(self.covar_module.batch_shape, inputs.shape[:-2])
        else: 
            batch_shape = self.covar_module.batch_shape

        # Randomly initialise the weights or resample them. 
        if resample or self._weights is None: 
            self._weights = torch.randn(*batch_shape, self.num_features, num_samples) # [M, O]
        else: 
            assert self._weights.shape[-1] == num_samples, f"Sample shape mismatch. Resample or use sample_shape with product {self._weights.shape[-1]}."
        return self._weights

    def compute_features(self, inputs):
        _, features = self.feature_map(
            inputs, 
            params=self.base_kernel.geometric_kernel_params, 
            key=self.base_kernel.key, 
            normalize=True
        )
        return features * self.base_kernel.batch_shape_scaling_factor.sqrt()

    def sample(self, inputs, weights): 
        """
        :param inputs: [..., D]
        """
        features = self.compute_features(inputs=inputs) # [..., batch_shape, N, num_eigenfunctions]
        res = torch.einsum('...ne, ...es -> s...n', features, weights)
        return self.covar_module.outputscale.sqrt().unsqueeze(-1) * res + self.mean_module(inputs)
    
    def forward(self, inputs: torch.Tensor, sample_shape: torch.Size = torch.Size([]), resample=True) -> torch.Tensor: 
        """
        :return: A sample from the model. [*S, O, N]
        """ 
        # flatten sample shape
        weights = self.weights(num_samples=math.prod(sample_shape), inputs=inputs, resample=resample)

        # Sample with flattened sample shape 
        sample = self.sample(inputs=inputs, weights=weights)

        # Reshape sample to desired shape
        return sample.view(*sample_shape, *sample.shape[1:])
    