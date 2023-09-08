import torch 
import gpytorch 
import math 
from mdgp.samplers import GeometricKernelsSampler
from geometric_kernels.kernels.feature_maps import deterministic_feature_map_compact, random_phase_feature_map_compact, random_phase_feature_map_noncompact
from geometric_kernels.spaces import DiscreteSpectrumSpace, NoncompactSymmetricSpace


class RFFSampler(torch.nn.Module):

    def __init__(self, covar_module, mean_module, feature_map='deterministic') -> None: 
        super().__init__()

        # Is covar_module as ScaleKernel? 
        if isinstance(covar_module, gpytorch.kernels.ScaleKernel):
            self.covar_module = covar_module
            self.base_kernel = covar_module.base_kernel
            geometric_kernel = self.base_kernel.geometric_kernel
            space = geometric_kernel.space 
        else:
            raise NotImplementedError(f"RFFSampler only implemented for ScaleKernel. Got {type(covar_module)}.")

        # Pick feature map to make sampler 
        if feature_map == 'deterministic': 
            if isinstance(space, DiscreteSpectrumSpace):
                feature_map = deterministic_feature_map_compact(space=space, kernel=geometric_kernel)
            else:
                raise NotImplementedError(f"Deterministic feature map only implemented for DiscreteSpectrumSpace. Got {type(space)}.")
        elif feature_map == 'random_phase':
            if isinstance(space, DiscreteSpectrumSpace):
                feature_map = random_phase_feature_map_compact(space=space, kernel=geometric_kernel)
            elif isinstance(space, NoncompactSymmetricSpace):
                feature_map = random_phase_feature_map_noncompact(space=space)
            else:
                raise NotImplementedError(f"Random phase feature map only implemented for DiscreteSpectrumSpace and NoncompactSymmetricSpace. Got {type(space)}.")

        # Set up sampler 
        self.geometric_kernels_sampler = GeometricKernelsSampler(feature_map=feature_map)
        self.mean_module = mean_module

        assert len(self.covar_module.batch_shape) <= 1, "RFFSampler supports at most a single output dimension"
        self.output_dims = self.covar_module.batch_shape[0] if self.covar_module.batch_shape else None

        self._weights = None

    @property
    def num_features(self): 
        return self.base_kernel.state['eigenfunctions'].num_eigenfunctions

    def weights(self, num_samples: int, resample=True) -> torch.Tensor:
        if resample: 
            self._weights = torch.randn(*self.covar_module.batch_shape, self.num_features, num_samples) # [M, O]
        else: 
            assert self._weights.shape[-1] == num_samples, "Sample shape mismatch. Resample or use the same sample shape."
        return self._weights
    
    def sample_unnormalized(self, x, num_samples: int, params, state, weights): 
        return self.geometric_kernels_sampler(X=x, s=num_samples, params=params, state=state, weights=weights, key=None)[1] # geometric_kernels_sampler returns (key, sample)

    def sample_standard(self, x, num_samples: int, params, state, weights):
        s = self.sample_unnormalized(x, num_samples, params, state, weights)
        x_single = x[..., 0, :][None]
        c = self.base_kernel.normalizing_constant(params=params, x=x_single).sqrt()
        return s / c 
    
    def sample_standard_multioutput(self, x, num_samples, weights, normalize_kernel=True):
        sample_fnc = self.sample_standard if normalize_kernel else self.sample_unnormalized
        state = self.base_kernel.state 

        # If empty output dimension, go to single output case
        if self.output_dims is None: 
            params = dict(
                lengthscale=self.base_kernel.lengthscale, 
                nu=self.base_kernel.nu, 
            )
            return sample_fnc(x, num_samples, params, state, weights) # [N, S]
        
        return torch.stack([
            sample_fnc(x, num_samples, dict(lengthscale=self.base_kernel.lengthscale[output_dim], nu=self.base_kernel.nu), state, weights[output_dim])
            for output_dim in range(self.output_dims)
        ], dim=-1) # [N, S, O]
    
    def sample(self, inputs, num_samples, weights, normalize_kernel=True):
        # if output dimension is empty, the standard sample is of shape [N, S]
        if self.output_dims is None:
            return self.covar_module.outputscale.sqrt() * self.sample_standard_multioutput(inputs, num_samples, weights, normalize_kernel=normalize_kernel).T + self.mean_module(inputs) # [N, S] -> [S, N]
        
        # otherwise, the standard sample is of shape [N, S, O]
        return (self.covar_module.outputscale.sqrt() * self.sample_standard_multioutput(inputs, num_samples, weights, normalize_kernel=normalize_kernel)).permute(
            1, 2, 0 # [N, S, O] -> [S, O, N]
        ) + self.mean_module(inputs)
    
    def forward(self, inputs: torch.Tensor, num_samples: int, resample_weights=True, normalize_kernel=True) -> torch.Tensor: 
        """
        :return: A sample from the model. [S, O, N]
        """ 
        weights = self.weights(num_samples=num_samples, resample=resample_weights) # [M, O, S]
        return self.sample(inputs=inputs, num_samples=num_samples, weights=weights, normalize_kernel=normalize_kernel)
    
    def __call__(self, inputs, sample_shape: torch.Size = torch.Size([]), resample_weights=True, normalize_kernel=True) -> torch.Tensor:
        # preprocessing 
        inputs_shape = inputs.shape 
        inputs = inputs.flatten(end_dim=-2) # [N, D]
        num_samples = math.prod(sample_shape)

        # forward 
        outputs = super().__call__(inputs=inputs, num_samples=num_samples, resample_weights=resample_weights, normalize_kernel=normalize_kernel) # [S, N] or [S, O, N]

        # postprocessing 
        outputs = outputs.view(*sample_shape, *outputs.shape[1:-1], *inputs_shape[:-1])
        return outputs 
