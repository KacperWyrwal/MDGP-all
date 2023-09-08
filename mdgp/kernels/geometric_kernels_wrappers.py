import torch 
import gpytorch
from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces import Space
from typing import Tuple 


class SingleOutputGPytorchGeometricKernel(gpytorch.kernels.Kernel):
    """
    Pytorch wrapper for `BaseGeometricKernel`
    """

    has_lengthscale = True

    def __init__(self, geometric_kernel: BaseGeometricKernel, nu: float = 2.5, optimize_nu: bool = False, **kwargs) -> None: 
        super().__init__(**kwargs)
        self.geometric_kernel = geometric_kernel
        _, self.state = self.geometric_kernel.init_params_and_state()

        # Add nu either as a parameter or as a buffer depending on whether it should be optimized
        self.optimize_nu = optimize_nu
        if self.optimize_nu:
            self.register_parameter(
                name="raw_nu", parameter=torch.nn.Parameter(torch.tensor(0.))
            )
            self.register_constraint("raw_nu", gpytorch.constraints.Positive())
        else: 
            self.register_buffer(
                name="raw_nu", tensor=torch.tensor(0.)
            )
        self.nu = nu 

    @property
    def space(self) -> Space:
        """Alias to kernel Space"""
        return self._kernel.space

    @property
    def nu(self) -> torch.Tensor:
        """A smoothness parameter"""
        if self.optimize_nu:
            return self.raw_nu_constraint.transform(self.raw_nu)
        else:
            return self.raw_nu

    @nu.setter
    def nu(self, value):
        value = torch.as_tensor(value).to(self.raw_nu)
        if self.optimize_nu:
            self.initialize(raw_nu=self.raw_nu_constraint.inverse_transform(value))
        else:
            self.raw_nu = value

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **kwargs):
        """Eval kernel"""
        # TODO: check batching dimensions

        params = dict(lengthscale=self.lengthscale, nu=self.nu)
        if diag:
            return self._kernel.K_diag(params, self.state, x1)
        return self._kernel.K(params, self.state, x1, x2)
    

def broadcast_batch_dims(x1: torch.Tensor, x2: torch.Tensor, batch_shape: torch.Size = torch.Size([])) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given two tensors x1 and x2 broadcast them so that their batch dimensions are the same. 
    If this is not possible, raise an error. 

    :param x1: A tensor with batch dimensions. [..., N, D1]
    :param x2: A tensor with batch dimensions. [..., M, D2]

    :returns: Tuple of tensors with dimensions [..., N, D1], [..., M, D2]
    """
    batch_shape1 = x1.shape[:-2]
    batch_shape2 = x2.shape[:-2]
    common_batch_shape = torch.broadcast_shapes(batch_shape1, batch_shape2, batch_shape)
    return x1.expand(*common_batch_shape, -1, -1), x2.expand(*common_batch_shape, -1, -1)


class GPytorchGeometricKernel(SingleOutputGPytorchGeometricKernel):
    """
    Pytorch wrapper for `BaseGeometricKernel`. Now works with batch dimensions of inputs and multiple output dimensions.
    """

    def normalizing_constant(self, params, x):
        return self.geometric_kernel.K_diag(params, self.state, x)

    def _forward_single_output_no_batch(self, x1, x2, params, diag=False, last_dim_is_batch=False, normalize=True, **kwargs):
        """
        Eval kernel with one output dimension and a single output dimension.
        """

        # Get unnormalized output (it need not have variance 1)
        out = self.geometric_kernel.K_diag(params, self.state, x1) if diag else self.geometric_kernel.K(params, self.state, x1, x2)

        # Maybe normalize 
        if normalize is True: 
            normalizing_constant = self.normalizing_constant(params, x1[..., 0, :][None])
            out = out / normalizing_constant
        
        return out 

    def _forward_single_output(self, x1, x2, params, diag=False, last_dim_is_batch=False, normalize=True, **kwargs):
        """
        Eval kernel for a single output dimension.
        """
        # If x1 is 2D, then both x1 and x2 have no batch dimensions, since we have already broadcasted them to the same batch dimensions.
        if x1.ndim <= 2: 
            return self._forward_single_output_no_batch(x1, x2, params, diag, last_dim_is_batch, normalize=normalize, **kwargs)
        
        # Otherwise, we have to iterate over the batch dimensions of x1 and x2. For now we only support a single batch dimension.
        assert x1.ndim <= 3, "both x1 and x2 must have at most 3 dimensions"
        out = torch.stack([
            self._forward_single_output_no_batch(x1_, x2_, params, diag, last_dim_is_batch, normalize=False, **kwargs)
            for x1_, x2_ in zip(x1.unbind(0), x2.unbind(0))
        ], dim=0)
        return out / self.normalizing_constant(params, x1[..., 0, :][None]) if normalize else out 

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, normalize=True, **kwargs):

        # Broadcast batch dimensions of x1 and x2 to batch_shape 
        x1, x2 = broadcast_batch_dims(x1, x2, batch_shape=self.batch_shape)

        # Temporarily, for simplicity, assume that batch shape is a single dimension.
        assert len(self.batch_shape) <= 1, "batch_shape must be a single dimension"

        # If single output, we do not need to iterate over parameters for each output dimension
        if not self.batch_shape:
            return self._forward_single_output(x1, x2, {'nu': self.nu, 'lengthscale': self.lengthscale}, diag, last_dim_is_batch, normalize=normalize, **kwargs)

        return torch.stack([
            # Step 2a. iterate over the corresponding batch entries of x1 and x2
            self._forward_single_output(x1_, x2_, {'nu': self.nu, 'lengthscale': lengthscale}, diag=diag, last_dim_is_batch=last_dim_is_batch, normalize=normalize, **kwargs)
            for x1_, x2_, lengthscale in zip(x1.unbind(-3), x2.unbind(-3), self.lengthscale.unbind(-3)) # unbind the batch dimension which is third to last 
        ], dim=-3 if not diag else -2)
