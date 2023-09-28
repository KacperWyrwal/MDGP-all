from gpytorch.variational import _VariationalDistribution

import torch 


class VISampler(torch.nn.Module):

    def __init__(self, variational_distribution: _VariationalDistribution):
        super().__init__()
        self.variational_distribution = variational_distribution
        self._base_samples = None

    def get_base_samples(self, sample_shape: torch.Size, resample=True):
        if resample is True or self._base_samples is None: 
            self._base_samples = torch.randn(*sample_shape, *self.variational_distribution.batch_shape, self.variational_distribution.num_inducing_points)
        return self._base_samples

    def forward(self, sample_shape: torch.Size = torch.Size([]), resample: bool = True):
        return self.variational_distribution().rsample(sample_shape=sample_shape, base_samples=self.get_base_samples(sample_shape, resample=resample))
