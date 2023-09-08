import torch 
from torch import Tensor 
from gpytorch.distributions import MultivariateNormal

def sample_naive(mvn: MultivariateNormal) -> Tensor:
    return torch.distributions.Normal(loc=mvn.mean, scale=mvn.stddev).rsample()
