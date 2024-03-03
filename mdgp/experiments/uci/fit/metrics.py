from torch import Tensor 
from gpytorch.distributions import MultivariateNormal


import torch 
from gpytorch.metrics import negative_log_predictive_density as gpytorch_nlpd
from scipy.special import logsumexp


def negative_log_predictive_density(outputs: MultivariateNormal, targets: Tensor, y_std: Tensor | None = None) -> Tensor:
    return gpytorch_nlpd(outputs, targets)


def test_log_likelihood(outputs: MultivariateNormal, targets: Tensor, y_std: Tensor) -> Tensor:
    mean, stddev = outputs.mean, outputs.stddev
    logpdf = torch.distributions.Normal(loc=mean, scale=stddev).log_prob(targets) - torch.log(y_std)
    # average over likelihood samples 
    logpdf = logsumexp(logpdf.numpy(), axis=0, b=1 / mean.size(0))
    # average over data points
    return torch.from_numpy(logpdf).mean()


def mean_squared_error(outputs: MultivariateNormal, targets: Tensor, y_std: Tensor) -> Tensor:
    mean = outputs.mean.mean(0)
    return y_std ** 2 * ((mean - targets) ** 2).mean(0)        
        