from torch import Tensor 
from gpytorch.distributions import MultivariateNormal


import torch 
from scipy.special import logsumexp


def squared_error_batch(outputs: MultivariateNormal, targets: Tensor, y_std: Tensor | None = None) -> Tensor:
    mean = outputs.mean
    y_std = y_std if y_std is not None else targets.new_ones(1)
    return y_std ** 2 * ((mean - targets) ** 2).sum(-1) # Don't average over points yet in case the batches are different sizes (e.g. last batch)


def test_log_likelihood_batch(outputs: MultivariateNormal, targets: Tensor, y_std: Tensor | None = None) -> Tensor:
    mean, stddev = outputs.mean, outputs.stddev
    y_std = y_std if y_std is not None else targets.new_ones(1)
    logpdf: Tensor = torch.distributions.Normal(loc=mean, scale=stddev).log_prob(targets) - torch.log(y_std) 
    logpdf = logsumexp(logpdf.cpu().numpy(), axis=0, b=1 / mean.size(0))
    return torch.from_numpy(logpdf).to(mean)


def test_log_likelihood(outputs: MultivariateNormal, targets: Tensor, y_std: Tensor | None = None) -> Tensor:
    mean, stddev = outputs.mean, outputs.stddev
    y_std = y_std if y_std is not None else targets.new_ones(1)

    logpdf: Tensor = torch.distributions.Normal(loc=mean, scale=stddev).log_prob(targets) - torch.log(y_std)
    logpdf = logsumexp(logpdf.cpu().numpy(), axis=0, b=1 / mean.size(0))
    return torch.from_numpy(logpdf).to(mean).mean()
