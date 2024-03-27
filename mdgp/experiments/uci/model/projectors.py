import torch 
from torch import Tensor 
from linear_operator.operators import DiagLinearOperator
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal


class Projector(torch.nn.Module):
    def forward(self, x: Tensor, y: Tensor | None = None) -> tuple[Tensor, Tensor] | Tensor:
        raise NotImplementedError
    
    def inverse(self, mvn: MultivariateNormal) -> MultivariateNormal:
        raise NotImplementedError


class SphereProjector(Projector):
    def __init__(self, b: float = 1.0):
        super().__init__()
        self.b = torch.nn.Parameter(torch.tensor(b))
        self.norm = None 

    def forward(self, x: Tensor, y: Tensor | None = None) -> tuple[Tensor, Tensor] | Tensor:
        b = self.b.expand(*x.shape[:-1], 1)
        x_cat_b = torch.cat([x, b], dim=-1)
        self.norm = x_cat_b.norm(dim=-1, keepdim=True)
        if y is None:
            return x_cat_b / self.norm
        else:
            return x_cat_b / self.norm, y / self.norm
    
    def inverse(self, mvn: MultivariateNormal) -> MultivariateNormal:
        L = DiagLinearOperator(self.norm.squeeze(-1))
        mean = L @ mvn.mean
        cov = L @ mvn.lazy_covariance_matrix @ L

        if isinstance(mvn, MultitaskMultivariateNormal):
            return MultitaskMultivariateNormal(mean=mean, covariance_matrix=cov)
        else:
            return MultivariateNormal(mean=mean, covariance_matrix=cov)
    

class IdentityProjector(Projector):
    def forward(self, x: Tensor, y: Tensor | None = None) -> tuple[Tensor, Tensor] | Tensor:
        if y is None:
            return x
        else:
            return x, y
    
    def inverse(self, mvn: MultivariateNormal) -> MultivariateNormal:
        return mvn
