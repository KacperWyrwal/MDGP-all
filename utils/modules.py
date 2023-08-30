from torch import nn 


class Normalize(nn.Module):
    def __init__(self, p=2, dim=-1, eps=1e-12):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return nn.functional.normalize(x, p=self.p, dim=self.dim, eps=self.eps)
    