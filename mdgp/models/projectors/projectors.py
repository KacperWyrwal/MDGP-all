import torch 
from geometric_kernels.spaces import Space, Hypersphere 
from mdgp.frames import HypersphereFrame
from mdgp.utils import space_to_manifold


class ProjectToTangentIntrinsic(torch.nn.Module): 
    def __init__(self, space: Space, get_normal_vector=None) -> None: 
        print(f"Got {get_normal_vector=}")
        assert isinstance(space, Hypersphere) and space.dim == 2, f"Only Hypersphere supported. Got space={space}"
        super().__init__()
        self.frame = HypersphereFrame(dim=space.dim, get_normal_vector=get_normal_vector)

    def forward(self, x, coeff): 
        return self.frame.coeff_to_tangent(x=x, coeff=coeff)
    

class ProjectToTangentExtrinsic(torch.nn.Module):
    def __init__(self, space: Space) -> None:
        super().__init__()
        self.manifold = space_to_manifold(space)

    def forward(self, x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        return self.manifold.proju(x=x, u=coeff)
    

class ExponentialMap(torch.nn.Module): 
    def __init__(self, space: Space) -> None:
        super().__init__()
        self.manifold = space_to_manifold(space)

    def forward(self, x, u): 
        return self.manifold.expmap(x=x, u=u)
    

class Retraction(torch.nn.Module): 
    def __init__(self, space: Space) -> None:
        super().__init__()
        self.manifold = space_to_manifold(space)

    def forward(self, x, u): 
        return self.manifold.retr(x=x, u=u)