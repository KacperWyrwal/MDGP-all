from torch import Tensor
from typing import Optional, Union, Callable

import torch 
from torch import nn 
from mdgp.utils import Normalize
from abc import ABC, abstractmethod


class Frame(nn.Module, ABC): 

    @abstractmethod 
    def frame(self, x: Tensor) -> Tensor: 
        pass 

    def coeff_to_tangent(self, x: Tensor, coeff: Tensor) -> Tensor:
        return torch.einsum('...ij, ...j -> ...i', self.frame(x), coeff)

    def forward(self, x: Tensor, coeff: Tensor) -> Tensor: 
        return self.coeff_to_tangent(x=x, coeff=coeff)
    

class HypersphereFrame(Frame): 
    def __init__(self, dim: int, get_normal_vector: Optional[Union[str, Callable[[Tensor], Tensor]]] = None) -> None:
        assert dim == 2, f"Only Hypersphere of dimension 2 supported. Got dim={dim}"
        super().__init__()
        # override get_normal_vector method if given
        if get_normal_vector == 'nn':
            self.get_normal_vector = nn.Linear(3, 3)
        else: 
            self.get_normal_vector = torch.tensor([[0., 0., 1.]]).expand_as

    def frame(self, x):
        """
        Compute the orthonormal frame of the 2-sphere at the given points. Not well defined at the poles. 
        """
        # Compute unit vector normal to u and x 
        u = self.get_normal_vector(x)
        e_1 = torch.cross(u, x)
        e_1 = e_1 / torch.norm(e_1, dim=-1, keepdim=True)

        # Compute unit vector normal to x and e_1 
        e_2 = torch.cross(x, e_1)
        e_2 = e_2 / torch.norm(e_2, dim=-1, keepdim=True)

        return torch.stack([e_1, e_2], dim=-1) # [N, 3], [N, 3] -> [N, 3, 2]
    