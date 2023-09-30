from torch import Tensor
from typing import Optional, Union, Callable

import torch 
from torch import nn 
from mdgp.utils import Normalize, cart_to_sph
from abc import ABC, abstractmethod


def canonical_frame(x):
    """
    Compute the orthonormal frame of the 2-sphere at the given points. Not well defined at the poles. 
    """
    # Compute unit vector normal to u and x 
    u = torch.tensor([[0., 0., 1.]]).expand_as(x)
    e_1 = torch.cross(u, x)
    e_1 = e_1 / torch.norm(e_1, dim=-1, keepdim=True)

    # Compute unit vector normal to x and e_1 
    e_2 = torch.cross(x, e_1)
    e_2 = e_2 / torch.norm(e_2, dim=-1, keepdim=True)

    return torch.stack([e_1, e_2], dim=-1) # [N, 3], [N, 3] -> [N, 3, 2]


class Frame(nn.Module, ABC): 

    @abstractmethod 
    def frame(self, x: Tensor) -> Tensor: 
        pass 

    def coeff_to_tangent(self, x: Tensor, coeff: Tensor) -> Tensor:
        return torch.einsum('...ij, ...j -> ...i', self.frame(x), coeff)

    def forward(self, x: Tensor, coeff: Tensor) -> Tensor: 
        return self.coeff_to_tangent(x=x, coeff=coeff)


class ExpandAs(nn.Module):
    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.tensor = tensor

    def forward(self, x: Tensor) -> Tensor:
        return self.tensor.expand_as(x)
    

class CartesianToSpherical(nn.Module): 
    def forward(self, x):
        return torch.stack(cart_to_sph(x), dim=-1)


class NN(nn.Module):
    def __init__(self, hidden_layers: list[int], in_dim=2, out_dim=1): 
        super().__init__()
        layers = [CartesianToSpherical()]
        prev_input_dims = in_dim
        for input_dims in hidden_layers: 
            layers.append(nn.Linear(prev_input_dims, input_dims))
            layers.append(nn.ReLU())
            prev_input_dims = input_dims
        layers.append(nn.Linear(prev_input_dims, out_dim))
        self.sequential = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.sequential(x)


class HypersphereFrame(Frame): 
    def __init__(self, dim: int, get_normal_vector: Optional[Union[str, Callable[[Tensor], Tensor]]] = None) -> None:
        assert dim == 2, f"Only Hypersphere of dimension 2 supported. Got dim={dim}"
        super().__init__()
        # override get_normal_vector method if given
        if get_normal_vector is None: 
            self.get_normal_vector = ExpandAs(torch.tensor([[0., 0., 1.]]))
        elif get_normal_vector == 'nn' or isinstance(get_normal_vector, list):
            self.get_normal_vector = NN(get_normal_vector)
        else:
            raise NotImplementedError(f"Expected get_normal_vector either None, 'nn', or list of ints. Got {get_normal_vector}.")

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


# class HypersphereFrame(Frame):
#     def __init__(self, dim, get_normal_vector: list[int]): 
#         assert dim == 2, f"Only Hypersphere of dimension 2 supported. Got dim={dim}"
#         super().__init__()
#         if isinstance(get_normal_vector, list):
#             self.get_coeff = NN(get_normal_vector)
#         else:
#             self.get_coeff = get_normal_vector
    
#     def frame(self, x): 
#         frame = canonical_frame(x)
#         # Rotate frame 
#         rotation_angle = self.get_coeff(x)
#         cos_angle, sin_angle = torch.cos(rotation_angle), torch.sin(rotation_angle)
#         e1_coeff = torch.cat([cos_angle, sin_angle], dim=-1)
#         e2_coeff = torch.cat([-sin_angle, cos_angle], dim=-1)
#         e1 = torch.einsum('...ij, ...j -> ...i', frame, e1_coeff)
#         e2 = torch.einsum('...ij, ...j -> ...i', frame, e2_coeff)
#         return torch.stack([e1, e2], dim=-1)
    