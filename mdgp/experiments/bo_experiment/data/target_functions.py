"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier and Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
"""
from torch import Tensor 
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch
from mdgp.utils import spherical_antiharmonic, spherical_distance


from pymanopt.manifolds import *


def get_base(manifold):
    if isinstance(manifold, Euclidean):
        return np.zeros((1, manifold._shape[0]))

    elif isinstance(manifold, Sphere):
        # Dimension of the manifold
        dimension = manifold._shape[0]

        # The base is fixed at (1, 0, 0, ...) for simplicity. Therefore, the tangent plane is aligned with the axis x.
        # The first coordinate of x_proj is always 0, so that vectors in the tangent space can be expressed in a dim-1
        # dimensional space by simply ignoring the first coordinate.
        base = np.zeros((1, dimension))
        base[0, 0] = 1.
        return base

    elif isinstance(manifold, SpecialOrthogonalGroup):
        dimension = manifold._n
        base = np.eye(dimension)
        return base.reshape((dimension**2))[None]

    else:
        raise RuntimeError("The base is not implemented for this manifold.")


def preprocess_manifold_data(manifold, x, cholesky=False):
    base = get_base(manifold)

    if isinstance(manifold, Euclidean):
        return x

    elif isinstance(manifold, Sphere):
        x_proj = manifold.log(base, x)

        # Remove first dim
        return x_proj[:, 1:]

    elif isinstance(manifold, SpecialOrthogonalGroup):
        # Dimension
        dimension = manifold._n

        base = base.reshape((dimension, dimension))
        x = x.reshape((dimension, dimension))
        x_proj = manifold.log(base, x)

        return x_proj.reshape(dimension**2)[None]

    raise NotImplementedError(f"Preprocessing is not implemented for manifold: {manifold}")


class BenchmarkFunction(ABC):
    def __init__(self, manifold, cholesky=False):
        self.manifold = manifold

        if not isinstance(manifold, SymmetricPositiveDefinite):
            self.cholesky = False
        else:
            self.cholesky = cholesky

    @abstractmethod
    def compute_function(self, x):
        pass

    def compute_function_torch(self, x):
        # Data to numpy
        torch_type = x.dtype
        x = x.cpu().detach().numpy()

        # Compute function
        y = self.compute_function(x)
        return torch.tensor(y, dtype=torch_type)

    def optimum(self):
        # Optimum x
        opt_x = get_base(self.manifold)
        # Optimum y
        opt_y = self.compute_function(opt_x)

        return opt_x, opt_y

    def get_base(self):
        return get_base(self.manifold)
    
    def __call__(self, x): 
        if x.ndim == 2 and x.shape[0] == 1:
            return self.compute_function_torch(x)
        # x is of shape [..., dim]
        y = torch.empty(x.shape[:-1], dtype=x.dtype, device=x.device)
        for index in np.ndindex(x.shape[:-1]):
            y[index] = self.compute_function_torch(x[index])
        return y 


class Ackley(BenchmarkFunction):
    def __init__(self, manifold):
        super(Ackley, self).__init__(manifold)

    def compute_function(self, x):
        if np.ndim(x) < 2:
            x = x[None]

        # Preprocess the input
        x = preprocess_manifold_data(self.manifold, x, self.cholesky)[0]

        # Dimension of the input
        dimension = x.shape[0]

        # Ackley function parameters
        a = 20
        b = 0.2
        c = 2 * np.pi

        # Ackley function
        aexp_term = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / dimension))
        expcos_term = - np.exp(np.sum(np.cos(c * x) / dimension))
        y = aexp_term + expcos_term + a + np.exp(1.)

        return y[None, None]


class Rosenbrock(BenchmarkFunction):
    def __init__(self, manifold):
        super(Rosenbrock, self).__init__(manifold)

    def compute_function(self, x):
        if np.ndim(x) < 2:
            x = x[None]

        # Preprocess the input
        x = preprocess_manifold_data(self.manifold, x, self.cholesky)[0]

        # Center optimum
        x = x + 1.

        # Rosenbrock function
        # y = 0
        # for i in range(reduced_dimension - 1):
        #     y += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        a = (x[1:] - x[:-1] ** 2)
        b = (1 - x[:-1])
        y = np.sum(100 * a * a + b * b)

        return y[None, None]


class Levy(BenchmarkFunction):
    def __init__(self, manifold, rescaling_factor=1.):
        super(Levy, self).__init__(manifold)

        if rescaling_factor is None:
            if isinstance(manifold, Sphere):
                self.rescaling_factor = 3.
            else:
                self.rescaling_factor = 1.
        else:
            self.rescaling_factor = rescaling_factor

    def compute_function(self, x):
        if np.ndim(x) < 2:
            x = x[None]

        # Preprocess the input
        x = preprocess_manifold_data(self.manifold, x, self.cholesky)[0]

        # Center optimum and rescale
        x = self.rescaling_factor * (x - 1.)

        # Dimension of the input
        dimension = x.shape[0]

        # Levy function
        pi = np.pi
        w1 = 1 + (x[0] - 1) / 4.
        y = np.sin(pi * w1) ** 2
        for i in range(dimension - 1):
            wi = 1 + (x[i] - 1) / 4.
            y += (wi - 1) ** 2 * (1 + 10 * np.sin(pi * wi + 1) ** 2)
        wd = 1 + (x[-1] - 1) / 4.
        y += (wd - 1) ** 2 * (1 + np.sin(2 * pi * wd) ** 2)

        return y[None, None]


class StyblinskiTang(BenchmarkFunction):
    def __init__(self, manifold, rescaling_factor=None):
        super(StyblinskiTang, self).__init__(manifold)
        if rescaling_factor is None:
            if isinstance(manifold, Sphere):
                self.rescaling_factor = 3.
            elif isinstance(manifold, SymmetricPositiveDefinite):
                self.rescaling_factor = 5.
            else:
                self.rescaling_factor = 1.
        else:
            self.rescaling_factor = rescaling_factor

    def compute_function(self, x):
        if np.ndim(x) < 2:
            x = x[None]

        # Preprocess the input
        x = preprocess_manifold_data(self.manifold, x, self.cholesky)[0]

        # Center the optimum and rescale
        x = self.rescaling_factor * x - 2.903534

        # Styblinski-tang function
        y = 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5. * x)

        return y[None, None]


class ProductOfSines(BenchmarkFunction):
    def __init__(self, manifold, coefficient=100.):
        super(ProductOfSines, self).__init__(manifold)
        self.coefficient = coefficient

    def compute_function(self, x):
        if np.ndim(x) < 2:
            x = x[None]

        # Preprocess the input
        x = preprocess_manifold_data(self.manifold, x, self.cholesky)[0]

        # Dimension of the input
        dimension = x.shape[0]

        # Center
        opt_x = np.pi / 2. * np.ones(dimension)
        opt_x[1] = - np.pi / 2.
        x = x + opt_x

        # Sinus
        sin_x = np.sin(x)
        # Product of sines function
        y = self.coefficient * sin_x[0] * np.prod(sin_x)

        return y[None, None]


import gpytorch 
from mdgp.bo_experiment.model import ModelArguments, create_model
from mdgp.utils import sphere_random_uniform
from pymanopt.manifolds.manifold import Manifold


class DGPSample(torch.nn.Module):
    def __init__(self, manifold: Manifold, seed=0, num_inducing=60, project_to_tangent='extrinsic'):
        super().__init__()
        self.seed = seed 
        model_args = ModelArguments(project_to_tangent=project_to_tangent, space_dim=int(manifold.dim), num_eigenfunctions=10)

        inducing_points = sphere_random_uniform(num_inducing, manifold.dim + 1)
        self.model = create_model(model_args, inducing_points).base_model


    def forward(self, x: Tensor) -> Tensor: 
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1):
            initial_seed = torch.initial_seed()
            torch.manual_seed(self.seed)
            out = self.model(x, sample_hidden='pathwise', sample_output='pathwise', resample_weights=False)[0, ..., 0]
            torch.manual_seed(initial_seed)
        return out
    
    
class PermutedSphericalHarmonic:
    def __init__(self, manifold, degree: int = 2, order: int = 3): 
        self.degree = degree 
        self.order = order 

    def __call__(self, x: Tensor) -> Tensor: 
        return spherical_antiharmonic(x, 2, 3) * (x[..., 2] + 1) * (1 - spherical_distance(x, torch.tensor([[0., 0., 1.]])))
    