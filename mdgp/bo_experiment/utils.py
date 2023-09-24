from geometric_kernels import spaces 
from pymanopt import manifolds



def space_class_from_name(name):
    if name == "hypersphere": 
        return spaces.Hypersphere 
    raise ValueError(f"Unknown space {name}")


def manifold_class_from_name(name): 
    if name == "hypersphere": 
        return manifolds.Sphere
    raise ValueError(f"Unknown manifold {name}")