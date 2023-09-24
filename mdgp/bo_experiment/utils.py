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


class ExcludeFromNameMixin:

    def __post_init__(self) -> None:
        self._exclude_from_name_names: set = set()

    def excluded_from_name(self, name):
        return name in self._exclude_from_name_names

    def exclude_from_name(self, name): 
        if isinstance(name, str): 
            self._exclude_from_name_names.add(name)
            return 
        if isinstance(name, list): 
            for n in name: 
                self.exclude_from_name(n)
            return 
        raise TypeError(f"name must be str or list, got {type(name)}")