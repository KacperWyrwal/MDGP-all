from geometric_kernels.spaces import Hypersphere


def space_class_from_name(name):
    if name == "hypersphere": 
        return Hypersphere
    raise ValueError(f"Unknown space {name}")


