from geometric_kernels.spaces import Space, Euclidean


def extrinsic_dimension(space: Space) -> int: 
    if isinstance(space, Euclidean):
        return space.dim
    return space.dimension - 1 # not sure if this always holds