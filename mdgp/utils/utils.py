from geometric_kernels.spaces import Space, Euclidean, Hypersphere


def extrinsic_dimension(space: Space) -> int: 
    if isinstance(space, Euclidean):
        return space.dim
    if isinstance(space, Hypersphere):
        return space.dim - 1
    raise NotImplementedError(f"Extrinsic dimension not implemented for {space.__class__.__name__}")