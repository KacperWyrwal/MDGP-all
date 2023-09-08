import lab as B 
from typing import Any, Tuple
from geometric_kernels.types import FeatureMap


class Sampler:

    def __init__(self, feature_map: FeatureMap):
        self.feature_map = feature_map

    def __call__(self, s, X: B.Numeric, params, state, key=None, weights=None) -> Tuple[Any, Any]:
        """
        Given a `feature_map`, compute `s` samples at `X` defined by random state `key`.

        Added a weights variable which is helpful when evaluating the same sampled function at points in batches.  
        """

        key = key or B.global_random_state(B.dtype(X))

        features, _context = self.feature_map(X, params, state, key=key)  # [N, M]

        if "key" in _context:
            key = _context["key"]

        num_features = B.shape(features)[-1]

        if weights is None: 
            key, random_weights = B.randn(key, B.dtype(X), num_features, s)  # [M, S]
        else:
            random_weights = weights

        random_sample = B.matmul(features, random_weights)  # [N, S]

        return key, random_sample
