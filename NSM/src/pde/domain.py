from typing import List

import numpy as np
import torch
from torch import Tensor

from src.dists import Ω, Uniform


class R(Ω):

    """
        Euclidean space
    """

    ndim: int
    boundary: List[Ω]

# ---------------------------------------------------------------------------- #
#                                     RECT                                     #
# ---------------------------------------------------------------------------- #


class Rect(Uniform, R):

    """
        N-d unit rectangle
    """

    def __init__(self, ndim: int):
        super().__init__(torch.zeros(ndim), torch.ones(ndim))

        class Boundary(Uniform):

            def __init__(self, dim: int):
                super().__init__(torch.zeros(ndim - 1), torch.ones(ndim - 1))

                self.dim = dim

            def sample(self, shape=()) -> Tensor:

                x = super().sample(shape)
                return np.insert(x, self.dim, torch.zeros(shape), axis=-1), \
                    np.insert(x, self.dim, torch.ones(shape), axis=-1)

        self.ndim = ndim
        self.boundary = list(map(Boundary, range(ndim)))
