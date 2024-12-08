from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor

from src.utils import Fx
from src.devices import DEVICE


class Ω(ABC):

    """
        Distribution
    """

    @abstractmethod
    def sample(self) -> Tensor:
        pass

# ---------------------------------------------------------------------------- #
#                                    UNIFORM                                   #
# ---------------------------------------------------------------------------- #


class Uniform:

    """
        Uniform distribution
    """

    min: Tensor
    max: Tensor

    def __init__(self, min: Tensor, max: Tensor):

        self.min = min
        self.max = max

    def sample(self, shape=()) -> Tensor:

        scale = self.max - self.min
        x = torch.normal(0, 1, shape + scale.shape)

        return x * scale + self.min

# ---------------------------------------------------------------------------- #
#                                    NORMAL                                    #
# ---------------------------------------------------------------------------- #


class Normal:

    """
        Normal distribution
    """

    μ: Tensor
    λ: Tensor

    def __init__(self, μ: Tensor, Σ: Tensor):

        self.μ = μ

        U, Λ, _ = torch.linalg.svd(Σ)
        self.λ = U * torch.sqrt(Λ)

    def sample(self, shape=()) -> Tensor:

        
        # from math import prod
        # var = torch.linspace(1, 2, prod(shape + self.μ.shape)).reshape(shape + self.μ.shape)

        var = torch.normal(0, 1, shape + self.μ.shape).to(DEVICE)
        ε = torch.einsum("...ij,...j->...i", self.λ, var)

        return self.μ + ε

# ---------------------------------------------------------------------------- #
#                                   GAUSSIAN                                   #
# ---------------------------------------------------------------------------- #


class Gaussian(Normal):

    """
        Gaussian Process
    """

    dim: Tuple[int]

    def __init__(self, grid: Tensor, kernel: Fx):

        *dim, ndim = grid.shape
        assert len(dim) == ndim

        X = grid.reshape(-1, ndim)
        K = torch.vmap(kernel, (0, None))
        Σ = torch.vmap(lambda y: K(X, y))(X)

        super().__init__(torch.zeros(len(Σ)), Σ)
        self.dim = tuple(dim)

    def sample(self, shape=()) -> Tensor:

        x = super().sample(shape)
        return x.reshape(shape + self.dim)

# ---------------------------------- KERNEL ---------------------------------- #

    def RBF(ƛ): return lambda x, y: torch.exp(-torch.sum((x - y)**2) / ƛ**2 / 2)

    def Per(ƛ): return lambda x, y: torch.exp(-torch.sum(
        (torch.sin(torch.pi * (x - y)) / 2)**2) / ƛ**2 * 2)
