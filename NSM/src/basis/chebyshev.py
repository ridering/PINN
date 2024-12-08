from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from src import utils
from src.basis.series import Series
from src.devices import DEVICE


class Chebyshev(Series):

    """
        Chebyshev polynomial of T kind
            - Tn(x) = cos(n cos^-1(x))
            - Tn^*(x) = Tn(2 x - 1)
    """

    @staticmethod
    def repr() -> str: return "C"

    @staticmethod
    def grid(n: int) -> Tensor:
        return torch.cos(torch.pi * utils.grid(n)) / 2 + 0.5

    @staticmethod
    def ix(n: int) -> Tensor:
        return torch.arange(n).to(DEVICE)

    @staticmethod
    def fn(n: int, x: Tensor) -> Tensor:
        return torch.cos(torch.arange(n) * torch.arccos(x * 2 - 1))

    def __getitem__(self, s: int) -> Tensor:
        if isinstance(s, Tuple):
            s, = s
        return self(utils.grid(s))

# --------------------------------- TRANSFORM -------------------------------- #

    @staticmethod
    def transform(x: Tensor):

        coef = torch.fft.hfft(x, axis=0, norm="forward")[:len(x)]
        coef[1:-1] *= 2

        assert len(x) > 1, "sharp bits!"
        return Chebyshev(coef)

    def inv(self) -> Tensor:

        coef = self.coef.clone()
        coef[1:-1] /= 2
        inv_coef = coef.flip([0])
        coef = torch.concatenate([coef, inv_coef[1:-1]])

        return torch.fft.ihfft(coef, axis=0, norm="forward").real

# --------------------------------- OPERATOR --------------------------------- #

    def grad(self, k: int = 1):

        padded = np.pad(gradient(len(self)).cpu(), [(0, 1), (0, 0)])
        coef = torch.linalg.matrix_power(torch.tensor(padded), k)

        return Chebyshev(torch.einsum("ij,jk...->ik...", coef,
                         self.coef)[..., torch.newaxis])

    def int(self, k: int = 1):

        coef = torch.linalg.matrix_power(integrate(len(self))[:-1], k)
        return Chebyshev(torch.tensordot(
            coef, self.coef, (1, 0))[..., torch.newaxis])

# ---------------------------------------------------------------------------- #
#                                    MATRIX                                    #
# ---------------------------------------------------------------------------- #


"""
    Chebyshev gradient and integrate matrix

        - gradient ∈ R ^ n-1⨉n; integrate ∈ R ^ n+1⨉n
        - When aligned, they are pseudo-inverse of each other:
            `gradient(n+1) @ integrate(n) == identity(n)`
"""


def gradient(n: int) -> Tensor:

    alternate = torch.tensor(
        np.pad(
            torch.eye(2).cpu(), [
                (0, n - 3), (0, n - 3)], mode="reflect"))
    alternate[0] /= 2
    coef = torch.concatenate(
        [torch.zeros(n - 1)[:, torch.newaxis], torch.triu(alternate)], axis=1)

    return coef * 4 * torch.arange(n)


def integrate(n: int) -> torch:

    shift = torch.identity(n).at[0, 0].set(2) - torch.eye(n, k=2)
    coef = torch.concatenate([torch.zeros(n)[torch.newaxis], shift])

    return coef.at[1:].divide(4 * torch.arange(1, n + 1)[:, torch.newaxis])
