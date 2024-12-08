from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from src import utils
from src.basis.series import Series
from src.devices import DEVICE


class Fourier(Series):

    """
        Trigonometric series

            fk(x) = e^{ik·x}
    """

    @staticmethod
    def repr() -> str: return "F"

    @staticmethod
    def grid(n: int) -> Tensor:
        return utils.grid(n, mode="left")

    @staticmethod
    def ix(n: int) -> Tensor:
        return torch.tensor(np.r_[-n // 2 + 1:n // 2 + 1], device=DEVICE)

    @staticmethod
    def fn(n: int, x: Tensor) -> Tensor:

        return torch.moveaxis(
            real(
                torch.moveaxis(
                    torch.exp(
                        x * -freq(n)), -1, 0), n), 0, -1)

    def __getitem__(self, s: int) -> Tensor:
        if isinstance(s, Tuple):
            s, = s

        return torch.concatenate([x := self.to(s - 1).inv(), x[np.r_[0]]])

# --------------------------------- TRANSFORM -------------------------------- #

    @staticmethod
    def transform(x: Tensor) -> 'Fourier':

        coef = torch.fft.rfft(x, axis=0, norm="forward")
        coef[1:-(len(x) // -2)] *= 2

        return Fourier(real(coef, len(x)))

    def inv(self) -> Tensor:

        coef = comp(self.coef, n := len(self))
        coef[1:-(n // -2)] /= 2

        return torch.fft.irfft(coef, len(self), axis=0, norm="forward")

# --------------------------------- OPERATOR --------------------------------- #

    def grad(self, k: int = 1) -> 'Fourier':

        coef = (freq(len(self)) ** k).reshape(-1, *[1 for _ in range(1, self.coef.ndim)])
        return Fourier(
            real(comp(self.coef, n := len(self)) * coef, n)[..., None])

    def int(self, k: int = 1) -> 'Fourier':

        coef = np.expand_dims(self.freq(len(self)), range(1, self.coef.ndim))
        return Fourier((self.coef / coef ** k)
                       [..., torch.newaxis].at[0].set(0))

# ---------------------------------------------------------------------------- #
#                                    HELPER                                    #
# ---------------------------------------------------------------------------- #


def freq(n: int) -> Tensor:
    return torch.arange(n // 2 + 1) * 2j * torch.pi


def real(coef: Tensor, n: int) -> Tensor:
    """Complex coef -> Real coef"""

    cos, sin = coef.real, coef.imag[1:-(n // -2)]

    return torch.concatenate((cos, sin.flip([0])), 0)


def comp(coef: Tensor, n: int) -> Tensor:
    """Real coef -> Complex coef"""

    cos, sin = torch.split(coef, (m := n // 2 + 1, n - m))

    res = cos + 0j

    rev_sin = sin.flip([0])
    res[1:n - m + 1] += rev_sin[-(n - m):] * 1j

    return res
