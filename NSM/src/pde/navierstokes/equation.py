import os
import functools as F
from typing import Tuple, Optional

import numpy as np
import torch
from torch import Tensor

from src.basis.series import Basis, series
from src.basis.fourier import Fourier
from src.basis.chebyshev import Chebyshev
from src.pde.pde import PDE
from src.utils import Fx, laplassian
from src.dists import Gaussian
from src.pde.domain import Rect
from src.pde.params import Interpolate
from src.pde.mollifier import initial_condition


def k(nx: int, ny: int) -> Tuple[Tensor]:

    return (
        torch.tile(torch.fft.fftfreq(nx)[:, None] *
                   nx * 2 * torch.pi, (1, ny)),
        torch.tile(torch.fft.fftfreq(ny)[None, :] *
                   ny * 2 * torch.pi, (nx, 1)),
    )


def velocity(what: Tensor = None, *, w: Tensor = None) -> Tensor:

    if what is None:
        what = torch.fft.fft2(w)
    kx, ky = k(*what.shape)

    Δ = kx ** 2 + ky ** 2
    Δ[0, 0] = 1

    vx = torch.fft.irfft2(what * 1j * ky / Δ, what.shape)
    vy = torch.fft.irfft2(what * -1j * kx / Δ, what.shape)

    return vx, vy


class Initial(Gaussian):

    grid = Fourier[2].grid(64, 64)

    def __str__(self): return f"{self.length}x{self.scaling}"

    def __init__(self, length: float, scaling: float = 1.0):
        super().__init__(Initial.grid, Gaussian.Per(length))

        self.length = length
        self.scaling = scaling

    def sample(self, shape=()) -> Tensor:

        x = super().sample(shape)
        x -= torch.mean(x, (-2, -1), keepdims=True)

        x = self.scaling * x[..., torch.newaxis, :, :]
        return torch.broadcast_to(x, shape + (2, *self.dim))


class NavierStokes(PDE):

    """
        wt + v ∇w = nu ∆w
        where
            ∇⨉v = w
            ∇·v = 0
    """

    T: int      # end time
    nu: float   # viscosity
    l: float    # length scale

    # forcing term?
    fn: Optional[Fx]

    # def __str__(self): return f"Re={int(self.Re)}:T={self.T}:{self.F}"
    def __str__(self): return f"Re{int(self.Re)}_T{self.T}_{self.F}"

    def __init__(self, ic: Initial, T: float, nu: float, fn: Fx = None):

        self.odim = 1
        self.ic = ic

        self.T = T
        self.nu = nu
        self.fn = fn

        self.l = (l := ic.length)
        self.Re = l / nu * ic.scaling

        if fn is None:
            self.F = None
        else:
            self.F = fn.__name__

        self.domain = Rect(3)

        self.basis = series(Chebyshev, Fourier, Fourier)
        self.params = Interpolate(ic, self.basis)

        self.mollifier = initial_condition

    @F.cached_property
    def solution(self):

        dir = os.path.dirname(__file__)

        w = torch.tensor(np.load(f"{dir}/w.{self.ic}.npy"))
        u = torch.tensor(np.load(f"{dir}/u.{self}.npy"))

        return self.basis(w), u.shape[1:-1], u

    def spectral(self, w0: Basis, w: Basis) -> Basis:
        w1 = w.grad()
        w2 = w1.grad()

        wt = self.basis(w1.coef[..., 0, 0])
        wx = self.basis(w1.coef[..., 0, 1])
        wy = self.basis(w1.coef[..., 0, 2])
        Δw = self.basis(laplassian(w2.coef[..., 0, 1:, 1:]))

        vel = lambda smth: velocity(w=smth)
        vx, vy = torch.vmap(vel)(w.inv().squeeze(-1))
        Dwdt = self.basis.add(
            wt.map(
                lambda coef: coef /
                self.T),
            self.basis.transform(
                vx *
                wx.inv() +
                vy *
                wy.inv()))

        if self.fn is None:
            f = self.basis(torch.zeros_like(Dwdt.coef))
        else:
            f = self.basis.transform(
                torch.broadcast_to(self.fn(*w.mode[1:]), w.mode))

        return self.basis.add(Dwdt,
                              self.basis(-self.nu * Δw.coef),
                              f.map(torch.negative))


ic = Initial(0.8)

# # ------------------------------- UNFORCED FLOW ------------------------------ #

# re2 = NavierStokes(ic, T=3, nu=1e-2)
re3 = NavierStokes(ic, T=3, nu=1e-3)
# re4 = NavierStokes(ic, T=3, nu=1e-4)

# # ------------------------------ TRANSIENT FLOW ------------------------------ #

# def transient(nx: int, ny: int) -> Tensor:

#     xy = utils.grid(nx, ny, mode="left").sum(-1)
#     return 0.1*(torch.sin(2*torch.pi*xy) + torch.cos(2*torch.pi*xy))

# tf = NavierStokes(ic, T=50, nu=2e-3, fn=transient)
