import torch
from torch import Tensor

from src.basis.series import Basis

SCALE = 1e-3

# ---------------------------------------------------------------------------- #
#                                   PERIODIC                                   #
# ---------------------------------------------------------------------------- #


def periodic(ϕ: Tensor, u: Tensor) -> Tensor:

    if isinstance(ϕ, Basis):
        base = (0, ) * u.ndim()
        origin = torch.tensor(base)

        return u.map(lambda coef: coef.at[base].add(-u(origin)) * SCALE)

    if isinstance(ϕ, Tensor):
        x, uofx = u

        return (uofx - uofx[(0,) * (uofx.ndim - 1)]) * SCALE

# ---------------------------------------------------------------------------- #
#                                   DIRICHLET                                  #
# ---------------------------------------------------------------------------- #


def dirichlet(ϕ: Tensor, u: Tensor) -> Tensor:

    if isinstance(ϕ, Basis):
        x = u.grid(*u.mode)

        mol = torch.prod(torch.sin(torch.pi * x), axis=-1, keepdims=True)
        return u.transform(u.inv() * mol * SCALE)

    if isinstance(ϕ, Tensor):
        x, uofx = u

        mol = torch.prod(torch.sin(torch.pi * x), axis=-1, keepdims=True)
        return uofx * mol * SCALE

# ---------------------------------------------------------------------------- #
#                               INITIAL-CONDITION                              #
# ---------------------------------------------------------------------------- #


def initial_condition(ϕ: Tensor, u: Tensor) -> Tensor:
    """
        Initial condition problem. The first dimension is temporal and the rest
        of them have periodic boundaries.
    """

    if isinstance(ϕ, Basis):

        mol = u.grid(*u.mode)[..., [0]] * SCALE
        return u.__class__.add(u.transform(u.inv() * mol), ϕ)

    if isinstance(ϕ, Tensor):
        x, uofx = u

        mol = x[..., [0]] * SCALE
        return uofx * mol + ϕ
