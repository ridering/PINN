import functools as F
from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn, Tensor

from src.pde.pde import PDE
from src.utils import Fx
from src.basis.series import Basis 


# ---------------------------------------------------------------------------- #
#                                    SOLVER                                    #
# ---------------------------------------------------------------------------- #

class Solver(ABC, nn.Module):

    def __init__(self, pde: PDE, cfg: Dict) -> None:
        super().__init__()

        self.pde = pde
        self.cfg = cfg

    @F.cached_property
    def activate(self) -> Fx:
        return getattr(nn.functional, self.cfg["activate"])

    @abstractmethod
    def u(self, phi: Fx, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss(self, phi: Fx) -> Dict[str, Tensor]:
        pass


# --------------------------------- SPECTRAL --------------------------------- #

class Spectral(Solver, ABC, nn.Module):

    @abstractmethod
    def forward(self, phi: Basis) -> Basis: pass

    def u(self, phi: Basis, x: Tensor) -> Tensor:

        u = self.forward(phi)
        
        return u[x]

    def loss(self, ϕ: Basis) -> Dict[str, Tensor]:

        R = self.pde.spectral(ϕ, self.forward(ϕ))
        return dict(residual=torch.sum(torch.square(R.coef)))
