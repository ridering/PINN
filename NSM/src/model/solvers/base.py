import math
import functools as F
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from torch import nn, Tensor

from src.pde.pde import PDE
from src.utils import Fx
from src.basis.series import Basis 


def lecun_init(param: Tensor):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(param)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.normal_(param, -bound, bound)


def linear_lecun_weight_zero_bias(in_features: int, out_features: int) -> nn.Module:
    layer = nn.Linear(in_features, out_features)
    # lecun_init(layer.weight)
    # nn.init.zeros_(layer.bias)

    return layer

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
        
        if isinstance(x, Tuple): return u[x]
        if isinstance(x, Tensor): return u(x)

    def loss(self, ϕ: Basis) -> Dict[str, Tensor]:

        R = self.pde.spectral(ϕ, self.forward(ϕ))
        return dict(residual=torch.sum(torch.square(R.coef)))
