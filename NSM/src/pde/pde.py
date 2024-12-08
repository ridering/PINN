from abc import ABC
from typing import Any

from src.utils import Fx
from src.pde.params import G
from src.pde.domain import R
from src.basis.series import Basis

class PDE(ABC):

    odim: int                           # output dimension

    domain: R                           # interior domain
    params: G                           # parameter function

    mollifier: Fx                       # transformation

    equation: Fx                        # PDE (equation)
    boundary: Fx                        # PDE (boundary)

    basis: Basis                        # basis function
    spectral: Fx                        # PDE (spectral)

    solution: Any
