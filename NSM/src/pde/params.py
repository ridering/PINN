from abc import ABC, abstractmethod
from typing import Type

from src.basis.series import Basis
from src.utils import Fx, nmap
from src.dists import Ω


class G(ABC):

    """
        Function space
    """

    idim: int
    odim: int

    @abstractmethod
    def sample(self) -> Fx:
        pass

# ---------------------------------------------------------------------------- #
#                                  INTERPOLATE                                 #
# ---------------------------------------------------------------------------- #


class Interpolate(G):

    """
        Interpolated function
    """

    def __init__(self, dist: Ω, basis: Type[Basis]):

        self.dist = dist
        self.basis = basis

        self.idim = len(dist.dim)
        self.odim = 1

    def sample(self, shape=()) -> Basis:

        x = self.dist.sample(shape)[..., None]
        for_map = lambda smth: self.basis.transform(smth).coef
        return self.basis(nmap(for_map, len(shape))(x))
