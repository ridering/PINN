import math
import operator as O
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
import functools as F
from typing import Tuple, Type

import numpy as np
import torch
from torch import Tensor

from src import utils
from src.devices import DEVICE
from src.utils import Fx


@dataclass
class Basis(ABC):

    coef: Tensor

    """
        Basis function on [0, 1]^ndim. The leading `ndim` axes are for:

            - (spectral) coefficients of basis functions
            - (physical) values evaluated on collocation points
    """

    @staticmethod
    @abstractmethod
    def repr(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def ndim() -> int:
        pass

    @property
    def mode(self):
        return self.coef.shape[:self.ndim()]

    def map(self, f: Fx):
        return self.__class__(f(self.coef))

    @staticmethod
    @abstractmethod
    def grid(*mode: int) -> Tensor:
        pass

    @staticmethod
    @abstractmethod
    def ix(*mode: int) -> Tensor:
        pass

    @staticmethod
    @abstractmethod
    def fn(*mode: int, x: Tensor) -> Tensor:
        pass

    def __call__(self, x: Tensor) -> Tensor:

        assert x.shape[-1] == self.ndim(), f"{x.shape[-1]=} =/= {self.ndim()=}"
        return torch.tensordot(
            self.fn(
                *self.mode,
                x=x),
            self.coef,
            self.ndim())

    def to(self, *mode: int):

        if self.mode == mode:
            return self
        ax = self.ix(*map(min, mode, self.mode))

        zero = self.coef.new_zeros(mode + self.coef.shape[self.ndim():])
        zero[ax] = self.coef[ax]
        return self.__class__(zero)

    @classmethod
    def add(cls, *terms):
        return cls(sum(map(O.attrgetter("coef"), align(*terms, scheme=max))))

    @classmethod
    def mul(cls, *terms):
        return cls.transform(
            math.prod(map(cls.inv, align(*terms, scheme=sum))))

# --------------------------------- TRANSFORM -------------------------------- #

    @staticmethod
    @abstractmethod
    def transform(x: Tensor) -> 'Basis': pass

    @abstractmethod
    def inv(self) -> Tensor: pass

# --------------------------------- OPERATOR --------------------------------- #

    @abstractmethod
    def grad(self, k: int = 1): pass

    @abstractmethod
    def int(self, k: int = 1): pass


def align(*basis: Basis, scheme: Fx = max) -> Tuple[Basis]:

    # asserting uniform properties:

    _ = set(map(lambda cls: cls.repr(), basis))
    _ = set(map(lambda cls: cls.ndim(), basis))
    _ = set(map(lambda self: self.coef.ndim, basis))

    mode = tuple(map(scheme, zip(*map(O.attrgetter("mode"), basis))))
    return tuple(map(lambda self: self.to(*mode), basis))

# ---------------------------------------------------------------------------- #
#                                    SERIES                                    #
# ---------------------------------------------------------------------------- #


class SeriesMeta(ABCMeta, type):

    def __getitem__(cls, n: int):
        return series(*(cls,) * n)


@dataclass
class Series(Basis, metaclass=SeriesMeta):

    """1-dimensional series on interval"""

    @staticmethod
    def ndim() -> int:
        return 1  # on [0, 1]

    def __len__(self):
        return len(self.coef)

    @abstractmethod
    def __getitem__(self, s: int) -> Tensor:
        pass


def series(*types: Type[Series]) -> Type[Basis]:
    """
        Generate new basis using finite product of given series. Each argument
        type corresponds to certain kind of series used for each dimension.
    """

    @dataclass
    class Class(Basis):

        @staticmethod
        def repr() -> str: return "".join(map(O.methodcaller("repr"), types))

        @staticmethod
        def ndim() -> int: return len(types)

        @staticmethod
        def grid(*mode: int) -> Tensor:

            assert len(mode) == len(types)

            axes = mesh(lambda i, cls: cls.grid(mode[i]).squeeze(1))
            return torch.stack(axes, axis=-1)

        def ix(self, *mode: int) -> Tuple:

            return np.ix_(*map(lambda self, n: self.ix(n).cpu(), types, mode))

        def fn(self, *mode: int, x: Tensor) -> Tensor:

            axes = mesh(lambda i, self: self.fn(mode[i], x=x[..., [i]]))
            return torch.prod(torch.stack(axes, axis=-1), axis=-1)

        def __getitem__(self, s: Tuple[int]) -> Tensor:

            def gett(smth): return F.partial(
                Super.__getitem__, s=s[1:])(Super(smth))

            got = Self(self.coef)[s[0]]

            return torch.vmap(gett)(got)

# --------------------------------- TRANSFORM -------------------------------- #

        @staticmethod
        def transform(x: Tensor) -> 'Class':

            def transform(smth): return Super.transform(smth).coef
            return Class(Self.transform(torch.vmap(transform)(x)).coef)

        def inv(self) -> Tensor:

            def inv(smth): return Super.inv(Super(smth))

            inversed = Self(self.coef).inv()

            return torch.vmap(inv)(inversed)

# --------------------------------- OPERATOR --------------------------------- #

        def grad(self, k: int = 1):

            def partial(smth): return Super.grad(Super(smth), k=k).coef

            coef = torch.vmap(partial)(self.coef)

            return Class(torch.concatenate(
                [Self(self.coef).grad(k).coef, coef], axis=-1))

        def int(self, k: int = 1):

            coef = torch.vmap(F.partial(Super.int, k=k))(Super(self.coef)).coef
            return Class(torch.concatenate(
                [Self(self.coef).int(k).coef, coef], axis=-1))

    def mesh(call: Fx) -> Tuple[Tensor]:
        def cat(*x: Tensor) -> Tuple[Tensor]:
            n, = set(map(np.ndim, x))

            if n != 1:
                return torch.vmap(cat)(*x)
            return torch.meshgrid(*x, indexing="ij")

        args = zip(*enumerate(types))
        return cat(*map(call, *args))

    try:
        cls, = types
        return cls
    except BaseException:

        Self, *other = types
        Super = series(*other)

        return Class
