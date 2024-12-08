import functools as F
from typing import Callable, List

import torch
from torch import Tensor

Fx = Callable[..., Tensor]      # real-valued function
laplassian = F.partial(torch.einsum, "...ii -> ...")


def grid(*s: int, mode: str = None, flatten: bool = False) -> Tensor:
    """
        Return grid on [0, 1)^n. If not flatten, shape=(*s, len(s));
        else shape=(∏s, len(s)).

        Mode:
            - `None`: uniformly spaced
            - "left": exclude endpoint
            - "cell": centers of rects
    """

    axes = F.partial(torch.linspace, 0, 1)
    grid = torch.stack(torch.meshgrid(*map(axes, s), indexing="ij"), -1)

    if mode == "cell":
        grid += .5 / torch.array(s)
    if flatten:
        return grid.reshape(-1, len(s))

    return grid


def nmap(f: Fx, n: int = 1, **kwargs) -> Fx:
    """
        Nested vmap. Keeps the same semantics as `jax.vmap` except that arbitrary
        `n` leading dimensions are vectorized. Returns the vmapped function.
    """

    if not n:
        return f

    if n > 1:
        f = nmap(f, n - 1)
    return torch.vmap(f, **kwargs)
