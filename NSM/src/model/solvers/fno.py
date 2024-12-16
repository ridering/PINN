import math
from typing import Dict

import torch
from torch import nn, Tensor

from src.basis.series import Basis
from src.basis.fourier import Fourier
from src.pde.pde import PDE
from src.model.solvers.base import Spectral, linear_lecun_weight_zero_bias


def create_weight_bias(mode: tuple[int], idim: int, odim: int) -> Tensor:

    weight_shape = (*mode, idim, odim)
    # weight = torch.randn(weight_shape) * scale
    weight = torch.empty(weight_shape)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(*mode, idim)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)

    return weight, bias

# ---------------------------------------------------------------------------- #
#                                    SOLVER                                    #
# ---------------------------------------------------------------------------- #


class SpectralConv(nn.Module):

    def __init__(
        self,
        mode: tuple[int],
        idim: int,
        odim: int,
        init: tuple[int] = None
    ) -> None:
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.mode = mode
        self.init = init

        w, b = create_weight_bias(mode, idim, odim)

        self.weight = nn.Parameter(w, True)
        self.bias = nn.Parameter(b, True)

    def forward(self, u):
        def W(a):
            result = torch.einsum("ijkl,ijklm->ijkm", a, self.weight)
            return result + self.bias

        mode = self.mode or u.mode
        return u.to(*mode).map(W).to(*u.mode)


class FNO(Spectral, nn.Module):

    def __repr__(self): return "NSM"

    def __init__(self, pde: PDE, cfg: Dict) -> None:
        super().__init__(pde, cfg)

        self.layer_1 = linear_lecun_weight_zero_bias(4, self.cfg['hdim'] * 4)
        self.layer_2 = linear_lecun_weight_zero_bias(
            self.cfg['hdim'] * 4, self.cfg['hdim'])

        self.convs_list = nn.ModuleList()
        self.lins_list = nn.ModuleList()

        for _ in range(self.cfg['depth']):
            conv = SpectralConv(
                self.cfg['mode'],
                self.cfg['hdim'],
                self.cfg['hdim'])
            linear = linear_lecun_weight_zero_bias(
                self.cfg['hdim'], self.cfg['hdim'])
            self.convs_list.append(conv)
            self.lins_list.append(linear)

        self.layer_n_minus_3 = linear_lecun_weight_zero_bias(
            self.cfg['hdim'], self.cfg['hdim'])
        self.layer_n_minus_2 = linear_lecun_weight_zero_bias(
            self.cfg['hdim'], self.cfg['hdim'] * 4)
        self.layer_n_minus_1 = linear_lecun_weight_zero_bias(
            self.cfg['hdim'] * 4, self.pde.odim)

    def forward(self, phi: Basis) -> Basis:

        if not self.cfg["fourier"]:
            T = self.pde.basis
        else:
            T = Fourier[self.pde.domain.ndim]

        u = phi.to(*self.cfg["mode"])

        bias = T.transform(u.grid(*u.mode)).coef
        u = u.map(lambda coef: torch.concatenate([coef, bias], axis=-1))

        u = u.map(self.layer_1)
        u = T.transform(self.activate(u.inv()))

        u = u.map(self.layer_2)
        u = T.transform(self.activate(u.inv()))

        for conv, linear in zip(self.convs_list, self.lins_list):
            convolved = conv(u)
            fc = u.map(linear)

            u = T.add(convolved, fc)
            u = T.transform(self.activate(u.inv()))

        u = u.map(self.layer_n_minus_3)
        u = T.transform(self.activate(u.inv()))

        u = u.map(self.layer_n_minus_2)
        u = T.transform(self.activate(u.inv()))

        u = u.map(self.layer_n_minus_1)
        return self.pde.mollifier(phi, u)
