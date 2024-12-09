from typing import Dict

import torch
from torch import nn, Tensor

from src.basis.series import Basis
from src.model.solvers.base import Spectral, lecun_init, \
    linear_lecun_weight_zero_bias
from src.pde.pde import PDE

# ---------------------------------------------------------------------------- #
#                                    SOLVER                                    #
# ---------------------------------------------------------------------------- #


class DenseGeneral(nn.Module):
    def __init__(self, out_shape, axes):
        super().__init__()
        self.out_shape = out_shape
        self.axes = tuple(axes)

        self.weight = nn.Parameter(
            torch.empty(
                *self.out_shape,
                *self.out_shape),
            True)
        lecun_init(self.weight)
        self.bias = nn.Parameter(torch.zeros(*self.out_shape), True)

        self._initialize_parameters()

    def forward(self, x):
        return torch.tensordot(
            x, self.weight, dims=(
                self.axes, self.axes)) + self.bias


class SNO(Spectral):

    def __repr__(self): return "SNO"

    def __init__(self, pde: PDE, cfg: Dict) -> None:
        super().__init__(pde, cfg)

        self.layer_1 = linear_lecun_weight_zero_bias(4, self.cfg['hdim'] * 4)
        self.layer_2 = linear_lecun_weight_zero_bias(
            self.cfg['hdim'] * 4, self.cfg['hdim'])

        self.integral_list = nn.ModuleList()
        self.linear_list = nn.ModuleList()

        for _ in range(self.cfg['depth']):
            conv = DenseGeneral(self.cfg['mode'], axes=range(-3, 0))
            linear = linear_lecun_weight_zero_bias(
                self.cfg['hdim'], self.cfg['hdim'])
            self.integral_list.append(conv)
            self.linear_list.append(linear)

        self.layer_n_minus_3 = linear_lecun_weight_zero_bias(
            self.cfg['hdim'], self.cfg['hdim'])
        self.layer_n_minus_2 = linear_lecun_weight_zero_bias(
            self.cfg['hdim'], self.cfg['hdim'] * 4)
        self.layer_n_minus_1 = linear_lecun_weight_zero_bias(
            self.cfg['hdim'] * 4, self.pde.odim)

    def forward(self, phi: Basis) -> Basis:

        u = phi.to(*self.cfg["mode"])

        bias = u.transform(u.grid(*u.mode)).coef
        u = u.map(lambda coef: torch.concatenate([coef, bias], axis=-1))

        u = u.map(self.layer_1)
        u = u.map(self.activate)

        u = u.map(self.layer_2)
        u = u.map(self.activate)

        for K, linear in zip(self.integral_list, self.linear_list):

            def Integral(coef: Tensor) -> Tensor:
                return torch.moveaxis(K(torch.moveaxis(coef, -1, 0)), 0, -1)

            u = u.map(Integral)

            u = u.map(linear)
            u = u.map(self.activate)

        u = u.map(self.layer_n_minus_3)
        u = u.map(self.activate)

        u = u.map(self.layer_n_minus_2)
        u = u.map(self.activate)

        u = u.map(self.layer_n_minus_1)
        return self.pde.mollifier(phi, u)
