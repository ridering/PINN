from typing import Tuple, Dict

import numpy as np
import torch
from torch import Tensor

from src.basis.series import Basis
from src.pde.pde import PDE
from src.model.solvers.base import Solver
from src.utils import Fx, grid


def evaluate(model: Solver) -> Tuple[Dict, Tensor]:

    if isinstance(model.pde.solution, Tuple):
        ϕ, s, u = model.pde.solution

    v = []
    metr = []
    metr = {'erra': [], 'errr': []}
    metric_caller = metric(model.pde, s)

    for batch, u_i in zip(ϕ.coef, u):
        v_i = model.u(model.pde.basis(batch), s)
        v.append(v_i.cpu().detach())

        metr_i = metric_caller(model.pde.basis(batch), u_i, v_i)
        metr['erra'].append(metr_i['erra'].item())
        metr['errr'].append(metr_i['errr'].item())
        # metr.append(metr_i)

    metr['erra'] = np.mean(metr['erra'])
    metr['errr'] = np.mean(metr['errr'])

    return metr, (u, torch.stack(v))


def metric(pde: PDE, s: Tuple[int]) -> Fx:

    def call(ϕ: Basis, u: Tensor, v: Tensor) -> Dict[str, Tensor]:

        return dict(
            erra=torch.mean(torch.abs(torch.ravel(u - v))),
            errr=torch.linalg.norm(torch.ravel(u - v)) /
            torch.linalg.norm(torch.ravel(u)),
            # residual=torch.mean(torch.abs(pde.equation(grid(*s), ϕ[s], v))),
        )

    return call
