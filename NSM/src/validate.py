from typing import Tuple, Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.basis.series import Basis
from src.pde.pde import PDE
from src.model.solvers.base import Solver
from src.utils import Fx, grid


def evaluate(model: Solver) -> Tuple[Dict, Tensor]:

    if isinstance(model.pde.solution, Tuple):
        ϕ, s, u = model.pde.solution
    
    phi_dataloader = DataLoader(ϕ.coef, batch_size=model.cfg['bs'], shuffle=False)
    u_dataloader = DataLoader(u, batch_size=model.cfg['bs'], shuffle=False)

    mapper = lambda smth: model.u(model.pde.basis(smth), s)
    metric_caller = metric(model.pde, s)
    metricer = lambda a, b, c: metric_caller(model.pde.basis(a), b, c)
    
    v = []
    metr = {'erra': [], 'errr': []}
    for phi_batch, u_batch in zip(phi_dataloader, u_dataloader):
        v_i = torch.vmap(mapper, chunk_size=model.cfg['vmap'])(phi_batch)
        v.append(v_i)

        metr_i = torch.vmap(metricer)(phi_batch, u_batch, v_i)
        metr['erra'].extend(metr_i['erra'])
        metr['errr'].extend(metr_i['errr'])
    
    metr['erra'] = torch.mean(torch.stack(metr['erra']))
    metr['errr'] = torch.mean(torch.stack(metr['errr']))
    v = torch.cat(v)

    return metr, (u, v)


def metric(pde: PDE, s: Tuple[int]) -> Fx:

    def call(ϕ: Basis, u: Tensor, v: Tensor) -> Dict[str, Tensor]:

        return dict(
            erra=torch.mean(torch.abs(torch.ravel(u - v))),
            errr=torch.linalg.norm(torch.ravel(u - v)) /
            torch.linalg.norm(torch.ravel(u)),
            # residual=torch.mean(torch.abs(pde.equation(grid(*s), ϕ[s], v))),
        )

    return call
