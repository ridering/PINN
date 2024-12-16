import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from tqdm import trange
from torch.utils.tensorboard import writer

from src.model.solvers.base import Solver
from src.model.solvers.fno import FNO
from src.model.solvers.sno import SNO
from src.pde.pde import PDE
from src.devices import DEVICE
from src.validate import evaluate

import torch

torch.set_default_device(DEVICE)


def main(cfg: Dict[str, Any]):

    from importlib import import_module

    if cfg["pde"] is not None:

        col, name = cfg["pde"].split(".", 2)
        mod = import_module(f"src.pde.{col}.equation")

        pde: PDE = getattr(mod, name)
        pde.solution  # load solution

        if cfg['model'] == 'fno':
            model = FNO(pde, cfg)
        else:
            model = SNO(pde, cfg)
        model = model.to(DEVICE)

# ---------------------------------------------------------------------------- #
#                                     TRAIN                                    #
# ---------------------------------------------------------------------------- #

    if cfg["action"] == "train":

        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

        if cfg["schd"] is None:
            def scheduler_fn(epoch):
                return 1.0
        elif cfg["schd"] == "exp":
            decay_rate = 1e-3 ** (1.0 / cfg["iter"])

            def scheduler_fn(epoch):
                return decay_rate ** epoch

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=scheduler_fn)

        pbar = trange(cfg['iter'])
        loss_hist = []

        if cfg['tensorboard'] > 0:
            wr = writer.SummaryWriter()
            wr.add_scalar('lr', scheduler.get_last_lr()[0], 0)
            for param_name, weights in model.named_parameters():
                flattened_weights = weights.flatten()
                wr.add_histogram(param_name, flattened_weights,
                                 0, bins='tensorflow')

        avg_avg_loss = 0.0

        for epoch in pbar:

            phi = pde.params.sample((cfg['bs'], ))

            def mapper(smth): return model.loss(pde.basis(smth))['residual']
            loss = torch.vmap(mapper, chunk_size=cfg['vmap'])(phi.coef)
            avg_loss = loss.cpu().detach().mean()
            loss.mean().backward()

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            avg_avg_loss += avg_loss

            with torch.no_grad():
                if cfg['tensorboard'] > 0 and (
                        epoch + 1) % cfg['tensorboard'] == 0:

                    wr.add_scalar('loss', avg_avg_loss / cfg['tensorboard'],
                                  epoch + 1)

                    model.eval()
                    metric, predictions = evaluate(model)

                    for key, val in metric.items():
                        wr.add_scalar(key, val, epoch + 1)

                    wr.add_image('u[1]_pred',
                                 predictions[1][1][-1].squeeze(),
                                 dataformats="HW",
                                 global_step=epoch + 1)
                    model.train()

                    wr.add_scalar('lr', scheduler.get_last_lr()[0], epoch + 1)
                    for param_name, weights in model.named_parameters():
                        flattened_weights = weights.flatten()
                        wr.add_histogram(
                            param_name,
                            flattened_weights,
                            epoch + 1,
                            bins='tensorflow')
                    avg_avg_loss = 0.0

                if (epoch + 1) % cfg['ckpt'] == 0:
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': loss,
                        },
                        'saves/' +
                        datetime.now().strftime("%Y-%m-%d--%H-%M-%S") +
                        f'--epoch-{epoch + 1}')

            pbar.set_description(
                f"LR: {scheduler.get_last_lr()} Loss: {avg_loss:10}")
            loss_hist.append(avg_loss)

        wr.flush()
        wr.close()

        return pde, model, loss_hist


# ---------------------------------------------------------------------------- #
#                                   ARGPARSE                                   #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":

    import argparse
    args = argparse.ArgumentParser()
    action = args.add_subparsers(dest="action")

    args.add_argument("--pde", type=str, help="PDE name")
    args.add_argument(
        "--model",
        type=str,
        help="model name",
        choices=[
            "fno",
            "sno"])  # --cheb=cno

# ----------------------------------- MODEL ---------------------------------- #

    args.add_argument("--hdim", type=int, help="hidden dimension")
    args.add_argument("--depth", type=int, help="number of layers")
    args.add_argument("--activate", type=str, help="activation name")

    args.add_argument(
        "--mode",
        type=int,
        nargs=3,
        help="number of modes per dim")
    args.add_argument(
        "--grid",
        type=int,
        default=256,
        help="training grid size")

    # ablation study

    args.add_argument(
        "--fourier",
        dest="fourier",
        action="store_true",
        help="fourier basis only")
    args.add_argument(
        "--cheb",
        dest="cheb",
        action="store_true",
        help="using chebyshev")

# ----------------------------------- TRAIN ---------------------------------- #

    args_train = action.add_parser("train", help="train model from scratch")

    args_train.add_argument("--bs", type=int, required=True, help="batch size")
    args_train.add_argument(
        "--lr",
        type=float,
        required=True,
        help="learning rate")
    args_train.add_argument(
        "--schd",
        type=str,
        required=True,
        help="scheduler name")
    args_train.add_argument(
        "--iter",
        type=int,
        required=True,
        help="total iterations")
    args_train.add_argument(
        "--ckpt",
        type=int,
        required=True,
        help="checkpoint every n iters")
    args_train.add_argument(
        "--tensorboard",
        type=int,
        required=False,
        default=0,
        help="watch training on tensorboard")
    args_train.add_argument(
        "--vmap",
        type=int,
        required=False,
        default=1,
        help="parallel computation")


# ----------------------------------- TEST ----------------------------------- #

# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

    args = args.parse_args()
    cfg = vars(args)
    print(f"{cfg=}")

    os.makedirs("saves", exist_ok=True)

    curr_datetime = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

    pde, model, loss_hist = main(cfg)

    torch.save(model.state_dict(), 'saves/final-model--' + curr_datetime)
