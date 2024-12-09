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

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


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

        if cfg['tensorboard']:
            wr = writer.SummaryWriter()

        for epoch in pbar:

            phi = pde.params.sample((cfg['bs'], ))

            ls = []

            optimizer.zero_grad()
            for batch in phi.coef:

                loss = model.loss(pde.basis(batch))['residual']

                loss.backward()

                ls.append(loss.cpu().detach().numpy())
            optimizer.step()
            scheduler.step()

            avg_loss = np.average(ls)

            kernel_type = 'conv' if cfg['model'] == 'fno' else 'integr'

            if cfg['tensorboard']:
                wr.add_scalar('loss', avg_loss, epoch + 1)
                wr.add_scalar('lr', scheduler.get_last_lr()[0], epoch + 1)
                for param_name, weights in model.named_parameters():
                    if param_name.find(kernel_type) == -1:
                        flattened_weights = weights.flatten()
                        wr.add_histogram(
                            param_name,
                            flattened_weights,
                            epoch + 1,
                            bins='tensorflow')
                    else:
                        if (epoch + 1) % 10 == 0:
                            flattened_weights = weights.flatten()
                            wr.add_histogram(
                                param_name,
                                flattened_weights,
                                epoch + 1,
                                bins='tensorflow')
                            wr.flush()

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
                    f'epoch_{epoch + 1}')

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
    args.add_argument(
        "--spectral",
        dest="spectral",
        action="store_true",
        help="spectral training")

# ----------------------------------- MODEL ---------------------------------- #

    args.add_argument("--hdim", type=int, help="hidden dimension")
    args.add_argument("--depth", type=int, help="number of layers")
    args.add_argument("--activate", type=str, help="activation name")

    args.add_argument(
        "--mode",
        type=int,
        nargs="+",
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
        action=argparse.BooleanOptionalAction,
        help="watch training on tensorboard")

# ----------------------------------- TEST ----------------------------------- #

# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

    args = args.parse_args()
    cfg = vars(args)
    print(f"{cfg=}")

    pde, model, loss_hist = main(cfg)

    torch.save([
        model.state_dict(),
        loss_hist,
    ], datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
