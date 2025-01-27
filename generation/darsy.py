import torch
import numpy as np
from numba import njit
from tqdm import trange

from _base import Dataset
from utils import GaussianRF


@njit
def sor(u, a, f, beta=1.3, eps=1e-3):
    err = np.inf

    alpha = 0

    N = len(a)
    h = 1 / (N - 1)

    a_x = a[:, 2:] - a[:, :-2]
    a_y = a[2:, :] - a[:-2, :]

    while err > eps:

        alpha += 1

        u_plus_1 = u.copy()

        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if a[i, j] == 0:
                    continue

                u_plus_1[i, j] = (beta / 16 / a[i, j]) * (
                    a_x[i, j - 1] * (u[i, j + 1] - u_plus_1[i, j - 1])
                    + a_y[i - 1, j] * (u[i, j + 1] - u_plus_1[i, j - 1])
                    + 4
                    * a[i, j]
                    * (
                        u[i + 1, j]
                        + u[i, j + 1]
                        + u_plus_1[i - 1, j]
                        + u_plus_1[i, j - 1]
                    )
                    + 4 * h * h * f[i, j]
                ) + (1 - beta) * u[i, j]

        err = np.sqrt(
            np.sum((u_plus_1[1:-1, 1:-1] - u[1:-1, 1:-1]) ** 2) / ((N - 1) ** 2)
        )

        u = u_plus_1

    return u, alpha


class DarcyFlowDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        x_steps: int,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__(num_samples, True, ((0, 1), (0, 1)), (x_steps, x_steps))

        GRF = GaussianRF(2, x_steps, alpha=2.5, tau=7, device="cpu", seed=seed)
        a = GRF.sample(num_samples).numpy()
        f = GRF.sample(1)[0].numpy()

        data = np.empty(self.config["mesh"])

        for i in trange(num_samples):
            a[i] -= a[i].min()
            a[i] += 1e-3
            u = np.zeros_like(a[i])
            data[i], _ = sor(u, a[i], f, 1.9, 1e-8)

        self.config["a"] = torch.tensor(a, dtype=torch.float32, device=device)
        self.config["f"] = torch.tensor(f, dtype=torch.float32, device=device)
        self.config["data"] = torch.tensor(data, dtype=torch.float32, device=device)
