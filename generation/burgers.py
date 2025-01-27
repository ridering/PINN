import tqdm
import numpy as np
import torch
from numba import njit
from tqdm import trange

from _base import Dataset
from utils import GaussianRF


@njit
def burgers_explicit(f, x_max, x_steps, t_max, t_steps, visc):

    h = x_max / (x_steps - 1)
    target_tau = t_max / (t_steps - 1)

    f_t_x = np.zeros((t_steps, x_steps))
    f_t_x[0] = f.copy()

    t_idx_next = 1
    if_next = False
    t = 0
    N = len(f)

    while t < t_max:
        tau = min(h * h / (4 * visc), visc / np.max(f**2))
        if tau < 1e-8:
            break
        if t + tau > t_idx_next * target_tau:
            tau = t_idx_next * target_tau - t
            if_next = True
        
        f2 = f.copy()

        f2[1:-1] = (
            f[1:-1]
            - f[1:-1] * tau / (2 * h) * (f[2:] - f[:-2])
            + visc * tau / (h * h) * (f[:-2] - 2 * f[1:-1] + f[2:])
        )
        
        left = f[N - 2]
        right = f[1]

        f2[0] = f[0] - f[0] * tau / (2 * h) * (right - left) + \
            visc * tau / (h * h) * (left - 2 * f[0] + right)
        
        f2[N - 1] = f[N - 1] - f[N - 1] * tau / (2 * h) * (right - left) + \
            visc * tau / (h * h) * (left - 2 * f[N - 1] + right)
        
        f = f2.copy()

        if if_next:
            f_t_x[t_idx_next] = f.copy()
            t_idx_next += 1
            if_next = False

        t += tau

    return f_t_x


class BurgersDataset(Dataset):
    def __init__(
        self,
        visc: float,
        num_samples: int,
        t_max: float,
        t_steps: int,
        x_steps: int,
        seed: int = 42,
        device: str = 'cpu',
    ) -> None:
        super().__init__(num_samples, False, ((0, t_max), (0, 1)), (t_steps, x_steps))
        self._config["visc"] = visc

        GRF = GaussianRF(1, x_steps, alpha=2.5, tau=7, device='cpu', seed=seed)
        ics = GRF.sample(num_samples).numpy()

        data = np.empty(self._config["mesh"])
        data[:, 0] = ics

        for i in trange(num_samples):
            data[i] = burgers_explicit(ics[i], 1, x_steps, t_max, t_steps, visc)

        self._config['initial_conditions'] = torch.tensor(ics, dtype=torch.float32, device=device)
        self._config['data'] = torch.tensor(data, dtype=torch.float32, device=device)
    


    # def generate_random(self, min_factor: int, max_factor: int):
    #     assert self._config is not None, "Dataset settings were't initialized correctly"

    #     num_samples = self._config["mesh"][0]
    #     x_steps = self._config["X"]["steps"]

    #     assert (
    #         0 < min_factor <= max_factor <= x_steps
    #     ), "factors should be in [1, x_steps]"

    #     factors = torch.randint(min_factor, max_factor + 1, (num_samples,))
    #     mask = torch.arange(x_steps) < factors[:, None]

    #     freqs = torch.zeros((num_samples, x_steps, 2))
    #     freqs[mask] = torch.normal(0, 1, (mask.sum(), 2))

    #     inversed = torch.fft.ifft(torch.view_as_complex(freqs), axis=1).real
    #     # first = inversed[:, 0][:, None]
    #     # last = inversed[:, -1][:, None]

    #     # k = last - first
    #     # lines = k * np.linspace(0, 1, x_steps) + first

    #     # inversed -= lines

    #     self._config["initial_conditions"] = inversed.numpy()

    # def solve(self):
    #     assert (
    #         self._config is not None
    #         and self._config.get("initial_conditions") is not None
    #     ), "Initial conditions aren't generated"

    #     ics = self._config["initial_conditions"]
    #     data = np.empty(self._config["mesh"])
    #     data[:, 0] = ics

    #     x_steps = self._config["X"]["steps"]
    #     x_max = self._config["X"]["max"] - self._config["X"]["min"]
    #     t_steps = self._config["T"]["steps"]
    #     t_max = self._config["T"]["max"] - self._config["T"]["min"]
    #     visc = self._config["visc"]

    #     for i in trange(len(ics)):
    #         data[i] = burgers_explicit(ics[i], x_max, x_steps, t_max, t_steps, visc)

    #     return data
