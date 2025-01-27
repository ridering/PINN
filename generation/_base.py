import torch


class Dataset:
    def __init__(
        self,
        num_samples: int,
        steady: bool,
        time_spatial_bounds: tuple[tuple[float, float], ...],
        time_spatial_num_steps: tuple[int, ...],
    ) -> None:

        assert len(time_spatial_bounds) == len(
            time_spatial_num_steps
        ), "Different num-steps and boundary dimensions"
        assert (
            1 <= len(time_spatial_bounds) <= 4
        ), "Can be used for 1 to 4 dimension problems"
        assert (
            isinstance(num_samples, int) and 1 <= num_samples
        ), "num_samples must be positive integer"

        self._config = {}

        config = {}

        dims = ["X", "Y", "Z"]
        if steady:
            assert len(time_spatial_bounds) <= 3, "Only 3 spatial dimensions supported"
            config["steady"] = True
        else:
            dims.insert(0, "T")
            config["steady"] = False

        for dim, bounds, num_steps in zip(
            dims, time_spatial_bounds, time_spatial_num_steps
        ):
            config[dim] = {"min": bounds[0], "max": bounds[1], "steps": num_steps}

        config["ndim"] = len(time_spatial_num_steps)
        config["mesh"] = [num_samples, *time_spatial_num_steps]

        self._config = config
        self._data = None

    def generate_random(self): ...

    def solve(self): ...

    def save(self, path):
        torch.save(self._config, path)

    @staticmethod
    def load(path) -> "Dataset":
        dataset = __class__.__new__(__class__)
        dataset._config = torch.load(path)
        return dataset
