"""Feature integrity and train-only normalization statistics."""
import numpy as np

from .errors import PreprocessError


class CMVN:
    """Accumulate raw training features in float64; never normalize the cache."""

    def __init__(self):
        import torch

        self.count = 0
        self.total = torch.zeros(40, dtype=torch.float64)
        self.squares = torch.zeros(40, dtype=torch.float64)

    def add(self, features):
        import torch

        data = np.asarray(features)
        if data.ndim != 2 or data.shape[1] != 40 or not len(data) or not np.isfinite(data).all():
            raise PreprocessError("CMVN requires finite nonempty [frames, 40] features")
        tensor = torch.as_tensor(data, dtype=torch.float64)
        self.total += tensor.sum(dim=0)
        self.squares += tensor.square().sum(dim=0)
        self.count += len(data)

    def result(self):
        if not self.count:
            raise PreprocessError("No training frames for CMVN")
        mean = self.total / self.count
        variance = (self.squares / self.count - mean.square()).clamp_min(1e-10)
        return {"mean": mean.tolist(), "var": variance.tolist(),
                "istd": variance.rsqrt().tolist(), "frame_num": self.count, "fit_split": "train"}
