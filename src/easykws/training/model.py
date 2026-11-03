"""Causal depthwise-separable TCN with persistent global CMVN.

The layer layout is adapted from an earlier local implementation attributed to
WeKWS. Its exact upstream revision has not been established. Layer/state names,
default dimensions, operation order, and normalization buffers are preserved;
this module does not claim a verified upstream commit or training result.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


_DEFAULT_CONFIG = {
    "input_dim": 40, "hidden_dim": 256, "num_layers": 4,
    "kernel_size": 8, "dropout": 0.1, "norm_var": True,
}


def model_config(config: dict | None = None) -> dict:
    """Return a validated complete configuration; never mutate the input."""
    if config is not None and not isinstance(config, dict):
        raise ValueError("model config must be an object")
    supplied = {} if config is None else config
    if set(supplied) - set(_DEFAULT_CONFIG):
        raise ValueError("unknown model config field")
    result = {**_DEFAULT_CONFIG, **supplied}
    for name in ("input_dim", "hidden_dim", "num_layers", "kernel_size"):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if result["input_dim"] != 40:
        raise ValueError("input_dim must match the 40-bin frontend")
    dropout = result["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, Real) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout must be finite and in [0, 1)")
    result["dropout"] = float(dropout)
    if type(result["norm_var"]) is not bool:
        raise ValueError("norm_var must be boolean")
    return result


class GlobalCMVN(nn.Module):
    """Normalize raw fbank once; statistics travel in the state dictionary."""

    def __init__(self, mean: torch.Tensor, istd: torch.Tensor, norm_var: bool = True):
        super().__init__()
        if mean.ndim != 1 or mean.shape != istd.shape:
            raise ValueError("CMVN mean and istd must be matching vectors")
        if not torch.isfinite(mean).all() or not torch.isfinite(istd).all() or not (istd > 0).all():
            raise ValueError("CMVN requires finite mean and positive finite istd")
        self.norm_var = norm_var
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("istd", istd.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x


class DsCnnBlock(nn.Module):
    """Causal depthwise/pointwise convolution with a residual connection."""

    def __init__(self, channel: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.cnn = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size, stride=1, dilation=dilation, groups=channel),
            nn.BatchNorm1d(channel), nn.ReLU(),
            nn.Conv1d(channel, channel, kernel_size=1, stride=1),
            nn.BatchNorm1d(channel), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cache: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if cache is None or cache.size(0) == 0:
            y = F.pad(x, (self.padding, 0), value=0.0)
        else:
            y = torch.cat((cache, x), dim=2)
        new_cache = y[:, :, -self.padding:] if self.padding else y[:, :, :0]
        return self.cnn(y) + x, new_cache


class TCN(nn.Module):
    def __init__(self, num_layers: int, channel: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.padding = 0
        self.network = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.padding += (kernel_size - 1) * dilation
            self.network.append(DsCnnBlock(channel, kernel_size, dilation, dropout))

    def forward(self, x: torch.Tensor, in_cache: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        out_caches = []
        offset = 0
        for block in self.network:
            cache = in_cache[:, :, offset:offset + block.padding] if in_cache is not None and in_cache.size(0) > 0 else None
            x, new_cache = block(x, cache)
            out_caches.append(new_cache)
            offset += block.padding
        return x.transpose(1, 2), torch.cat(out_caches, dim=2)


class KWSModel(nn.Module):
    """Return sigmoid probabilities [B,T,1] and the causal convolution cache."""

    def __init__(self, input_dim: int = 40, hidden_dim: int = 256, num_layers: int = 4,
                 kernel_size: int = 8, dropout: float = 0.1,
                 cmvn_mean: torch.Tensor | None = None, cmvn_istd: torch.Tensor | None = None,
                 norm_var: bool = True):
        super().__init__()
        self.config = model_config(dict(input_dim=input_dim, hidden_dim=hidden_dim,
                                       num_layers=num_layers, kernel_size=kernel_size,
                                       dropout=dropout, norm_var=norm_var))
        if (cmvn_mean is None) != (cmvn_istd is None):
            raise ValueError("both CMVN vectors are required together")
        if cmvn_mean is not None and cmvn_mean.shape != (input_dim,):
            raise ValueError("CMVN dimension does not match input_dim")
        self.cmvn = GlobalCMVN(cmvn_mean, cmvn_istd, norm_var) if cmvn_mean is not None else None
        self.preprocess = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.backbone = TCN(num_layers, hidden_dim, kernel_size, dropout)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, in_cache: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != self.config["input_dim"]:
            raise ValueError("features must be nonempty [batch, frames, input_dim]")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("features must have finite floating values")
        if in_cache is not None and (in_cache.ndim != 3 or (in_cache.shape[0] == 0 and tuple(in_cache.shape) != (0, 0, 0))):
            raise ValueError("cache must be [batch, hidden_dim, context] or [0,0,0]")
        if in_cache is not None and in_cache.size(0) > 0:
            expected = (x.shape[0], self.config["hidden_dim"], self.backbone.padding)
            if tuple(in_cache.shape) != expected or in_cache.device != x.device or in_cache.dtype != x.dtype:
                raise ValueError("cache shape, dtype and device must match features and model")
            if not torch.isfinite(in_cache).all():
                raise ValueError("cache must be finite")
        if self.cmvn is not None:
            x = self.cmvn(x)
        x = self.preprocess(x)
        x, out_cache = self.backbone(x, in_cache)
        return self.activation(self.classifier(x)), out_cache


def build_model(cmvn: dict, config: dict | None = None) -> KWSModel:
    """Build with float32 persistent statistics from a preprocessing CMVN record."""
    resolved = model_config(config)
    if not isinstance(cmvn, dict) or "mean" not in cmvn or "istd" not in cmvn:
        raise ValueError("CMVN record requires mean and istd")
    try:
        mean = torch.as_tensor(cmvn["mean"], dtype=torch.float32)
        istd = torch.as_tensor(cmvn["istd"], dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("invalid CMVN vectors") from exc
    return KWSModel(**resolved, cmvn_mean=mean, cmvn_istd=istd)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
