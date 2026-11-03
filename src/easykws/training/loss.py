"""Single-keyword max-pooling loss adapted from a WeKWS-derived local source.

The exact upstream revision is unknown. Positive samples use their maximum
eligible probability; negative samples use the minimum complementary
probability. Padding is excluded. Label 0 means positive and -1 means negative.
"""
from __future__ import annotations

import torch


_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def padding_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Return [B,T] with True only at padding positions."""
    if lengths.ndim != 1 or lengths.numel() == 0 or lengths.dtype not in _INTEGER_DTYPES:
        raise ValueError("lengths must be a nonempty integer vector")
    if not (lengths > 0).all():
        raise ValueError("all valid lengths must be positive")
    largest = int(lengths.max().item())
    width = largest if max_len is None else max_len
    if type(width) is not int or width < largest:
        raise ValueError("mask width must be an integer at least the largest valid length")
    return torch.arange(width, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


def validate_targets(target: torch.Tensor, num_keywords: int = 1) -> None:
    if num_keywords != 1:
        raise ValueError("only one keyword output is supported")
    if target.ndim != 1 or target.numel() == 0 or target.dtype not in _INTEGER_DTYPES:
        raise ValueError("targets must be a nonempty integer vector")
    if not ((target == 0) | (target == -1)).all():
        raise ValueError("labels must be 0 (positive) or -1 (negative)")


def _validate_probs(probs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if probs.ndim != 3 or probs.shape[0] < 1 or probs.shape[1] < 1 or probs.shape[2] != 1:
        raise ValueError("probabilities must be nonempty [batch, frames, 1]")
    if not probs.is_floating_point() or not torch.isfinite(probs).all() or not ((probs >= 0) & (probs <= 1)).all():
        raise ValueError("probabilities must be finite floating values in [0, 1]")
    mask = padding_mask(lengths, probs.shape[1])
    if lengths.shape[0] != probs.shape[0]:
        raise ValueError("lengths and probabilities have different batch sizes")
    return mask.to(probs.device)


def frame_max_score(probs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Return one score per sample, pooling over valid frames only."""
    mask = _validate_probs(probs, lengths)
    return probs.masked_fill(mask.unsqueeze(-1), 0.0).max(dim=1).values.reshape(probs.shape[0])


def max_pooling_loss(probs: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor,
                     min_duration: int = 0) -> tuple[torch.Tensor, float]:
    """Return mean per-sample loss and strict-threshold (> or < 0.5) accuracy.

    min_duration skips an initial prefix for positive loss only, preserving the
    historical rule. Accuracy still considers all valid frames. A positive
    sample must retain at least one eligible frame after this prefix.
    """
    mask = _validate_probs(probs, lengths)
    validate_targets(target)
    if target.shape[0] != probs.shape[0]:
        raise ValueError("targets and probabilities have different batch sizes")
    if type(min_duration) is not int or min_duration < 0:
        raise ValueError("min_duration must be a nonnegative integer")
    target = target.to(probs.device)
    positive = target == 0
    negative = ~positive
    if (lengths.to(probs.device)[positive] <= min_duration).any():
        raise ValueError("min_duration excludes every valid frame of a positive sample")
    positive_mask = mask.clone()
    positive_mask[:, :min_duration] = True
    loss = probs.new_tensor(0.0)
    if positive.any():
        p = probs[positive, :, 0].masked_fill(positive_mask[positive], 0.0).clamp(1e-8, 1.0)
        loss = loss - torch.log(p.max(dim=1).values).sum()
    if negative.any():
        p = (1 - probs[negative, :, 0]).masked_fill(mask[negative], 1.0).clamp(1e-8, 1.0)
        loss = loss - torch.log(p.min(dim=1).values).sum()
    scores = probs.masked_fill(mask.unsqueeze(-1), 0.0).max(dim=1).values.reshape(probs.shape[0])
    correct = ((scores > 0.5) & positive) | ((scores < 0.5) & negative)
    return loss / probs.shape[0], float(correct.float().mean().item())
