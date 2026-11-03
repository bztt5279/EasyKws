"""Bounded 16 kHz mono FP32 replay shared by calibration and runtime.

The frame/cache loop is adapted from the retained local runtime. The frontend
uses the same complete Kaldi configuration as offline preprocessing. No audio
device is opened, and no input audio, scores or events are accumulated here.
"""
from __future__ import annotations

import numpy as np
import torch

from .preprocess.frontend import extract_fbank


class StreamingScorer:
    """Score complete 25 ms frames with 10 ms shifts, without end padding.

    Each push consumes finite one-dimensional float32 PCM in [-1, 1], already
    sampled at 16 kHz. Digital silence is valid and advances every state. The
    model is put in evaluation mode and must already reside on CPU in FP32.
    One model call per frame makes inference independent of push chunk length.
    """

    def __init__(self, model: torch.nn.Module):
        if not isinstance(model, torch.nn.Module):
            raise ValueError("model must be a torch module")
        self.model = model
        self._validate_model()
        self.model.eval()
        self.reset()

    def _validate_model(self) -> None:
        for value in (*self.model.parameters(), *self.model.buffers()):
            if value.device.type != "cpu" or value.is_complex() or (value.is_floating_point() and value.dtype != torch.float32):
                raise ValueError("streaming scoring requires a CPU FP32 model")
            if not torch.isfinite(value).all():
                raise ValueError("streaming model contains non-finite values")

    def reset(self) -> None:
        """Start a distinct uninterrupted session; discard residual and cache."""
        self.buffer = np.empty(0, dtype=np.float32)
        self.cache = torch.empty((0, 0, 0), dtype=torch.float32)
        self.sample_count = self.frames_processed = 0
        self.finished = False

    @property
    def audio_sec(self) -> float:
        return self.sample_count / 16000.0

    def push(self, samples: np.ndarray) -> list[dict]:
        """Return only new frame records; invalid blocks leave state unchanged."""
        data = np.asarray(samples)
        if data.ndim != 1 or data.dtype != np.float32:
            raise ValueError("input must be one-dimensional float32 mono 16 kHz PCM")
        if not np.isfinite(data).all() or np.any(np.abs(data) > 1):
            raise ValueError("input must be finite PCM in [-1, 1]")
        if self.finished:
            raise ValueError("stream is flushed; reset before pushing a new session")
        if self.model.training:
            raise ValueError("streaming model must remain in evaluation mode")
        if torch.is_autocast_enabled("cpu"):
            raise ValueError("streaming scoring does not support autocast")
        if not data.size:
            return []
        rows = []
        offset = 0
        # Add only enough samples for one frame at a time. The owned waveform
        # buffer never exceeds 400 samples, even for very large caller blocks.
        while offset < data.size:
            take = min(400 - self.buffer.size, data.size - offset)
            self.buffer = np.concatenate((self.buffer, data[offset:offset + take]))
            offset += take
            if self.buffer.size < 400:
                break
            features = torch.from_numpy(extract_fbank(self.buffer)).reshape(1, 1, 40)
            with torch.inference_mode():
                probs, cache = self.model(features, self.cache)
            if probs.shape != (1, 1, 1) or probs.dtype != torch.float32 or probs.device.type != "cpu" or not torch.isfinite(probs).all() or not (0 <= probs).all() or not (probs <= 1).all():
                raise ValueError("model must return finite CPU FP32 [1,1,1] probabilities")
            if not torch.is_tensor(cache) or cache.device.type != "cpu" or cache.dtype != torch.float32 or not torch.isfinite(cache).all():
                raise ValueError("model must return a finite CPU FP32 causal cache")
            self.cache = cache.detach().clone()
            rows.append({"frame_index": self.frames_processed,
                         "t_audio_sec": (self.frames_processed * 160 + 400) / 16000.0,
                         "score": float(probs[0, 0, 0])})
            self.frames_processed += 1
            self.buffer = self.buffer[160:].copy()
        self.sample_count += int(data.size)
        return rows

    def flush(self) -> list[dict]:
        """Close the session; drop the incomplete residual without padding.

        All complete frames were scored during push. Repeated flush is a no-op.
        Input sample count and its audio duration still include the residual.
        """
        self.finished = True
        self.buffer = np.empty(0, dtype=np.float32)
        return []
