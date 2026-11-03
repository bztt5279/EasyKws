"""Explicit frontend preserving the retained project's numerical contract.

This adapter is newly written. See docs/reference/preprocessing.md for its scope and
compatibility checks; no historical upstream revision is inferred.
"""
from __future__ import annotations

import io
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .errors import PreprocessError

FBANK = {
    "sample_frequency": 16000.0,
    "num_mel_bins": 40,
    "frame_length": 25.0,
    "frame_shift": 10.0,
    "dither": 0.0,
    "energy_floor": 0.0,
    "blackman_coeff": 0.42,
    "channel": -1,
    "high_freq": 0.0,
    "htk_compat": False,
    "low_freq": 20.0,
    "min_duration": 0.0,
    "preemphasis_coefficient": 0.97,
    "raw_energy": True,
    "remove_dc_offset": True,
    "round_to_power_of_two": True,
    "snip_edges": True,
    "subtract_mean": False,
    "use_energy": False,
    "use_log_fbank": True,
    "use_power": True,
    "vtln_high": -500.0,
    "vtln_low": 100.0,
    "vtln_warp": 1.0,
    "window_type": "povey",
}


def decode_audio(raw_bytes, config):
    """Decode a verified PCM WAV payload; convert only when explicitly allowed."""
    try:
        samples, rate = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    except (RuntimeError, ValueError, OSError) as exc:
        raise PreprocessError("Cannot decode PCM WAV") from exc
    if samples.size == 0 or not np.isfinite(samples).all():
        raise PreprocessError("Audio is empty or non-finite")
    channels = samples.shape[1]
    if channels != 1 and config["channel_policy"] != "mean":
        raise PreprocessError("Multichannel input requires channel_policy='mean'")
    if rate != 16000 and not config["resample"]:
        raise PreprocessError("Non-16k input requires resample=true")
    if np.any(samples >= 32767 / 32768) or np.any(samples <= -1):
        raise PreprocessError("Full-scale audio requires quality review before preprocessing")
    # Keeping the same float32 average and polyphase resampler preserves the
    # previously used conversion contract. No peak normalization is applied.
    waveform = samples.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, rate // divisor)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all() or not np.any(waveform):
        raise PreprocessError("Processed audio is non-finite or all zero")
    if len(waveform) < 400:
        raise PreprocessError("Audio is shorter than one 25 ms frame; no implicit padding")
    info = {"source_sample_rate": int(rate), "source_channels": int(channels),
            "processed_sample_rate": 16000, "processed_samples": len(waveform),
            "channel_policy": config["channel_policy"], "resampled": rate != 16000}
    return waveform, info


def extract_fbank(waveform):
    """Return raw, unnormalized float32 [frames, 40] Kaldi fbank features."""
    import torch
    import torchaudio

    data = np.asarray(waveform, dtype=np.float32)
    if data.ndim != 1 or len(data) < 400 or not np.isfinite(data).all():
        raise PreprocessError("fbank requires finite mono audio with at least 400 samples")
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0) * 32768.0
        features = torchaudio.compliance.kaldi.fbank(tensor, **FBANK)
    if features.ndim != 2 or features.shape[1] != 40 or not torch.isfinite(features).all():
        raise PreprocessError("Invalid fbank result")
    return features.numpy().astype(np.float32, copy=False)
