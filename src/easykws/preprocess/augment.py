"""Explicit offline waveform operations, with no implicit crop or normalization.

Resources are already verified, decoded arrays supplied by the caller. Their
logical IDs, rather than local file paths, are retained in the operation trace.
The caller owns scheduling, resource provenance and feature extraction.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

from .errors import PreprocessError

IMPLEMENTATION = "easykws-offline-augment/v1"


def stable_seed(global_seed: int, source_id: str, variant_id: int | str) -> int:
    """Return a seed independent of manifest order and process hash state."""
    if type(global_seed) is not int or not isinstance(source_id, str) or not source_id:
        raise PreprocessError("Seed must be an integer and source_id a nonempty string")
    if type(variant_id) not in (int, str):
        raise PreprocessError("variant_id must be an integer or string")
    payload = json.dumps([global_seed, source_id, variant_id], ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def rms(x: np.ndarray) -> float:
    """Float64 RMS used for SNR mixing and reverb level matching."""
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _number(op, name, *, positive=False, unit=False):
    value = op.get(name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise PreprocessError(f"Operation {name} must be a finite number")
    if not math.isfinite(value) or (positive and value <= 0) or (unit and not 0 <= value <= 1):
        raise PreprocessError(f"Operation {name} is outside its valid range")
    return value


def prepare_resource(samples: np.ndarray, source_rate: int, *, channel: int | None = None,
                     sample_rate: int = 16000) -> tuple[np.ndarray, dict]:
    """Average channels for noise or select a RIR channel, then resample.

    Channel selection precedes resampling. Output is float32, matching decoding
    an audio resource as float32 before the same conversion operations.
    """
    if type(source_rate) is not int or source_rate <= 0 or sample_rate != 16000:
        raise PreprocessError("Resource rate must be positive; output rate must be 16000")
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or not data.shape[0] or not data.shape[1] or not np.isfinite(data).all():
        raise PreprocessError("Resource must contain finite, nonempty audio")
    if channel is not None and (type(channel) is not int or not 0 <= channel < data.shape[1]):
        raise PreprocessError("Resource channel is out of range")
    output = data.mean(axis=1) if channel is None else data[:, channel]
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        output = resample_poly(output, sample_rate // divisor, source_rate // divisor)
    if not np.isfinite(output).all() or not np.any(output):
        raise PreprocessError("Converted resource is non-finite or all zero")
    return np.asarray(output, dtype=np.float32), {
        "source_sample_rate": source_rate, "source_samples": len(data),
        "source_channels": data.shape[1], "channel": channel,
        "channel_policy": "mean" if channel is None else "select",
        "processed_sample_rate": sample_rate, "processed_samples": len(output),
        "resampled": source_rate != sample_rate,
    }


def transform(x: np.ndarray, ops: Sequence[Mapping], peak_limit: float = 0.98, *,
              resources: Mapping[str, tuple[np.ndarray, int]] | None = None,
              sample_rate: int = 16000) -> tuple[np.ndarray, dict]:
    """Apply ordered operations and return float32 audio plus an exact trace.

    ``gain`` and ``speed`` take ``value``; ``filter`` takes ``low``/``high`` Hz;
    ``noise`` takes ``resource_id``, ``snr_db`` and ``position`` in [0,1]; ``rir``
    takes ``resource_id`` and ``channel``. Explicit ``crop`` takes ``length`` and
    ``only_if_longer_than`` in seconds plus ``position`` in [0,1]. Noise positions
    refer to the repeated resource, if repetition was needed. RIR removes the
    main-peak delay and keeps all remaining convolution samples. RMS matching
    uses only the first input-length samples of that delay-adjusted convolution.

    An empty operation list is a full-length clean copy, without peak limiting.
    Any nonempty list applies peak protection once, after the complete chain.
    """
    if sample_rate != 16000:
        raise PreprocessError("Offline augmentation requires 16000 Hz audio")
    if isinstance(peak_limit, bool) or not isinstance(peak_limit, (int, float)) or not 0 < peak_limit <= 1:
        raise PreprocessError("peak_limit must be in (0,1]")
    y = np.asarray(x, dtype=np.float32).copy()
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or not np.any(y):
        raise PreprocessError("Input must be finite, nonempty, nonzero mono audio")
    if not isinstance(ops, Sequence) or isinstance(ops, (str, bytes)):
        raise PreprocessError("Operations must be an ordered sequence")
    trace = []
    for op in ops:
        if not isinstance(op, Mapping):
            raise PreprocessError("Each operation must be an object")
        kind = op.get("kind")
        # Retain declared scalar parameters (including resource content hashes),
        # but never accept filesystem locations as operation identities.
        if "path" in op:
            raise PreprocessError("Operations must use logical resource_id, not path")
        step = dict(op)
        step["input_samples"] = len(y)
        if kind == "crop":
            length = _number(op, "length", positive=True)
            threshold = _number(op, "only_if_longer_than", positive=True)
            position = _number(op, "position", unit=True)
            original_length = len(y)
            n, start = original_length, 0
            triggered = original_length > round(threshold * sample_rate)
            if triggered:
                n = min(original_length, round(length * sample_rate))
                if n < 1:
                    raise PreprocessError("Crop length must retain at least one sample")
                start = round(position * (original_length - n))
                y = y[start:start + n]
            step.update(applied=triggered, source_sample_rate=sample_rate,
                        original_samples=original_length, start_sample=start, end_sample=start + n,
                        original_duration_sec=original_length / sample_rate,
                        retained_duration_sec=n / sample_rate,
                        discarded_duration_sec=(original_length - n) / sample_rate)
        elif kind == "gain":
            y = y * _number(op, "value", positive=True)
        elif kind == "speed":
            factor = _number(op, "value", positive=True)
            rate = round(sample_rate * factor)
            if rate < 1:
                raise PreprocessError("Speed factor rounds to an invalid sample rate")
            divisor = math.gcd(rate, sample_rate)
            up, down = sample_rate // divisor, rate // divisor
            y = resample_poly(y, up, down)
            step.update(effective_rate=rate, resample_up=up, resample_down=down)
        elif kind == "filter":
            low, high = _number(op, "low", positive=True), _number(op, "high", positive=True)
            if not low < high < sample_rate / 2:
                raise PreprocessError("Filter requires 0 < low < high < Nyquist")
            y = sosfilt(butter(4, [low, high], fs=sample_rate, btype="bandpass", output="sos"), y)
            step.update(filter_type="butterworth_bandpass", order=4, implementation="sosfilt")
        elif kind in ("noise", "rir"):
            resource_id = op.get("resource_id")
            if not isinstance(resource_id, str) or not resource_id or resources is None or resource_id not in resources:
                raise PreprocessError("Operation references an unavailable resource_id")
            channel = op.get("channel") if kind == "rir" else None
            if kind == "rir" and type(channel) is not int:
                raise PreprocessError("RIR channel must be explicit")
            resource = resources[resource_id]
            if not isinstance(resource, tuple) or len(resource) != 2:
                raise PreprocessError("Resource must be a (samples, sample_rate) tuple")
            signal, resource_info = prepare_resource(resource[0], resource[1], channel=channel,
                                                     sample_rate=sample_rate)
            step["resource"] = resource_info
            if kind == "noise":
                snr = _number(op, "snr_db")
                position = _number(op, "position", unit=True)
                repeats = max(1, math.ceil(len(y) / len(signal)))
                if repeats > 1:
                    signal = np.tile(signal, repeats)
                start = round(position * (len(signal) - len(y)))
                noise = signal[start:start + len(y)]
                noise_gain = (rms(y) / max(rms(noise), 1e-12)) * 10 ** (-snr / 20)
                # Preserve multiplication order, including float32 rounding.
                y = y + noise * (rms(y) / max(rms(noise), 1e-12)) * 10 ** (-snr / 20)
                step.update(repeats=repeats, repeated_samples=len(signal), start_sample=start,
                            end_sample=start + len(noise), noise_gain=noise_gain)
            else:
                peak = int(np.argmax(np.abs(signal)))
                original_rms = rms(y)
                conv = fftconvolve(y, signal)[peak:]
                reference_rms = rms(conv[:len(y)])
                gain = original_rms / max(reference_rms, 1e-12)
                y = conv * gain
                step.update(peak_sample=peak, removed_delay_samples=peak,
                            impulse_samples=len(signal), input_rms=original_rms,
                            normalization_rms=reference_rms, rms_gain=gain,
                            full_convolution_samples=step["input_samples"] + len(signal) - 1)
        else:
            raise PreprocessError(f"Unknown augmentation operation: {kind}")
        step["output_samples"] = len(y)
        trace.append(step)
    if not len(y) or not np.isfinite(y).all() or rms(y) == 0:
        raise PreprocessError("Augmentation result is empty, non-finite or all zero")
    peak = float(np.max(np.abs(y)))
    gain = min(1.0, peak_limit / peak) if ops else 1.0
    return np.asarray(y * gain, dtype=np.float32), {
        "implementation": IMPLEMENTATION, "sample_rate": sample_rate,
        "input_samples": len(x), "samples": len(y), "prelimit_peak": peak,
        "safety_gain": gain, "operations": trace,
    }
