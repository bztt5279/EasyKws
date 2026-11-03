"""Shared fresh-frame trigger for calibration and continuous replay.

The episode state machine is adapted from the retained project's final local
implementation. Thresholds are always supplied by the caller; no historical
model threshold is a default. Diagnostics never participate in decisions.
"""
from __future__ import annotations

import math
from collections import deque
from numbers import Real


_DEFAULT_CONFIG = {
    "confirm_frames": 3, "warmup_frames": 50,
    "rearm_sec": 0.3, "cooldown_sec": 2.0, "comparison": "ge",
}


def trigger_config(config: dict | None = None) -> dict:
    """Resolve rules, excluding the separately calibrated threshold.

    Warmup uses the count of actual frames, including the current frame: with
    warmup_frames=50, frame index 49 is the first eligible frame. Confirmation
    can span warmup. Rearm requires a continuous below-threshold interval.
    """
    if config is not None and not isinstance(config, dict):
        raise ValueError("trigger config must be an object")
    supplied = {} if config is None else config
    if set(supplied) - set(_DEFAULT_CONFIG):
        raise ValueError("unknown trigger config field")
    result = {**_DEFAULT_CONFIG, **supplied}
    for name, minimum in (("confirm_frames", 1), ("warmup_frames", 0)):
        if type(result[name]) is not int or result[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in ("rearm_sec", "cooldown_sec"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0 or (name == "rearm_sec" and value == 0):
            raise ValueError("rearm must be positive and cooldown nonnegative")
        result[name] = float(value)
    if result["comparison"] not in ("ge", "gt"):
        raise ValueError("comparison must be 'ge' or 'gt'")
    return result


class FreshFrameTrigger:
    """One report per activation episode, using only newly scored frames.

    A reported episode stays reported through cooldown and a held-high signal.
    Only the configured low interval rearms it. An unreported episode that
    begins during cooldown may report when cooldown expires. History and recent
    events are bounded at 200 entries; event_count is the lifetime count.
    """

    def __init__(self, threshold: float, config: dict | None = None):
        if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and in [0, 1]")
        self.threshold = float(threshold)
        self.config = trigger_config(config)
        self.reset()

    def reset(self) -> None:
        """Begin a new session, including warmup, episode and event indices."""
        self.active = self.reported = False
        self.low_start = None
        self.last_time = self.last_report = -math.inf
        self.high_streak = self.frames_seen = self.event_count = 0
        self.history = deque(maxlen=200)
        self.events = deque(maxlen=200)

    def update(self, score: float, t: float) -> list[dict]:
        """Consume exactly one new frame at a strictly increasing audio time."""
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("frame score must be finite and in [0, 1]")
        if isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t) or t < 0 or t <= self.last_time:
            raise ValueError("frame time must be nonnegative and strictly increase")
        score, t = float(score), float(t)
        self.last_time = t
        self.frames_seen += 1
        self.history.append(score)
        high = score >= self.threshold if self.config["comparison"] == "ge" else score > self.threshold
        self.high_streak = self.high_streak + 1 if high else 0
        if not high:
            if self.low_start is None:
                self.low_start = t
            if t - self.low_start + 1e-9 >= self.config["rearm_sec"]:
                self.active = self.reported = False
            return []
        self.low_start = None
        if not self.active:
            self.active, self.reported = True, False
        eligible = (self.frames_seen >= self.config["warmup_frames"]
                    and self.high_streak >= self.config["confirm_frames"])
        if self.reported or not eligible or t + 1e-9 < self.last_report + self.config["cooldown_sec"]:
            return []
        self.reported, self.last_report = True, t
        event = {
            "index": self.event_count, "frame_index": self.frames_seen - 1,
            "t_audio_sec": t, "score": score, "threshold": self.threshold,
            "cooldown_until_sec": t + self.config["cooldown_sec"],
        }
        self.event_count += 1
        # Store an independent diagnostic copy so callers cannot alter history.
        self.events.append(dict(event))
        return [event]
