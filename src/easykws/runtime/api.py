"""Public streaming scoring object over one loaded deployment package.

The scorer and trigger are the single shared implementations; nothing here
re-approximates frame handling or episode rules. Sessions are explicit: a
new session requires reset(), gets a new identifier and restarts its clock.
"""
from collections import deque

import torch

from ..streaming import StreamingScorer
from ..trigger import FreshFrameTrigger

_RECENT_FRAMES = 200


class DeploymentError(ValueError):
    """Raised when approved execution conditions or package rules are violated."""


class DeploymentModel:
    """push/reset/flush scoring bound to one verified package.

    Every push re-checks the mutable execution settings that would silently
    change numbers (thread count, deterministic algorithms, autocast, model
    training mode). Returned events carry the session identifier so an
    interrupted stream can never masquerade as one continuous observation.
    """

    def __init__(self, model, trigger, execution, *, purpose, data_flags, is_test_model,
                 parameter_count, state_sha256, allow_test_model, apply_execution_profile):
        self._execution = dict(execution)
        self._threshold = float(trigger["threshold"])
        self._rules = dict(trigger["rules"])
        self.purpose = purpose
        self.data_flags = list(data_flags)
        self.is_test_model = bool(is_test_model)
        self.parameter_count = int(parameter_count)
        self.state_sha256 = state_sha256
        self._allow_test_model = allow_test_model
        self._apply_execution_profile = apply_execution_profile
        self._scorer = StreamingScorer(model)
        self._trigger = FreshFrameTrigger(self._threshold, self._rules)
        self._recent = deque(maxlen=_RECENT_FRAMES)
        self._session = 0
        self._closed = False

    @property
    def threshold(self):
        return self._threshold

    @property
    def rules(self):
        return dict(self._rules)

    @property
    def execution(self):
        return dict(self._execution)

    @property
    def session_id(self):
        return self._session

    @property
    def session_samples(self):
        return self._scorer.sample_count

    @property
    def session_sec(self):
        return self._scorer.audio_sec

    @property
    def frames_processed(self):
        return self._scorer.frames_processed

    @property
    def recent_frames(self):
        return [dict(row) for row in self._recent]

    @property
    def event_count(self):
        return self._trigger.event_count

    def summary(self):
        """Load-time facts for display; never a quality or provenance claim."""
        return {"purpose": self.purpose, "data_flags": self.data_flags, "test_model": self.is_test_model,
                "test_model_allowed": self.is_test_model and self._allow_test_model,
                "threshold": self._threshold, "rules": dict(self._rules),
                "parameter_count": self.parameter_count, "state_sha256": self.state_sha256,
                "execution": dict(self._execution),
                "execution_profile_applied": self._apply_execution_profile,
                "sample_rate": 16000, "channels": 1, "dtype": "float32"}

    def _guard_execution(self):
        """Reject any drift from the approved mutable execution settings.

        Runs before the scorer consumes any input, so a rejected block leaves
        the session state untouched. Covers every mutable field recorded in
        the binding's execution contract; deleting fields or bypassing this
        check is not part of the contract.
        """
        approved = self._execution
        model = self._scorer.model
        if model.training or any(module.training for module in model.modules()):
            raise DeploymentError("deployment model or a submodule left evaluation mode; outputs are no longer "
                                  "inference under the approved configuration")
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise DeploymentError("autocast became active after loading; outputs are no longer "
                                  "inference under the approved configuration")
        actual = {"torch_threads": torch.get_num_threads(),
                  "torch_interop_threads": torch.get_num_interop_threads(),
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                  "float32_matmul_precision": torch.get_float32_matmul_precision(),
                  "mkldnn_enabled": torch.backends.mkldnn.enabled,
                  "default_dtype": str(torch.get_default_dtype()),
                  "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                  "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
        drift = {key: (approved[key], value) for key, value in actual.items()
                 if key in approved and approved[key] != value}
        if drift:
            details = ", ".join(f"{key}: approved {want!r}, actual {got!r}"
                                for key, (want, got) in sorted(drift.items()))
            raise DeploymentError("execution conditions drifted from the approved profile (" + details + "); "
                                  "outputs are no longer inference under the approved configuration")

    def push(self, samples):
        """Score one block of new PCM; return this call's frames and events.

        Input validation errors leave the session state unchanged and raise
        ValueError; execution drift raises DeploymentError. Neither consumes
        or hides the offending data.
        """
        self._guard_execution()
        rows = self._scorer.push(samples)
        events = []
        for row in rows:
            for event in self._trigger.update(row["score"], row["t_audio_sec"]):
                events.append(dict(event, session_id=self._session))
        for row in rows:
            self._recent.append(dict(row, session_id=self._session))
        return {"session_id": self._session, "frames": rows, "events": events,
                "session_samples": self._scorer.sample_count, "session_sec": self._scorer.audio_sec}

    def reset(self):
        """Start a new distinct session with a new identifier and clock."""
        self._scorer.reset()
        self._trigger.reset()
        self._recent.clear()
        self._session += 1
        self._closed = False
        return {"session_id": self._session, "samples": 0, "frames": 0}

    def flush(self):
        """Close the current session; push stays invalid until reset()."""
        self._scorer.flush()
        self._closed = True
        return {"session_id": self._session, "samples": self._scorer.sample_count,
                "sec": self._scorer.audio_sec, "frames": self._scorer.frames_processed}
