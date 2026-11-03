"""Smoke replay verification: sessions, events, clocks and failure paths."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

from easykws import smoke
from easykws.trigger import FreshFrameTrigger, trigger_config as resolve_trigger_config

SESSIONS = ("silence", "proxy", "mixed")
THRESHOLD = 0.5
RULES = resolve_trigger_config({"confirm_frames": 1, "warmup_frames": 0,
                                "rearm_sec": 0.005, "cooldown_sec": 0})


def _rows(scores):
    return [{"frame_index": index, "t_audio_sec": (index * 160 + 400) / 16000.0, "score": score}
            for index, score in enumerate(scores)]


def _events(rows, session_id=0):
    trigger = FreshFrameTrigger(THRESHOLD, RULES)
    return [dict(event, session_id=session_id) for row in rows
            for event in trigger.update(row["score"], row["t_audio_sec"])]


def _build_root(root):
    package = root / "package"
    package.mkdir(parents=True)
    payloads = {}
    for name in ("model.pt", "cmvn.json", "threshold.json", "calibration.json", "manifest.json"):
        (package / name).write_bytes(b"payload:" + name.encode())
        payloads[name] = hashlib.sha256((package / name).read_bytes()).hexdigest()
    ready = {"schema": "easykws-deployment-package/v1", "files": payloads,
             "purpose": "synthetic-test", "data_flags": ["synthetic-test"], "test_model": True}
    (package / "READY.json").write_text(json.dumps(ready), encoding="utf-8")
    calibration = root / "calibration"
    calibration.mkdir()
    # Two trigger episodes per session: high, low long enough to rearm, high.
    saved = {name: _rows([0.9, 0.1, 0.1, 0.9]) for name in SESSIONS}
    (calibration / "scores_0000.json").write_text(
        json.dumps({"candidate": "c1.pt", "sessions": saved}), encoding="utf-8")
    selection = {"candidate": "c1.pt", "threshold": THRESHOLD,
                 "binding": {"trigger_config": {"threshold": THRESHOLD, "rules": RULES}}}
    (calibration / "selected_checkpoint.json").write_text(json.dumps(selection), encoding="utf-8")
    audio = root / "session-audio"
    audio.mkdir()
    for name in SESSIONS:
        (audio / (name + ".wav")).write_bytes(b"RIFF-fake")
    return root, saved


def _replay_payload(saved, *, sessions=SESSIONS, drop_event=False, clock_error=False):
    payload = {}
    for name in sessions:
        rows = saved[name]
        events = _events(rows)
        if drop_event and name == "mixed":
            events = events[:-1]
        last_sec = rows[-1]["t_audio_sec"] if rows else 0.0
        if clock_error and name == "mixed":
            last_sec = last_sec + 0.01
        payload[name] = {"frames": rows, "events": events, "samples": smoke.SESSION_SAMPLES,
                         "last_sec": last_sec}
    return payload


def test_verify_package_accepts_exact_replay(tmp_path):
    root, saved = _build_root(tmp_path)
    result = smoke.verify_package(root, replay_results=_replay_payload(saved))
    assert result["ready_files_verified"] == 5
    assert set(result["fresh_process_frames"].values()) == {4}
    assert result["events"] == {"silence": 2, "proxy": 2, "mixed": 2}
    assert result["clocks_sec"]["mixed"] == (3 * 160 + 400) / 16000.0


def test_verify_package_rejects_missing_event(tmp_path):
    """Frames intact but one event dropped: the smoke gate must fail."""
    root, saved = _build_root(tmp_path)
    payload = _replay_payload(saved, drop_event=True)
    with pytest.raises(AssertionError, match="replay events differ"):
        smoke.verify_package(root, replay_results=payload)


def test_verify_package_rejects_wrong_clock(tmp_path):
    root, saved = _build_root(tmp_path)
    payload = _replay_payload(saved, clock_error=True)
    with pytest.raises(AssertionError, match="replay clock wrong"):
        smoke.verify_package(root, replay_results=payload)


def test_verify_package_rejects_incomplete_session_set(tmp_path):
    root, saved = _build_root(tmp_path)
    payload = _replay_payload(saved, sessions=("silence", "proxy"))
    with pytest.raises(AssertionError, match="session set differs"):
        smoke.verify_package(root, replay_results=payload)


def test_verify_package_rejects_frame_mismatch(tmp_path):
    root, saved = _build_root(tmp_path)
    payload = _replay_payload(saved)
    payload["mixed"]["frames"] = _rows([0.1, 0.9, 0.9])
    with pytest.raises(AssertionError, match="replay frames differ"):
        smoke.verify_package(root, replay_results=payload)
