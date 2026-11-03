"""Loaded-package scoring: numbers, events, sessions, execution drift, purpose gate."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from easykws.runtime import DeploymentError, load_model
from easykws.trigger import FreshFrameTrigger

from helpers import rewrite_manifest_hashes, rewrite_ready

SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def model(world):
    return load_model(world.package, allow_test_model=True)


def push_all(runtime, audio, chunk):
    frames, events = [], []
    for start in range(0, len(audio), chunk):
        result = runtime.push(np.ascontiguousarray(audio[start:start + chunk]))
        frames.extend(result["frames"])
        events.extend(result["events"])
    return frames, events


def test_package_replay_matches_calibration_scores_and_events(world, model):
    saved = json.loads(sorted(world.calibration.glob("scores_*.json"))[0].read_text(encoding="utf-8"))
    audio, rate = sf.read(world.audio_root / "proxy.wav", dtype="float32")
    assert rate == 16000
    frames, events = push_all(model, audio, 512)
    model.flush()
    assert frames == saved["sessions"]["proxy"]
    report = json.loads((world.calibration / "report.json").read_text(encoding="utf-8"))
    record = [r for r in report["records"] if r["threshold"] == world.threshold][0]
    triggers = FreshFrameTrigger(world.threshold, model.rules)
    replayed = [event for frame in frames for event in triggers.update(frame["score"], frame["t_audio_sec"])]
    assert [e["t_audio_sec"] for e in replayed] == [e["t_audio_sec"] for e in events]
    total_events = (record["metrics"]["positive_hits"] + record["metrics"]["negative_events"]
                    + record["metrics"]["excluded_events"] + record["metrics"]["duplicate_positive_events"])
    assert len(events) == total_events


@pytest.mark.parametrize("chunk", [160, 512, 4096])
def test_chunk_lengths_give_identical_frames_and_events(world, model, chunk):
    audio = world.proxy_samples
    reference = push_all(load_model(world.package, allow_test_model=True), audio, 160)
    frames, events = push_all(model, audio, chunk)
    assert frames == reference[0] and events == reference[1]
    expected_frames = 1 + (len(audio) - 400) // 160
    assert len(frames) == expected_frames
    assert frames[-1]["t_audio_sec"] == ((expected_frames - 1) * 160 + 400) / 16000.0
    total = model.flush()
    assert total["samples"] == len(audio)
    assert abs(total["sec"] - len(audio) / 16000) < 1e-12


def test_silence_signal_silence_block_accounting(world, model):
    audio = world.proxy_samples
    quiet = np.zeros(16000, np.float32)
    block = np.concatenate((quiet, audio, quiet))
    result = model.push(block)
    total = model.flush()
    assert result["session_samples"] == len(block)
    assert model.session_id == 0
    assert result["frames"][0]["t_audio_sec"] == .025
    assert total["samples"] == len(block)
    assert len(result["frames"]) == 1 + (len(block) - 400) // 160


def test_empty_blocks_do_not_advance_and_zero_blocks_do(world, model):
    before = (model.session_samples, model.frames_processed)
    for _ in range(3):
        assert model.push(np.empty(0, np.float32))["events"] == []
    assert (model.session_samples, model.frames_processed) == before
    result = model.push(np.zeros(1600, np.float32))
    assert result["session_samples"] == 1600
    assert abs(result["session_sec"] - 0.1) < 1e-12


def test_invalid_input_leaves_state_unchanged(world, model):
    model.push(np.zeros(512, np.float32))
    before = (model.session_samples, model.frames_processed, model.recent_frames)
    for invalid in (np.zeros(3, np.float64), np.array([np.nan], np.float32),
                    np.array([2.0], np.float32), np.zeros((2, 2), np.float32)):
        with pytest.raises(ValueError):
            model.push(invalid)
        assert (model.session_samples, model.frames_processed, model.recent_frames) == before
    model.push(np.zeros(512, np.float32))
    assert model.session_samples == 1024


def test_flush_reset_session_identifiers(world, model):
    model.push(np.zeros(1600, np.float32))
    closed = model.flush()
    assert closed["session_id"] == 0 and closed["samples"] == 1600
    with pytest.raises(ValueError, match="flushed"):
        model.push(np.zeros(1600, np.float32))
    fresh = model.reset()
    assert fresh["session_id"] == 1
    assert model.session_samples == 0 and model.frames_processed == 0
    result = model.push(np.zeros(1600, np.float32))
    assert result["session_id"] == 1 and result["frames"][0]["frame_index"] == 0


def test_execution_drift_during_push_is_rejected(world, model):
    model.push(np.zeros(1600, np.float32))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(DeploymentError, match="autocast"):
            model.push(np.zeros(1600, np.float32))
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(max(2, previous))
        with pytest.raises(DeploymentError, match="thread"):
            model.push(np.zeros(1600, np.float32))
    finally:
        torch.set_num_threads(previous)
    # Simulate external code flipping training mode on the loaded model.
    model._scorer.model.train()
    try:
        with pytest.raises(DeploymentError, match="evaluation mode"):
            model.push(np.zeros(1600, np.float32))
    finally:
        model._scorer.model.eval()
    model.push(np.zeros(1600, np.float32))


def test_load_rejects_thread_mismatch_and_apply_profile(world):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(max(2, previous))
        with pytest.raises(DeploymentError, match="thread"):
            load_model(world.package, allow_test_model=True)
        loaded = load_model(world.package, allow_test_model=True, apply_execution_profile=True)
        assert torch.get_num_threads() == loaded.execution["torch_threads"]
        assert loaded.push(np.zeros(1600, np.float32))["frames"]
    finally:
        torch.set_num_threads(previous)


def test_default_runtime_rejects_test_package_and_display_persists(world):
    with pytest.raises(DeploymentError, match="test"):
        load_model(world.package)
    loaded = load_model(world.package, allow_test_model=True)
    summary = loaded.summary()
    assert summary["test_model"] is True and summary["test_model_allowed"] is True
    assert summary["purpose"] == "synthetic-test" and "synthetic-test" in summary["data_flags"]


def test_purpose_cannot_be_washed_to_normal(world, package_copy):
    manifest = json.loads((package_copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["purpose"] = "recorded-data"
    manifest["data_flags"] = []
    manifest["test_model"] = False
    (package_copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rewrite_manifest_hashes(package_copy)
    ready = json.loads((package_copy / "READY.json").read_text(encoding="utf-8"))
    ready["purpose"] = "recorded-data"
    ready["data_flags"] = []
    ready["test_model"] = False
    (package_copy / "READY.json").write_text(json.dumps(ready), encoding="utf-8")
    rewrite_ready(package_copy)
    with pytest.raises(DeploymentError, match="purpose or flags"):
        load_model(package_copy, allow_test_model=True)


def test_missing_purpose_fields_are_rejected(world, package_copy):
    manifest = json.loads((package_copy / "manifest.json").read_text(encoding="utf-8"))
    del manifest["purpose"]
    (package_copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rewrite_manifest_hashes(package_copy)
    rewrite_ready(package_copy)
    with pytest.raises(DeploymentError, match="purpose"):
        load_model(package_copy, allow_test_model=True)


def test_allow_test_model_does_not_bypass_identity(world, package_copy):
    manifest = json.loads((package_copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["threshold"]["threshold"] = 0.123
    (package_copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rewrite_manifest_hashes(package_copy)
    rewrite_ready(package_copy)
    with pytest.raises(DeploymentError, match="threshold"):
        load_model(package_copy, allow_test_model=True)


def test_package_rejects_cmvn_payload_mismatch_with_fresh_hashes(world, package_copy):
    """P1: only the CMVN payload changes; hashes are recomputed; binding untouched."""
    cmvn = json.loads((package_copy / "cmvn.json").read_text(encoding="utf-8"))
    cmvn["mean"][0] += 10.0
    (package_copy / "cmvn.json").write_text(json.dumps(cmvn, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rewrite_manifest_hashes(package_copy)
    rewrite_ready(package_copy)
    with pytest.raises(DeploymentError, match="CMVN"):
        load_model(package_copy, allow_test_model=True)


def test_package_rejects_cmvn_semantic_fields_mismatch(world, package_copy):
    cmvn = json.loads((package_copy / "cmvn.json").read_text(encoding="utf-8"))
    cmvn["frame_num"] += 1
    (package_copy / "cmvn.json").write_text(json.dumps(cmvn, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rewrite_manifest_hashes(package_copy)
    rewrite_ready(package_copy)
    with pytest.raises(DeploymentError, match="CMVN"):
        load_model(package_copy, allow_test_model=True)


def test_push_rejects_execution_drift_before_consuming(world, model):
    """P1: mutable approved settings must be re-checked before audio is consumed."""
    model.push(np.zeros(1600, np.float32))
    baseline = (model.session_samples, model.frames_processed, model.recent_frames)
    mkldnn = torch.backends.mkldnn.enabled
    try:
        torch.backends.mkldnn.enabled = not mkldnn
        with pytest.raises(DeploymentError, match="mkldnn"):
            model.push(np.zeros(1600, np.float32))
    finally:
        torch.backends.mkldnn.enabled = mkldnn
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(DeploymentError, match="default_dtype"):
            model.push(np.zeros(1600, np.float32))
    finally:
        torch.set_default_dtype(torch.float32)
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        with pytest.raises(DeploymentError, match="float32_matmul_precision"):
            model.push(np.zeros(1600, np.float32))
    finally:
        torch.set_float32_matmul_precision(precision)
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        with pytest.raises(DeploymentError, match="deterministic_warn_only"):
            model.push(np.zeros(1600, np.float32))
    finally:
        torch.use_deterministic_algorithms(False)
    submodule = model._scorer.model.backbone.network[0]
    try:
        submodule.train()
        with pytest.raises(DeploymentError, match="evaluation mode"):
            model.push(np.zeros(1600, np.float32))
    finally:
        submodule.eval()
    assert (model.session_samples, model.frames_processed, model.recent_frames) == baseline
    model.push(np.zeros(1600, np.float32))


def test_deployment_purpose_and_flag_helpers_cover_diagnostic():
    from easykws.calibration.binding import deployment_is_test, deployment_purpose
    assert deployment_purpose("training-data", "diagnostic-subset") == "diagnostic-subset"
    assert deployment_purpose("diagnostic-subset", "recorded-data") == "diagnostic-subset"
    assert deployment_purpose("synthetic-test", "diagnostic-subset") == "synthetic-test"
    assert deployment_is_test("training-data", ["diagnostic-subset"]) is True
    assert deployment_is_test("diagnostic-subset", []) is True
    assert deployment_is_test("recorded-data", ["unrelated-flag"]) is False


def test_cli_subprocess_check_and_replay(world, tmp_path):
    """Fresh subprocess --check, then a fresh-process push replay of the package."""
    env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-B", "-c",
               "import json,sys; from easykws.runtime import load_model; "
               "m = load_model(sys.argv[1], allow_test_model=True); print(json.dumps(m.summary()))",
               str(world.package)]
    done = subprocess.run(command, capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    summary = json.loads(done.stdout.strip().splitlines()[-1])
    assert summary["test_model"] is True and summary["threshold"] == world.threshold
    replay = [sys.executable, "-B", "-c",
              "import json,sys; import numpy as np, soundfile as sf; "
              "from easykws.runtime import load_model; "
              "m = load_model(sys.argv[1], allow_test_model=True); "
              "audio, _ = sf.read(sys.argv[2], dtype='float32'); rows = []; "
              "rows = [f for start in range(0, len(audio), 512) "
              "for f in m.push(np.ascontiguousarray(audio[start:start+512]))['frames']]; "
              "print(json.dumps({'frames': len(rows), 'first': rows[0], 'last': rows[-1]}))",
              str(world.package), str(world.audio_root / "proxy.wav")]
    done = subprocess.run(replay, capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["frames"] == 298 and result["first"]["frame_index"] == 0
