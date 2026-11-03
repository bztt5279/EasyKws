"""Diagnostic-subset deployment chain over the formal prep/preprocess/train path."""
import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocess"))

from test_augmentation_pipeline import bundle as pipeline_bundle  # noqa: F401  (fixture reuse)

from easykws.calibration import calibrate
from easykws.export import export_package
from easykws.preprocess import run as preprocess_run
from easykws.runtime import DeploymentError, load_model
from easykws.training import runner


@pytest.fixture(scope="module")
def diagnostic_world(tmp_path_factory):
    """A real diagnostic-subset feature bundle, trained model, calibration and export."""
    root = tmp_path_factory.mktemp("diagnostic-world")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        document, audio_root_source, prepare = pipeline_bundle.__wrapped__(root)
        document["selection"] = {"kind": "diagnostic-subset", "reason": "generated regression fixture",
                                 "available_count": 6}
        prepared, roots = prepare(name="diagnostic-prepared")
        features = root / "features"
        preprocess_run(prepared, None, features, path_map=roots)
        status = runner.run(features, {"model": {"hidden_dim": 4, "num_layers": 1, "kernel_size": 3},
                                       "training": {"num_epochs": 1, "batch_size": 2},
                                       "sampler": {"group_weights": {"examples": 1}}}, root / "run")
        candidate = status["candidates"][0]["path"]
        # One generated tone replayed offline; never a concatenation passed off as
        # a continuous recording, and no train/test audio or speaker is reused.
        n = 48000
        times = np.arange(n, dtype=np.float64) / 16000
        audio = (.1 * np.sin(2 * np.pi * 733 * times)).astype(np.float32)
        audio_root = root / "session-audio"
        audio_root.mkdir()
        path = audio_root / "diagnostic.wav"
        sf.write(path, audio, 16000, subtype="PCM_16")
        session = {"id": "diagnostic", "path": path.name,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "duration_samples": n,
                   "recording_kind": "offline-replay", "speaker_ids": [],
                   "positive_windows": [{"id": "event-1", "start_sec": .9, "end_sec": 2.2}],
                   "negative_intervals": [{"start_sec": 0., "end_sec": .9}, {"start_sec": 2.2, "end_sec": 3.}]}
        manifest = {"schema": "easykws-calibration-sessions/v1", "purpose": "diagnostic-subset",
                    "split": "calibration", "sample_rate": 16000, "channels": 1, "clock": "samples",
                    "boundary_policy": "reset-each-session", "gap_policy": "split-session",
                    "provenance": {"source_id": "diagnostic-fixtures",
                                   "description": "generated tone for interface regression only"},
                    "sessions": [session]}
        manifest_path = audio_root / "sessions.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        config_path = root / "calibration-config.json"
        config_path.write_text(json.dumps({"schema": "easykws-calibration-config/v1", "candidates": [candidate],
            "threshold_grid": [.1, .5], "min_recall": 0, "max_fah": None, "trigger": {}, "chunk_samples": 512}),
            encoding="utf-8")
        report = calibrate(root / "run", features, manifest_path, root / "calibration", config=str(config_path))
        assert report["feasible"] and report["selected_checkpoint"], "diagnostic fixture calibration must be feasible"
        assert report["purpose"] == "diagnostic-subset"
        summary = export_package(root / "run", features, root / "calibration", manifest_path, root / "package")
    finally:
        torch.set_num_threads(previous_threads)
    return SimpleNamespace(root=root, package=root / "package", summary=summary, manifest=manifest_path,
                           calibration=root / "calibration", threshold=report["selected_threshold"])


def test_diagnostic_bundle_purpose_and_flags(diagnostic_world):
    assert diagnostic_world.summary["purpose"] == "diagnostic-subset"
    assert "diagnostic-subset" in diagnostic_world.summary["data_flags"]
    assert diagnostic_world.summary["test_model"] is True


def test_diagnostic_package_default_rejected_and_allow_test_displays(diagnostic_world):
    with pytest.raises(DeploymentError, match="diagnostic"):
        load_model(diagnostic_world.package)
    loaded = load_model(diagnostic_world.package, allow_test_model=True)
    summary = loaded.summary()
    assert summary["test_model"] is True and summary["test_model_allowed"] is True
    assert summary["purpose"] == "diagnostic-subset" and "diagnostic-subset" in summary["data_flags"]
    assert summary["threshold"] == diagnostic_world.threshold
    assert loaded.push(np.zeros(1600, np.float32))["frames"]


def test_allow_test_does_not_bypass_identity_checks(diagnostic_world, tmp_path):
    import hashlib
    import torch
    target = tmp_path / "package"
    target.mkdir()
    for item in diagnostic_world.package.iterdir():
        (target / item.name).write_bytes(item.read_bytes())
    saved = torch.load(target / "model.pt", weights_only=True)
    saved["state_dict"]["classifier.weight"][0, 0] += 0.5
    torch.save(saved, target / "model.pt")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    payloads = {name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in manifest["payloads"]}
    manifest["payloads"] = payloads
    manifest["model"]["file_sha256"] = payloads["model.pt"]
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ready = json.loads((target / "READY.json").read_text(encoding="utf-8"))
    ready["files"] = {name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in ready["files"]}
    (target / "READY.json").write_text(json.dumps(ready), encoding="utf-8")
    with pytest.raises(DeploymentError, match="inference content"):
        load_model(target, allow_test_model=True)
