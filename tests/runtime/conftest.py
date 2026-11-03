"""Shared M6 fixtures: one small synthetic world with a real calibration and export."""
import hashlib
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import SAMPLES, waveform_proxy
from easykws.calibration import calibrate
from easykws.export import export_package
from easykws.training import runner


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _bundle(root):
    audio = root / "audio"
    audio.mkdir(parents=True)
    from easykws.prep import v2
    from easykws.preprocess import run as preprocess_run
    doc = {"schema": v2.SCHEMA, "datasets": [{"dataset_id": "demo", "source": "generated", "version": "v1",
            "license": "test-only", "split_policy": "speaker"}], "resources": [],
           "selection": {"kind": "full", "reason": "", "available_count": 6}, "rows": []}
    for i in range(6):
        samples = (2500 * np.sin(np.arange(3200 + 160 * i) * (.08 + i * .01))).astype("<i2")
        path = audio / f"{i}.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(samples.tobytes())
        doc["rows"].append({"sample_id": f"s{i}", "dataset_id": "demo", "speaker_id": f"p{i // 2}",
                            "source_recording_id": f"r{i}", "audio_path": path.name,
                            "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "label": 0 if i % 2 == 0 else -1, "group": "positive" if i % 2 == 0 else "negative",
                            "expected_phrase": "x" if i % 2 == 0 else "", "review_status": "pass",
                            "origin": "test-fixture", "purpose": "synthetic-test", "parent_sample_ids": [],
                            "resource_ids": [], "requested_split": ("train", "val", "test")[i // 2],
                            "generation": None})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    prepared = root / "prep"
    output = root / "features"
    v2.prepare(manifest, {"demo": audio}, prepared)
    preprocess_run(prepared, None, output, path_map={"demo": audio})
    return output


def _sessions(root):
    audio_root = root / "session-audio"
    audio_root.mkdir(parents=True)
    times = np.arange(SAMPLES, dtype=np.float64) / 16000
    waveform = np.zeros(SAMPLES, np.float32)
    waveform[16000:32000] = (.1 * np.sin(2 * np.pi * 733 * times[16000:32000])).astype(np.float32)
    sessions = []
    for name, audio, positives, negatives in (
            ("silence", np.zeros(SAMPLES, np.float32), [], [{"start_sec": 0., "end_sec": 3.}]),
            ("proxy", waveform, [{"id": "event-1", "start_sec": .9, "end_sec": 2.2}],
             [{"start_sec": 0., "end_sec": .9}, {"start_sec": 2.2, "end_sec": 3.}])):
        path = audio_root / (name + ".wav")
        sf.write(path, audio, 16000, subtype="PCM_16")
        sessions.append({"id": name, "path": path.name,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "duration_samples": SAMPLES, "recording_kind": "synthetic-concatenation",
                         "speaker_ids": [], "positive_windows": positives, "negative_intervals": negatives})
    manifest = {"schema": "easykws-calibration-sessions/v1", "purpose": "synthetic-test", "split": "calibration",
                "sample_rate": 16000, "channels": 1, "clock": "samples", "boundary_policy": "reset-each-session",
                "gap_policy": "split-session", "provenance": {"source_id": "m6-tests",
                "description": "generated silence and tone for interface tests"}, "sessions": sessions}
    path = audio_root / "sessions.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path, audio_root


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    """Bundle, two differently seeded runs, a feasible calibration and an exported package."""
    root = tmp_path_factory.mktemp("m6-world")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bundle = _bundle(root)
        config = {"model": {"hidden_dim": 4, "num_layers": 1, "kernel_size": 3},
                  "training": {"num_epochs": 1, "batch_size": 2},
                  "sampler": {"group_weights": {"positive": 1, "negative": 1}}}
        run = runner.run(bundle, config, root / "run")
        runner.run(bundle, dict(config, seed=42), root / "run-b")
        manifest, audio_root = _sessions(root)
        calibration_config = root / "calibration-config.json"
        calibration_config.write_text(json.dumps({"schema": "easykws-calibration-config/v1",
            "candidates": [run["candidates"][0]["path"]], "threshold_grid": [.1, .5], "min_recall": 0,
            "max_fah": None, "trigger": {}, "chunk_samples": 512}), encoding="utf-8")
        report = calibrate(root / "run", bundle, manifest, root / "calibration", config=str(calibration_config))
        assert report["feasible"] and report["selected_checkpoint"], "fixture calibration must be feasible"
        summary = export_package(root / "run", bundle, root / "calibration", manifest, root / "package")
    finally:
        torch.set_num_threads(previous_threads)
    return SimpleNamespace(root=root, bundle=bundle, run=root / "run", run_b=root / "run-b",
                           manifest=manifest, audio_root=audio_root, calibration=root / "calibration",
                           package=root / "package", summary=summary, selected=report["selected_checkpoint"],
                           threshold=report["selected_threshold"], proxy_samples=waveform_proxy())


@pytest.fixture
def package_copy(world, tmp_path):
    """A writable copy of the verified package for negative tests."""
    target = tmp_path / "package"
    target.mkdir()
    for item in world.package.iterdir():
        (target / item.name).write_bytes(item.read_bytes())
    return target
