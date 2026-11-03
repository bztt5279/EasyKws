"""Behavioral checks on small synthetic signals; no recognition claims."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
import pytest

from easykws.prep import prepare
from easykws.preprocess import PreprocessError, plan, run


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_wave(path, samples, rate=16000, channels=1):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(np.asarray(samples, dtype="<i2").tobytes())


@pytest.fixture
def prepared(dataset, tmp_path):
    manifest, audio, rows = dataset
    output = tmp_path / "prepared"
    prepare(manifest, audio, output)
    return output, audio, rows


def test_plan_reads_without_mutating_sources(prepared):
    source, audio, _ = prepared
    paths = list(source.rglob("*")) + list(audio.rglob("*"))
    before = {path: digest(path) for path in paths if path.is_file()}
    result = plan(source, audio)
    assert result["purpose"] == "synthetic-test"
    assert len(result["rows"]) == 10
    assert before == {path: digest(path) for path in before}
    assert set(paths) == set(source.rglob("*")) | set(audio.rglob("*"))


def test_artifacts_preserve_identity_and_train_only_cmvn(prepared, tmp_path):
    source, audio, _ = prepared
    originals = {p: digest(p) for root in (source, audio) for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "features"
    result = run(source, audio, output)
    assert json.loads((output / "READY.json").read_text(encoding="utf-8")) == result
    assert result["schema"] == "easykws-preprocessed/v1"
    assert result["purpose"] == "synthetic-test"
    for name, sha in result["files"].items():
        assert digest(output / name) == sha
    train = []
    for split in ("train", "val", "test"):
        old = {r["sample_id"]: r for r in read_rows(source / f"{split}.jsonl")}
        new = read_rows(output / f"{split}.jsonl")
        assert {r["sample_id"] for r in new} == set(old)
        for row in new:
            assert all(row[key] == value for key, value in old[row["sample_id"]].items())
            safe_name = hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest()
            assert row["feature_path"] == f"features/{safe_name}.npy"
            feature = output / row["feature_path"]
            assert digest(feature) == row["feature_sha256"]
            values = np.load(feature, allow_pickle=False)
            assert values.dtype == np.float32
            assert values.shape == (row["num_frames"], 40)
            assert row["feature_dim"] == 40 and np.isfinite(values).all()
            if split == "train":
                train.append(values.astype(np.float64))
    joined = np.concatenate(train)
    cmvn = json.loads((output / "cmvn.json").read_text(encoding="utf-8"))
    assert cmvn["fit_split"] == "train"
    assert cmvn["frame_num"] == len(joined)
    np.testing.assert_allclose(cmvn["mean"], joined.mean(axis=0), rtol=1e-7, atol=1e-7)
    variance = np.maximum(joined.var(axis=0), 1e-10)
    np.testing.assert_allclose(cmvn["var"], variance, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(cmvn["istd"], 1 / np.sqrt(variance), rtol=1e-6)
    assert originals == {p: digest(p) for p in originals}


def test_eval_audio_changes_do_not_affect_fitted_cmvn(dataset, tmp_path):
    manifest, audio, rows = dataset
    assignment = {f"person_{i}": "train" if i < 3 else ("val" if i == 3 else "test") for i in range(5)}
    first_prep = tmp_path / "prep_one"
    prepare(manifest, audio, first_prep, {"speaker_splits": assignment})
    run(first_prep, audio, tmp_path / "out_one")
    for index, row in enumerate(rows):
        if assignment[row["speaker_id"]] != "train":
            n = np.arange(3200)
            signal = 9000 * np.sin(2 * np.pi * (1000 + index * 47) * n / 16000)
            write_wave(audio / row["audio_path"], signal)
    second_prep = tmp_path / "prep_two"
    prepare(manifest, audio, second_prep, {"speaker_splits": assignment})
    run(second_prep, audio, tmp_path / "out_two")
    assert (tmp_path / "out_one/cmvn.json").read_bytes() == (tmp_path / "out_two/cmvn.json").read_bytes()
    old_val = read_rows(tmp_path / "out_one/val.jsonl")
    new_val = read_rows(tmp_path / "out_two/val.jsonl")
    assert old_val[0]["feature_sha256"] != new_val[0]["feature_sha256"]


@pytest.mark.parametrize("filename", ["train.jsonl", "config.json", "excluded.jsonl"])
def test_prepared_artifact_tampering_fails_before_output(prepared, tmp_path, filename):
    source, audio, _ = prepared
    path = source / filename
    path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "bad"
    with pytest.raises(PreprocessError):
        run(source, audio, output)
    assert not (output / "READY.json").exists()


def test_changed_audio_is_rejected_even_when_wav_is_valid(prepared, tmp_path):
    source, audio, rows = prepared
    path = audio / rows[0]["audio_path"]
    write_wave(path, np.arange(3200) % 1000)
    with pytest.raises(PreprocessError):
        run(source, audio, tmp_path / "changed")
    assert not (tmp_path / "changed/READY.json").exists()


@pytest.mark.parametrize("rate,channels,config", [(8000, 1, {"resample": True}),
                                                 (48000, 1, {"resample": True}),
                                                 (16000, 2, {"channel_policy": "mean"})])
def test_conversion_requires_explicit_policy(dataset, tmp_path, rate, channels, config):
    manifest, audio, rows = dataset
    n = np.arange(rate // 5)
    signal = 3000 * np.sin(2 * np.pi * 377 * n / rate)
    if channels == 2:
        signal = np.column_stack([signal, signal * .6])
    write_wave(audio / rows[0]["audio_path"], signal, rate, channels)
    source = tmp_path / "converted_prep"
    prepare(manifest, audio, source)
    with pytest.raises(PreprocessError):
        plan(source, audio)
    result = run(source, audio, tmp_path / "converted", config)
    assert result["schema"] == "easykws-preprocessed/v1"


@pytest.mark.parametrize("kind", ["zero", "short", "positive_fullscale", "negative_fullscale"])
def test_invalid_signal_never_publishes_ready(dataset, tmp_path, kind):
    manifest, audio, rows = dataset
    samples = np.arange(3200) % 1000
    if kind == "zero":
        samples[:] = 0
    elif kind == "short":
        samples = samples[:399]
    elif kind == "positive_fullscale":
        samples[100] = 32767
    else:
        samples[100] = -32768
    write_wave(audio / rows[0]["audio_path"], samples)
    source = tmp_path / "invalid_prep"
    prepare(manifest, audio, source)
    with pytest.raises(PreprocessError):
        run(source, audio, tmp_path / "invalid")
    assert not (tmp_path / "invalid/READY.json").exists()


@pytest.mark.parametrize("config", [{"resample": 1}, {"channel_policy": "first"},
                                  {"resampl": True}, {"schema": "wrong"}])
def test_invalid_configuration_is_rejected(prepared, config):
    source, audio, _ = prepared
    with pytest.raises(PreprocessError):
        plan(source, audio, config)


def test_output_paths_and_existing_results_are_protected(prepared, tmp_path):
    source, audio, _ = prepared
    for output in (source / "nested", audio / "nested"):
        with pytest.raises(PreprocessError):
            run(source, audio, output)
        assert not output.exists()
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(PreprocessError):
        run(source, audio, output)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_failed_artifact_write_never_publishes_ready(prepared, tmp_path, monkeypatch):
    source, audio, _ = prepared
    original = Path.write_bytes

    def fail(path, payload):
        if path.name == "val.jsonl":
            raise OSError("synthetic disk failure")
        return original(path, payload)

    monkeypatch.setattr(Path, "write_bytes", fail)
    output = tmp_path / "failed"
    with pytest.raises((OSError, PreprocessError)):
        run(source, audio, output)
    assert not (output / "READY.json").exists()


def test_cli_runs_from_unrelated_directory(prepared, tmp_path):
    source, audio, _ = prepared
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    output = tmp_path / "cli_features"
    command = [sys.executable, "-B", "-m", "easykws.preprocess", "run",
               "--prepared", str(source), "--audio-root", str(audio), "--output", str(output)]
    proc = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                          text=True, encoding="utf-8", timeout=40)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["purpose"] == "synthetic-test"
    proc = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                          text=True, encoding="utf-8", timeout=40)
    assert proc.returncode != 0
