"""Small generated DSP fixtures verify scheduling, cache integration and isolation."""
import hashlib
import json
import shutil
import wave

import numpy as np
import pytest

from easykws.prep import v2
from easykws.preprocess import PreprocessError, plan, run
from easykws.preprocess.augmentation_pipeline import configure, schedules


def wav(path, signal, rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(np.asarray(signal, dtype="<i2").tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def config(kinds=None, regular=1, challenge=0):
    return {"augmentation": {"seed": 73,
            "selectors": {"groups": ["examples"], "labels": [0, -1], "origins": ["recorded"]},
            "variants": {"regular": regular, "challenge": challenge}, "kinds": kinds or ["speed"],
            "speed": [.8, .8], "noise": {"resource_ids": ["noise_b", "noise_a"], "snr_db": [10, 10]},
            "rir": {"resource_ids": ["impulse"], "channel": 0}}}


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "audio"
    document = {"schema": v2.SCHEMA, "datasets": [
        {"dataset_id": "examples", "source": "generated-test-signals", "version": "test-v1",
         "license": "test-only", "split_policy": "speaker"}], "resources": [],
        "selection": {"kind": "full", "reason": "", "available_count": 6}, "rows": []}
    for person, split in enumerate(("train", "val", "test")):
        for label in (0, -1):
            index = len(document["rows"])
            name = "sample_" + str(index)
            n = np.arange(64000 if index == 0 else 3200)
            sha = wav(root / (name + ".wav"), 3000 * np.sin(2 * np.pi * (220 + 37 * index) * n / 16000))
            document["rows"].append({"sample_id": name, "dataset_id": "examples", "speaker_id": f"speaker_{person}",
                "source_recording_id": f"recording_{person}", "audio_path": name + ".wav", "audio_sha256": sha,
                "label": label, "group": "examples", "expected_phrase": "example" if label == 0 else "",
                "review_status": "pass", "origin": "recorded", "purpose": "training-data",
                "parent_sample_ids": [], "resource_ids": [], "requested_split": split, "generation": None})
    signals = {"noise_a": np.random.default_rng(7).normal(0, 1500, 1000),
               "noise_b": np.random.default_rng(8).normal(0, 1600, 1500),
               "impulse": np.r_[np.zeros(20), 12000, np.ones(2979) * 100]}
    for key, signal in signals.items():
        document["resources"].append({"resource_id": key, "dataset_id": "examples", "audio_path": key + ".wav",
            "sha256": wav(root / (key + ".wav"), signal), "kind": "rir" if key == "impulse" else "noise",
            "purpose": "synthetic-test" if key == "noise_a" else "training-data", "split": "external",
            "speaker_id": None, "source_sample_id": None})

    def prepare(document=document, name="prepared"):
        manifest = tmp_path / (name + ".json")
        manifest.write_text(json.dumps(document), encoding="utf-8")
        output = tmp_path / name
        v2.prepare(manifest, {"examples": root}, output)
        return output, {"examples": root}

    return document, root, prepare


def test_full_length_clean_train_variants_eval_unchanged_and_final_cmvn(bundle, tmp_path):
    _, _, prepare = bundle
    prepared, roots = prepare()
    baseline = tmp_path / "clean"
    run(prepared, None, baseline, path_map=roots)
    output = tmp_path / "augmented"
    result = run(prepared, None, output, config(["gain", "speed"], regular=1, challenge=1), path_map=roots)
    assert result["splits"]["train"]["samples"] == 6
    assert result["source_included_count"] == 6 and result["included_count"] == 10
    assert result["augmentation"]["generated_count"] == 4
    for split in ("val", "test"):
        assert (output / (split + ".jsonl")).read_bytes() == (baseline / (split + ".jsonl")).read_bytes()
    train = rows(output / "train.jsonl")
    clean = next(row for row in train if row["sample_id"] == "sample_0")
    assert clean["augmentation"]["waveform_samples"] == 64000
    assert clean["num_frames"] == 398
    source = next(row for row in rows(baseline / "train.jsonl") if row["sample_id"] == "sample_0")
    assert source["feature_sha256"] == clean["feature_sha256"]
    for row in train:
        metadata = row["augmentation"]
        assert len(metadata["operations"]) == {"clean": 0, "regular": 1, "challenge": 2}[metadata["variant"]]
        if metadata["variant_index"]:
            assert row["parent_sample_ids"] == [metadata["source_sample_id"]]
    combined = np.concatenate([np.load(output / row["feature_path"]).astype(np.float64) for row in train])
    cmvn = json.loads((output / "cmvn.json").read_text())
    assert cmvn["frame_num"] == len(combined) == result["train_frames"]
    np.testing.assert_allclose(cmvn["mean"], combined.mean(axis=0), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(cmvn["var"], np.maximum(combined.var(axis=0), 1e-10), rtol=1e-6)


def test_rir_tail_and_fixture_resource_purpose_are_retained(bundle, tmp_path):
    _, _, prepare = bundle
    prepared, roots = prepare()
    cfg = config(["rir"])
    cfg["augmentation"]["rir"]["resource_ids"] = ["impulse"]
    result = run(prepared, None, tmp_path / "rir", cfg, path_map=roots)
    variant = next(row for row in rows(tmp_path / "rir/train.jsonl")
                   if row.get("augmentation", {}).get("source_sample_id") == "sample_0"
                   and row["augmentation"]["variant_index"] == 1)
    assert variant["augmentation"]["waveform_samples"] == 64000 + 3000 - 1 - 20
    assert variant["num_frames"] > 398
    cfg = config(["noise"])
    cfg["augmentation"]["noise"]["resource_ids"] = ["noise_a"]
    purpose = run(prepared, None, tmp_path / "noise", cfg, path_map=roots)
    assert purpose["purpose"] == "synthetic-test" and "synthetic-test" in purpose["data_flags"]
    noisy = [r for r in rows(tmp_path / "noise/train.jsonl") if r["augmentation"]["variant_index"]]
    assert all(r["effective_purpose"] == "synthetic-test" for r in noisy)
    assert noisy[0]["augmentation"]["resources"][0]["sha256"]
    assert result["train_frames"] > 0


def test_explicit_crop_replaces_clean_and_same_interval_precedes_variants(bundle):
    _, _, prepare = bundle
    prepared, roots = prepare()
    cfg = config(["gain"])
    cfg["augmentation"]["crop"] = {"mode": "replace-clean",
        "selectors": {"groups": ["examples"], "labels": [0], "origins": ["recorded"]},
        "seconds": [2, 2], "only_if_longer_than": 3}
    result = plan(prepared, None, cfg, path_map=roots)
    selected = [r for r in result["rows"] if r.get("augmentation", {}).get("source_sample_id") == "sample_0"]
    assert len(selected) == 2 and result["augmentation"]["clean_replaced_count"] == 1
    intervals = [r["augmentation"]["processing"]["operations"][0] for r in selected]
    assert intervals[0] == intervals[1]
    assert intervals[0]["end_sample"] - intervals[0]["start_sample"] == 32000
    assert intervals[0]["discarded_duration_sec"] == 2
    assert all(r["augmentation"]["waveform_samples"] == 32000 for r in selected)
    assert all("augmentation" not in r for r in result["rows"] if r["split"] != "train")


def test_seed_and_pool_selection_ignore_order_and_relocation(bundle, tmp_path):
    document, root, prepare = bundle
    prepared, roots = prepare()
    cfg = config(["noise", "speed"], regular=4, challenge=2)
    first = plan(prepared, None, cfg, path_map=roots)
    moved = tmp_path / "moved_audio"
    shutil.copytree(root, moved)
    cfg["augmentation"]["kinds"].reverse()
    cfg["augmentation"]["noise"]["resource_ids"].reverse()
    assert first == plan(prepared, None, cfg, path_map={"examples": moved})
    document["rows"].reverse()
    document["resources"].reverse()
    second, roots = prepare(document, "reordered")
    assert first["rows"] == plan(second, None, cfg, path_map=roots)["rows"]


@pytest.mark.parametrize("leak", ["split", "speaker", "source", "pcm"])
def test_pool_rejects_eval_split_speaker_source_or_rewrapped_pcm(bundle, leak):
    document, root, prepare = bundle
    resource = document["resources"][0]
    if leak == "split": resource["split"] = "test"
    elif leak == "speaker": resource["speaker_id"] = "speaker_1"
    elif leak == "source": resource["source_sample_id"] = "sample_2"
    else:
        raw = (root / "sample_2.wav").read_bytes()
        # Add an ancillary RIFF chunk while preserving exact PCM and dimensions.
        raw = raw[:4] + (len(raw) + 4).to_bytes(4, "little") + raw[8:] + b"JUNK\x04\x00\x00\x00test"
        (root / "noise_a.wav").write_bytes(raw)
        resource["sha256"] = hashlib.sha256(raw).hexdigest()
    prepared, roots = prepare()
    with pytest.raises(PreprocessError, match="val/test|not train"):
        plan(prepared, None, config(["noise"]), path_map=roots)


def test_resource_change_after_features_never_publishes_ready(bundle, tmp_path, monkeypatch):
    from easykws.preprocess import frontend

    _, root, prepare = bundle
    prepared, roots = prepare()
    original = frontend.extract_fbank
    changed = False

    def extract(waveform):
        nonlocal changed
        output = original(waveform)
        if not changed:
            changed = True
            wav(root / "noise_a.wav", np.arange(1000))
        return output

    monkeypatch.setattr(frontend, "extract_fbank", extract)
    output = tmp_path / "failed"
    with pytest.raises(PreprocessError, match="hash mismatch"):
        run(prepared, None, output, config(["noise"]), path_map=roots)
    assert not (output / "READY.json").exists()
    assert (output / "FAILED.json").exists()


def test_excluded_declared_eval_content_cannot_be_recataloged_external(bundle):
    document, root, prepare = bundle
    sha = wav(root / "excluded.wav", 2000 * np.sin(np.arange(5000) * .173))
    excluded = dict(document["rows"][2], sample_id="excluded_eval", speaker_id="other_eval_speaker",
                    source_recording_id="excluded_session", audio_path="excluded.wav", audio_sha256=sha,
                    review_status="pending")
    document["rows"].append(excluded)
    document["selection"]["available_count"] += 1
    document["resources"][0].update(audio_path="excluded.wav", sha256=sha)
    prepared, roots = prepare()
    assert "audio_file_sha256" not in rows(prepared / "excluded.jsonl")[0]
    with pytest.raises(PreprocessError, match="val/test"):
        plan(prepared, None, config(["noise"]), path_map=roots)


def add_excluded_source(bundle, *, fixture=False, rewrap=False):
    document, root, prepare = bundle
    source = root / "excluded.wav"
    sha = wav(source, 2000 * np.sin(np.arange(5000) * .173))
    excluded = dict(document["rows"][2], sample_id="excluded_source", speaker_id="other_speaker",
                    source_recording_id="excluded_session", audio_path=source.name,
                    audio_sha256=sha, review_status="pending")
    if fixture:
        excluded.update(requested_split="train", purpose="synthetic-test", origin="test-fixture")
    document["rows"].append(excluded)
    document["selection"]["available_count"] += 1
    if rewrap:
        raw = source.read_bytes()
        wrapped = raw[:4] + (len(raw) + 4).to_bytes(4, "little") + raw[8:] + b"JUNK\x04\x00\x00\x00test"
        resource_path = root / "noise_b.wav"
        resource_path.write_bytes(wrapped)
        resource = next(r for r in document["resources"] if r["resource_id"] == "noise_b")
        resource["sha256"] = hashlib.sha256(wrapped).hexdigest()
        assert resource["sha256"] != sha
        with wave.open(str(source), "rb") as original, wave.open(str(resource_path), "rb") as recataloged:
            assert original.getparams() == recataloged.getparams()
            assert original.readframes(original.getnframes()) == recataloged.readframes(recataloged.getnframes())
    cfg = config(["noise"])
    cfg["augmentation"]["noise"]["resource_ids"] = ["noise_b"]
    return excluded, source, prepare, cfg


def test_run_rejects_rewrapped_excluded_eval_pcm_without_ready(bundle, tmp_path):
    _, _, prepare, cfg = add_excluded_source(bundle, rewrap=True)
    prepared, roots = prepare()
    output = tmp_path / "rejected_eval"
    with pytest.raises(PreprocessError, match="val/test"):
        run(prepared, None, output, cfg, path_map=roots)
    assert not (output / "READY.json").exists()


def test_run_rewrapped_excluded_fixture_propagates_synthetic_purpose(bundle, tmp_path):
    _, _, prepare, cfg = add_excluded_source(bundle, fixture=True, rewrap=True)
    prepared, roots = prepare()
    output = tmp_path / "fixture_noise"
    result = run(prepared, None, output, cfg, path_map=roots)
    assert (output / "READY.json").exists()
    assert result["purpose"] == "synthetic-test" and "synthetic-test" in result["data_flags"]
    assert result["augmentation"]["resources"]["noise_b"]["effective_purpose"] == "synthetic-test"
    checks = result["augmentation"]["excluded_identity_checks"]
    assert [(check["sample_id"], check["status"]) for check in checks] == [("excluded_source", "verified-pcm")]
    augmented = [row for row in rows(output / "train.jsonl") if row["augmentation"]["variant_index"]]
    assert augmented and all(row["effective_purpose"] == "synthetic-test" for row in augmented)


@pytest.mark.parametrize("status", ["verified-pcm", "unavailable", "unverified-identity"])
def test_excluded_identity_status_is_reported_by_plan_and_run(bundle, tmp_path, status):
    excluded, source, prepare, cfg = add_excluded_source(bundle)
    if status == "unverified-identity":
        excluded["audio_sha256"] = None
    prepared, roots = prepare()
    if status == "unavailable":
        source.unlink()
    planned = plan(prepared, None, cfg, path_map=roots)
    result = run(prepared, None, tmp_path / "identity_status", cfg, path_map=roots)
    for response in (planned, result):
        checks = response["augmentation"]["excluded_identity_checks"]
        assert isinstance(checks, list)
        assert [(check["sample_id"], check["status"]) for check in checks] == [("excluded_source", status)]


def test_excluded_declared_hash_mismatch_aborts_without_ready(bundle, tmp_path):
    _, source, prepare, cfg = add_excluded_source(bundle)
    prepared, roots = prepare()
    wav(source, np.arange(5000))
    with pytest.raises(PreprocessError, match="hash mismatch"):
        plan(prepared, None, cfg, path_map=roots)
    output = tmp_path / "changed_excluded"
    with pytest.raises(PreprocessError, match="hash mismatch"):
        run(prepared, None, output, cfg, path_map=roots)
    assert not (output / "READY.json").exists()


@pytest.mark.parametrize("change", ["missing_selectors", "negative_counts", "missing_pool", "one_kind_challenge", "unknown_field"])
def test_configuration_rejects_ambiguous_or_invalid_schedules(change):
    cfg = config()["augmentation"]
    if change == "missing_selectors": del cfg["selectors"]
    elif change == "negative_counts": cfg["variants"]["regular"] = -1
    elif change == "missing_pool": cfg.update(kinds=["noise"], noise={"resource_ids": [], "snr_db": [5, 20]})
    elif change == "one_kind_challenge": cfg["variants"]["challenge"] = 1
    else: cfg["automatic_recipe"] = True
    with pytest.raises(PreprocessError): configure(cfg)


def test_selectors_are_intersection_and_clean_is_always_scheduled():
    cfg = configure(config(["gain"], regular=2)["augmentation"])
    row = {"sample_id": "source", "split": "train", "group": "other", "label": 0, "origin": "recorded"}
    assert [(i, variant) for i, variant, _, _ in schedules(row, cfg)] == [(0, "clean")]
    row["group"] = "examples"
    assert len(list(schedules(row, cfg))) == 3
    row["split"] = "test"
    assert len(list(schedules(row, cfg))) == 1
