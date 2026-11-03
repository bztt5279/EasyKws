"""v2 reader checks on generated tones and simulated VC interface declarations.

These are data-contract and feature-path tests, not speech/VC-engine evaluations.
"""
import hashlib
import json
import math
import shutil
import struct
import wave

import pytest

from easykws.prep import v2
from easykws.preprocess import PreprocessError, plan, run


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tone(path, index):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [int(3500 * math.sin(2 * math.pi * (233 + 41 * index) * n / 16000))
               for n in range(3200)]
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(struct.pack("<3200h", *samples))
    return digest(path)


def dump(path, document):
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def bundle(tmp_path):
    roots = {name: tmp_path / name for name in ("main", "refs")}
    for root in roots.values(): root.mkdir()
    rows = []
    for person, split in enumerate(("train", "val", "test")):
        for label in (0, -1):
            index = len(rows)
            name = f"sample_{index}"
            rows.append({"sample_id": name, "dataset_id": "main", "speaker_id": f"person_{person}",
                         "source_recording_id": f"session_{person}", "audio_path": name + ".wav",
                         "audio_sha256": tone(roots["main"] / (name + ".wav"), index),
                         "label": label, "group": "target" if label == 0 else "background",
                         "expected_phrase": "example trigger" if label == 0 else "",
                         "review_status": "pass", "origin": "recorded", "purpose": "training-data",
                         "parent_sample_ids": [], "resource_ids": [], "requested_split": split,
                         "generation": None})
    rows[0].update(origin="test-fixture", purpose="synthetic-test")
    rows.append(dict(rows[0], sample_id="derived", audio_path="derived.wav", origin="vc",
                     purpose="training-data", audio_sha256=tone(roots["main"] / "derived.wav", 10),
                     parent_sample_ids=["sample_0"], resource_ids=["reference"],
                     generation={"engine": "interface-simulator", "model": "test-model",
                                 "config_sha256": "a" * 64, "seed": 9, "status": "success"}))
    document = {"schema": "easykws-recordings/v2", "datasets": [
        {"dataset_id": key, "source": "test-generated-schema-input", "version": "test-v1",
         "license": "test-only", "split_policy": "speaker"} for key in roots],
        "resources": [{"resource_id": "reference", "dataset_id": "refs", "audio_path": "reference.wav",
                       "sha256": tone(roots["refs"] / "reference.wav", 11), "kind": "reference-voice",
                       "purpose": "training-data", "split": "external", "speaker_id": None,
                       "source_sample_id": None}],
        "selection": {"kind": "diagnostic-subset", "reason": "small interface check", "available_count": 20},
        "rows": rows}
    manifest = tmp_path / "manifest.json"
    dump(manifest, document)
    prepared = tmp_path / "prepared"
    v2.prepare(manifest, roots, prepared)
    return prepared, roots


def rehash(prepared, ready):
    """Update the public hash inventory; this deliberately provides no authenticity."""
    for filename in ready["files"]:
        ready["files"][filename] = digest(prepared / filename)
    ready["manifest_sha256"] = digest(prepared / "manifest.json")
    dump(prepared / "READY.json", ready)


def test_v2_features_preserve_manifest_resources_lineage_and_separate_flags(bundle, tmp_path):
    prepared, roots = bundle
    paths = [p for root in (prepared, *roots.values()) for p in root.rglob("*") if p.is_file()]
    before = {path: digest(path) for path in paths}
    output = tmp_path / "features"
    result = run(prepared, None, output, path_map=roots)
    assert result["schema"] == "easykws-preprocessed/v2"
    assert result["purpose"] == "synthetic-test"
    assert set(result["data_flags"]) == {"synthetic-test", "diagnostic-subset"}
    assert result["selection"]["available_count"] == 20
    assert (output / "source_manifest.json").read_bytes() == (prepared / "manifest.json").read_bytes()
    assert json.loads((output / "source_manifest.json").read_text())["resources"]
    row = next(row for row in read_rows(output / "train.jsonl") if row["sample_id"] == "derived")
    assert row["purpose"] == "training-data" and row["effective_purpose"] == "synthetic-test"
    assert row["parent_sample_ids"] == ["sample_0"] and row["resource_ids"] == ["reference"]
    assert result["cmvn_fit_split"] == "train"
    assert all(digest(output / filename) == expected for filename, expected in result["files"].items())
    assert before == {path: digest(path) for path in before}


@pytest.mark.parametrize("kind", ["fixture_training", "reference_val", "reference_hidden_eval_content",
                                 "generation_failed", "selection_full_count", "official_missing_split"])
def test_reader_rejects_illegal_declarations_even_after_inventory_hash_recomputation(bundle, tmp_path, kind):
    prepared, roots = bundle
    manifest_path = prepared / "manifest.json"
    document = json.loads(manifest_path.read_text())
    ready = json.loads((prepared / "READY.json").read_text())
    changes = {}
    if kind == "fixture_training":
        document["rows"][0]["purpose"] = "training-data"
        changes["sample_0"] = {"purpose": "training-data", "effective_purpose": "training-data"}
    elif kind == "reference_val": document["resources"][0]["split"] = "val"
    elif kind == "reference_hidden_eval_content":
        source = document["rows"][2]
        document["resources"][0].update(dataset_id="main", audio_path=source["audio_path"],
                                           sha256=source["audio_sha256"])
    elif kind == "generation_failed":
        document["rows"][-1]["generation"]["status"] = "failed"
        changes["derived"] = {"generation": document["rows"][-1]["generation"]}
    elif kind == "selection_full_count":
        document["selection"]["kind"] = "full"
        ready["selection"] = document["selection"]
        ready["data_flags"].remove("diagnostic-subset")
    else:
        document["datasets"][0]["split_policy"] = "official"
        document["rows"][0]["requested_split"] = None
        changes["sample_0"] = {"requested_split": None}
    dump(manifest_path, document)
    for split in ("train", "val", "test"):
        path = prepared / (split + ".jsonl")
        rows = read_rows(path)
        for row in rows: row.update(changes.get(row["sample_id"], {}))
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rehash(prepared, ready)
    assert all(digest(prepared / name) == expected for name, expected in ready["files"].items())
    with pytest.raises(PreprocessError):
        run(prepared, None, tmp_path / "rejected", path_map=roots)
    assert not (tmp_path / "rejected" / "READY.json").exists()


@pytest.mark.parametrize("kind", ["ready_purpose", "ready_flags", "ready_selection", "index_purpose"])
def test_reader_reconstructs_ready_and_index_semantics_with_matching_file_hashes(bundle, kind):
    prepared, roots = bundle
    ready = json.loads((prepared / "READY.json").read_text())
    if kind == "ready_purpose": ready["purpose"] = "training-data"
    elif kind == "ready_flags": ready["data_flags"] = []
    elif kind == "ready_selection": ready["selection"]["reason"] = "different declaration"
    else:
        path = prepared / "train.jsonl"
        rows = read_rows(path)
        next(row for row in rows if row["sample_id"] == "derived")["effective_purpose"] = "training-data"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    rehash(prepared, ready)
    with pytest.raises(PreprocessError):
        plan(prepared, None, path_map=roots)


@pytest.mark.parametrize("kind", ["sample", "resource", "missing_resource"])
def test_reader_rechecks_actual_audio_and_referenced_resource_files(bundle, kind):
    prepared, roots = bundle
    if kind == "sample": tone(roots["main"] / "sample_0.wav", 99)
    elif kind == "resource": tone(roots["refs"] / "reference.wav", 98)
    else: (roots["refs"] / "reference.wav").unlink()
    with pytest.raises(PreprocessError):
        plan(prepared, None, path_map=roots)


def test_reader_supports_relocated_multiroot_audio_and_rejects_audio_root_ambiguity(bundle, tmp_path):
    prepared, roots = bundle
    moved = {}
    for key, root in roots.items():
        moved[key] = tmp_path / "moved" / key
        shutil.copytree(root, moved[key])
    first = plan(prepared, None, path_map=roots)
    second = plan(prepared, None, path_map=moved)
    assert first == second
    with pytest.raises(PreprocessError):
        plan(prepared, roots["main"], path_map=roots)
