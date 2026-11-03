"""Independent v2 schema/lineage checks using tones, never a speech benchmark.

The recorded/public/VC declarations below exercise schema inputs only. No recording
session or VC engine is run or validated by these tests.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import wave

import pytest

from easykws.prep import PrepError, plan as plan_v1, prepare as prepare_v1
from easykws.prep import v2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tone(path, index):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [int(3500 * math.sin(2 * math.pi * (211 + 37 * index) * n / 16000))
               for n in range(3200)]
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(struct.pack("<3200h", *samples))
    return sha(path)


def write_json(path, document):
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def case(tmp_path):
    roots = {key: tmp_path / key for key in ("main", "refs")}
    for root in roots.values():
        root.mkdir()
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
    document = {"schema": "easykws-recordings/v2", "datasets": [
        {"dataset_id": key, "source": "test-generated-schema-input", "version": "test-v1",
         "license": "test-only", "split_policy": "speaker"} for key in roots],
        "resources": [], "selection": {"kind": "full", "reason": "", "available_count": len(rows)},
        "rows": rows}
    manifest = tmp_path / "manifest.json"
    write_json(manifest, document)
    return manifest, roots, document


def save(case):
    manifest, roots, document = case
    write_json(manifest, document)
    return manifest, roots


def add_vc(case, purpose="training-data"):
    _, roots, document = case
    resource = {"resource_id": "reference", "dataset_id": "refs", "audio_path": "reference.wav",
                "sha256": tone(roots["refs"] / "reference.wav", 20), "kind": "reference-voice",
                "purpose": purpose, "split": "external", "speaker_id": None, "source_sample_id": None}
    document["resources"].append(resource)
    row = dict(document["rows"][0], sample_id="derived", audio_path="derived.wav", origin="vc",
               audio_sha256=tone(roots["main"] / "derived.wav", 21),
               parent_sample_ids=["sample_0"], resource_ids=["reference"],
               generation={"engine": "interface-simulator", "model": "test-model",
                           "config_sha256": "a" * 64, "seed": 7, "status": "success"})
    document["rows"].append(row)
    document["selection"]["available_count"] = len(document["rows"])
    return row, resource


def by_id(result):
    return {row["sample_id"]: row for row in result["rows"]}


def test_legal_vc_is_train_data_and_all_input_pcm_are_distinct(case):
    row, _ = add_vc(case)
    result = v2.plan(*save(case))
    assert len({r["audio_pcm_sha256"] for r in result["rows"]}) == 7
    assert result["purpose"] == "training-data" and result["data_flags"] == []
    assert by_id(result)[row["sample_id"]]["split"] == "train"
    assert by_id(result)[row["sample_id"]]["effective_purpose"] == "training-data"


def test_excluded_fixture_ancestor_still_propagates_test_purpose(case):
    row, _ = add_vc(case)
    document = case[2]
    ancestor = document["rows"][0]
    ancestor.update(origin="test-fixture", purpose="synthetic-test", review_status="flagged",
                    audio_path="absent.wav", audio_sha256=None)
    result = v2.plan(*save(case))
    output = by_id(result)
    assert not output[ancestor["sample_id"]]["included"]
    assert output[row["sample_id"]]["purpose"] == "training-data"
    assert output[row["sample_id"]]["effective_purpose"] == "synthetic-test"
    assert result["purpose"] == "synthetic-test" and "synthetic-test" in result["data_flags"]


def test_diagnostic_selection_and_qc_exclusions_are_separate(case):
    document = case[2]
    document["selection"] = {"kind": "diagnostic-subset", "reason": "small interface check",
                             "available_count": 100}
    rejected = dict(document["rows"][0], sample_id="rejected", review_status="pending",
                    audio_path="missing.wav", audio_sha256=None)
    document["rows"].append(rejected)
    result = v2.plan(*save(case))
    assert result["selection"] == document["selection"]
    assert result["input_count"] == 7 and result["included_count"] == 6 and result["excluded_count"] == 1
    assert "diagnostic-subset" in result["data_flags"]
    assert "synthetic-test" not in result["data_flags"]


@pytest.mark.parametrize("mutation", ["unknown_top", "unknown_row", "unknown_dataset", "version",
                                    "fixture_training", "unsafe_excluded", "full_count", "missing_root",
                                    "extra_root", "pending_allowed"])
def test_strict_schema_paths_selection_and_status_cannot_be_relaxed(case, mutation):
    document = case[2]
    config = None
    roots = dict(case[1])
    if mutation == "unknown_top": document["extra"] = True
    elif mutation == "unknown_row": document["rows"][0]["extra"] = True
    elif mutation == "unknown_dataset": document["datasets"][0]["extra"] = True
    elif mutation == "version": document["schema"] = "easykws-recordings/v999"
    elif mutation == "fixture_training": document["rows"][0]["origin"] = "test-fixture"
    elif mutation == "unsafe_excluded":
        document["rows"][0].update(review_status="flagged", audio_path="../escape.wav")
    elif mutation == "full_count": document["selection"]["available_count"] = 100
    elif mutation == "missing_root": del roots["refs"]
    elif mutation == "extra_root": roots["extra"] = roots["main"]
    else: config = {"include_review_status": ["pass", "pending"]}
    save(case)
    with pytest.raises(PrepError):
        v2.plan(case[0], roots, config)


def test_official_unknown_speakers_keep_official_splits_and_are_counted(case):
    document = case[2]
    document["datasets"][0]["split_policy"] = "official"
    for row in document["rows"]:
        row.update(origin="public", speaker_id=None,
                   source_recording_id="official_" + row["requested_split"])
    result = v2.plan(*save(case))
    for split in ("train", "val", "test"):
        assert result["splits"][split]["unknown_speaker_samples"] == 2
        assert result["splits"][split]["speakers"] == []
    assert all(row["actual_speaker_id"] is None for row in result["rows"])


@pytest.mark.parametrize("kind", ["missing_official", "shared_source", "shared_speaker", "parent_bridge",
                                 "explicit_assignment", "fixed_counts", "null_recorded"])
def test_fixed_official_and_connected_component_conflicts_are_rejected(case, kind):
    document = case[2]
    rows = document["rows"]
    config = None
    if kind in ("missing_official", "shared_source"):
        document["datasets"][0]["split_policy"] = "official"
        for row in rows: row.update(origin="public", speaker_id=None)
        if kind == "missing_official": rows[0]["requested_split"] = None
        else: rows[2]["source_recording_id"] = rows[0]["source_recording_id"]
    elif kind == "shared_speaker": rows[2]["speaker_id"] = rows[0]["speaker_id"]
    elif kind == "parent_bridge":
        rows.append(dict(rows[0], sample_id="bridge", parent_sample_ids=[rows[2]["sample_id"]],
                         review_status="flagged", audio_path="missing.wav", audio_sha256=None))
        document["selection"]["available_count"] = len(rows)
    elif kind == "explicit_assignment":
        config = {"speaker_splits": {"person_0": "val", "person_1": "val", "person_2": "test"}}
    elif kind == "fixed_counts": config = {"split_counts": {"train": 1, "val": 1, "test": 1}}
    else: rows[0]["speaker_id"] = None
    with pytest.raises(PrepError):
        v2.plan(*save(case), config)


@pytest.mark.parametrize("kind", ["no_parent", "two_parents", "no_reference", "negative", "speaker",
                                 "phrase", "val", "unknown_parent", "cycle"])
def test_vc_requires_valid_target_lineage_and_train_assignment(case, kind):
    row, resource = add_vc(case)
    if kind == "no_parent": row["parent_sample_ids"] = []
    elif kind == "two_parents": row["parent_sample_ids"].append("sample_1")
    elif kind == "no_reference": resource["kind"] = "noise"
    elif kind == "negative": row["label"] = -1
    elif kind == "speaker": row["speaker_id"] = "different_person"
    elif kind == "phrase": row["expected_phrase"] = "different trigger"
    elif kind == "val": row["requested_split"] = "val"
    elif kind == "unknown_parent": row["parent_sample_ids"] = ["absent"]
    else: case[2]["rows"][0]["parent_sample_ids"] = [row["sample_id"]]
    with pytest.raises(PrepError):
        v2.plan(*save(case))


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_unsuccessful_generation_is_excluded_even_after_manual_review(case, status):
    row, _ = add_vc(case)
    row.update(review_status="manual", audio_path="not-generated.wav", audio_sha256=None)
    row["generation"]["status"] = status
    result = v2.plan(*save(case))
    derived = by_id(result)[row["sample_id"]]
    assert not derived["included"] and derived["exclusion_reasons"] and derived["split"] is None
    assert result["included_count"] == 6


def test_reference_test_purpose_propagates_without_counting_resource_as_speaker(case):
    row, _ = add_vc(case, purpose="synthetic-test")
    result = v2.plan(*save(case))
    assert by_id(result)[row["sample_id"]]["effective_purpose"] == "synthetic-test"
    assert result["data_flags"] == ["synthetic-test"]
    assert result["splits"]["train"]["speakers"] == ["person_0"]


@pytest.mark.parametrize("kind", ["val_split", "test_speaker", "val_source", "hidden_val_content",
                                 "missing", "bad_hash", "hidden_fixture_content"])
def test_references_cannot_hide_eval_audio_or_missing_invalid_provenance(case, kind):
    _, resource = add_vc(case)
    if kind == "val_split": resource["split"] = "val"
    elif kind == "test_speaker": resource["speaker_id"] = "person_2"
    elif kind == "val_source": resource["source_sample_id"] = "sample_2"
    elif kind == "hidden_val_content":
        source = case[2]["rows"][2]
        resource.update(dataset_id="main", audio_path=source["audio_path"], sha256=source["audio_sha256"])
    elif kind == "missing": resource["audio_path"] = "absent.wav"
    elif kind == "bad_hash": resource["sha256"] = "0" * 64
    else:
        source = case[2]["rows"][0]
        source.update(origin="test-fixture", purpose="synthetic-test")
        resource.update(dataset_id="main", audio_path=source["audio_path"], sha256=source["audio_sha256"])
    with pytest.raises(PrepError):
        v2.plan(*save(case))


def test_multiroot_relocation_keeps_artifacts_byte_identical_and_does_not_copy_audio(case, tmp_path):
    add_vc(case)
    manifest, roots = save(case)
    first = tmp_path / "prepared_first"
    second = tmp_path / "prepared_second"
    one = v2.prepare(manifest, roots, first)
    moved = {}
    for key, root in roots.items():
        moved[key] = tmp_path / "relocated" / key
        shutil.copytree(root, moved[key])
    two = v2.prepare(manifest, moved, second)
    assert one == two
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    for output in (first, second):
        assert not list(output.rglob("*.wav"))
        for path in output.iterdir():
            assert all(str(root).encode() not in path.read_bytes() for root in [*roots.values(), *moved.values()])
    for name, expected in one["files"].items():
        assert sha(first / name) == expected


def test_v1_upgrade_is_conservative_fixes_old_splits_and_preserves_sources(case, tmp_path):
    _, roots, document = case
    legacy_rows = []
    for original in document["rows"]:
        row = {key: value for key, value in original.items() if key in {
            "sample_id", "speaker_id", "source_recording_id", "audio_path", "label", "group",
            "expected_phrase", "review_status", "origin"}}
        row["origin"] = "synthetic"
        legacy_rows.append(row)
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text("".join(json.dumps(row) + "\n" for row in legacy_rows), encoding="utf-8")
    config = {"speaker_splits": {f"person_{i}": split for i, split in enumerate(("train", "val", "test"))}}
    old = tmp_path / "old_prepared"
    prepare_v1(legacy, roots["main"], old, config)
    paths = [legacy, *old.iterdir(), *roots["main"].iterdir()]
    before = {path: sha(path) for path in paths}
    upgrade = tmp_path / "upgraded"
    v2.upgrade_v1(legacy, roots["main"], upgrade, config)
    converted = json.loads((upgrade / "manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((upgrade / "config.json").read_text(encoding="utf-8"))
    assert all(row["origin"] == "test-fixture" and row["purpose"] == "synthetic-test" for row in converted["rows"])
    previous = {r["sample_id"]: r["split"] for r in plan_v1(legacy, roots["main"], config)["rows"]}
    assert {r["sample_id"]: r["requested_split"] for r in converted["rows"]} == previous
    assert v2.plan(upgrade / "manifest.json", {"recordings": roots["main"]}, cfg)["purpose"] == "synthetic-test"
    assert before == {path: sha(path) for path in before}
