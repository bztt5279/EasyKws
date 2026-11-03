import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from easykws.prep import PrepError, plan, prepare
from easykws.prep.core import load_json


def mapping(result):
    return {r["sample_id"]: r["split"] for r in result["rows"] if r["included"]}


def test_plan_is_readonly_and_order_independent(dataset, rewrite_manifest):
    manifest, audio, rows = dataset
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in audio.iterdir()}
    first = plan(manifest, audio)
    rewrite_manifest(manifest, list(reversed(rows)))
    second = plan(manifest, audio)
    assert mapping(first) == mapping(second)
    assert first["rows"] == second["rows"]
    assert first["purpose"] == "synthetic-test"
    assert [first["splits"][s]["independent_units"] for s in ("train", "val", "test")] == [3, 1, 1]
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in audio.iterdir()}
    assert sorted(p.name for p in manifest.parent.iterdir()) == ["audio", "recordings.jsonl"]


def test_aliases_merge_recording_ids_of_same_speaker(dataset, rewrite_manifest):
    manifest, audio, rows = dataset
    rows[1]["speaker_id"] = "alternate_name"
    rows[1]["source_recording_id"] = "another_session"
    rewrite_manifest(manifest, rows)
    result = plan(manifest, audio, {"speaker_aliases": {"alternate_name": "person_0"}})
    output = {r["sample_id"]: r for r in result["rows"]}
    assert output[rows[0]["sample_id"]]["split"] == output[rows[1]["sample_id"]]["split"]
    assert output[rows[1]["sample_id"]]["actual_speaker_id"] == "person_0"


def test_source_links_keep_different_speakers_together(dataset, rewrite_manifest):
    manifest, audio, rows = dataset
    for row in rows[2:4]:
        row["source_recording_id"] = "session_0"
    rewrite_manifest(manifest, rows)
    result = plan(manifest, audio)
    assignments = mapping(result)
    assert len({assignments[r["sample_id"]] for r in rows[:4]}) == 1
    assert sum(s["independent_units"] for s in result["splits"].values()) == 4


def test_parent_link_cannot_be_split_by_explicit_assignment(dataset, rewrite_manifest):
    manifest, audio, rows = dataset
    rows[2]["parent_sample_id"] = rows[0]["sample_id"]
    rewrite_manifest(manifest, rows)
    assignment = {"person_0": "train", "person_1": "val", "person_2": "test",
                  "person_3": "train", "person_4": "train"}
    with pytest.raises(PrepError, match="linked"):
        plan(manifest, audio, {"speaker_splits": assignment})


def test_excluded_parent_still_links_related_samples(dataset, rewrite_manifest):
    manifest, audio, rows = dataset
    rows.append(dict(rows[0], sample_id="bridge", audio_path="missing.wav",
                     review_status="flagged", parent_sample_id=rows[2]["sample_id"]))
    rewrite_manifest(manifest, rows)
    assignments = mapping(plan(manifest, audio))
    assert len({assignments[r["sample_id"]] for r in rows[:4]}) == 1


def test_seed_selects_repeatable_splits(dataset):
    manifest, audio, _ = dataset
    assignments = [mapping(plan(manifest, audio, {"seed": seed})) for seed in range(6)]
    assert assignments[3] == mapping(plan(manifest, audio, {"seed": 3}))
    assert any(value != assignments[0] for value in assignments[1:])


@pytest.mark.parametrize("aliases", [{"a": "a"}, {"a": "b", "b": "a"}])
def test_alias_cycles_fail_before_writing(dataset, tmp_path, aliases):
    manifest, audio, _ = dataset
    with pytest.raises(PrepError, match="cycle"):
        prepare(manifest, audio, tmp_path / "prepared", {"speaker_aliases": aliases})
    assert not (tmp_path / "prepared").exists()


def test_exclusions_are_explicit_and_do_not_read_excluded_audio(dataset, rewrite_manifest):
    manifest, audio, rows = dataset
    extra = dict(rows[0], sample_id="rejected", audio_path="missing.wav", review_status="flagged")
    rows.append(extra)
    rewrite_manifest(manifest, rows)
    result = plan(manifest, audio)
    rejected = next(r for r in result["rows"] if r["sample_id"] == "rejected")
    assert result["excluded_count"] == 1 and result["included_count"] == 10
    assert rejected["exclusion_reasons"] == ["review_status_not_included"]
    assert rejected["split"] is None and "audio_file_sha256" not in rejected


def test_explicit_exclusions_and_unknown_ids(dataset, rewrite_manifest):
    manifest, audio, rows = dataset
    rows.append(dict(rows[0], sample_id="user_excluded", audio_path="not_needed.wav"))
    rewrite_manifest(manifest, rows)
    result = plan(manifest, audio, {"exclude_sample_ids": ["user_excluded"]})
    assert result["excluded_count"] == 1
    with pytest.raises(PrepError, match="unknown samples"):
        plan(manifest, audio, {"exclude_sample_ids": ["user_excluded", "typo"]})


@pytest.mark.parametrize("path", ["../outside.wav", "/outside.wav", "Z:/outside.wav", "sub\\audio.wav"])
def test_paths_cannot_escape_root(dataset, rewrite_manifest, path):
    manifest, audio, rows = dataset
    rows[0]["audio_path"] = path
    rewrite_manifest(manifest, rows)
    with pytest.raises(PrepError, match="relative"):
        plan(manifest, audio)


def test_duplicate_ids_and_content_are_rejected(dataset, rewrite_manifest):
    manifest, audio, rows = dataset
    rewrite_manifest(manifest, rows + [rows[0]])
    with pytest.raises(PrepError, match="Duplicate sample_id"):
        plan(manifest, audio)
    rewrite_manifest(manifest, rows + [dict(rows[0], sample_id="copy")])
    with pytest.raises(PrepError, match="Duplicate PCM"):
        plan(manifest, audio)


def test_included_audio_must_exist_and_not_be_truncated(dataset):
    manifest, audio, rows = dataset
    path = audio / rows[0]["audio_path"]
    original = path.read_bytes()
    path.write_bytes(original[:-10])
    with pytest.raises(PrepError, match="truncated"):
        plan(manifest, audio)
    path.unlink()
    with pytest.raises(PrepError, match="Cannot read"):
        plan(manifest, audio)


def test_too_few_independent_units_is_not_random_clip_split(dataset, rewrite_manifest):
    manifest, audio, rows = dataset
    for row in rows:
        row["source_recording_id"] = "shared_recording"
    rewrite_manifest(manifest, rows)
    with pytest.raises(PrepError, match="independent units"):
        plan(manifest, audio)


def test_missing_classes_and_required_groups_fail(dataset, rewrite_manifest):
    manifest, audio, rows = dataset
    with pytest.raises(PrepError, match="required groups"):
        plan(manifest, audio, {"required_groups": {"train": ["not_recorded"], "val": [], "test": []}})
    rewrite_manifest(manifest, [r for r in rows if r["label"] == 0])
    with pytest.raises(PrepError, match="non-target"):
        plan(manifest, audio)


def test_label_is_explicit_not_guessed_from_phrase(dataset, rewrite_manifest):
    manifest, audio, rows = dataset
    rows[1]["expected_phrase"] = "你好小A"
    rewrite_manifest(manifest, rows)
    result = plan(manifest, audio)
    assert next(r for r in result["rows"] if r["sample_id"] == rows[1]["sample_id"])["label"] == -1


@pytest.mark.parametrize("label", [True, "0", 1, 0.0])
def test_labels_are_strict(dataset, rewrite_manifest, label):
    manifest, audio, rows = dataset
    rows[0]["label"] = label
    rewrite_manifest(manifest, rows)
    with pytest.raises(PrepError, match="label"):
        plan(manifest, audio)


def test_parent_cycles_and_unknown_parents_fail(dataset, rewrite_manifest):
    manifest, audio, rows = dataset
    rows[0]["parent_sample_id"] = rows[1]["sample_id"]
    rows[1]["parent_sample_id"] = rows[0]["sample_id"]
    rewrite_manifest(manifest, rows)
    with pytest.raises(PrepError, match="cycle"):
        plan(manifest, audio)
    rows[0]["parent_sample_id"] = "unknown"
    rewrite_manifest(manifest, rows)
    with pytest.raises(PrepError, match="unknown parent"):
        plan(manifest, audio)


def test_explicit_split_counts_and_assignments(dataset):
    manifest, audio, _ = dataset
    result = plan(manifest, audio, {"split_counts": {"train": 1, "val": 2, "test": 2}})
    assert [result["splits"][s]["independent_units"] for s in ("train", "val", "test")] == [1, 2, 2]
    assignments = {"person_0": "test", "person_1": "val", "person_2": "train",
                   "person_3": "train", "person_4": "train"}
    explicit = plan(manifest, audio, {"speaker_splits": assignments})
    assert all(r["split"] == assignments[r["speaker_id"]] for r in explicit["rows"])
    with pytest.raises(PrepError, match="sum"):
        plan(manifest, audio, {"split_counts": {"train": 4, "val": 1, "test": 1}})
    with pytest.raises(PrepError, match="exactly"):
        plan(manifest, audio, {"speaker_splits": {"person_0": "train"}})


def test_bad_configuration_is_not_silently_ignored(dataset):
    manifest, audio, _ = dataset
    for config in [{"sead": 3}, {"seed": True}, {"split_ratios": {"train": 1, "val": 0, "test": 0}}]:
        with pytest.raises(PrepError):
            plan(manifest, audio, config)


def test_alias_chain_validates_all_values_before_traversal(dataset):
    manifest, audio, _ = dataset
    with pytest.raises(PrepError, match="canonical speaker"):
        plan(manifest, audio, {"speaker_aliases": {"first": "second", "second": []}})


@pytest.mark.parametrize("text", ['{"label": 0, "label": -1}', '{"seed": NaN}',
                                  '{"speaker_aliases": {"a": "b", "a": "c"}}'])
def test_ambiguous_json_is_rejected(text):
    with pytest.raises(PrepError):
        load_json(text)


def test_output_is_readback_verified_and_no_overwrite(dataset, tmp_path):
    manifest, audio, _ = dataset
    output = tmp_path / "prepared"
    summary = prepare(manifest, audio, output)
    assert json.loads((output / "READY.json").read_text(encoding="utf-8")) == summary
    for name, expected in summary["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert not list(output.glob("*.wav"))
    assert not list(output.glob("*.tmp"))
    before = (output / "READY.json").read_bytes()
    with pytest.raises(PrepError, match="exists"):
        prepare(manifest, audio, output)
    assert (output / "READY.json").read_bytes() == before
    with pytest.raises(PrepError, match="outside"):
        prepare(manifest, audio, audio / "output")


def test_failed_write_never_publishes_ready(dataset, tmp_path, monkeypatch):
    manifest, audio, _ = dataset
    original = Path.write_bytes
    output = tmp_path / "failed"

    def fail(path, data):
        if path.name == "val.jsonl":
            raise OSError("simulated disk error")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(OSError, match="simulated"):
        prepare(manifest, audio, output)
    assert not (output / "READY.json").exists()
    assert (output / "FAILED.json").exists()


def test_cli_runs_without_old_workspace_or_optional_ml_dependencies(dataset, tmp_path):
    manifest, audio, _ = dataset
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", "-m", "easykws.prep", "run",
               "--manifest", str(manifest), "--audio-root", str(audio),
               "--output", str(tmp_path / "cli")]
    proc = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["purpose"] == "synthetic-test"
    again = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True,
                           encoding="utf-8", timeout=20)
    assert again.returncode == 2
