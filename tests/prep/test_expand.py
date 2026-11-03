"""Small expansion imports exercise decisions, not VC recognition quality."""
import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from easykws.prep import PrepError, v2, expand
from easykws.preprocess.core import read_prepared


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def import_case(dataset, tmp_path):
    old, audio, _ = dataset
    upgrade = tmp_path / "upgrade"
    v2.upgrade_v1(old, audio, upgrade)
    base_manifest = upgrade / "manifest.json"
    cfg = json.loads((upgrade / "config.json").read_text(encoding="utf-8"))
    roots = {"recordings": audio}
    prepared = tmp_path / "base"
    v2.prepare(base_manifest, roots, prepared, cfg)
    _, _, _, base_rows = read_prepared(prepared)
    source = next(r for r in base_rows if r["split"] == "train" and r["label"] == 0)
    original = json.loads(base_manifest.read_text(encoding="utf-8"))
    resource = {"resource_id": "voice_reference", "dataset_id": "recordings", "audio_path": source["audio_path"],
                "sha256": source["audio_sha256"], "kind": "reference-voice", "purpose": "synthetic-test",
                "split": "train", "speaker_id": source["speaker_id"], "source_sample_id": source["sample_id"]}
    additions = {"schema": v2.SCHEMA, "datasets": original["datasets"], "resources": [resource],
                 "selection": {"kind": "full", "reason": "", "available_count": 2}, "rows": []}
    qc = {}
    for i in range(2):
        row = {k: copy.deepcopy(source[k]) for k in v2.ROW_FIELDS}
        name = f"conversion_{i}"
        with wave.open(str(audio / source["audio_path"]), "rb") as wav:
            params, raw = wav.getparams(), wav.readframes(wav.getnframes())
        samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)
        with wave.open(str(audio / (name + ".wav")), "wb") as wav:
            wav.setparams(params)
            wav.writeframes(struct.pack("<" + "h" * len(samples), *(int(v * (0.4 + 0.2 * i)) for v in samples)))
        file_sha = hashlib.sha256((audio / (name + ".wav")).read_bytes()).hexdigest()
        row.update(sample_id=name, audio_path=name + ".wav", audio_sha256=file_sha, origin="vc",
                   purpose="training-data", parent_sample_ids=[source["sample_id"]], resource_ids=[resource["resource_id"]],
                   requested_split=None, generation={"engine": "fixture", "model": "fixture", "config_sha256": "a" * 64,
                                                    "seed": i, "status": "success"})
        additions["rows"].append(row)
        qc[name] = {"source_sha256": source["audio_sha256"], "output_sha256": file_sha,
                    "resource_sha256": {resource["resource_id"]: resource["sha256"]}, "status": "ok", "reasons": []}
    path = tmp_path / "candidates.json"
    save(path, additions)
    return prepared, path, roots, additions, qc


def test_import_is_consumable_freezes_base_and_retains_fixture_purpose(import_case, tmp_path):
    base, candidate, roots, doc, qc = import_case
    before = {p.name: p.read_bytes() for p in base.iterdir()}
    output = tmp_path / "expanded"
    report = expand.run(base, candidate, roots, output, qc=qc, expected_ids=[r["sample_id"] for r in doc["rows"]])
    assert report["counts"] == {"accepted": 2}
    assert report["coverage"] == "expected_ids_matched"
    assert report["purpose"] == "synthetic-test"
    ready, _, _, rows = read_prepared(output / "prepared")
    assert ready["included_count"] == 12
    assert all(r["split"] == "train" and r["effective_purpose"] == "synthetic-test"
               for r in rows if r["origin"] == "vc")
    assert before == {p.name: p.read_bytes() for p in base.iterdir()}
    with pytest.raises(PrepError, match="exists"):
        expand.run(base, candidate, roots, output, qc=qc)


@pytest.mark.parametrize("change,status", [("missing_qc", "pending"), ("flagged_qc", "flagged"),
                                         ("failed_qc", "failed"), ("review", "flagged"),
                                         ("failed_generation", "failed"), ("pending_generation", "pending"),
                                         ("missing_audio", "failed"), ("zero_audio", "failed")])
def test_unapproved_and_bad_outputs_are_reported_without_becoming_limits(import_case, change, status):
    base, path, roots, doc, qc = import_case
    row = doc["rows"][0]
    sid = row["sample_id"]
    if change == "missing_qc": del qc[sid]
    elif change.endswith("_qc"): qc[sid]["status"] = change.split("_")[0]
    elif change == "review": row["review_status"] = "flagged"
    elif change.endswith("_generation"): row["generation"]["status"] = change.split("_")[0]
    elif change == "missing_audio": (roots["recordings"] / row["audio_path"]).unlink()
    else:
        audio = roots["recordings"] / row["audio_path"]
        with wave.open(str(audio), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(bytes(6400))
        row["audio_sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
        qc[sid]["output_sha256"] = row["audio_sha256"]
    save(path, doc)
    result = expand.plan(base, path, roots, qc=qc)
    rejected = next(d for d in result["report"]["decisions"] if d["sample_id"] == sid)
    assert rejected["status"] == status
    assert len(result["report"]["accepted_ids"]) == 1
    assert "diagnostic-subset" not in result["report"]["data_flags"]


@pytest.mark.parametrize("field", ["source_sha256", "output_sha256", "resource_sha256"])
def test_qc_must_describe_this_source_output_and_reference(import_case, field):
    base, path, roots, _, qc = import_case
    qc["conversion_0"][field] = {} if field == "resource_sha256" else "0" * 64
    with pytest.raises(PrepError, match="hash"):
        expand.plan(base, path, roots, qc=qc)


def test_limits_are_deterministic_and_only_actual_truncation_marks_subset(import_case):
    base, path, roots, doc, qc = import_case
    first = expand.plan(base, path, roots, qc=qc, max_per_source=1, seed=9)
    doc["rows"].reverse()
    save(path, doc)
    second = expand.plan(base, path, roots, qc=qc, max_per_source=1, seed=9)
    assert first["report"]["accepted_ids"] == second["report"]["accepted_ids"]
    assert first["report"]["counts"] == {"accepted": 1, "limited": 1}
    assert "diagnostic-subset" in first["report"]["data_flags"]
    assert "diagnostic-subset" not in expand.plan(base, path, roots, qc=qc, max_total=3)["report"]["data_flags"]


def test_plan_ids_cannot_be_missing_or_duplicated(import_case):
    base, path, roots, doc, qc = import_case
    with pytest.raises(PrepError, match="coverage"):
        expand.plan(base, path, roots, qc=qc, expected_ids=["conversion_0", "missing"])
    doc["rows"].append(copy.deepcopy(doc["rows"][0]))
    doc["selection"]["available_count"] += 1
    save(path, doc)
    with pytest.raises(PrepError, match="Duplicate"):
        expand.plan(base, path, roots, qc=qc)


def test_duplicate_pcm_is_reported_before_limits(import_case):
    base, path, roots, doc, qc = import_case
    row = doc["rows"][1]
    row["audio_path"] = doc["rows"][0]["audio_path"]
    row["audio_sha256"] = doc["rows"][0]["audio_sha256"]
    qc[row["sample_id"]]["output_sha256"] = row["audio_sha256"]
    save(path, doc)
    assert expand.plan(base, path, roots, qc=qc)["report"]["counts"] == {"accepted": 1, "duplicate": 1}


def test_cli_expansion_from_unrelated_directory(import_case, tmp_path):
    base, path, roots, _, qc = import_case
    mapping, quality = tmp_path / "paths.json", tmp_path / "qc.json"
    save(mapping, {k: str(v) for k, v in roots.items()})
    save(quality, qc)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    proc = subprocess.run([sys.executable, "-B", "-m", "easykws.prep", "expand-run",
                           "--prepared", str(base), "--candidates", str(path), "--path-map", str(mapping),
                           "--qc", str(quality), "--output", str(tmp_path / "cli")], cwd=tmp_path,
                          env=env, capture_output=True, text=True, encoding="utf-8", timeout=40)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["counts"] == {"accepted": 2}


def test_reference_repacked_from_eval_audio_is_rejected(import_case):
    base, path, roots, doc, qc = import_case
    _, _, _, base_rows = read_prepared(base)
    source = next(r for r in base_rows if r["split"] == "val")
    raw = bytearray((roots["recordings"] / source["audio_path"]).read_bytes())
    raw.extend(b"JUNK\x04\x00\x00\x00test")
    raw[4:8] = (len(raw) - 8).to_bytes(4, "little")
    target = roots["recordings"] / "repacked.wav"
    target.write_bytes(raw)
    resource = doc["resources"][0]
    resource.update(audio_path=target.name, sha256=hashlib.sha256(raw).hexdigest(), split="external",
                    speaker_id=None, source_sample_id=None)
    for decision in qc.values():
        decision["resource_sha256"] = {resource["resource_id"]: resource["sha256"]}
    save(path, doc)
    with pytest.raises(PrepError, match="Resource PCM"):
        expand.plan(base, path, roots, qc=qc)


def test_public_adapter_merges_without_changing_base_splits(import_case, tmp_path):
    from easykws.prep.mobvoi import build, SETS
    from easykws.preprocess import plan as feature_plan
    base, path, roots, _, _ = import_case
    metadata, audio = tmp_path / "metadata", tmp_path / "public_audio"
    metadata.mkdir()
    audio.mkdir()
    for i, name in enumerate(SETS):
        uid = "official_" + str(i)
        save(metadata / (name + ".json"), [{"utt_id": uid, "keyword_id": -1 if name.startswith("n_") else 1,
                                          "speaker_id": None}])
        if name.startswith("n_"):
            with wave.open(str(audio / (uid + ".wav")), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(struct.pack("<3200h", *((100 + i * 100) if n % (3 + i) else 0 for n in range(3200))))
    doc, audit = build(metadata, audio, dataset_id="public", source="fixture", version="test",
                       license="test", keyword_labels={"-1": -1, "1": "exclude"}, purpose="synthetic-test")
    save(path, doc)
    paths = dict(roots, public=audio)
    output = tmp_path / "public_import"
    report = expand.run(base, path, paths, output, expected_ids=[entry["sample_id"] for entry in audit["entries"]])
    assert report["counts"] == {"accepted": 3, "flagged": 3}
    result = feature_plan(output / "prepared", None, path_map=paths)
    assert result["included_count"] == 13
    imported = [r for r in result["rows"] if r["origin"] == "public"]
    assert {r["split"] for r in imported} == {"train", "val", "test"}
    assert all(r["actual_speaker_id"] is None and r["label"] == -1 for r in imported)
