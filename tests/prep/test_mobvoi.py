"""Tiny synthetic metadata and PCM fixtures; no public speech is bundled."""
import hashlib
import json
import struct
import wave

import pytest

from easykws.prep import PrepError, core, mobvoi, v2


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def case(tmp_path):
    metadata, audio = tmp_path / "metadata", tmp_path / "audio"
    metadata.mkdir()
    audio.mkdir()
    for index, name in enumerate(mobvoi.SETS):
        record = dict(utt_id=f"utterance_{index}", keyword_id=7 if index < 3 else 8,
                      speaker_id=["person-a", None, 123, "person-d", "名字", ""][index])
        write_json(metadata / (name + ".json"), [record])
        if index < 3:
            with wave.open(str(audio / (record["utt_id"] + ".wav")), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(struct.pack("<16h", *[index * 20 + n for n in range(16)]))
    return metadata, audio


def build(case, **kwargs):
    options = dict(dataset_id="public-fixture", source="synthetic-local-metadata",
                   version="fixture-v1", license="test-only", keyword_labels={"7": -1, "8": "exclude"})
    options.update(kwargs)
    return mobvoi.build(*case, **options)


def mutate(case, name, change):
    path = case[0] / (name + ".json")
    records = json.loads(path.read_text(encoding="utf-8"))
    change(records)
    write_json(path, records)


def test_full_local_metadata_and_official_splits_survive(case, tmp_path):
    document, report = build(case)
    assert document["schema"] == v2.SCHEMA
    assert document["datasets"][0]["split_policy"] == "official"
    assert document["resources"] == []
    assert document["selection"] == dict(kind="full", reason="", available_count=6)
    assert len(document["rows"]) == len(report["entries"]) == 6
    assert report["original_uids"] == [f"utterance_{i}" for i in range(6)]
    assert report["included_uids"] == [f"utterance_{i}" for i in range(3)]
    assert report["excluded_uids"] == [f"utterance_{i}" for i in range(3, 6)]
    for row, entry in zip(document["rows"], report["entries"]):
        assert set(row) == v2.ROW_FIELDS
        assert row["source_recording_id"] == entry["utt_id"]
        assert entry["metadata_sha256"] == hashlib.sha256(
            (case[0] / entry["metadata_file"]).read_bytes()).hexdigest()
        assert entry["official_set"] == entry["metadata_file"][:-5]
        assert row["requested_split"] == {"dev": "val"}.get(entry["official_split"], entry["official_split"])
        assert row["origin"] == "public" and row["generation"] is None
        assert row["parent_sample_ids"] == row["resource_ids"] == []
        if entry["included"]:
            assert row["audio_sha256"] == hashlib.sha256((case[1] / row["audio_path"]).read_bytes()).hexdigest()
            assert entry["audio"]["num_samples"] == 16
        else:
            assert row["audio_sha256"] is None and row["review_status"] == "flagged"
            assert not (case[1] / row["audio_path"]).exists()
    assert document["rows"][0]["speaker_id"] == "person-a"
    assert document["rows"][1]["speaker_id"] is None
    assert document["rows"][2]["speaker_id"].startswith("speaker-hash-")
    assert report["entries"][2]["original_speaker_id"] == 123
    assert document["rows"][4]["speaker_id"] is not None
    assert document["rows"][5]["speaker_id"] is not None
    manifest = tmp_path / "manifest.json"
    write_json(manifest, document)
    result = v2.plan(manifest, {"public-fixture": case[1]}, {"require_both_labels": False})
    assert result["included_count"] == 3 and result["excluded_count"] == 3
    assert [r["split"] for r in result["rows"] if r["included"]].count("val") == 1
    assert build(case) == (document, report)


@pytest.mark.parametrize("labels", [{}, {"7": -1}, {7: -1, "8": "exclude"},
                                      {"7": 0, "8": "exclude"}, {"7": True}, {"7": -1.0}])
def test_explicit_mapping_required_before_audio_access(case, monkeypatch, labels):
    monkeypatch.setattr(core, "_audio_info", lambda _: pytest.fail("audio read before metadata validation"))
    with pytest.raises(PrepError):
        build(case, keyword_labels=labels)


@pytest.mark.parametrize("uid", ["../outside", "nested/name", "nested\\name", "C:drive", ".", "..",
                                  "CON", "name.wav", "name.WaV", "name.", "x" * 129])
def test_unsafe_utt_id_rejected(case, uid):
    mutate(case, "n_train", lambda rows: rows[0].update(utt_id=uid))
    with pytest.raises(PrepError):
        build(case)


@pytest.mark.parametrize("within", [False, True])
def test_global_uid_duplicates_rejected(case, within):
    if within:
        mutate(case, "n_train", lambda rows: rows.append(dict(rows[0])))
    else:
        mutate(case, "p_test", lambda rows: rows[0].update(utt_id="utterance_0"))
    with pytest.raises(PrepError, match="Duplicate utt_id"):
        build(case)


@pytest.mark.parametrize("damage", ["missing", "truncated", "riff-truncated", "compressed", "invalid"])
def test_included_wav_is_readable_complete_and_at_exact_path(case, damage):
    path = case[1] / "utterance_0.wav"
    raw = path.read_bytes()
    if damage == "missing":
        nested = case[1] / "nested"
        nested.mkdir()
        path.replace(nested / path.name)
    else:
        changed = {"truncated": raw[:-2], "riff-truncated": raw[:4] + (len(raw) + 20).to_bytes(4, "little") + raw[8:],
                   "compressed": raw[:20] + b"\x03\x00" + raw[22:], "invalid": b"not a wave"}[damage]
        path.write_bytes(changed)
    with pytest.raises(PrepError):
        build(case)


@pytest.mark.parametrize("bad", [None, {}, ["not-an-object"], [{"utt_id": "valid", "keyword_id": True}]])
def test_invalid_metadata_rejected(case, bad):
    write_json(case[0] / "p_test.json", bad)
    with pytest.raises(PrepError):
        build(case)


def test_all_six_files_required_and_hashes_track_changes(case):
    _, original = build(case)
    mutate(case, "p_test", lambda rows: rows[0].update(note="fixture annotation"))
    _, changed = build(case)
    assert original["metadata_files"]["p_test.json"]["sha256"] != changed["metadata_files"]["p_test.json"]["sha256"]
    (case[0] / "p_test.json").unlink()
    with pytest.raises(PrepError, match="p_test.json"):
        build(case)


@pytest.mark.parametrize("raw", ['[{"utt_id":"a","utt_id":"b","keyword_id":7}]', '[NaN]', '['])
def test_ambiguous_or_malformed_json_rejected(case, raw):
    (case[0] / "p_test.json").write_text(raw, encoding="utf-8")
    with pytest.raises(PrepError):
        build(case)


def test_long_dataset_id_does_not_lengthen_sample_ids(case):
    document, _ = build(case, dataset_id="d" * 128, purpose="synthetic-test")
    assert all(len(row["sample_id"]) <= 128 and row["purpose"] == "synthetic-test" for row in document["rows"])
