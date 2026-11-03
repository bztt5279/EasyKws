"""Independent integrity regressions for internally inconsistent prepared bundles."""
import hashlib
import json
from pathlib import Path

import pytest

from easykws.prep import prepare
from easykws.preprocess import PreprocessError, plan, run
from easykws.preprocess import core


@pytest.fixture
def bundle(dataset, tmp_path):
    manifest, audio, _ = dataset
    prepared = tmp_path / "prepared"
    prepare(manifest, audio, prepared)
    return prepared, audio


def rewrite_index(prepared, change):
    index = prepared / "train.jsonl"
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    change(rows[0])
    index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    ready = json.loads((prepared / "READY.json").read_text(encoding="utf-8"))
    ready["files"]["train.jsonl"] = hashlib.sha256(index.read_bytes()).hexdigest()
    (prepared / "READY.json").write_text(json.dumps(ready), encoding="utf-8")


@pytest.mark.parametrize("key,value", [("audio_pcm_sha256", []), ("audio_file_sha256", "short"),
                                      ("channels", True), ("sample_width", 2.0),
                                      ("num_samples", 3200.0), ("duration_sec", True)])
def test_invalid_metadata_types_fail_even_with_updated_file_hashes(bundle, key, value):
    prepared, audio = bundle
    rewrite_index(prepared, lambda row: row.update({key: value}))
    with pytest.raises(PreprocessError):
        plan(prepared, audio)


@pytest.mark.parametrize("kind", ["missing_identity", "invalid_identity", "purpose_downgrade"])
def test_ready_requires_complete_source_identity_and_purpose(bundle, kind):
    prepared, audio = bundle
    path = prepared / "READY.json"
    ready = json.loads(path.read_text(encoding="utf-8"))
    if kind == "missing_identity":
        del ready["manifest_sha256"]
    elif kind == "invalid_identity":
        ready["manifest_sha256"] = 42
    else:
        ready["purpose"] = "recorded-data"
    path.write_text(json.dumps(ready), encoding="utf-8")
    with pytest.raises(PreprocessError):
        plan(prepared, audio)


def test_changed_code_identity_cannot_publish_completed_run(bundle, tmp_path, monkeypatch):
    prepared, audio = bundle
    identities = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(core, "code_identity", lambda: next(identities))
    output = tmp_path / "changed_code"
    with pytest.raises(PreprocessError, match="code changed"):
        run(prepared, audio, output)
    assert not (output / "READY.json").exists()
    assert (output / "FAILED.json").exists()


def test_final_source_mutation_prevents_ready(bundle, tmp_path, monkeypatch):
    prepared, audio = bundle
    original = Path.write_bytes

    def modify_source(path, payload):
        result = original(path, payload)
        if path.name == "plan.json":
            with (prepared / "excluded.jsonl").open("ab") as stream:
                stream.write(b" ")
        return result

    monkeypatch.setattr(Path, "write_bytes", modify_source)
    output = tmp_path / "changed_input"
    with pytest.raises(PreprocessError, match="hash mismatch"):
        run(prepared, audio, output)
    assert not (output / "READY.json").exists()
