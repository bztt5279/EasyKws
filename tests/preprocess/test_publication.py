"""A source changing after feature extraction must not publish a ready run."""
import pytest

from easykws.prep import prepare
from easykws.preprocess import frontend
from easykws.preprocess.core import run
from easykws.preprocess.errors import PreprocessError


def test_source_changed_after_read_prevents_ready(dataset, tmp_path, monkeypatch):
    manifest, audio, rows = dataset
    prepared = tmp_path / "prepared"
    prepare(manifest, audio, prepared)
    output = tmp_path / "features"
    original = frontend.extract_fbank
    first = min(rows, key=lambda row: row["sample_id"])
    path = audio / first["audio_path"]
    changed = False

    def extract(waveform):
        nonlocal changed
        value = original(waveform)
        if not changed:
            raw = bytearray(path.read_bytes())
            raw[-2] ^= 1
            path.write_bytes(raw)
            changed = True
        return value

    monkeypatch.setattr(frontend, "extract_fbank", extract)
    with pytest.raises(PreprocessError, match="hash mismatch"):
        run(prepared, audio, output)
    assert changed
    assert not (output / "READY.json").exists()
    assert (output / "FAILED.json").exists()
