import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "synthetic_example", ROOT / "examples" / "make_synthetic_recordings.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.fixture
def dataset(tmp_path):
    manifest = example.create(tmp_path / "recordings")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    return manifest, manifest.parent / "audio", rows


@pytest.fixture
def rewrite_manifest():
    def write(path, rows):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
    return write
