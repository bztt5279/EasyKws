"""Shared helpers for M6 tests; no fixtures, no pytest dependency."""
import hashlib
import json
from pathlib import Path

import numpy as np

SAMPLES = 48000
SRC = Path(__file__).resolve().parents[2] / "src"


def waveform_proxy():
    times = np.arange(SAMPLES, dtype=np.float64) / 16000
    waveform = np.zeros(SAMPLES, np.float32)
    waveform[16000:32000] = (.1 * np.sin(2 * np.pi * 733 * times[16000:32000])).astype(np.float32)
    return waveform


def rewrite_ready(root):
    """Recompute the package READY file hashes after a deliberate content change."""
    ready = json.loads((Path(root) / "READY.json").read_text(encoding="utf-8"))
    ready["files"] = {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
                      for name in ready["files"]}
    (Path(root) / "READY.json").write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_manifest_hashes(root):
    """Recompute manifest payload hashes after a deliberate payload change."""
    manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    payloads = {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
                for name in manifest["payloads"]}
    manifest["payloads"] = payloads
    manifest["model"]["file_sha256"] = payloads["model.pt"]
    manifest["cmvn"]["file_sha256"] = payloads["cmvn.json"]
    manifest["threshold"]["file_sha256"] = payloads["threshold.json"]
    (Path(root) / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inventory(root):
    return {str(path.relative_to(Path(root))): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(root).rglob("*") if path.is_file()}
