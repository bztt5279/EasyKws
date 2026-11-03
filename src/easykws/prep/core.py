"""Prepare user-owned PCM WAV manifests without editing or copying audio."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import io
import json
import math
import ntpath
from pathlib import Path, PurePosixPath
import re
import wave

SPLITS = ("train", "val", "test")
STATUSES = {"pending", "pass", "manual", "flagged"}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
DEFAULT_CONFIG = {
    "schema": "easykws-prep-config/v1",
    "seed": 0,
    "speaker_aliases": {},
    "exclude_sample_ids": [],
    "include_review_status": ["pass", "manual"],
    "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
    "split_counts": None,
    "speaker_splits": {},
    "require_both_labels": True,
    "required_groups": {"train": [], "val": [], "test": []},
}


class PrepError(ValueError):
    """Input or split constraints do not satisfy the preparation contract."""


def load_json(text):
    """Reject ambiguous duplicate keys and nonstandard numeric constants."""
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PrepError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise PrepError(f"Nonstandard JSON constant: {value}")

    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def _require(condition, message):
    if not condition:
        raise PrepError(message)


def _identifier(value, name):
    _require(isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None,
             f"{name} must be a nonempty ASCII identifier (up to 128 characters)")
    return value


def _digest(data: bytes):
    return hashlib.sha256(data).hexdigest()


def _text(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _canonical(speaker, aliases):
    seen = set()
    while speaker in aliases:
        _require(speaker not in seen, f"speaker_aliases contains a cycle at {speaker}")
        seen.add(speaker)
        speaker = aliases[speaker]
    return speaker


def _config(raw):
    _require(raw is None or isinstance(raw, dict), "config must be an object")
    raw = raw or {}
    _require(not set(raw) - set(DEFAULT_CONFIG), f"Unknown config keys: {sorted(set(raw)-set(DEFAULT_CONFIG))}")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(copy.deepcopy(raw))
    _require(cfg["schema"] == DEFAULT_CONFIG["schema"], "Unsupported config schema")
    _require(type(cfg["seed"]) is int and cfg["seed"] >= 0, "seed must be a nonnegative integer")
    _require(type(cfg["require_both_labels"]) is bool, "require_both_labels must be boolean")
    aliases = cfg["speaker_aliases"]
    _require(isinstance(aliases, dict), "speaker_aliases must be an object")
    for key, value in aliases.items():
        _identifier(key, "alias"); _identifier(value, "canonical speaker")
    for key in aliases:
        _canonical(key, aliases)
    for key in ("exclude_sample_ids", "include_review_status"):
        values = cfg[key]
        _require(isinstance(values, list) and all(isinstance(v, str) for v in values),
                 f"{key} must be an array of strings")
        _require(len(values) == len(set(values)), f"{key} contains duplicates")
    _require(bool(cfg["include_review_status"]) and set(cfg["include_review_status"]) <= STATUSES,
             "include_review_status must contain known statuses")
    groups = cfg["required_groups"]
    _require(isinstance(groups, dict) and set(groups) == set(SPLITS),
             "required_groups must contain train, val, test")
    for values in groups.values():
        _require(isinstance(values, list) and all(isinstance(v, str) and v for v in values),
                 "required_groups values must be arrays of nonempty strings")
    assignments = cfg["speaker_splits"]
    _require(isinstance(assignments, dict), "speaker_splits must be an object")
    for key, value in assignments.items():
        _identifier(key, "speaker_splits key")
        _require(value in SPLITS, f"Unknown assigned split for {key}")
        _require(_canonical(key, aliases) == key, f"speaker_splits must use canonical speaker IDs: {key}")
    counts = cfg["split_counts"]
    ratios = cfg["split_ratios"]
    if counts is not None:
        _require(isinstance(counts, dict) and set(counts) == set(SPLITS),
                 "split_counts must contain train, val, test")
        _require(all(type(v) is int and v > 0 for v in counts.values()),
                 "split_counts must be positive integers")
    _require(isinstance(ratios, dict) and set(ratios) == set(SPLITS),
             "split_ratios must contain train, val, test")
    _require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in ratios.values()),
             "split_ratios must be finite positive numbers")
    _require(abs(sum(ratios.values()) - 1.0) < 1e-9, "split_ratios must sum to one")
    _require(not (assignments and counts is not None),
             "Use speaker_splits or split_counts, not both")
    return cfg


def _audio_path(raw, root):
    _require(isinstance(raw, str) and raw, "audio_path must be a relative path")
    path = PurePosixPath(raw)
    _require(not path.is_absolute() and not ntpath.splitdrive(raw)[0]
             and "\\" not in raw and ".." not in path.parts,
             "audio_path must be relative, use '/', and not contain '..'")
    _require(path.suffix.lower() == ".wav", "Only uncompressed PCM WAV inputs are supported")
    resolved = (root / path).resolve()
    _require(resolved.is_relative_to(root), f"audio_path escapes audio root: {raw}")
    return resolved


def _audio_info(path):
    try:
        data = path.read_bytes()
        with wave.open(io.BytesIO(data), "rb") as wav:
            rate, channels, width, frames = (
                wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes())
            _require(wav.getcomptype() == "NONE", "WAV must be uncompressed PCM")
            _require(rate > 0 and channels > 0 and frames > 0 and width in (1, 2, 3, 4),
                     "WAV has invalid or empty audio dimensions")
            pcm = wav.readframes(frames)
            _require(len(pcm) == frames * channels * width, "WAV payload is truncated")
    except (OSError, wave.Error, EOFError) as exc:
        raise PrepError(f"Cannot read PCM WAV {path.name}: {exc}") from exc
    identity = f"{rate}/{channels}/{width}/".encode("ascii") + pcm
    return {"audio_file_sha256": _digest(data), "audio_pcm_sha256": _digest(identity),
            "sample_rate": rate, "channels": channels, "sample_width": width,
            "num_samples": frames, "duration_sec": frames / rate}


def _load_rows(manifest, audio_root, cfg):
    try:
        raw_bytes = manifest.read_bytes()
        lines = raw_bytes.decode("utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PrepError(f"Cannot read manifest: {exc}") from exc
    rows, seen, content = [], set(), {}
    required = {"sample_id", "speaker_id", "source_recording_id", "audio_path",
                "label", "group", "expected_phrase", "review_status", "origin"}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = load_json(line)
        except (ValueError, TypeError) as exc:
            raise PrepError(f"Manifest line {number} is not valid JSON") from exc
        _require(isinstance(row, dict), f"Manifest line {number} must be an object")
        _require(required <= row.keys(), f"Manifest line {number} missing {sorted(required-row.keys())}")
        _require(not row.keys() - (required | {"parent_sample_id"}),
                 f"Manifest line {number} has unknown fields")
        row = dict(row)
        for key in ("sample_id", "speaker_id", "source_recording_id"):
            _identifier(row[key], f"line {number}: {key}")
        sid = row["sample_id"]
        _require(sid not in seen, f"Duplicate sample_id: {sid}")
        seen.add(sid)
        _require(type(row["label"]) is int and row["label"] in (0, -1),
                 f"{sid}: label must be integer 0 (target) or -1 (non-target)")
        _require(isinstance(row["group"], str) and bool(row["group"].strip()), f"{sid}: group is empty")
        _require(isinstance(row["expected_phrase"], str), f"{sid}: expected_phrase must be a string")
        _require(row["label"] != 0 or bool(row["expected_phrase"].strip()),
                 f"{sid}: target requires expected_phrase")
        _require(isinstance(row["review_status"], str) and row["review_status"] in STATUSES,
                 f"{sid}: unknown review_status")
        _require(isinstance(row["origin"], str) and row["origin"] in {"recorded", "synthetic"},
                 f"{sid}: origin must be recorded or synthetic")
        if row.get("parent_sample_id") is not None:
            _identifier(row["parent_sample_id"], f"{sid}: parent_sample_id")
        path = _audio_path(row["audio_path"], audio_root)
        reasons = []
        if sid in cfg["exclude_sample_ids"]:
            reasons.append("explicit_exclusion")
        if row["review_status"] not in cfg["include_review_status"]:
            reasons.append("review_status_not_included")
        row.update(actual_speaker_id=_canonical(row["speaker_id"], cfg["speaker_aliases"]),
                   included=not reasons, exclusion_reasons=reasons, split=None)
        if row["included"]:
            row.update(_audio_info(path))
            sha = row["audio_pcm_sha256"]
            _require(sha not in content,
                     f"Duplicate PCM content: {sid} and {content.get(sha)}; deduplicate explicitly")
            content[sha] = sid
        rows.append(row)
    _require(bool(rows), "Manifest is empty")
    _require(not set(cfg["exclude_sample_ids"]) - seen, "exclude_sample_ids references unknown samples")
    by_id = {r["sample_id"]: r for r in rows}
    for row in rows:
        current, ancestry = row, set()
        while current.get("parent_sample_id") is not None:
            key = current["sample_id"]
            _require(key not in ancestry, f"Parent relationship cycle at {key}")
            ancestry.add(key)
            parent = current["parent_sample_id"]
            _require(parent in by_id, f"{key}: unknown parent_sample_id {parent}")
            current = by_id[parent]
    _require(any(r["included"] for r in rows), "All samples were excluded")
    return sorted(rows, key=lambda r: r["sample_id"]), _digest(raw_bytes)


def _units(rows):
    """Connected components enforce speaker, original recording, and parent isolation."""
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[max(ra, rb)] = min(ra, rb)

    for row in rows:
        node = ("sample", row["sample_id"])
        union(node, ("speaker", row["actual_speaker_id"]))
        union(node, ("source", row["source_recording_id"]))
        if row.get("parent_sample_id"):
            union(node, ("sample", row["parent_sample_id"]))
    groups = defaultdict(list)
    for row in rows:
        if row["included"]:
            groups[find(("sample", row["sample_id"]))].append(row)
    return list(groups.values())


def _split(rows, cfg):
    groups = _units(rows)
    _require(len(groups) >= len(SPLITS),
             f"Need at least three independent units after speaker/source linking; got {len(groups)}")
    speakers = {r["actual_speaker_id"] for r in rows if r["included"]}
    explicit = cfg["speaker_splits"]
    if explicit:
        _require(set(explicit) == speakers, "speaker_splits must cover exactly the included canonical speakers")
        for group in groups:
            requested = {explicit[r["actual_speaker_id"]] for r in group}
            _require(len(requested) == 1, "speaker_splits separates linked recordings or parents")
            for row in group:
                row["split"] = next(iter(requested))
    else:
        counts = cfg["split_counts"]
        if counts is None:
            raw = {key: len(groups) * cfg["split_ratios"][key] for key in SPLITS}
            counts = {key: math.floor(raw[key]) for key in SPLITS}
            remaining = len(groups) - sum(counts.values())
            order = sorted(SPLITS, key=lambda key: (-(raw[key] - counts[key]), SPLITS.index(key)))
            for key in order[:remaining]:
                counts[key] += 1
        _require(sum(counts.values()) == len(groups), "split_counts must sum to independent unit count")
        _require(all(n > 0 for n in counts.values()), "Split has zero units; use explicit counts or more speakers")

        def key(group):
            identity = sorted({r["actual_speaker_id"] for r in group})
            return _digest(json.dumps([cfg["seed"], identity], separators=(",", ":")).encode("utf-8"))

        groups.sort(key=key)
        offset = 0
        for split in SPLITS:
            for group in groups[offset:offset + counts[split]]:
                for row in group:
                    row["split"] = split
            offset += counts[split]
    stats = {}
    for split in SPLITS:
        selected = [r for r in rows if r["included"] and r["split"] == split]
        _require(bool(selected), f"{split} is empty")
        labels = Counter(r["label"] for r in selected)
        if cfg["require_both_labels"]:
            _require(set(labels) == {0, -1},
                     f"{split} lacks target or non-target examples; change seed or explicit speaker_splits")
        present = {r["group"] for r in selected}
        _require(set(cfg["required_groups"][split]) <= present,
                 f"{split} lacks required groups; change seed or explicit speaker_splits")
        stats[split] = {"samples": len(selected), "labels": {str(k): v for k, v in labels.items()},
                        "speakers": sorted({r["actual_speaker_id"] for r in selected}),
                        "groups": sorted(present),
                        "independent_units": sum(g[0]["split"] == split for g in groups)}
    return stats


def plan(manifest, audio_root, config=None):
    """Read and validate inputs; return deterministic indices without writing anything."""
    cfg = _config(config)
    manifest, audio_root = Path(manifest).resolve(), Path(audio_root).resolve()
    _require(audio_root.is_dir(), "audio_root must be an existing directory")
    rows, manifest_sha = _load_rows(manifest, audio_root, cfg)
    stats = _split(rows, cfg)
    included = [r for r in rows if r["included"]]
    return {"schema": "easykws-prepared/v1", "config": cfg, "manifest_sha256": manifest_sha,
            "config_sha256": _digest(_text(cfg).encode("utf-8")),
            "purpose": "synthetic-test" if any(r["origin"] == "synthetic" for r in included) else "recorded-data",
            "input_count": len(rows), "included_count": len(included),
            "excluded_count": len(rows) - len(included), "splits": stats, "rows": rows}


def prepare(manifest, audio_root, output, config=None):
    """Write a new index directory. READY.json is written last; never overwrite a run."""
    output, audio_root = Path(output).resolve(), Path(audio_root).resolve()
    _require(not output.exists(), "Output already exists; use a new directory")
    _require(not output.is_relative_to(audio_root), "Output must be outside audio_root")
    result = plan(manifest, audio_root, config)
    payloads = {}
    for split in SPLITS:
        selected = [r for r in result["rows"] if r["included"] and r["split"] == split]
        payloads[split + ".jsonl"] = "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in selected)
    payloads["excluded.jsonl"] = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for r in result["rows"] if not r["included"])
    payloads["config.json"] = _text(result["config"])
    summary = {key: value for key, value in result.items() if key not in {"rows", "config"}}
    summary["files"] = {name: _digest(body.encode("utf-8")) for name, body in payloads.items()}
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, body in payloads.items():
            (output / name).write_bytes(body.encode("utf-8"))
        for name, expected in summary["files"].items():
            _require(_digest((output / name).read_bytes()) == expected, f"Written file hash mismatch: {name}")
        ready_tmp = output / ".READY.json.tmp"
        ready_tmp.write_bytes(_text(summary).encode("utf-8"))
        ready_tmp.replace(output / "READY.json")
    except BaseException as exc:
        # The incomplete directory remains for diagnosis; it must not be treated as ready.
        (output / "FAILED.json").write_bytes(_text({"error": type(exc).__name__}).encode("utf-8"))
        raise
    return summary
