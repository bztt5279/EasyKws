"""Consume a verified prep bundle and publish portable raw feature artifacts."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import re

from ..prep import core as prep
from ..prep import v2
from .errors import PreprocessError

DEFAULT_CONFIG = {"schema": "easykws-preprocess-config/v1", "channel_policy": "reject", "resample": False}
PREP_FILES = {"train.jsonl", "val.jsonl", "test.jsonl", "excluded.jsonl", "config.json"}
READY_FIELDS = {"schema", "manifest_sha256", "config_sha256", "purpose", "input_count", "included_count",
                "excluded_count", "splits", "files"}
HEX = re.compile(r"[a-f0-9]{64}\Z")
BASE_FIELDS = {"sample_id", "speaker_id", "source_recording_id", "audio_path", "label", "group",
               "expected_phrase", "review_status", "origin", "actual_speaker_id", "included",
               "exclusion_reasons", "split"}
AUDIO_FIELDS = {"audio_file_sha256", "audio_pcm_sha256", "sample_rate", "channels", "sample_width",
                "num_samples", "duration_sec"}


def require(condition, message):
    if not condition:
        raise PreprocessError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def configure(config):
    require(config is None or isinstance(config, dict), "Config must be an object")
    config = config or {}
    require(not config.keys() - (DEFAULT_CONFIG.keys() | {"augmentation"}), "Unknown preprocessing configuration field")
    result = {**DEFAULT_CONFIG, **config}
    require(result["schema"] == DEFAULT_CONFIG["schema"], "Unsupported preprocessing config schema")
    require(result["channel_policy"] in ("reject", "mean"), "channel_policy must be reject or mean")
    require(type(result["resample"]) is bool, "resample must be boolean")
    if "augmentation" in result:
        from .augmentation_pipeline import configure as configure_augmentation

        result["augmentation"] = configure_augmentation(result["augmentation"])
    return result


def read_json(data, context):
    try:
        return prep.load_json(data.decode("utf-8-sig"))
    except (UnicodeError, ValueError) as exc:
        raise PreprocessError(f"Invalid JSON in {context}") from exc


def source_file(root, name):
    # Bundle names come from a fixed schema, never from caller-supplied paths.
    path = root / name
    require(path.resolve().is_relative_to(root), f"Bundle file escapes prepared root: {name}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PreprocessError(f"Cannot read prepared file: {name}") from exc


def read_prepared(root):
    root = Path(root).resolve()
    require(root.is_dir(), "Prepared directory does not exist")
    require(not (root / "FAILED.json").exists(), "Prepared directory has a failure marker")
    ready_bytes = source_file(root, "READY.json")
    ready = read_json(ready_bytes, "READY.json")
    if isinstance(ready, dict) and ready.get("schema") == v2.PREP_SCHEMA:
        return read_prepared_v2(root, ready, ready_bytes)
    require(isinstance(ready, dict) and ready.get("schema") == "easykws-prepared/v1", "Unsupported prepared schema")
    require(set(ready) == READY_FIELDS, "Prepared READY fields are incomplete or unexpected")
    for name in ("manifest_sha256", "config_sha256"):
        require(isinstance(ready[name], str) and HEX.fullmatch(ready[name]), "Invalid prepared identity hash")
    hashes = ready.get("files")
    require(isinstance(hashes, dict) and set(hashes) == PREP_FILES, "Prepared file inventory is incomplete or unexpected")
    payloads = {}
    for name in sorted(PREP_FILES):
        expected = hashes[name]
        require(isinstance(expected, str) and HEX.fullmatch(expected), "Invalid prepared file hash")
        payloads[name] = source_file(root, name)
        require(digest(payloads[name]) == expected, f"Prepared file hash mismatch: {name}")
    config = read_json(payloads["config.json"], "config.json")
    try:
        require(prep._config(config) == config, "Prepared config must contain expanded defaults")
    except prep.PrepError as exc:
        raise PreprocessError(str(exc)) from exc
    require(digest(payloads["config.json"]) == ready.get("config_sha256"), "Prepared config identity mismatch")
    rows = []
    for split in (*prep.SPLITS, "excluded"):
        filename = split + ".jsonl"
        for number, line in enumerate(payloads[filename].splitlines(), 1):
            if not line.strip():
                continue
            row = read_json(line, f"{filename}:{number}")
            require(isinstance(row, dict), "Prepared row must be an object")
            required = BASE_FIELDS | (AUDIO_FIELDS if split != "excluded" else set())
            require(required <= row.keys() <= required | {"parent_sample_id"}, "Unexpected prepared row fields")
            require(type(row["included"]) is bool and row["included"] == (split != "excluded"), "Row inclusion mismatch")
            require(row["split"] == (None if split == "excluded" else split), "Row split mismatch")
            rows.append(row)
    validate_rows(rows, config, ready)
    return ready, digest(ready_bytes), payloads, sorted(rows, key=lambda row: row["sample_id"])


def read_prepared_v2(root, ready, ready_bytes):
    """Rebuild v2 semantics, not merely trust matching inventory hashes."""
    require(set(ready) == READY_FIELDS | {"data_flags", "selection"}, "Unexpected v2 READY fields")
    for name in ("manifest_sha256", "config_sha256"):
        require(isinstance(ready[name], str) and HEX.fullmatch(ready[name]), "Invalid v2 identity hash")
    hashes = ready["files"]
    require(isinstance(hashes, dict) and set(hashes) == v2.PREP_FILES, "Unexpected v2 file inventory")
    payloads = {}
    for name in sorted(v2.PREP_FILES):
        expected = hashes[name]
        require(isinstance(expected, str) and HEX.fullmatch(expected), "Invalid v2 file hash")
        payloads[name] = source_file(root, name)
        require(digest(payloads[name]) == expected, f"Prepared file hash mismatch: {name}")
    require(digest(payloads["manifest.json"]) == ready["manifest_sha256"], "Manifest identity mismatch")
    require(digest(payloads["config.json"]) == ready["config_sha256"], "Config identity mismatch")
    document = read_json(payloads["manifest.json"], "manifest.json")
    config = read_json(payloads["config.json"], "config.json")
    rows, by_id = [], {}
    for split in (*prep.SPLITS, "excluded"):
        for line in payloads[split + ".jsonl"].splitlines():
            if not line.strip():
                continue
            row = read_json(line, split + ".jsonl")
            required = v2.ROW_FIELDS | v2.DERIVED_FIELDS | (AUDIO_FIELDS if split != "excluded" else set())
            require(isinstance(row, dict) and set(row) == required, "Unexpected v2 row fields")
            require(isinstance(row["sample_id"], str), "Invalid sample_id")
            require(row["sample_id"] not in by_id, "Duplicate prepared sample ID")
            require(type(row["included"]) is bool and row["included"] == (split != "excluded"), "Inclusion mismatch")
            require(row["split"] == (None if split == "excluded" else split), "Split mismatch")
            rows.append(row)
            by_id[row["sample_id"]] = row

    def metadata(row):
        require(row["sample_id"] in by_id, "Declared sample missing from prepared indices")
        actual = by_id[row["sample_id"]]
        require(AUDIO_FIELDS <= actual.keys(), "Included sample lacks audio metadata")
        return {key: actual[key] for key in AUDIO_FIELDS}

    try:
        require(v2.configure(config) == config, "V2 config must contain expanded defaults")
        reconstructed = v2.evaluate(document, config, metadata, lambda resource: None)
    except prep.PrepError as exc:
        raise PreprocessError(str(exc)) from exc
    rows.sort(key=lambda row: row["sample_id"])
    require(encode(reconstructed["rows"]) == encode(rows), "V2 row/lineage/purpose contract mismatch")
    reconstructed["manifest_sha256"] = digest(payloads["manifest.json"])
    expected = {key: value for key, value in reconstructed.items() if key not in {"config", "rows"}}
    actual = {key: value for key, value in ready.items() if key != "files"}
    require(encode(actual) == encode(expected), "V2 READY summary or purpose mismatch")
    return ready, digest(ready_bytes), payloads, rows


def input_roots(ready, payloads, audio_root, path_map):
    if ready["schema"] == v2.PREP_SCHEMA:
        require(audio_root is None and path_map is not None, "V2 requires path_map instead of audio_root")
        document = read_json(payloads["manifest.json"], "manifest.json")
        try:
            roots = v2.resolve_roots(document, path_map)
            v2.check_paths(document, roots)
        except prep.PrepError as exc:
            raise PreprocessError(str(exc)) from exc
        return roots
    require(path_map is None and audio_root is not None, "V1 requires audio_root instead of path_map")
    root = Path(audio_root).resolve()
    require(root.is_dir(), "Audio root does not exist")
    return {"__legacy__": root}


def row_root(row, roots):
    return roots[row.get("dataset_id", "__legacy__")]


def verify_resources(ready, payloads, rows, roots):
    if ready["schema"] != v2.PREP_SCHEMA:
        return
    document = read_json(payloads["manifest.json"], "manifest.json")
    resources = {r["resource_id"]: r for r in document["resources"]}
    used = {key for row in rows if row["included"] for key in row["resource_ids"]}
    try:
        for key in sorted(used):
            v2.check_resource(resources[key], roots)
    except prep.PrepError as exc:
        raise PreprocessError(str(exc)) from exc


def validate_rows(rows, config, ready):
    by_id = {}
    for row in rows:
        try:
            for key in ("sample_id", "speaker_id", "source_recording_id", "actual_speaker_id"):
                prep._identifier(row[key], key)
            require(row["actual_speaker_id"] == prep._canonical(row["speaker_id"], config["speaker_aliases"]),
                    "Canonical speaker does not match prep configuration")
        except prep.PrepError as exc:
            raise PreprocessError(str(exc)) from exc
        sid = row["sample_id"]
        require(sid not in by_id, "Duplicate sample ID in prepared data")
        by_id[sid] = row
        require(type(row["label"]) is int and row["label"] in (0, -1), "Invalid label")
        require(isinstance(row["expected_phrase"], str) and
                (row["label"] != 0 or row["expected_phrase"].strip()), "Invalid expected phrase")
        require(isinstance(row["group"], str) and row["group"].strip(), "Invalid group")
        require(isinstance(row["origin"], str) and row["origin"] in ("recorded", "synthetic"), "Invalid origin")
        require(isinstance(row["review_status"], str) and row["review_status"] in prep.STATUSES, "Invalid review status")
        reasons = []
        if sid in config["exclude_sample_ids"]:
            reasons.append("explicit_exclusion")
        if row["review_status"] not in config["include_review_status"]:
            reasons.append("review_status_not_included")
        require(row["exclusion_reasons"] == reasons and row["included"] == (not reasons), "Exclusion rules mismatch")
        if row["included"]:
            for name in ("audio_file_sha256", "audio_pcm_sha256"):
                require(isinstance(row[name], str) and HEX.fullmatch(row[name]), "Invalid audio identity hash")
            for name in ("sample_rate", "channels", "sample_width", "num_samples"):
                require(type(row[name]) is int and row[name] > 0, f"{name} must be a positive integer")
            require(row["sample_width"] in (1, 2, 3, 4), "Invalid PCM sample width")
            require(type(row["duration_sec"]) in (int, float) and math.isfinite(row["duration_sec"])
                    and row["duration_sec"] > 0, "Invalid audio duration")
        if row.get("parent_sample_id") is not None:
            try:
                prep._identifier(row["parent_sample_id"], "parent_sample_id")
            except prep.PrepError as exc:
                raise PreprocessError(str(exc)) from exc
    require(not set(config["exclude_sample_ids"]) - by_id.keys(), "Unknown excluded sample in config")
    for row in rows:
        current, visited = row, set()
        while current.get("parent_sample_id") is not None:
            sid = current["sample_id"]
            require(sid not in visited, "Prepared parent cycle")
            visited.add(sid)
            parent = current["parent_sample_id"]
            require(parent in by_id, "Prepared parent is missing")
            current = by_id[parent]
    included = [row for row in rows if row["included"]]
    require(included, "No included recordings")
    require(len({row["audio_pcm_sha256"] for row in included}) == len(included), "Duplicate PCM identity")
    for field, expected in (("input_count", len(rows)), ("included_count", len(included)),
                            ("excluded_count", len(rows) - len(included))):
        require(type(ready.get(field)) is int and ready[field] == expected, "Prepared count mismatch")
    expected_purpose = "synthetic-test" if any(row["origin"] == "synthetic" for row in included) else "recorded-data"
    require(ready.get("purpose") == expected_purpose, "Prepared purpose does not match sample origins")
    reconstructed = copy.deepcopy(rows)
    try:
        stats = prep._split(reconstructed, config)
    except prep.PrepError as exc:
        raise PreprocessError(str(exc)) from exc
    require(encode(stats) == encode(ready.get("splits")), "Prepared split statistics mismatch")
    expected_splits = {row["sample_id"]: row["split"] for row in reconstructed}
    require(all(row["split"] == expected_splits[row["sample_id"]] for row in rows), "Prepared split violates recorded policy")


def audio_bytes(row, root):
    try:
        path = prep._audio_path(row["audio_path"], root)
        raw = path.read_bytes()
        require(digest(raw) == row["audio_file_sha256"], f"Audio file hash mismatch: {row['sample_id']}")
        actual = prep._audio_info(path)
    except (OSError, prep.PrepError) as exc:
        raise PreprocessError(f"Invalid source audio: {row['sample_id']}: {exc}") from exc
    require(all(row[key] == actual[key] for key in AUDIO_FIELDS), f"Audio metadata mismatch: {row['sample_id']}")
    return raw


def versions():
    import soundfile
    import torch

    return {"python": platform.python_version(), "platform": platform.system(),
            "machine": platform.machine(), "libsndfile": soundfile.__libsndfile_version__,
            "torch_num_threads": torch.get_num_threads(),
            **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "soundfile", "torch", "torchaudio")}}


def code_identity():
    root = Path(__file__).resolve().parents[1]
    files = {path.relative_to(root).as_posix(): digest(path.read_bytes())
             for directory in (root / "prep", root / "preprocess") for path in sorted(directory.glob("*.py"))}
    return digest(encode(files))


def plan(prepared, audio_root=None, config=None, *, path_map=None):
    """Verify the prepared bundle and every included WAV; write no artifacts."""
    from .frontend import FBANK, decode_audio

    config = configure(config)
    prepared = Path(prepared).resolve()
    ready, ready_sha, payloads, all_rows = read_prepared(prepared)
    roots = input_roots(ready, payloads, audio_root, path_map)
    verify_resources(ready, payloads, all_rows, roots)
    augmentation = config.get("augmentation")
    resources = None
    if augmentation is not None:
        from .augmentation_pipeline import IMPLEMENTATION, Resources, expand

        resources = Resources(augmentation, ready, payloads, all_rows, roots, read_json)
    rows = []
    for row in all_rows:
        # Excluded rows still must have safe relative paths, but need not exist.
        try:
            prep._audio_path(row["audio_path"], row_root(row, roots))
        except prep.PrepError as exc:
            raise PreprocessError(str(exc)) from exc
        if row["included"]:
            raw = audio_bytes(row, row_root(row, roots))
            waveform, info = decode_audio(raw, config)
            source = {**row, "processing": info}
            variants = expand(source, waveform, augmentation, resources) if augmentation is not None else [(source, waveform)]
            for variant, transformed in variants:
                rows.append({**variant, "expected_frames": 1 + (len(transformed) - 400) // 160})
    require(len({row["sample_id"] for row in rows}) == len(rows), "Augmentation sample ID collision")
    if resources is not None:
        resources.verify()
    # Use a digest in feature filenames: IDs can differ only by case or match a
    # reserved Windows device name without colliding on disk.
    version = "v2" if ready["schema"] == v2.PREP_SCHEMA else "v1"
    result = {"schema": "easykws-preprocess-plan/" + version, "purpose": ready["purpose"],
            "prepared_ready_sha256": ready_sha, "prepared_files": ready["files"],
            "config": config, "config_sha256": digest(encode(config)),
            "frontend": {"implementation": "torchaudio.compliance.kaldi.fbank", "parameters": dict(FBANK),
                         "waveform_scale": 32768.0, "cmvn_applied": False},
            "versions": versions(), "code_sha256": code_identity(),
            "input_count": ready["input_count"], "included_count": len(rows),
            "excluded_count": ready["excluded_count"], "rows": rows,
            "splits": {split: {"samples": sum(row["split"] == split for row in rows),
                               "frames": sum(row["expected_frames"] for row in rows if row["split"] == split)}
                       for split in prep.SPLITS}}
    if version == "v2":
        result.update(data_flags=ready["data_flags"], selection=ready["selection"])
    if augmentation is not None:
        result["source_included_count"] = ready["included_count"]
        result["augmentation"] = {"implementation": IMPLEMENTATION,
                                  "resources": resources.metadata,
                                  "excluded_identity_checks": resources.excluded_identity_checks,
                                  "generated_count": len(rows) - ready["included_count"],
                                  "clean_replaced_count": sum(row.get("augmentation", {}).get("clean_replaced", False)
                                                               and row.get("augmentation", {}).get("variant_index") == 0 for row in rows)}
        if any(row.get("effective_purpose") == "synthetic-test" for row in rows):
            result["purpose"] = "synthetic-test"
            if version == "v2":
                result["data_flags"] = sorted(set(result["data_flags"]) | {"synthetic-test"})
    return result


def run(prepared, audio_root, output, config=None, *, path_map=None):
    """Write raw .npy features and train-only CMVN; publish READY only on success."""
    import numpy as np

    from .features import CMVN
    from .frontend import decode_audio, extract_fbank

    prepared, output = (Path(path).resolve() for path in (prepared, output))
    require(not output.exists(), "Output already exists; choose a new directory")
    source_ready, _, source_payloads, _ = read_prepared(prepared)
    roots = input_roots(source_ready, source_payloads, audio_root, path_map)
    require(not output.is_relative_to(prepared) and not any(output.is_relative_to(root) for root in roots.values()),
            "Output must be outside input directories")
    result = plan(prepared, audio_root, config, path_map=path_map)
    source_ready, ready_sha, source_payloads, source_rows = read_prepared(prepared)
    require(ready_sha == result["prepared_ready_sha256"], "Prepared bundle changed during planning")
    output.mkdir(parents=True, exist_ok=False)
    files, indices = {}, {split: [] for split in prep.SPLITS}

    def write(name, payload):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        actual = digest(destination.read_bytes())
        require(actual == digest(payload), f"Artifact readback mismatch: {name}")
        files[name] = actual

    try:
        augmentation = result["config"].get("augmentation")
        resources = None
        if augmentation is not None:
            from .augmentation_pipeline import Resources, apply

            resources = Resources(augmentation, source_ready, source_payloads, source_rows, roots, read_json)
            require(resources.metadata == result["augmentation"]["resources"], "Augmentation resource metadata changed")
            require(resources.excluded_identity_checks == result["augmentation"]["excluded_identity_checks"],
                    "Excluded identity checks changed after planning")
        cmvn = CMVN()
        for row in result["rows"]:
            raw = audio_bytes(row, row_root(row, roots))
            waveform, processing = decode_audio(raw, result["config"])
            require(processing == row["processing"], "Audio processing identity changed")
            if "augmentation" in row:
                waveform, actual = apply(waveform, row["augmentation"], augmentation, resources)
                require(actual == row["augmentation"]["processing"], "Augmentation processing identity changed")
                require(len(waveform) == row["augmentation"]["waveform_samples"], "Augmentation length changed")
            features = extract_fbank(waveform)
            require(features.shape == (row["expected_frames"], 40), "Actual fbank frame count differs from plan")
            name = "features/" + digest(row["sample_id"].encode("utf-8")) + ".npy"
            buffer = io.BytesIO()
            np.save(buffer, features, allow_pickle=False)
            write(name, buffer.getvalue())
            restored = np.load(io.BytesIO((output / name).read_bytes()), allow_pickle=False)
            require(restored.dtype == np.float32 and np.array_equal(restored, features), "Feature round-trip mismatch")
            if row["split"] == "train":
                cmvn.add(restored)
            new_row = {key: value for key, value in row.items() if key != "expected_frames"}
            new_row.update(feature_path=name, feature_sha256=files[name], num_frames=len(features), feature_dim=40)
            indices[row["split"]].append(new_row)
        for split in prep.SPLITS:
            body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in indices[split])
            write(split + ".jsonl", body.encode("utf-8"))
        stats = cmvn.result()
        require(stats["frame_num"] == result["splits"]["train"]["frames"], "Train CMVN frame count mismatch")
        write("cmvn.json", encode(stats))
        write("config.json", encode(result["config"]))
        write("excluded.jsonl", source_payloads["excluded.jsonl"])
        write("prep_config.json", source_payloads["config.json"])
        write("source_ready.json", encode(source_ready))
        if source_ready["schema"] == v2.PREP_SCHEMA:
            write("source_manifest.json", source_payloads["manifest.json"])
        write("plan.json", encode(result))
        summary = {key: value for key, value in result.items() if key not in {"rows", "config", "schema"}}
        version = "v2" if source_ready["schema"] == v2.PREP_SCHEMA else "v1"
        summary.update(schema="easykws-preprocessed/" + version, feature_format="numpy-float32-raw-v1", files=files,
                       cmvn_fit_split="train", train_frames=stats["frame_num"])
        for name, expected in files.items():
            require(digest((output / name).read_bytes()) == expected, f"Final artifact hash mismatch: {name}")
        _, final_ready_sha, _, _ = read_prepared(prepared)
        require(final_ready_sha == result["prepared_ready_sha256"], "Prepared bundle changed during run")
        for row in result["rows"]:
            audio_bytes(row, row_root(row, roots))
        verify_resources(source_ready, source_payloads, source_rows, roots)
        if resources is not None:
            resources.verify()
        require(code_identity() == result["code_sha256"], "Processing code changed during run")
        ready_tmp = output / ".READY.json.tmp"
        ready_tmp.write_bytes(encode(summary))
        ready_tmp.replace(output / "READY.json")
        return summary
    except BaseException as exc:
        try:
            (output / "FAILED.json").write_bytes(encode({"error": type(exc).__name__}))
        except OSError:
            pass
        raise
