"""Versioned data identities, lineage, declared split policies and resource roots.

This module validates declarations; hashes and purpose flags are not provenance
authentication. Audio generation and automatic quality decisions are not provided.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import heapq
import json
import math
import ntpath
from pathlib import Path, PurePosixPath
import re

from . import core as legacy

PrepError = legacy.PrepError
require = legacy._require
digest = legacy._digest
text = legacy._text
SPLITS = legacy.SPLITS
HEX = re.compile(r"[a-f0-9]{64}\Z")
SCHEMA = "easykws-recordings/v2"
CONFIG_SCHEMA = "easykws-prep-config/v2"
PREP_SCHEMA = "easykws-prepared/v2"
FIELDS = {"schema", "datasets", "resources", "selection", "rows"}
ROW_FIELDS = {"sample_id", "dataset_id", "speaker_id", "source_recording_id", "audio_path",
              "audio_sha256", "label", "group", "expected_phrase", "review_status", "origin",
              "purpose", "parent_sample_ids", "resource_ids", "requested_split", "generation"}
RESOURCE_FIELDS = {"resource_id", "dataset_id", "audio_path", "sha256", "kind", "purpose",
                   "split", "speaker_id", "source_sample_id"}
AUDIO_FIELDS = {"audio_file_sha256", "audio_pcm_sha256", "sample_rate", "channels", "sample_width",
                "num_samples", "duration_sec"}
DERIVED_FIELDS = {"actual_speaker_id", "included", "exclusion_reasons", "split", "effective_purpose"}
PREP_FILES = {"manifest.json", "config.json", "train.jsonl", "val.jsonl", "test.jsonl", "excluded.jsonl"}


def exact(value, fields, name):
    require(isinstance(value, dict) and set(value) == fields, f"{name}: missing or unexpected fields")


def word(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be a nonempty string")


def choice(value, allowed, name):
    require(isinstance(value, str) and value in allowed, f"Invalid {name}")


def sha(value, name):
    require(isinstance(value, str) and HEX.fullmatch(value) is not None, f"Invalid {name} hash")


def identifiers(values, name):
    require(isinstance(values, list), f"{name} must be an array")
    for value in values:
        legacy._identifier(value, name)
    require(len(values) == len(set(values)), f"Duplicate {name}")


def configure(config=None):
    require(config is None or isinstance(config, dict), "config must be an object")
    raw = copy.deepcopy(config or {})
    require(raw.get("schema", CONFIG_SCHEMA) == CONFIG_SCHEMA, "Unsupported v2 config schema")
    raw["schema"] = legacy.DEFAULT_CONFIG["schema"]
    result = legacy._config(raw)
    require(set(result["include_review_status"]) <= {"pass", "manual"},
            "v2 review policy cannot include pending or flagged samples")
    result["schema"] = CONFIG_SCHEMA
    return result


def read_manifest(path):
    try:
        raw = Path(path).read_bytes()
        value = legacy.load_json(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PrepError("Cannot read v2 manifest JSON") from exc
    exact(value, FIELDS, "manifest")
    require(value["schema"] == SCHEMA, "Unsupported recording schema")
    return value, raw


def resolve_roots(document, path_map):
    require(isinstance(path_map, dict), "path_map must be a dataset-to-root object")
    require(isinstance(document["datasets"], list), "datasets must be an array")
    names = []
    for item in document["datasets"]:
        require(isinstance(item, dict), "dataset must be an object")
        legacy._identifier(item.get("dataset_id"), "dataset_id")
        names.append(item["dataset_id"])
    require(set(path_map) == set(names), "path_map must cover exactly the declared datasets")
    roots = {}
    for key, value in path_map.items():
        require(isinstance(value, (str, Path)), "path_map roots must be paths")
        root = Path(value).resolve()
        require(root.is_dir(), f"Missing root for dataset {key}")
        roots[key] = root
    return roots


def relative_path(value):
    # Lexical validation must not depend on an unrelated current directory or
    # on symlinks below it. Consumers separately check containment at real roots.
    require(isinstance(value, str) and value and "\x00" not in value, "Invalid relative audio path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and not ntpath.splitdrive(value)[0]
            and "\\" not in value and ".." not in path.parts and ":" not in value
            and path.as_posix() == value, "audio_path must be a canonical relative path")
    require(path.suffix.lower() == ".wav", "Only PCM WAV resource paths are supported")


def audio_metadata(info):
    exact(info, AUDIO_FIELDS, "audio metadata")
    sha(info["audio_file_sha256"], "audio file")
    sha(info["audio_pcm_sha256"], "audio PCM")
    for key in ("sample_rate", "channels", "sample_width", "num_samples"):
        require(type(info[key]) is int and info[key] > 0, f"Invalid audio {key}")
    require(info["sample_width"] in (1, 2, 3, 4), "Invalid sample width")
    duration = info["duration_sec"]
    require(type(duration) in (int, float) and math.isfinite(duration)
            and duration == info["num_samples"] / info["sample_rate"], "Invalid audio duration")


def _catalog(document):
    exact(document, FIELDS, "manifest")
    require(document["schema"] == SCHEMA, "Unsupported recording schema")
    datasets, resources = {}, {}
    require(isinstance(document["datasets"], list) and document["datasets"], "datasets must be a nonempty array")
    for item in document["datasets"]:
        exact(item, {"dataset_id", "source", "version", "license", "split_policy"}, "dataset")
        key = legacy._identifier(item["dataset_id"], "dataset_id")
        require(key not in datasets, "Duplicate dataset_id")
        for field in ("source", "version", "license"):
            word(item[field], field)
        choice(item["split_policy"], {"speaker", "official"}, "split_policy")
        datasets[key] = item
    require(isinstance(document["resources"], list), "resources must be an array")
    for item in document["resources"]:
        exact(item, RESOURCE_FIELDS, "resource")
        key = legacy._identifier(item["resource_id"], "resource_id")
        require(key not in resources, "Duplicate resource_id")
        legacy._identifier(item["dataset_id"], "resource dataset_id")
        require(item["dataset_id"] in datasets, "Unknown resource dataset")
        relative_path(item["audio_path"])
        sha(item["sha256"], "resource")
        choice(item["kind"], {"reference-voice", "noise", "rir"}, "resource kind")
        choice(item["purpose"], {"training-data", "synthetic-test"}, "resource purpose")
        choice(item["split"], {*SPLITS, "external"}, "resource split")
        for field in ("speaker_id", "source_sample_id"):
            if item[field] is not None:
                legacy._identifier(item[field], field)
        resources[key] = item
    return datasets, resources


def _selection(value, count):
    exact(value, {"kind", "reason", "available_count"}, "selection")
    choice(value["kind"], {"full", "fixed-selection", "diagnostic-subset"}, "selection kind")
    require(isinstance(value["reason"], str), "selection reason must be a string")
    require(type(value["available_count"]) is int and value["available_count"] >= count,
            "selection available_count must cover all manifest rows")
    if value["kind"] == "full":
        require(value["available_count"] == count, "full selection must contain all available rows")
    else:
        word(value["reason"], "selection reason")


def _rows(document, datasets, resources, cfg, loader):
    require(isinstance(document["rows"], list) and document["rows"], "rows must be a nonempty array")
    _selection(document["selection"], len(document["rows"]))
    rows, seen_pcm = {}, {}
    for original in document["rows"]:
        exact(original, ROW_FIELDS, "row")
        row = copy.deepcopy(original)
        for key in ("sample_id", "dataset_id", "source_recording_id"):
            legacy._identifier(row[key], key)
        sid = row["sample_id"]
        require(sid not in rows, "Duplicate sample_id")
        require(row["dataset_id"] in datasets, "Unknown sample dataset")
        policy = datasets[row["dataset_id"]]["split_policy"]
        choice(row["origin"], {"recorded", "public", "vc", "test-fixture"}, "origin")
        choice(row["purpose"], {"training-data", "synthetic-test"}, "purpose")
        if row["origin"] == "test-fixture":
            require(row["purpose"] == "synthetic-test", "test-fixture must declare synthetic-test purpose")
        if row["speaker_id"] is None:
            require(row["origin"] == "public" and policy == "official",
                    "Unknown speaker requires public data with official split policy")
            canonical = None
        else:
            legacy._identifier(row["speaker_id"], "speaker_id")
            canonical = legacy._canonical(row["speaker_id"], cfg["speaker_aliases"])
        require(type(row["label"]) is int and row["label"] in (0, -1), "Invalid label")
        word(row["group"], "group")
        require(isinstance(row["expected_phrase"], str), "expected_phrase must be a string")
        if row["label"] == 0:
            word(row["expected_phrase"], "target expected_phrase")
        choice(row["review_status"], legacy.STATUSES, "review_status")
        relative_path(row["audio_path"])
        if row["audio_sha256"] is not None:
            sha(row["audio_sha256"], "declared audio")
        for key in ("parent_sample_ids", "resource_ids"):
            identifiers(row[key], key)
            row[key] = sorted(row[key])
        require(set(row["resource_ids"]) <= resources.keys(), "Unknown resource reference")
        if row["requested_split"] is not None:
            choice(row["requested_split"], set(SPLITS), "requested_split")
        if policy == "official":
            require(row["requested_split"] in SPLITS, "Official data requires a requested_split")
        reasons = []
        if sid in cfg["exclude_sample_ids"]:
            reasons.append("explicit_exclusion")
        if row["review_status"] not in cfg["include_review_status"]:
            reasons.append("review_status_not_included")
        generation = row["generation"]
        if row["origin"] == "vc":
            exact(generation, {"engine", "model", "config_sha256", "seed", "status"}, "VC generation")
            word(generation["engine"], "generation engine")
            word(generation["model"], "generation model")
            sha(generation["config_sha256"], "generation config")
            require(type(generation["seed"]) is int and generation["seed"] >= 0, "Invalid generation seed")
            choice(generation["status"], {"success", "failed", "pending"}, "generation status")
            if generation["status"] != "success":
                reasons.append("generation_not_successful")
            require(len(row["parent_sample_ids"]) == 1 and row["label"] == 0,
                    "VC requires one positive parent")
            require(any(resources[r]["kind"] == "reference-voice" for r in row["resource_ids"]),
                    "VC requires a reference-voice resource")
        else:
            require(generation is None, "generation must be null for non-VC rows")
        row.update(actual_speaker_id=canonical, included=not reasons, exclusion_reasons=reasons,
                   split=None, effective_purpose=row["purpose"])
        if row["included"]:
            sha(row["audio_sha256"], "included audio")
            info = loader(row)
            audio_metadata(info)
            require(info["audio_file_sha256"] == row["audio_sha256"], "Declared audio hash mismatch")
            require(info["audio_pcm_sha256"] not in seen_pcm, "Duplicate PCM content; deduplicate explicitly")
            seen_pcm[info["audio_pcm_sha256"]] = sid
            row.update(info)
        rows[sid] = row
    require(not set(cfg["exclude_sample_ids"]) - rows.keys(), "Unknown excluded sample")
    require(any(row["included"] for row in rows.values()), "All samples were excluded")
    return rows


def _lineage(rows, resources):
    # Topological propagation also accounts for source-linked resources. Excluded
    # ancestors remain in the graph and cannot erase a fixture declaration.
    dependencies, children = {}, defaultdict(set)
    for resource in resources.values():
        source = resource["source_sample_id"]
        require(source is None or source in rows, "Missing resource source_sample_id")
    for sid, row in rows.items():
        parents = set(row["parent_sample_ids"])
        require(parents <= rows.keys(), "Missing parent sample")
        deps = parents | {resources[r]["source_sample_id"] for r in row["resource_ids"]
                          if resources[r]["source_sample_id"] is not None}
        dependencies[sid] = len(deps)
        for parent in deps:
            children[parent].add(sid)
        if row["origin"] == "vc":
            parent = rows[row["parent_sample_ids"][0]]
            require(parent["label"] == 0 and row["actual_speaker_id"] == parent["actual_speaker_id"]
                    and row["expected_phrase"] == parent["expected_phrase"], "VC parent identity or target mismatch")
    queue = sorted(sid for sid, degree in dependencies.items() if not degree)
    heapq.heapify(queue)
    visited = 0
    while queue:
        sid = heapq.heappop(queue)
        row = rows[sid]
        sources = [rows[p]["effective_purpose"] for p in row["parent_sample_ids"]]
        for key in row["resource_ids"]:
            resource = resources[key]
            sources.append(resource["purpose"])
            if resource["source_sample_id"] is not None:
                sources.append(rows[resource["source_sample_id"]]["effective_purpose"])
        if "synthetic-test" in sources:
            row["effective_purpose"] = "synthetic-test"
        visited += 1
        for child in sorted(children[sid]):
            dependencies[child] -= 1
            if not dependencies[child]:
                heapq.heappush(queue, child)
    require(visited == len(rows), "Sample/resource lineage contains a cycle")


def _components(rows):
    links = {}

    def find(node):
        links.setdefault(node, node)
        while links[node] != node:
            links[node] = links[links[node]]
            node = links[node]
        return node

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            links[max(a, b)] = min(a, b)

    for row in rows.values():
        node = ("sample", row["sample_id"])
        if row["actual_speaker_id"] is not None:
            union(node, ("speaker", row["actual_speaker_id"]))
        union(node, ("source", row["dataset_id"], row["source_recording_id"]))
        for parent in row["parent_sample_ids"]:
            union(node, ("sample", parent))
    groups = defaultdict(list)
    for row in rows.values():
        groups[find(("sample", row["sample_id"]))].append(row)
    return list(groups.values())


def _assign(rows, cfg):
    groups = _components(rows)
    included_groups = [g for g in groups if any(r["included"] for r in g)]
    require(len(included_groups) >= 3, "Need at least three independent source units")
    known = {r["actual_speaker_id"] for r in rows.values() if r["actual_speaker_id"] is not None}
    require(set(cfg["speaker_splits"]) <= known, "speaker_splits references unknown speakers")
    split_by_id, automatic, fixed = {}, [], []
    for group in groups:
        requests = {r["requested_split"] for r in group if r["requested_split"] is not None}
        requests.update(cfg["speaker_splits"][r["actual_speaker_id"]] for r in group
                        if r["actual_speaker_id"] in cfg["speaker_splits"])
        require(len(requests) <= 1, "Conflicting split assignments across linked samples")
        if requests:
            split = next(iter(requests))
            for row in group:
                split_by_id[row["sample_id"]] = split
            if any(r["included"] for r in group):
                fixed.append(group)
        elif any(r["included"] for r in group):
            automatic.append(group)
    require(not (cfg["split_counts"] is not None and fixed),
            "split_counts cannot be combined with fixed components")
    if automatic:
        counts = cfg["split_counts"]
        if counts is None:
            raw = {s: len(automatic) * cfg["split_ratios"][s] for s in SPLITS}
            counts = {s: math.floor(raw[s]) for s in SPLITS}
            order = sorted(SPLITS, key=lambda s: (-(raw[s] - counts[s]), SPLITS.index(s)))
            for split in order[:len(automatic) - sum(counts.values())]:
                counts[split] += 1
        require(sum(counts.values()) == len(automatic), "split_counts must match automatic component count")

        def key(group):
            speakers = sorted({r["actual_speaker_id"] for r in group if r["actual_speaker_id"] is not None})
            identity = speakers or sorted({(r["dataset_id"], r["source_recording_id"]) for r in group})
            return digest(text([cfg["seed"], identity]).encode("utf-8"))

        automatic.sort(key=key)
        start = 0
        for split in SPLITS:
            for group in automatic[start:start + counts[split]]:
                for row in group:
                    split_by_id[row["sample_id"]] = split
            start += counts[split]
    for row in rows.values():
        row["split"] = split_by_id.get(row["sample_id"]) if row["included"] else None
        if row["included"] and row["origin"] == "vc":
            require(row["split"] == "train", "VC derivatives must remain in train")
    stats = {}
    for split in SPLITS:
        selected = [r for r in rows.values() if r["included"] and r["split"] == split]
        require(selected, f"{split} is empty")
        labels = Counter(r["label"] for r in selected)
        if cfg["require_both_labels"]:
            require(set(labels) == {0, -1}, f"{split} lacks target or non-target examples")
        present = {r["group"] for r in selected}
        require(set(cfg["required_groups"][split]) <= present, f"{split} lacks required groups")
        stats[split] = {"samples": len(selected), "labels": {str(k): v for k, v in labels.items()},
                       "speakers": sorted({r["actual_speaker_id"] for r in selected if r["actual_speaker_id"] is not None}),
                       "unknown_speaker_samples": sum(r["actual_speaker_id"] is None for r in selected),
                       "groups": sorted(present),
                       "independent_units": sum(any(r["included"] and r["split"] == split for r in g)
                                                for g in included_groups)}
    return stats, split_by_id


def _resources(rows, resources, cfg, split_by_id, loader):
    speaker_splits = defaultdict(set)
    content_splits = defaultdict(set)
    content_purposes = defaultdict(set)
    for row in rows.values():
        speaker = row["actual_speaker_id"]
        declared = row["requested_split"]
        known_split = split_by_id.get(row["sample_id"], declared)
        if row["audio_sha256"] is not None:
            content_purposes[row["audio_sha256"]].add(row["effective_purpose"])
        if row["audio_sha256"] is not None and known_split is not None:
            content_splits[row["audio_sha256"]].add(known_split)
        if speaker is not None and known_split is not None:
            speaker_splits[speaker].add(known_split)
    used = set()
    for row in rows.values():
        if not row["included"]:
            continue
        for key in row["resource_ids"]:
            resource = resources[key]
            require(row["split"] == "train", "Resource-derived data must remain in train")
            require(resource["split"] in ("train", "external"), "Training resource belongs to val/test")
            require(not (content_splits[resource["sha256"]] & {"val", "test"}),
                    "Resource content matches known val/test audio")
            require("synthetic-test" not in content_purposes[resource["sha256"]]
                    or resource["purpose"] == "synthetic-test", "Known fixture resource must retain synthetic-test purpose")
            source = resource["source_sample_id"]
            if source is not None:
                require(split_by_id.get(source) == "train", "Resource source sample is not known to be train")
            speaker = resource["speaker_id"]
            if speaker is not None:
                speaker = legacy._canonical(speaker, cfg["speaker_aliases"])
                require(not (speaker_splits[speaker] & {"val", "test"}), "Resource speaker leaks val/test")
                if source is not None and rows[source]["actual_speaker_id"] is not None:
                    require(speaker == rows[source]["actual_speaker_id"], "Resource source speaker mismatch")
            used.add(key)
    for key in sorted(used):
        loader(resources[key])


def evaluate(document, config, audio_loader, resource_loader):
    """Reconstruct the same contract both at production and at consumption."""
    cfg = configure(config)
    datasets, resources = _catalog(document)
    rows = _rows(document, datasets, resources, cfg, audio_loader)
    _lineage(rows, resources)
    stats, split_by_id = _assign(rows, cfg)
    _resources(rows, resources, cfg, split_by_id, resource_loader)
    included = [r for r in rows.values() if r["included"]]
    flags = []
    if any(r["effective_purpose"] == "synthetic-test" for r in included):
        flags.append("synthetic-test")
    if document["selection"]["kind"] == "diagnostic-subset":
        flags.append("diagnostic-subset")
    return {"schema": PREP_SCHEMA, "config": cfg, "config_sha256": digest(text(cfg).encode("utf-8")),
            "purpose": flags[0] if flags else "training-data", "data_flags": flags,
            "selection": copy.deepcopy(document["selection"]), "input_count": len(rows),
            "included_count": len(included), "excluded_count": len(rows) - len(included),
            "splits": stats, "rows": sorted(rows.values(), key=lambda r: r["sample_id"])}


def check_resource(resource, roots):
    path = legacy._audio_path(resource["audio_path"], roots[resource["dataset_id"]])
    info = legacy._audio_info(path)
    require(info["audio_file_sha256"] == resource["sha256"], "Resource hash mismatch")


def check_paths(document, roots):
    for item in [*document["rows"], *document["resources"]]:
        legacy._audio_path(item["audio_path"], roots[item["dataset_id"]])


def plan(manifest, path_map, config=None):
    document, raw = read_manifest(manifest)
    # Validate structure before accessing root-keyed fields.
    _catalog(document)
    roots = resolve_roots(document, path_map)

    def load(row):
        return legacy._audio_info(legacy._audio_path(row["audio_path"], roots[row["dataset_id"]]))

    result = evaluate(document, config, load, lambda resource: check_resource(resource, roots))
    check_paths(document, roots)
    result["manifest_sha256"] = digest(raw)
    return result


def prepare(manifest, path_map, output, config=None):
    output = Path(output).resolve()
    require(not output.exists(), "Output already exists; use a new directory")
    result = plan(manifest, path_map, config)
    document, raw = read_manifest(manifest)
    roots = resolve_roots(document, path_map)
    require(not any(output.is_relative_to(root) for root in roots.values()), "Output must be outside audio roots")
    require(digest(raw) == result["manifest_sha256"], "Manifest changed during planning")
    payloads = {"manifest.json": raw, "config.json": text(result["config"]).encode("utf-8")}
    for split in (*SPLITS, "excluded"):
        selected = [r for r in result["rows"] if (not r["included"] if split == "excluded" else r["split"] == split)]
        payloads[split + ".jsonl"] = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                                         allow_nan=False) + "\n" for r in selected).encode("utf-8")
    summary = {k: v for k, v in result.items() if k not in {"config", "rows"}}
    summary["files"] = {name: digest(data) for name, data in payloads.items()}
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, data in payloads.items():
            (output / name).write_bytes(data)
        for name, expected in summary["files"].items():
            require(digest((output / name).read_bytes()) == expected, f"Written file hash mismatch: {name}")
        require(plan(manifest, path_map, config) == result, "Input changed during preparation")
        temp = output / ".READY.json.tmp"
        temp.write_bytes(text(summary).encode("utf-8"))
        temp.replace(output / "READY.json")
    except BaseException as exc:
        try:
            (output / "FAILED.json").write_bytes(text({"error": type(exc).__name__}).encode("utf-8"))
        except OSError:
            pass
        raise
    return summary


def upgrade_v1(manifest, audio_root, output, config=None):
    """Convert validated v1 declarations into a NEW manifest/config directory."""
    output, audio_root = Path(output).resolve(), Path(audio_root).resolve()
    require(not output.exists(), "Output already exists; use a new directory")
    require(not output.is_relative_to(audio_root), "Output must be outside audio_root")
    result = legacy.plan(manifest, audio_root, config)
    cfg = copy.deepcopy(result["config"])
    cfg["schema"] = CONFIG_SCHEMA
    # Frozen requested splits replace the old automatic/count assignment.
    cfg["split_counts"] = None
    cfg["speaker_splits"] = {}
    configure(cfg)
    converted = []
    for row in result["rows"]:
        converted.append({"sample_id": row["sample_id"], "dataset_id": "recordings",
                          "speaker_id": row["speaker_id"], "source_recording_id": row["source_recording_id"],
                          "audio_path": row["audio_path"], "audio_sha256": row.get("audio_file_sha256"),
                          "label": row["label"], "group": row["group"], "expected_phrase": row["expected_phrase"],
                          "review_status": row["review_status"],
                          "origin": "test-fixture" if row["origin"] == "synthetic" else "recorded",
                          "purpose": "synthetic-test" if row["origin"] == "synthetic" else "training-data",
                          "parent_sample_ids": [row["parent_sample_id"]] if row.get("parent_sample_id") else [],
                          "resource_ids": [], "requested_split": row["split"], "generation": None})
    document = {"schema": SCHEMA,
                "datasets": [{"dataset_id": "recordings", "source": "user-provided",
                              "version": "converted-v1", "license": "unspecified", "split_policy": "speaker"}],
                "resources": [], "selection": {"kind": "full", "reason": "", "available_count": len(converted)},
                "rows": converted}
    # Validate the upgraded declarations before writing, including more
    # conservative fixture ancestry and stricter review rules.
    by_id = {r["sample_id"]: r for r in result["rows"]}
    evaluate(document, cfg, lambda row: {k: by_id[row["sample_id"]][k] for k in AUDIO_FIELDS}, lambda _: None)
    output.mkdir(parents=True, exist_ok=False)
    payloads = {"manifest.json": text(document).encode("utf-8"), "config.json": text(cfg).encode("utf-8")}
    try:
        for name, payload in payloads.items():
            (output / name).write_bytes(payload)
            require((output / name).read_bytes() == payload, "Upgrade readback mismatch")
        report = {"schema": "easykws-manifest-upgrade/v1", "source_manifest_sha256": result["manifest_sha256"],
                  "source_config_sha256": result["config_sha256"],
                  "files": {k: digest(v) for k, v in payloads.items()}}
        (output / "UPGRADE.json").write_bytes(text(report).encode("utf-8"))
    except BaseException as exc:
        try:
            (output / "FAILED.json").write_bytes(text({"error": type(exc).__name__}).encode("utf-8"))
        except OSError:
            pass
        raise
    return report
