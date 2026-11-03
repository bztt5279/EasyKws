"""Explicit, train-only offline scheduling over the existing resource catalog."""
from __future__ import annotations

import copy
import hashlib
import io
import math
import random

from ..prep import core as prep
from ..prep import v2
from .augment import IMPLEMENTATION, stable_seed, transform
from .errors import PreprocessError

KINDS = ("gain", "speed", "noise", "rir", "filter")
DEFAULTS = {"schema": "easykws-offline-augmentation/v1", "kinds": list(KINDS),
            "gain": [0.3, 1.5], "speed": [0.8, 1.2],
            "filter": {"low_cut": [50, 300], "high_cut": [4000, 7600]},
            "noise": {"resource_ids": [], "snr_db": [5, 20]},
            "rir": {"resource_ids": [], "channel": 0}, "peak_limit": 0.98, "crop": None}


def require(condition, message):
    if not condition:
        raise PreprocessError(message)


def interval(value, name, positive=True):
    require(isinstance(value, list) and len(value) == 2
            and all(type(x) in (int, float) and math.isfinite(x) for x in value),
            f"{name} requires two finite numbers")
    require(value[0] <= value[1] and (not positive or value[0] > 0), f"Invalid {name} interval")


def selectors(value):
    require(isinstance(value, dict) and set(value) == {"groups", "labels", "origins"},
            "selectors requires explicit groups, labels and origins")
    for key in ("groups", "labels", "origins"):
        values = value[key]
        require(isinstance(values, list) and values, f"selectors.{key} must be nonempty")
        require(all(type(x) is int and x in (0, -1) for x in values) if key == "labels"
                else all(isinstance(x, str) and x.strip() for x in values), f"Invalid selectors.{key}")
        require(len(set(values)) == len(values), f"Duplicate selectors.{key}")
    return {key: sorted(value[key]) for key in value}


def configure(value):
    require(isinstance(value, dict), "augmentation must be an object")
    require(not value.keys() - (DEFAULTS.keys() | {"seed", "selectors", "variants"}),
            "Unknown augmentation configuration field")
    result = copy.deepcopy(DEFAULTS)
    result.update(copy.deepcopy(value))
    require(result["schema"] == DEFAULTS["schema"], "Unsupported augmentation schema")
    require(type(result.get("seed")) is int and result["seed"] >= 0, "augmentation.seed must be a nonnegative integer")
    result["selectors"] = selectors(result.get("selectors"))
    counts = result.get("variants")
    require(isinstance(counts, dict) and set(counts) == {"regular", "challenge"}
            and all(type(x) is int and x >= 0 for x in counts.values()),
            "variants requires explicit nonnegative regular and challenge counts")
    kinds = result["kinds"]
    require(isinstance(kinds, list) and kinds and all(isinstance(x, str) and x in KINDS for x in kinds)
            and len(set(kinds)) == len(kinds), "Invalid augmentation kinds")
    result["kinds"] = [kind for kind in KINDS if kind in kinds]
    require(not counts["challenge"] or len(kinds) >= 2, "challenge requires at least two operation kinds")
    for key in ("gain", "speed"):
        interval(result[key], key)
    for key, fields in (("filter", {"low_cut", "high_cut"}), ("noise", {"resource_ids", "snr_db"}),
                        ("rir", {"resource_ids", "channel"})):
        require(isinstance(result[key], dict) and set(result[key]) == fields, f"Invalid augmentation.{key} fields")
    interval(result["filter"]["low_cut"], "low_cut")
    interval(result["filter"]["high_cut"], "high_cut")
    require(result["filter"]["low_cut"][1] < result["filter"]["high_cut"][0]
            and result["filter"]["high_cut"][1] < 8000, "Invalid filter band")
    interval(result["noise"]["snr_db"], "snr_db", positive=False)
    for kind in ("noise", "rir"):
        pool = result[kind]["resource_ids"]
        require(isinstance(pool, list) and all(isinstance(x, str) and x for x in pool)
                and len(set(pool)) == len(pool), "resource_ids must contain unique logical IDs")
        require(kind not in kinds or not sum(counts.values()) or pool, f"{kind} requires an explicit resource pool")
        result[kind]["resource_ids"] = sorted(pool)
    require(type(result["rir"]["channel"]) is int and result["rir"]["channel"] >= 0, "Invalid RIR channel")
    require(type(result["peak_limit"]) in (int, float) and math.isfinite(result["peak_limit"])
            and 0 < result["peak_limit"] < 1, "Invalid peak_limit")
    crop = result["crop"]
    if crop is not None:
        require(isinstance(crop, dict) and set(crop) == {"mode", "selectors", "seconds", "only_if_longer_than"}
                and crop["mode"] == "replace-clean", "crop requires explicit replace-clean policy")
        crop["selectors"] = selectors(crop["selectors"])
        interval(crop["seconds"], "crop seconds")
        threshold = crop["only_if_longer_than"]
        require(type(threshold) in (int, float) and math.isfinite(threshold)
                and threshold >= crop["seconds"][1] and crop["seconds"][0] >= .025,
                "Invalid crop threshold or length")
    return result


def selected(row, selector):
    return row["split"] == "train" and all(row[key] in selector[plural]
            for key, plural in (("group", "groups"), ("label", "labels"), ("origin", "origins")))


class Resources:
    """Validate pools, including unused candidates, and load only each operation's assets."""

    def __init__(self, cfg, ready, payloads, rows, roots, read_json):
        self.items, self.roots, self.metadata = {}, roots, {}
        self.excluded_identity_checks, self.excluded_verified = [], []
        ids = {key for kind in ("noise", "rir") for key in cfg[kind]["resource_ids"]}
        if not ids:
            return
        require(ready["schema"] == v2.PREP_SCHEMA, "Augmentation resources require prepared v2 catalog")
        document = read_json(payloads["manifest.json"], "manifest.json")
        catalog = {r["resource_id"]: r for r in document["resources"]}
        require(ids <= catalog.keys(), "Unknown augmentation resource ID")
        aliases = read_json(payloads["config.json"], "config.json")["speaker_aliases"]
        by_id = {row["sample_id"]: row for row in rows}
        evaluation = [row for row in rows if row.get("split", row.get("requested_split")) in ("val", "test")
                      or row.get("requested_split") in ("val", "test")]
        eval_speakers = {row["actual_speaker_id"] for row in evaluation if row["actual_speaker_id"] is not None}
        eval_hashes = {row["audio_file_sha256"] for row in evaluation if "audio_file_sha256" in row}
        eval_hashes.update(row["audio_sha256"] for row in evaluation if row.get("audio_sha256") is not None)
        eval_pcm = {row["audio_pcm_sha256"] for row in evaluation if "audio_pcm_sha256" in row}
        fixture_pcm = {row["audio_pcm_sha256"] for row in rows if "audio_pcm_sha256" in row
                       and row.get("effective_purpose") == "synthetic-test"}
        fixture_hashes = {row.get("audio_sha256", row.get("audio_file_sha256")) for row in rows
                          if row.get("effective_purpose") == "synthetic-test"}
        evaluation_ids = {row["sample_id"] for row in evaluation}
        for row in rows:
            is_eval = row["sample_id"] in evaluation_ids
            is_fixture = row.get("effective_purpose") == "synthetic-test"
            if row["included"] or not (is_eval or is_fixture):
                continue
            check = {"sample_id": row["sample_id"], "status": "unverified-identity"}
            self.excluded_identity_checks.append(check)
            if row.get("audio_sha256") is None:
                continue
            try:
                path = prep._audio_path(row["audio_path"], roots[row["dataset_id"]])
            except prep.PrepError as exc:
                raise PreprocessError(str(exc)) from exc
            try:
                info = prep._audio_info(path)
            except (prep.PrepError, OSError):
                check["status"] = "unavailable"
                continue
            require(info["audio_file_sha256"] == row["audio_sha256"],
                    "Excluded audio identity hash mismatch")
            check.update(status="verified-pcm", audio_pcm_sha256=info["audio_pcm_sha256"])
            self.excluded_verified.append((row, info))
            if is_eval:
                eval_pcm.add(info["audio_pcm_sha256"])
            if is_fixture:
                fixture_pcm.add(info["audio_pcm_sha256"])
        datasets = {d["dataset_id"]: d for d in document["datasets"]}
        try:
            for key in sorted(ids):
                resource = catalog[key]
                for kind in ("noise", "rir"):
                    require(key not in cfg[kind]["resource_ids"] or resource["kind"] == kind,
                            "Augmentation resource kind mismatch")
                require(resource["split"] in ("train", "external"), "Augmentation resource belongs to val/test")
                source = by_id.get(resource["source_sample_id"])
                require(resource["source_sample_id"] is None or source is not None
                        and source["included"] and source["split"] == "train", "Resource source sample is not train")
                speaker = resource["speaker_id"]
                if speaker is not None:
                    speaker = prep._canonical(speaker, aliases)
                    require(speaker not in eval_speakers, "Resource speaker leaks val/test")
                    require(source is None or source["actual_speaker_id"] in (None, speaker),
                            "Resource source speaker mismatch")
                path = prep._audio_path(resource["audio_path"], roots[resource["dataset_id"]])
                info = prep._audio_info(path)
                require(info["audio_file_sha256"] == resource["sha256"], "Resource hash mismatch")
                require(info["audio_file_sha256"] not in eval_hashes and info["audio_pcm_sha256"] not in eval_pcm,
                        "Resource content leaks val/test PCM")
                purpose = resource["purpose"]
                if (info["audio_pcm_sha256"] in fixture_pcm or info["audio_file_sha256"] in fixture_hashes
                        or source is not None and source["effective_purpose"] == "synthetic-test"):
                    purpose = "synthetic-test"
                self.items[key] = resource
                self.metadata[key] = {**resource, "audio_pcm_sha256": info["audio_pcm_sha256"],
                                      "effective_purpose": purpose,
                                      "dataset": copy.deepcopy(datasets[resource["dataset_id"]])}
        except (prep.PrepError, OSError) as exc:
            raise PreprocessError(str(exc)) from exc

    def verify(self):
        try:
            for row, expected in self.excluded_verified:
                path = prep._audio_path(row["audio_path"], self.roots[row["dataset_id"]])
                require(prep._audio_info(path) == expected, "Excluded audio identity changed during processing")
            for resource in self.items.values():
                v2.check_resource(resource, self.roots)
        except (prep.PrepError, OSError) as exc:
            raise PreprocessError(str(exc)) from exc

    def load(self, ops):
        import soundfile as sf

        result = {}
        for key in sorted({op["resource_id"] for op in ops if "resource_id" in op}):
            resource = self.items[key]
            try:
                path = prep._audio_path(resource["audio_path"], self.roots[resource["dataset_id"]])
                raw = path.read_bytes()
                require(hashlib.sha256(raw).hexdigest() == resource["sha256"], "Resource hash mismatch")
                result[key] = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            except PreprocessError:
                raise
            except (OSError, ValueError, RuntimeError, prep.PrepError) as exc:
                raise PreprocessError("Cannot load augmentation resource") from exc
        return result


def schedules(row, cfg):
    """Yield clean then uniformly sampled single/two-kind permutations."""
    crop = []
    if cfg["crop"] is not None and selected(row, cfg["crop"]["selectors"]):
        rng = random.Random(stable_seed(cfg["seed"], row["sample_id"], 0))
        crop = [{"kind": "crop", "length": rng.uniform(*cfg["crop"]["seconds"]),
                 "position": rng.random(), "only_if_longer_than": cfg["crop"]["only_if_longer_than"]}]
    counts = cfg["variants"] if selected(row, cfg["selectors"]) else {"regular": 0, "challenge": 0}
    schedule = [("clean", 0)] + [("regular", 1)] * counts["regular"] + [("challenge", 2)] * counts["challenge"]
    for index, (variant, count) in enumerate(schedule):
        seed = stable_seed(cfg["seed"], row["sample_id"], index)
        rng = random.Random(seed)
        ops = copy.deepcopy(crop)
        for kind in rng.sample(cfg["kinds"], count):
            op = {"kind": kind}
            if kind in ("gain", "speed"):
                op["value"] = rng.uniform(*cfg[kind])
            elif kind == "filter":
                op.update(low=rng.uniform(*cfg[kind]["low_cut"]), high=rng.uniform(*cfg[kind]["high_cut"]))
            else:
                op["resource_id"] = rng.choice(cfg[kind]["resource_ids"])
                if kind == "noise":
                    op.update(snr_db=rng.uniform(*cfg[kind]["snr_db"]), position=rng.random())
                else:
                    op["channel"] = cfg[kind]["channel"]
            ops.append(op)
        yield index, variant, seed, ops


def apply(waveform, metadata, cfg, resources):
    return transform(waveform, metadata["operations"], cfg["peak_limit"],
                     resources=resources.load(metadata["operations"]))


def expand(row, waveform, cfg, resources):
    for index, variant, seed, ops in schedules(row, cfg):
        if index == 0 and not ops and not selected(row, cfg["selectors"]):
            yield dict(row), waveform
            continue
        used = sorted({op["resource_id"] for op in ops if "resource_id" in op})
        metadata = {"implementation": IMPLEMENTATION, "source_sample_id": row["sample_id"],
                    "source_parent_sample_ids": row.get("parent_sample_ids", [row["parent_sample_id"]]
                                                          if row.get("parent_sample_id") else []),
                    "variant_index": index, "variant": variant, "seed": seed, "operations": ops,
                    "resources": [resources.metadata[key] for key in used],
                    "clean_replaced": False,
                    "crop_coordinate_system": "processed-16000Hz-samples"}
        transformed, info = apply(waveform, metadata, cfg, resources)
        metadata["clean_replaced"] = bool(info["operations"] and info["operations"][0]["kind"] == "crop"
                                           and info["operations"][0]["applied"])
        metadata.update(processing=info, waveform_samples=len(transformed),
                        feature_frames=1 + (len(transformed) - 400) // 160)
        require(len(transformed) >= 400, "Augmentation produces less than one fbank frame")
        new = {**row, "augmentation": metadata}
        if index:
            identity = hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest()
            new["sample_id"] = f"augmentation/{identity}/{index}"
            if "parent_sample_ids" in row:
                new["parent_sample_ids"] = [row["sample_id"]]
                new["resource_ids"] = sorted(set(row["resource_ids"]) | set(used))
            else:
                new["parent_sample_id"] = row["sample_id"]
        if any(resources.metadata[key]["effective_purpose"] == "synthetic-test" for key in used):
            new["effective_purpose"] = "synthetic-test"
        yield new, transformed
