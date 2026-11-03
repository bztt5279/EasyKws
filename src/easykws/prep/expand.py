"""Import reviewed expansion outputs into an existing, frozen v2 preparation.

Generation and acoustic QC run externally. This importer checks their declared
identities and decisions, verifies actual WAVs, and reports every candidate ID.
"""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

from . import core, v2
from ..preprocess.core import read_prepared


def _merge(left, right, key):
    items = {item[key]: copy.deepcopy(item) for item in left}
    for item in right:
        require = core._require
        require(item[key] not in items or items[item[key]] == item, f"Conflicting {key}: {item[key]}")
        items[item[key]] = copy.deepcopy(item)
    return [items[key] for key in sorted(items)]


def plan(prepared, candidates, path_map, *, qc=None, expected_ids=None,
         max_total=None, max_per_source=None, seed=0):
    """Read-only import/QC/selection plan; limits affect added train samples only."""
    require = core._require
    for name, value in (("max_total", max_total), ("max_per_source", max_per_source)):
        require(value is None or (type(value) is int and value >= 0), f"{name} must be null or a nonnegative integer")
    require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
    ready, ready_sha, payloads, base_rows = read_prepared(Path(prepared).resolve())
    require(ready["schema"] == v2.PREP_SCHEMA, "Expansion requires v2; explicitly upgrade v1 first")
    base = core.load_json(payloads["manifest.json"].decode("utf-8-sig"))
    additions, candidate_bytes = v2.read_manifest(candidates)
    v2._catalog(additions)
    require(isinstance(additions["rows"], list) and additions["rows"], "Candidates must be a nonempty array")
    v2._selection(additions["selection"], len(additions["rows"]))
    base_by_id = {r["sample_id"]: r for r in base_rows}
    ids = []
    for row in additions["rows"]:
        v2.exact(row, v2.ROW_FIELDS, "candidate")
        core._identifier(row["sample_id"], "candidate sample_id")
        core._identifier(row["dataset_id"], "candidate dataset_id")
        v2.choice(row["origin"], {"vc", "public", "recorded", "test-fixture"}, "candidate origin")
        v2.choice(row["review_status"], core.STATUSES, "candidate review_status")
        ids.append(row["sample_id"])
    require(len(ids) == len(set(ids)), "Duplicate candidate sample_id")
    require(not set(ids) & base_by_id.keys(), "Candidate ID already exists in base")
    if expected_ids is not None:
        v2.identifiers(expected_ids, "expected_ids")
        missing, extra = sorted(set(expected_ids) - set(ids)), sorted(set(ids) - set(expected_ids))
        require(not missing and not extra, f"Candidate coverage mismatch: missing_count={len(missing)}, "
                f"extra_count={len(extra)}, missing_examples={missing[:10]}, extra_examples={extra[:10]}")
    require(qc is None or isinstance(qc, dict), "QC must be an object")
    qc = copy.deepcopy(qc or {})
    require(isinstance(qc, dict) and set(qc) <= set(ids), "QC must map known candidate IDs to decisions")
    document = {"schema": v2.SCHEMA,
                "datasets": _merge(base["datasets"], additions["datasets"], "dataset_id"),
                "resources": _merge(base["resources"], additions["resources"], "resource_id"),
                "rows": copy.deepcopy(base["rows"] + additions["rows"]),
                "selection": {"kind": "full", "reason": "", "available_count":
                              base["selection"]["available_count"] + additions["selection"]["available_count"]}}
    roots = v2.resolve_roots(document, path_map)
    resources = {r["resource_id"]: r for r in document["resources"]}
    cfg = core.load_json(payloads["config.json"].decode("utf-8-sig"))
    cfg.update(split_counts=None, speaker_splits={})
    for row in document["rows"][:len(base["rows"])]:
        previous = base_by_id[row["sample_id"]]
        if previous["included"]:
            row["requested_split"] = previous["split"]
    rows = document["rows"][len(base["rows"]):]
    decisions, metadata = {}, {}
    seen_pcm = {r["audio_pcm_sha256"]: (r["sample_id"], r["split"]) for r in base_rows if r["included"]}
    # Validate all base audio too, without recomputing its frozen assignment.
    for row in base_rows:
        if row["included"]:
            info = core._audio_info(core._audio_path(row["audio_path"], roots[row["dataset_id"]]))
            require(info["audio_file_sha256"] == row["audio_file_sha256"], "Base audio hash mismatch")
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        sid = row["sample_id"]
        decision = {"sample_id": sid, "status": "accepted", "reasons": [],
                    "review_status": row["review_status"], "qc_status": None}
        decisions[sid] = decision
        v2.identifiers(row["parent_sample_ids"], "parent_sample_ids")
        v2.identifiers(row["resource_ids"], "resource_ids")
        require(set(row["parent_sample_ids"]) <= base_by_id.keys(), "Expansion parents must belong to the frozen base")
        if row["origin"] == "vc":
            require(len(row["parent_sample_ids"]) == 1, "VC needs one frozen parent")
            parent = base_by_id[row["parent_sample_ids"][0]]
            require(parent["included"] and parent["split"] == "train" and parent["label"] == 0,
                    "VC source must be an included train target")
            require(row["requested_split"] in (None, "train"), "VC cannot request an evaluation split")
            row["requested_split"] = "train"
            v2.exact(row["generation"], {"engine", "model", "config_sha256", "seed", "status"}, "generation")
            generated = row["generation"]["status"]
            if generated != "success":
                decision.update(status="pending" if generated == "pending" else "failed", reasons=["generation_" + str(generated)])
            check = qc.get(sid)
            if check is None:
                if decision["status"] == "accepted":
                    decision.update(status="pending", reasons=["missing_qc"])
            else:
                v2.exact(check, {"source_sha256", "output_sha256", "resource_sha256", "status", "reasons"}, "QC decision")
                v2.choice(check["status"], {"ok", "flagged", "failed", "pending"}, "QC status")
                require(isinstance(check["reasons"], list) and all(isinstance(r, str) for r in check["reasons"]), "QC reasons must be strings")
                require(check["source_sha256"] == parent["audio_file_sha256"], "QC source hash mismatch")
                require(check["output_sha256"] == row["audio_sha256"], "QC output hash mismatch")
                require(set(row["resource_ids"]) <= resources.keys(), "Unknown candidate resource")
                expected_resources = {r: resources[r]["sha256"] for r in row["resource_ids"]}
                require(check["resource_sha256"] == expected_resources, "QC resource hashes mismatch")
                decision["qc_status"] = check["status"]
                if decision["status"] == "accepted" and check["status"] != "ok":
                    decision.update(status=check["status"], reasons=["qc_" + check["status"], *check["reasons"]])
        elif sid in qc:
            raise core.PrepError("QC sidecars are currently supported for VC candidates only")
        if row["review_status"] not in cfg["include_review_status"] and decision["status"] == "accepted":
            decision.update(status="pending" if row["review_status"] == "pending" else "flagged", reasons=["review_not_approved"])
        if decision["status"] != "accepted":
            continue
        try:
            require(row["dataset_id"] in roots, "Unknown dataset")
            info = core._audio_info(core._audio_path(row["audio_path"], roots[row["dataset_id"]]))
            require(info["audio_file_sha256"] == row["audio_sha256"], "output_hash_mismatch")
            if row["origin"] == "vc":
                from ..preprocess.frontend import decode_audio
                decode_audio(core._audio_path(row["audio_path"], roots[row["dataset_id"]]).read_bytes(),
                             {"channel_policy": "reject", "resample": False})
        except (ValueError, OSError):
            # Store a portable category, not an exception containing a local path.
            decision.update(status="failed", reasons=["invalid_or_missing_output"])
            continue
        duplicate = seen_pcm.get(info["audio_pcm_sha256"])
        if duplicate is not None:
            require(duplicate[1] == row["requested_split"], "Duplicate PCM crosses split boundaries")
            decision.update(status="duplicate", reasons=["duplicate_pcm"], duplicate_of=duplicate[0])
        else:
            seen_pcm[info["audio_pcm_sha256"]] = (sid, row["requested_split"])
            metadata[sid] = info
    counts, selected = defaultdict(int), 0
    order = sorted(rows, key=lambda r: (core._digest(f"{seed}:{r['sample_id']}".encode()), r["sample_id"]))
    for row in order:
        decision = decisions[row["sample_id"]]
        if decision["status"] != "accepted":
            continue
        # Unassigned personal additions are not capped before their split is
        # known: require an explicit assignment, just like official additions.
        require(row["requested_split"] in core.SPLITS, "Expansion rows require an explicit requested_split")
        if row["requested_split"] != "train":
            continue
        source = tuple(row["parent_sample_ids"]) or (row["dataset_id"], row["source_recording_id"])
        if (max_total is not None and selected >= max_total) or (max_per_source is not None and counts[source] >= max_per_source):
            decision.update(status="limited", reasons=["diagnostic_limit"])
        else:
            selected += 1
            counts[source] += 1
    excluded = {sid for sid, decision in decisions.items() if decision["status"] != "accepted"}
    cfg["exclude_sample_ids"] = sorted(set(cfg["exclude_sample_ids"]) | excluded)
    kinds = {base["selection"]["kind"], additions["selection"]["kind"]}
    if any(d["status"] == "limited" for d in decisions.values()):
        kinds.add("diagnostic-subset")
    if "diagnostic-subset" in kinds:
        document["selection"].update(kind="diagnostic-subset", reason="Inherited diagnostic selection or applied import limits")
    elif "fixed-selection" in kinds:
        document["selection"].update(kind="fixed-selection", reason="Combined declared fixed selections")
    document["rows"].sort(key=lambda r: r["sample_id"])

    def load(row):
        if row["sample_id"] in metadata:
            return metadata[row["sample_id"]]
        return {key: base_by_id[row["sample_id"]][key] for key in v2.AUDIO_FIELDS}

    eval_pcm = {r["audio_pcm_sha256"] for r in base_rows if r["included"] and r["split"] in ("val", "test")}
    eval_pcm.update(metadata[r["sample_id"]]["audio_pcm_sha256"] for r in rows
                    if r["requested_split"] in ("val", "test") and r["sample_id"] in metadata)
    fixture_pcm = {r["audio_pcm_sha256"] for r in base_rows
                   if r["included"] and r["effective_purpose"] == "synthetic-test"}

    def resource_check(resource):
        info = core._audio_info(core._audio_path(resource["audio_path"], roots[resource["dataset_id"]]))
        require(info["audio_file_sha256"] == resource["sha256"], "Resource hash mismatch")
        require(info["audio_pcm_sha256"] not in eval_pcm, "Resource PCM matches evaluation audio")
        require(info["audio_pcm_sha256"] not in fixture_pcm or resource["purpose"] == "synthetic-test",
                "Resource PCM matches a known test fixture")

    result = v2.evaluate(document, cfg, load, resource_check)
    v2.check_paths(document, roots)
    # Added links must never change a base record's inclusion or split.
    actual = {r["sample_id"]: r for r in result["rows"]}
    require(all(actual[sid]["split"] == row["split"] and actual[sid]["included"] == row["included"]
                for sid, row in base_by_id.items()), "Expansion changed a frozen base assignment")
    report = {"base_ready_sha256": ready_sha, "candidate_manifest_sha256": core._digest(candidate_bytes),
              "qc_sha256": core._digest(core._text(qc).encode()), "qc": qc, "seed": seed,
              "limits": {"max_total": max_total, "max_per_source": max_per_source},
              "coverage": "expected_ids_matched" if expected_ids is not None else "declared_manifest_only",
              "candidate_ids": sorted(ids), "accepted_ids": sorted(s for s, d in decisions.items() if d["status"] == "accepted"),
              "counts": dict(sorted(Counter(d["status"] for d in decisions.values()).items())),
              "decisions": [decisions[sid] for sid in sorted(decisions)],
              "purpose": result["purpose"], "data_flags": result["data_flags"]}
    return {"manifest": document, "config": cfg, "report": report}


def run(prepared, candidates, path_map, output, **options):
    """Write a new import report and a directly consumable prepared/ bundle."""
    output = Path(output).resolve()
    core._require(not output.exists(), "Output already exists")
    result = plan(prepared, candidates, path_map, **options)
    roots = v2.resolve_roots(result["manifest"], path_map)
    core._require(not output.is_relative_to(Path(prepared).resolve())
                  and not any(output.is_relative_to(r) for r in roots.values()), "Output must be outside input directories")
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, key in (("manifest.json", "manifest"), ("config.json", "config"), ("report.json", "report")):
            payload = core._text(result[key]).encode("utf-8")
            (output / name).write_bytes(payload)
            core._require((output / name).read_bytes() == payload, "Import artifact readback mismatch")
        core._require(core._digest(Path(candidates).read_bytes()) == result["report"]["candidate_manifest_sha256"],
                      "Candidates changed during import")
        core._require(read_prepared(Path(prepared).resolve())[1] == result["report"]["base_ready_sha256"], "Base changed during import")
        v2.prepare(output / "manifest.json", path_map, output / "prepared", result["config"])
    except BaseException as exc:
        (output / "FAILED.json").write_bytes(core._text({"error": type(exc).__name__}).encode())
        raise
    return result["report"]
