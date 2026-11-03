"""Adapt six user-supplied Mobvoi metadata files; never download or infer labels."""
from __future__ import annotations

from pathlib import Path

from . import core, v2

SETS = ("n_train", "n_dev", "n_test", "p_train", "p_dev", "p_test")


def _uid(value):
    core._identifier(value, "utt_id")
    # A single filename stem, including on Windows; never search subdirectories.
    core._require(not value.endswith(".") and not value.lower().endswith(".wav")
                  and value.split(".")[0].upper() not in
                  {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
                   *(f"LPT{i}" for i in range(10))}, "utt_id must be a safe filename stem")
    return value


def _speaker(value):
    if value is None:
        return None
    core._require(type(value) in (str, int, float, bool), "speaker_id must be a scalar or null")
    if (isinstance(value, str) and core.IDENTIFIER.fullmatch(value)
            and not value.startswith("speaker-hash-")):
        return value
    return "speaker-hash-" + core._digest(core._text(value).encode("utf-8"))


def build(metadata_dir, audio_root, *, dataset_id, source, version, license,
          keyword_labels, purpose="training-data"):
    """Return a v2 document and audit report for the supplied local metadata.

    Mapping keys are string keyword IDs; every value must be -1 or 'exclude'.
    'full' describes these six local files, not verified dataset completeness.
    """
    dataset = dict(dataset_id=dataset_id, source=source, version=version,
                   license=license, split_policy="official")
    document = dict(schema=v2.SCHEMA, datasets=[dataset], resources=[], rows=[],
                    selection=dict(kind="full", reason="", available_count=0))
    v2._catalog(document)
    v2.choice(purpose, {"training-data", "synthetic-test"}, "purpose")
    core._require(isinstance(keyword_labels, dict) and keyword_labels,
                  "keyword_labels must be an explicit nonempty mapping")
    for key, value in keyword_labels.items():
        core._require(isinstance(key, str) and bool(key), "keyword_labels keys must be strings")
        core._require((type(value) is int and value == -1) or value == "exclude",
                      "keyword_labels values must be -1 or 'exclude'")
    metadata_dir, audio_root = Path(metadata_dir), Path(audio_root).resolve()
    entries, seen, files = [], set(), {}
    for name in SETS:
        filename = name + ".json"
        try:
            raw = (metadata_dir / filename).read_bytes()
            records = core.load_json(raw.decode("utf-8-sig"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise core.PrepError(f"Cannot read metadata {filename}: {exc}") from exc
        core._require(isinstance(records, list), f"{filename} must contain an array")
        files[filename] = dict(sha256=core._digest(raw), count=len(records))
        for record in records:
            core._require(isinstance(record, dict), "metadata entry must be an object")
            uid = _uid(record.get("utt_id"))
            core._require(uid not in seen, f"Duplicate utt_id: {uid}")
            seen.add(uid)
            kid = record.get("keyword_id")
            core._require(type(kid) in (str, int) and str(kid) in keyword_labels,
                          "Unmapped or invalid keyword_id")
            split = name.split("_")[1]
            entries.append(dict(utt_id=uid, keyword_id=kid, official_set=name,
                                official_split=split, metadata_file=filename,
                                metadata_sha256=files[filename]["sha256"],
                                original_speaker_id=record.get("speaker_id"),
                                speaker_id=_speaker(record.get("speaker_id")),
                                included=keyword_labels[str(kid)] == -1))
    core._require(entries, "No metadata entries in the six supplied files")
    for entry in entries:
        uid = entry["utt_id"]
        path = uid + ".wav"
        info = None
        if entry["included"]:
            resolved = core._audio_path(path, audio_root)
            info = core._audio_info(resolved)
            # PCM frame checks alone can miss a truncated trailing RIFF chunk.
            with resolved.open("rb") as stream:
                header = stream.read(12)
            core._require(int.from_bytes(header[4:8], "little") + 8 <= resolved.stat().st_size,
                          "WAV RIFF container is truncated")
        sample_id = "public-" + core._digest(core._text([dataset_id, uid]).encode("utf-8"))
        row = dict(sample_id=sample_id, dataset_id=dataset_id, speaker_id=entry["speaker_id"],
                   source_recording_id=uid, audio_path=path,
                   audio_sha256=info["audio_file_sha256"] if info else None,
                   label=-1, group="public-negative", expected_phrase="",
                   review_status="pass" if entry["included"] else "flagged", origin="public",
                   purpose=purpose, parent_sample_ids=[], resource_ids=[],
                   requested_split={"dev": "val"}.get(entry["official_split"], entry["official_split"]),
                   generation=None)
        document["rows"].append(row)
        entry.update(sample_id=sample_id, audio=info,
                     exclusion_reason=None if entry["included"] else "keyword_labels:exclude")
    document["selection"]["available_count"] = len(entries)
    report = dict(dataset_id=dataset_id, metadata_files=files, keyword_labels=dict(keyword_labels),
                  original_uids=sorted(seen), included_uids=sorted(e["utt_id"] for e in entries if e["included"]),
                  excluded_uids=sorted(e["utt_id"] for e in entries if not e["included"]), entries=entries,
                  scope="Six supplied local JSON files; official full-release completeness not verified.")
    return document, report
