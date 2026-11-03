"""Verified export of a calibrated candidate into a self-contained package.

Every step re-derives facts from the actual inputs; a stored calibration
selection is never trusted on its own. READY.json is written last, so a
failed export can never be mistaken for a complete package by the runtime.
"""
import copy
from pathlib import Path

from ..calibration import binding as binding_module
from ..calibration.core import RULE
from ..calibration.metrics import choose, event_metrics
from ..calibration.sessions import load_sessions, read_audio
from ..streaming import StreamingScorer
from ..trigger import FreshFrameTrigger, trigger_config as resolve_trigger_config
from ..training import runner
from ..training.checkpoint import TrainingError, atomic_save, file_hash, require, restricted_load
from ..training.data import _json, load_bundle
from ..training.model import build_model, count_parameters

CALIBRATION_READY_SCHEMA = "easykws-calibration-output/v1"
REPORT_SCHEMA = "easykws-calibration-report/v1"
MANIFEST_SCHEMA = "easykws-deployment-manifest/v1"
MODEL_SCHEMA = "easykws-deployment-model/v1"
CMVN_SCHEMA = "easykws-deployment-cmvn/v1"
THRESHOLD_SCHEMA = "easykws-deployment-threshold/v1"
CALIBRATION_RECORD_SCHEMA = "easykws-deployment-calibration/v1"
READY_SCHEMA = "easykws-deployment-package/v1"
REPLAY_SCHEMA = "easykws-export-replay/v1"

_PAYLOADS = ("model.pt", "cmvn.json", "threshold.json", "calibration.json")
_SELECTION_FIELDS = {"candidate", "threshold", "metrics", "binding"}


def _reject_paths(output, protected):
    root = Path(output).resolve()
    for source in protected:
        source = Path(source).resolve()
        require(not root.is_relative_to(source) and not source.is_relative_to(root),
                "Output overlaps protected input path")
    return root


def _verify_calibration_directory(directory):
    """Strict JSON, complete READY, no FAILED, verified file inventory."""
    root = Path(directory).resolve()
    require(root.is_dir(), "Calibration output directory is missing")
    require(not (root / "FAILED.json").exists(), "Calibration directory has a failure marker")
    require((root / "READY.json").is_file(), "Calibration directory has no READY marker")
    ready = _json((root / "READY.json").read_bytes())
    require(isinstance(ready, dict) and ready.get("schema") == CALIBRATION_READY_SCHEMA,
            "Unsupported calibration READY schema")
    files = ready.get("files")
    require(isinstance(files, dict) and files, "Missing calibration file inventory")
    listed = set(files)
    present = {p.name for p in root.iterdir() if p.is_file() and p.name not in ("READY.json", "FAILED.json")}
    require(listed == present, "Calibration READY inventory does not match directory contents")
    for name, digest in sorted(files.items()):
        require((root / name).is_file(), "Calibration READY lists a missing file: " + name)
        require(file_hash(root / name) == digest, "Calibration file hash mismatch: " + name)
    require((root / "report.json").is_file(), "Calibration directory has no report")
    report = _json((root / "report.json").read_bytes())
    require(isinstance(report, dict) and report.get("schema") == REPORT_SCHEMA, "Unsupported calibration report schema")
    for field in ("feasible", "records", "config", "calibration_identity", "purpose", "data_flags"):
        require(field in report, "Calibration report is missing: " + field)
    require(report["feasible"] is True, "Calibration report is not feasible")
    require(isinstance(report["records"], list) and report["records"], "Calibration report has no records")
    return root, report


def _verify_selection(report, selection):
    """The chosen record must exist and remain the deterministic rule winner."""
    require(isinstance(selection, dict) and set(selection) == _SELECTION_FIELDS,
            "Missing or unsupported selection record")
    config = report["config"]
    require(isinstance(config, dict) and config.get("selection_rule") == RULE, "Unsupported selection rule")
    candidate, threshold = selection["candidate"], selection["threshold"]
    require(isinstance(candidate, str) and candidate in config["candidates"], "Selection is not a configured candidate")
    matches = [r for r in report["records"] if isinstance(r, dict) and r.get("candidate") == candidate
               and r.get("threshold") == threshold]
    require(len(matches) == 1, "Selection does not match exactly one calibration record")
    record = matches[0]
    require(record.get("metrics") == selection["metrics"], "Selection metrics disagree with the report record")
    require(record.get("binding") == selection["binding"], "Selection binding disagrees with the report record")
    require(report.get("selected_checkpoint") == candidate and report.get("selected_threshold") == threshold,
            "Report does not confirm the selection")
    binding = selection["binding"]
    require(isinstance(binding, dict) and binding.get("trigger_config", {}).get("threshold") == threshold,
            "Selection threshold disagrees with its binding")
    require(binding.get("calibration_identity") == report.get("calibration_identity"),
            "Selection binding disagrees with the report calibration identity")
    rules = binding["trigger_config"]["rules"]
    require(rules == config["trigger"], "Selection trigger rules disagree with the calibration config")
    require(binding["calibration_identity"]["grid"] == config["threshold_grid"]
            and threshold in config["threshold_grid"], "Selection threshold is outside the calibrated grid")
    require(binding["calibration_identity"]["constraints"] ==
            {k: config[k] for k in ("min_recall", "max_fah")}, "Selection constraints disagree with the calibration config")
    require(binding["calibration_identity"]["selection_policy"] == config["selection_rule"],
            "Selection policy disagrees with the calibration config")
    winner = choose(report["records"], config["min_recall"], config["max_fah"])
    require(winner is not None and winner["candidate"] == candidate and winner["threshold"] == threshold,
            "Selection is not the winner under the recorded selection rule")
    return record, binding


def _recompute_identity(manifest_path, bundle, path_root, config, expected_inputs, expected_protocol):
    """Reload sessions from disk and require the identity to match the binding."""
    sessions, manifest_sha, paths, _ = load_sessions(manifest_path, bundle, path_root)
    identity = {"schema": "easykws-calibration-identity/v1", "manifest_sha256": manifest_sha,
                "inputs": {s["id"]: s["sha256"] for s in sessions["sessions"]},
                "protocol": {k: v for k, v in sessions.items() if k != "sessions"},
                "grid": config["threshold_grid"], "constraints": {k: config[k] for k in ("min_recall", "max_fah")},
                "selection_policy": config["selection_rule"], "candidates": config["candidates"],
                "chunk_samples": config["chunk_samples"]}
    require(identity["inputs"] == expected_inputs, "Calibration inputs changed since calibration")
    require(identity["protocol"] == expected_protocol, "Calibration protocol changed since calibration")
    return sessions, paths, identity


def _replay_scores(model, sessions, paths, chunk_samples):
    frames = {}
    for session in sessions["sessions"]:
        scorer = StreamingScorer(model)
        audio = read_audio(paths[session["id"]], session["sha256"], session["duration_samples"])
        rows = []
        for start in range(0, len(audio), chunk_samples):
            rows.extend(scorer.push(audio[start:start + chunk_samples]))
        rows.extend(scorer.flush())
        frames[session["id"]] = rows
    return frames


def _saved_scores(calibration_root, candidate):
    for path in sorted(calibration_root.glob("scores_*.json")):
        value = _json(path.read_bytes())
        if value.get("candidate") == candidate:
            sessions = value.get("sessions")
            require(isinstance(sessions, dict) and sessions, "Saved calibration scores are incomplete")
            return sessions
    raise TrainingError("Calibration directory has no saved scores for the selected candidate")


def _write(root, name, value):
    (root / name).write_bytes(runner.encode(value))


def export_package(run_dir, features, calibration_dir, manifest, output, path_root=None):
    """Export the calibrated selection after full re-verification of every input.

    Returns a summary dictionary; the deployment package is written to a new
    output directory that must not overlap any input. READY.json is only
    written after every verification, including reload and replay, succeeds.
    """
    run_dir, features, manifest = Path(run_dir).resolve(), Path(features).resolve(), Path(manifest).resolve()
    calibration_input = Path(calibration_dir).resolve()
    audio_root = Path(path_root).resolve() if path_root is not None else manifest.parent
    output = _reject_paths(output, [run_dir, features, calibration_input, manifest, audio_root])
    # The existence check stays outside the failure handler: an existing
    # directory must never be touched, not even with a failure marker.
    require(not output.exists(), "Output directory already exists")
    try:
        bundle = load_bundle(features)
        calibration_root, report = _verify_calibration_directory(calibration_input)
        selection_path = calibration_root / "selected_checkpoint.json"
        require(selection_path.is_file(), "Calibration directory has no selection record")
        selection = _json(selection_path.read_bytes())
        record, binding = _verify_selection(report, selection)
        candidate = record["candidate"]
        calibration_identity = binding["calibration_identity"]
        sessions, paths, identity = _recompute_identity(
            manifest, bundle, audio_root, report["config"],
            calibration_identity["inputs"], calibration_identity["protocol"])
        require(identity == calibration_identity, "Recomputed calibration identity changed since calibration")
        model, metadata = binding_module.load_candidate(run_dir, candidate, bundle)
        trigger = {"threshold": binding["trigger_config"]["threshold"], "rules": binding["trigger_config"]["rules"]}
        require(resolve_trigger_config(trigger["rules"]) == trigger["rules"], "Invalid bound trigger rules")
        binding_module.verify_binding(binding, model, metadata, trigger, identity)
        purpose = binding_module.deployment_purpose(metadata["purpose"], sessions["purpose"])
        flags = binding_module.deployment_flags(metadata["data_flags"], report["data_flags"])
        require(report["purpose"] == sessions["purpose"], "Report purpose disagrees with the session manifest")
        # Package payload verification: reload the restricted model and require
        # identical inference content before anything else is published.
        output.mkdir(parents=True)
        model_sha = atomic_save(output / "model.pt", {"schema": MODEL_SCHEMA, "state_dict": model.state_dict(),
                                                      "model_config": copy.deepcopy(metadata["model_config"])})
        saved = restricted_load(output / "model.pt", model_sha)
        require(saved["schema"] == MODEL_SCHEMA and saved["model_config"] == metadata["model_config"],
                "Saved deployment model metadata mismatch")
        reloaded = build_model({k: bundle["cmvn"][k] for k in ("mean", "istd")}, metadata["model_config"])
        require(set(saved["state_dict"]) == set(reloaded.state_dict()), "Saved deployment model structure mismatch")
        reloaded.load_state_dict(saved["state_dict"], strict=True)
        reloaded.eval()
        reloaded_digest = binding_module.state_digest(reloaded.state_dict())
        require(binding_module.state_digest(saved["state_dict"]) == binding["state_sha256"],
                "Saved state digest mismatch")
        require(reloaded_digest == binding["state_sha256"], "Repacked model changed inference content")
        binding_module.verify_binding(binding, reloaded, metadata, trigger, identity)
        # Replay the verified calibration sessions with the reloaded model only.
        reference = _saved_scores(calibration_root, candidate)
        chunk_samples = identity["chunk_samples"]
        replay = _replay_scores(reloaded, sessions, paths, chunk_samples)
        require(set(replay) == set(reference), "Replay sessions do not match the saved calibration scores")
        for sid, rows in replay.items():
            require(rows == reference[sid], "Replayed frame scores differ from the saved calibration scores")
        events = {}
        for sid, rows in replay.items():
            trigger_instance = FreshFrameTrigger(trigger["threshold"], trigger["rules"])
            events[sid] = [event for row in rows for event in trigger_instance.update(row["score"], row["t_audio_sec"])]
        require(event_metrics(sessions["sessions"], events) == record["metrics"],
                "Replayed events disagree with the calibrated record metrics")
        cmvn_record = {"schema": CMVN_SCHEMA, "mean": bundle["cmvn"]["mean"], "istd": bundle["cmvn"]["istd"],
                       "var": bundle["cmvn"]["var"], "frame_num": bundle["cmvn"]["frame_num"],
                       "fit_split": bundle["cmvn"]["fit_split"], "identity": bundle["cmvn_identity"]}
        _write(output, "cmvn.json", cmvn_record)
        evidence = {"schema": REPLAY_SCHEMA, "chunk_samples": chunk_samples,
                    "sessions": {sid: {"frames": len(rows), "events": len(events[sid])}
                                 for sid, rows in sorted(replay.items())},
                    "scores_match_saved": True, "events_match_report": True,
                    "state_digest": reloaded_digest, "state_digest_version": binding["state_digest_version"]}
        calibration_record = {"schema": CALIBRATION_RECORD_SCHEMA, "binding": copy.deepcopy(binding),
                              "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
                              "training_data_identity": metadata["training_data_identity"],
                              "calibration_report_sha256": file_hash(calibration_root / "report.json"),
                              "calibration_ready_sha256": file_hash(calibration_root / "READY.json"),
                              "replay": evidence}
        _write(output, "calibration.json", calibration_record)
        threshold_record = {"schema": THRESHOLD_SCHEMA, "threshold": trigger["threshold"],
                            "comparison": trigger["rules"]["comparison"], "rules": copy.deepcopy(trigger["rules"]),
                            "calibration": {"manifest_sha256": identity["manifest_sha256"],
                                            "candidates": identity["candidates"], "grid": identity["grid"],
                                            "constraints": identity["constraints"],
                                            "selection_policy": identity["selection_policy"],
                                            "chunk_samples": identity["chunk_samples"],
                                            "purpose": sessions["purpose"], "selected_candidate": candidate},
                            "binding_reference": {"file": "calibration.json", "sha256": file_hash(output / "calibration.json")}}
        _write(output, "threshold.json", threshold_record)
        payloads = {name: file_hash(output / name) for name in _PAYLOADS}
        manifest_record = {
            "schema": MANIFEST_SCHEMA,
            "model": {"implementation": "easykws.training.model.KWSModel",
                      "config": copy.deepcopy(metadata["model_config"]),
                      "parameter_count": count_parameters(reloaded), "state_names": sorted(reloaded.state_dict()),
                      "file": "model.pt", "file_sha256": payloads["model.pt"]},
            "cmvn": {"file": "cmvn.json", "file_sha256": payloads["cmvn.json"], "identity": bundle["cmvn_identity"]},
            "threshold": {"file": "threshold.json", "file_sha256": payloads["threshold.json"],
                          "threshold": trigger["threshold"]},
            "frontend": copy.deepcopy(metadata["frontend"]),
            "state_digest": {"version": binding["state_digest_version"], "sha256": reloaded_digest},
            "execution": copy.deepcopy(binding["execution"]),
            "trigger": {"rules": copy.deepcopy(trigger["rules"]), "binding_schema": binding["schema"]},
            "purpose": purpose, "data_flags": flags,
            "test_model": binding_module.deployment_is_test(purpose, flags),
            "source": {"source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
                       "training_data_identity": metadata["training_data_identity"],
                       "calibration_manifest_sha256": identity["manifest_sha256"],
                       "calibration_report_sha256": file_hash(calibration_root / "report.json"),
                       "selected_candidate": candidate, "selected_threshold": trigger["threshold"]},
            "streaming": {"sample_rate": 16000, "channels": 1, "dtype": "float32",
                          "frame_ms": 25, "shift_ms": 10, "threshold_file": "threshold.json",
                          "calibration_file": "calibration.json"},
            "payloads": payloads,
        }
        _write(output, "manifest.json", manifest_record)
        _write(output, "READY.json", {"schema": READY_SCHEMA,
                                      "files": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
                                      "purpose": purpose, "data_flags": flags, "test_model": manifest_record["test_model"]})
        return {"schema": "easykws-export-summary/v1", "output": output.name, "candidate": candidate,
                "threshold": trigger["threshold"], "purpose": purpose, "data_flags": flags,
                "test_model": manifest_record["test_model"],
                "parameter_count": manifest_record["model"]["parameter_count"],
                "state_sha256": reloaded_digest, "model_sha256": model_sha, "payloads": payloads}
    except BaseException as exc:
        if output.exists():
            _write(output, "FAILED.json", {"error": type(exc).__name__, "message": str(exc)})
        raise
