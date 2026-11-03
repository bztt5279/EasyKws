"""Restricted loading and full verification of deployment packages.

Loading depends only on the package directory and the installed software:
no training run, feature bundle, calibration audio or absolute source path
is ever consulted. Every stored value is re-derived and cross-checked before
any scoring is allowed.
"""
from pathlib import Path

import torch

from ..calibration.binding import (RESTRICTED_PURPOSES, deployment_flags, deployment_is_test,
                                   deployment_purpose, execution_contract, state_digest, verify_binding)
from ..export.core import (CALIBRATION_RECORD_SCHEMA, CMVN_SCHEMA, MANIFEST_SCHEMA, MODEL_SCHEMA,
                           READY_SCHEMA, THRESHOLD_SCHEMA)
from ..training.checkpoint import file_hash, restricted_load
from ..training.data import _json
from ..training.model import build_model, count_parameters
from ..trigger import trigger_config as resolve_trigger_config
from .api import DeploymentError, DeploymentModel

_READY_SCHEMA = READY_SCHEMA
_PACKAGE_FILES = ("READY.json", "manifest.json", "model.pt", "cmvn.json", "threshold.json", "calibration.json")
_PAYLOADS = ("model.pt", "cmvn.json", "threshold.json", "calibration.json")
_HEX64 = set("0123456789abcdef")
_DTYPE_BY_NAME = {"torch.float32": torch.float32}


def require(condition, message):
    if not condition:
        raise DeploymentError(message)


def _strict(path, what):
    try:
        return _json(path.read_bytes())
    except OSError as exc:
        raise DeploymentError("Cannot read package file: " + what) from exc
    except ValueError as exc:
        raise DeploymentError(f"Invalid package JSON in {what}: {exc}") from exc


def _sha64(value, what):
    require(isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64,
            "Invalid SHA256 in package: " + what)
    return value


def _verify_inventory(root):
    present = {p.name for p in root.iterdir()}
    missing = [name for name in _PACKAGE_FILES if name not in present]
    require(not missing, "Package is incomplete; missing: " + ", ".join(missing))
    unexpected = sorted(present - set(_PACKAGE_FILES))
    require(not unexpected, "Package contains unexpected files: " + ", ".join(unexpected))
    ready = _strict(root / "READY.json", "READY.json")
    require(isinstance(ready, dict) and ready.get("schema") == _READY_SCHEMA, "Unsupported deployment package schema")
    files = ready.get("files")
    require(isinstance(files, dict) and set(files) == set(_PACKAGE_FILES) - {"READY.json"},
            "Package READY inventory mismatch")
    for name, digest in sorted(files.items()):
        require(_sha64(digest, name) == file_hash(root / name), "Package file hash mismatch: " + name)
    return ready, _strict(root / "manifest.json", "manifest.json")


def _apply_execution_profile(approved):
    """Apply the package's approved CPU settings after an explicit request."""
    require(approved.get("device") == "cpu", "Only CPU deployment packages are supported")
    torch.set_num_threads(approved["torch_threads"])
    if torch.get_num_interop_threads() != approved["torch_interop_threads"]:
        try:
            torch.set_num_interop_threads(approved["torch_interop_threads"])
        except RuntimeError as exc:
            raise DeploymentError("Cannot change interop threads after parallel work started; "
                                  "set OMP/MKL thread variables before starting Python") from exc
    torch.use_deterministic_algorithms(approved["deterministic_algorithms"],
                                       warn_only=approved["deterministic_warn_only"])
    torch.set_float32_matmul_precision(approved["float32_matmul_precision"])
    torch.backends.mkldnn.enabled = approved["mkldnn_enabled"]
    name = approved["default_dtype"]
    require(name in _DTYPE_BY_NAME, "Unsupported approved default dtype: " + str(name))
    if str(torch.get_default_dtype()) != name:
        torch.set_default_dtype(_DTYPE_BY_NAME[name])


def load_model(package, allow_test_model=False, apply_execution_profile=False):
    """Verify a deployment package and return a scoring object with push/reset/flush.

    The default rejects test or diagnostic packages; ``allow_test_model`` only
    lifts that purpose restriction and never any weight, threshold, frontend,
    execution or hash verification. ``apply_execution_profile`` explicitly
    applies the package's approved CPU settings before the actual values are
    verified; without it any mismatch is a clear error, never a silent fix.
    """
    allow_test_model, apply_execution_profile = bool(allow_test_model), bool(apply_execution_profile)
    root = Path(package).resolve()
    require(root.is_dir(), "Deployment package directory not found")
    try:
        ready, manifest = _verify_inventory(root)
    except (ValueError, OSError) as exc:
        raise DeploymentError(str(exc)) from exc
    require(isinstance(manifest, dict) and manifest.get("schema") == MANIFEST_SCHEMA,
            "Unsupported deployment manifest schema")
    for field in ("model", "cmvn", "threshold", "frontend", "state_digest", "execution", "trigger",
                  "purpose", "data_flags", "test_model", "source", "streaming", "payloads"):
        require(field in manifest, "Deployment manifest is missing: " + field)
    payloads = manifest["payloads"]
    require(isinstance(payloads, dict) and set(payloads) == set(_PAYLOADS), "Manifest payload inventory mismatch")
    for name, digest in sorted(payloads.items()):
        require(_sha64(digest, name) == file_hash(root / name), "Payload hash mismatch: " + name)
    require(_sha64(manifest["model"]["file_sha256"], "model") == payloads["model.pt"], "Manifest model hash mismatch")
    require(_sha64(manifest["cmvn"]["file_sha256"], "cmvn") == payloads["cmvn.json"], "Manifest CMVN hash mismatch")
    require(_sha64(manifest["threshold"]["file_sha256"], "threshold") == payloads["threshold.json"],
            "Manifest threshold hash mismatch")
    streaming = manifest["streaming"]
    require(isinstance(streaming, dict) and streaming == {"sample_rate": 16000, "channels": 1, "dtype": "float32",
            "frame_ms": 25, "shift_ms": 10, "threshold_file": "threshold.json",
            "calibration_file": "calibration.json"}, "Unsupported streaming contract")
    # Threshold, complete trigger rules and the hash-bound calibration record.
    threshold_record = _strict(root / "threshold.json", "threshold.json")
    require(isinstance(threshold_record, dict) and threshold_record.get("schema") == THRESHOLD_SCHEMA,
            "Unsupported threshold schema")
    require(manifest["threshold"]["threshold"] == threshold_record.get("threshold"),
            "Manifest threshold value disagrees with the threshold record")
    reference = threshold_record.get("binding_reference")
    require(isinstance(reference, dict) and reference.get("file") == "calibration.json"
            and _sha64(reference.get("sha256", ""), "binding reference") == file_hash(root / "calibration.json"),
            "Threshold record does not reference the packaged calibration binding")
    binding_record = _strict(root / "calibration.json", "calibration.json")
    require(isinstance(binding_record, dict) and binding_record.get("schema") == CALIBRATION_RECORD_SCHEMA,
            "Unsupported deployment calibration schema")
    binding = binding_record.get("binding")
    require(isinstance(binding, dict) and binding.get("schema") == "easykws-calibration-binding/v1",
            "Missing calibration binding")
    trigger = {"threshold": threshold_record["threshold"], "rules": threshold_record["rules"]}
    require(binding["trigger_config"] == trigger, "Threshold record disagrees with the calibration binding")
    require(resolve_trigger_config(trigger["rules"]) == trigger["rules"], "Incomplete bound trigger rules")
    require(threshold_record["comparison"] == trigger["rules"]["comparison"], "Comparison mismatch")
    calibration = threshold_record["calibration"]
    identity = binding["calibration_identity"]
    require(isinstance(calibration, dict)
            and all(calibration.get(k) == identity.get(k)
                    for k in ("manifest_sha256", "candidates", "grid", "constraints", "selection_policy", "chunk_samples"))
            and calibration.get("purpose") == identity["protocol"]["purpose"],
            "Threshold calibration source disagrees with the binding")
    # Re-derive the merged purpose and flags from the binding's own sources.
    # The export merged training flags with the calibration report flags; the
    # report flags equal the bundle flags plus the session synthetic marker,
    # both of which are carried inside the binding.
    metadata = binding["metadata"]
    protocol = identity["protocol"]
    purpose = deployment_purpose(metadata["purpose"], protocol["purpose"])
    flags = deployment_flags(metadata["data_flags"],
                             sorted({protocol["purpose"]} if protocol["purpose"] in RESTRICTED_PURPOSES else set()))
    require(manifest["purpose"] == purpose and manifest["data_flags"] == flags,
            "Manifest purpose or flags disagree with the recorded sources")
    require(manifest["test_model"] is deployment_is_test(purpose, flags), "Manifest test marker mismatch")
    require(ready.get("purpose") == purpose and ready.get("data_flags") == flags
            and ready.get("test_model") is manifest["test_model"], "READY purpose markers disagree with the manifest")
    is_test = manifest["test_model"]
    if is_test and not allow_test_model:
        raise DeploymentError("This package is a test or diagnostic model (purpose: " + purpose +
                              ", flags: " + ", ".join(flags) + "); pass allow_test_model=True / "
                              "--allow-test-model to use it for testing")
    # Restricted model load and inference-content verification.
    require(manifest["frontend"] == metadata["frontend"], "Manifest frontend mismatch")
    cmvn_record = _strict(root / "cmvn.json", "cmvn.json")
    require(isinstance(cmvn_record, dict) and cmvn_record.get("schema") == CMVN_SCHEMA, "Unsupported CMVN schema")
    require(_sha64(cmvn_record.get("identity", ""), "cmvn identity") == manifest["cmvn"]["identity"],
            "CMVN identity mismatch")
    # The package CMVN must agree with the calibration binding field by field;
    # file hashes alone cannot prove that the statistics were not rewritten.
    require(isinstance(metadata.get("cmvn"), dict), "Binding carries no CMVN record")
    for key in ("mean", "istd", "var", "frame_num", "fit_split"):
        require(key in cmvn_record and key in metadata["cmvn"] and cmvn_record[key] == metadata["cmvn"][key],
                "Package CMVN " + key + " disagrees with the calibration binding")
    require(cmvn_record["identity"] == metadata["cmvn_identity"],
            "Package CMVN identity disagrees with the calibration binding")
    saved = restricted_load(root / "model.pt")
    require(saved.get("schema") == MODEL_SCHEMA, "Unsupported deployment model schema")
    require(saved.get("model_config") == manifest["model"]["config"], "Model config disagrees with the manifest")
    model = build_model({"mean": cmvn_record["mean"], "istd": cmvn_record["istd"]}, saved["model_config"])
    require(set(saved["state_dict"]) == set(model.state_dict()), "Deployment model structure mismatch")
    expected = model.state_dict()
    require(all(saved["state_dict"][key].dtype == expected[key].dtype and saved["state_dict"][key].shape == expected[key].shape
                for key in saved["state_dict"]), "Deployment model tensor dtype or shape mismatch")
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    # After the state_dict overwrote the CMVN buffers, require them to equal
    # the packaged statistics in FP32, not just the binding record.
    require(torch.equal(model.cmvn.mean, torch.as_tensor(cmvn_record["mean"], dtype=torch.float32))
            and torch.equal(model.cmvn.istd, torch.as_tensor(cmvn_record["istd"], dtype=torch.float32)),
            "Loaded model CMVN buffers disagree with the package CMVN record")
    digest = state_digest(model.state_dict())
    require(manifest["state_digest"]["version"] == binding["state_digest_version"], "State digest version mismatch")
    require(digest == manifest["state_digest"]["sha256"] == binding["state_sha256"],
            "Deployment model inference content changed")
    require(manifest["model"]["parameter_count"] == count_parameters(model), "Parameter count mismatch")
    require(manifest["model"]["state_names"] == sorted(model.state_dict()), "Model state names mismatch")
    require(manifest["model"]["implementation"] == "easykws.training.model.KWSModel", "Unsupported model implementation")
    source = manifest["source"]
    require(source["source_checkpoint_sha256"] == metadata["source_checkpoint_sha256"]
            and source["training_data_identity"] == metadata["training_data_identity"]
            and source["calibration_manifest_sha256"] == identity["manifest_sha256"],
            "Manifest source identity disagrees with the binding")
    require(manifest["trigger"] == {"rules": trigger["rules"], "binding_schema": binding["schema"]},
            "Manifest trigger record mismatch")
    # Execution conditions: apply only on explicit request, then verify values.
    require(manifest["execution"] == binding["execution"], "Manifest execution profile disagrees with the binding")
    if apply_execution_profile:
        _apply_execution_profile(binding["execution"])
    current = execution_contract(model)
    drift = {key: (binding["execution"][key], current.get(key))
             for key in sorted(binding["execution"]) if current.get(key) != binding["execution"][key]}
    if drift:
        details = ", ".join(f"{key}: approved {want!r}, actual {got!r}" for key, (want, got) in drift.items())
        raise DeploymentError("Execution conditions do not match the calibrated profile (" + details + "); "
                              "set OMP_NUM_THREADS/MKL_NUM_THREADS before starting Python, or load with "
                              "apply_execution_profile=True to apply the package's approved CPU settings")
    verify_binding(binding, model, metadata, trigger, identity)
    return DeploymentModel(model, trigger, binding["execution"], purpose=purpose, data_flags=flags,
                           is_test_model=is_test, parameter_count=manifest["model"]["parameter_count"],
                           state_sha256=digest, allow_test_model=allow_test_model,
                           apply_execution_profile=apply_execution_profile)
