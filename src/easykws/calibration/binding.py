"""Bind calibration to actual inference content and execution conditions."""
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import struct
import sys

import numpy as np
import torch
import torchaudio

from ..training.checkpoint import TrainingError, file_hash, require, restricted_load, validate_checkpoint
from ..training.model import build_model, model_config
from ..training import runner
from ..preprocess.frontend import FBANK
from ..trigger import trigger_config as resolve_trigger_config


_STATE_VERSION = b"easykws-state-digest/v1\x00"
_BINDING_SCHEMA = "easykws-calibration-binding/v1"
_DTYPES = {torch.float16: "<f2", torch.float32: "<f4", torch.float64: "<f8",
           torch.uint8: "u1", torch.int8: "i1", torch.int16: "<i2", torch.int32: "<i4",
           torch.int64: "<i8", torch.bool: "?"}


def _frame(hasher, data):
    hasher.update(struct.pack("<Q", len(data)))
    hasher.update(data)


def state_digest(state_dict):
    """SHA256 over sorted names, dtype, shape and contiguous little-endian bytes.

    All fields are unsigned-64-bit length framed. Shapes are encoded as JSON
    arrays of integers; only dense, supported real and integer tensors qualify.
    """
    require(isinstance(state_dict, dict) and state_dict, "Missing model state")
    require(all(type(k) is str and k for k in state_dict), "Invalid state name")
    h = hashlib.sha256(_STATE_VERSION)
    for name in sorted(state_dict):
        value = state_dict[name]
        require(isinstance(value, torch.Tensor) and value.layout == torch.strided
                and not value.is_quantized and value.dtype in _DTYPES, "Unsupported state tensor")
        if value.is_floating_point():
            require(bool(torch.isfinite(value).all()), "Nonfinite state tensor")
        array = value.detach().cpu().contiguous().numpy().astype(np.dtype(_DTYPES[value.dtype]), copy=False)
        for part in (name.encode("utf-8"), str(value.dtype).encode("ascii"),
                     json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"), array.tobytes(order="C")):
            _frame(h, part)
    return h.hexdigest()


def execution_contract(model):
    """Inspect live settings and reject configurations not supported in v0.1."""
    tensors = list(model.parameters()) + list(model.buffers())
    require(tensors and all(t.device.type == "cpu" for t in tensors), "Only CPU scoring is supported")
    require(all(not t.is_floating_point() or t.dtype == torch.float32 for t in tensors),
            "Only FP32 scoring is supported")
    require(not model.training and all(not m.training for m in model.modules()), "Model must be in evaluation mode")
    cpu_autocast = torch.is_autocast_enabled("cpu")
    cuda_autocast = torch.is_autocast_enabled("cuda")
    require(not cpu_autocast and not cuda_autocast, "Autocast scoring is unsupported")
    require(torch.get_default_dtype() == torch.float32, "Default dtype must be float32")
    return {"device": "cpu", "backend": "pytorch", "weight_dtype": "float32", "compute_dtype": "float32",
            "cpu_autocast": cpu_autocast, "cuda_autocast": cuda_autocast,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "default_dtype": str(torch.get_default_dtype()), "python": platform.python_version(),
            "torch": str(torch.__version__), "torchaudio": str(torchaudio.__version__),
            "numpy": str(np.__version__), "platform": sys.platform, "machine": platform.machine()}


def implementation_identity():
    """Identify inference and calibration implementation present at binding time."""
    root = Path(__file__).resolve().parents[1]
    paths = [root / "training" / "model.py", root / "preprocess" / "frontend.py",
             root / "trigger.py", root / "streaming.py"]
    require(all(p.is_file() for p in paths), "Missing inference implementation")
    for directory in ("calibration", "runtime", "streaming"):
        paths.extend(sorted((root / directory).rglob("*.py")))
    return runner.digest({p.relative_to(root).as_posix(): file_hash(p) for p in paths})


def _check_metadata(model, metadata):
    required = {"model_config", "cmvn", "cmvn_identity", "frontend", "purpose", "data_flags", "source_checkpoint_sha256", "training_data_identity"}
    require(isinstance(metadata, dict) and required <= metadata.keys(), "Missing binding metadata")
    require(model_config(metadata["model_config"]) == model.config, "Model architecture mismatch")
    expected = build_model(metadata["cmvn"], metadata["model_config"])
    actual = model.state_dict()
    require(all(k in actual and torch.equal(actual[k], expected.state_dict()[k]) for k in ("cmvn.mean", "cmvn.istd")),
            "Model CMVN mismatch")
    for key in ("source_checkpoint_sha256", "cmvn_identity", "training_data_identity"):
        require(isinstance(metadata[key], str) and re.fullmatch(r"[0-9a-f]{64}", metadata[key]) is not None,
                "Invalid identity: " + key)
    require(metadata["frontend"] == {"implementation": "torchaudio.compliance.kaldi.fbank",
            "parameters": FBANK, "waveform_scale": 32768.0, "cmvn_applied": False}, "Unsupported frontend")
    require(isinstance(metadata["purpose"], str) and metadata["purpose"] in
            {"training-data", "recorded-data", "synthetic-test", "diagnostic-subset"}, "Invalid purpose")
    flags = metadata["data_flags"]
    require(isinstance(flags, list) and all(isinstance(x, str) and x for x in flags)
            and len(set(flags)) == len(flags), "Invalid data flags")


def load_candidate(run_dir, checkpoint_name, feature_bundle):
    """Load a recorded candidate with a mandatory hash and validated provenance."""
    root = Path(run_dir)
    status = runner._read_status(root)
    path = runner._checkpoint_path(root, checkpoint_name)
    candidates = status.get("candidates")
    require(isinstance(candidates, list), "Missing candidate list")
    matches = [x for x in candidates if isinstance(x, dict) and x.get("path") == checkpoint_name]
    require(len(matches) == 1, "Checkpoint must be a unique recorded candidate")
    candidate = matches[0]
    sha = candidate.get("sha256")
    require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None, "Missing candidate SHA256")
    identity = status.get("identity")
    require(isinstance(identity, dict) and isinstance(feature_bundle, dict), "Missing run or feature identity")
    config = status.get("config")
    require(isinstance(config, dict) and identity.get("config") == config
            and identity.get("config_sha256") == runner.digest(config)
            and identity.get("model") == config.get("model")
            and model_config(config["model"]) == config["model"], "Run config identity mismatch")
    for key, bundle_key in (("data", "identity"), ("cmvn", "cmvn_identity"), ("purpose", "purpose"),
                            ("data_flags", "data_flags"), ("frontend", "frontend"), ("selection", "selection")):
        require(key in identity and identity[key] == feature_bundle.get(bundle_key), "Feature identity mismatch: " + key)
    require(status.get("purpose") == identity["purpose"] and status.get("data_flags") == identity["data_flags"],
            "Run purpose or flags mismatch")
    checkpoint = validate_checkpoint(restricted_load(path, sha), identity)
    require(candidate.get("epoch") == checkpoint["completed_epochs"], "Candidate epoch mismatch")
    raw = checkpoint["model"]
    state_digest(raw)
    require(all(not t.is_floating_point() or t.dtype == torch.float32 for t in raw.values()), "Only FP32 model states supported")
    model = build_model(feature_bundle["cmvn"], config["model"])
    expected = model.state_dict()
    require(set(raw) == set(expected) and all(raw[k].dtype == expected[k].dtype and raw[k].shape == expected[k].shape for k in raw),
            "Checkpoint state structure or dtype mismatch")
    require(all(torch.equal(raw[k], expected[k]) for k in ("cmvn.mean", "cmvn.istd")), "Checkpoint CMVN mismatch")
    model.load_state_dict(raw, strict=True)
    model.eval()
    metadata = {"model_config": copy.deepcopy(config["model"]), "cmvn": copy.deepcopy(feature_bundle["cmvn"]),
                "cmvn_identity": feature_bundle["cmvn_identity"], "frontend": copy.deepcopy(feature_bundle["frontend"]),
                "purpose": feature_bundle["purpose"], "data_flags": copy.deepcopy(feature_bundle["data_flags"]),
                "training_data_identity": feature_bundle["identity"], "source_checkpoint_sha256": sha}
    _check_metadata(model, metadata)
    execution_contract(model)
    return model, metadata


def build_binding(model, metadata, trigger_config, calibration_identity):
    _check_metadata(model, metadata)
    require(isinstance(trigger_config, dict) and set(trigger_config) == {"threshold", "rules"}
            and type(trigger_config["threshold"]) in (int, float) and math.isfinite(trigger_config["threshold"])
            and 0 <= trigger_config["threshold"] <= 1 and isinstance(trigger_config["rules"], dict)
            and bool(trigger_config["rules"]), "Missing threshold or trigger rules")
    try:
        require(resolve_trigger_config(trigger_config["rules"]) == trigger_config["rules"],
                "Trigger rules must be complete and resolved")
    except (TypeError, ValueError) as exc:
        raise TrainingError("Invalid trigger rules") from exc
    fields = {"schema", "manifest_sha256", "inputs", "protocol", "grid", "constraints",
              "selection_policy", "candidates", "chunk_samples"}
    require(isinstance(calibration_identity, dict) and fields <= calibration_identity.keys()
            and calibration_identity["schema"] == "easykws-calibration-identity/v1", "Missing calibration identity")
    require(isinstance(calibration_identity["manifest_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", calibration_identity["manifest_sha256"]) is not None,
            "Invalid calibration manifest identity")
    inputs = calibration_identity["inputs"]
    require(isinstance(inputs, dict) and bool(inputs) and all(isinstance(k, str) and k and isinstance(v, str)
            and re.fullmatch(r"[0-9a-f]{64}", v) is not None for k, v in inputs.items()), "Invalid calibration inputs")
    grid = calibration_identity["grid"]
    require(isinstance(grid, list) and grid and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in grid)
            and trigger_config["threshold"] in grid, "Invalid calibration threshold grid")
    require(isinstance(calibration_identity["protocol"], dict) and bool(calibration_identity["protocol"])
            and isinstance(calibration_identity["constraints"], dict)
            and isinstance(calibration_identity["selection_policy"], str) and bool(calibration_identity["selection_policy"])
            and isinstance(calibration_identity["candidates"], list) and bool(calibration_identity["candidates"])
            and type(calibration_identity["chunk_samples"]) is int and calibration_identity["chunk_samples"] > 0,
            "Incomplete calibration protocol")
    result = {"schema": _BINDING_SCHEMA, "state_digest_version": "easykws-state-digest/v1",
              "state_sha256": state_digest(model.state_dict()), "metadata": copy.deepcopy(metadata),
              "cmvn_content_sha256": runner.digest(metadata["cmvn"]),
              "trigger_config": copy.deepcopy(trigger_config), "calibration_identity": copy.deepcopy(calibration_identity),
              "execution": execution_contract(model), "implementation_sha256": implementation_identity()}
    try:
        runner.encode(result)
    except (TypeError, ValueError) as exc:
        raise TrainingError("Binding must contain finite JSON data") from exc
    return result


# Purposes that the runtime refuses by default; every diagnostic or test
# marker keeps a deployment restricted until explicitly allowed.
RESTRICTED_PURPOSES = ("diagnostic-subset", "synthetic-test")


def deployment_purpose(model_purpose, calibration_purpose):
    """Merge training and calibration purposes; the most restrictive wins."""
    rank = {"training-data": 0, "recorded-data": 1, "diagnostic-subset": 2, "synthetic-test": 3}
    require(model_purpose in rank and calibration_purpose in rank, "Unknown deployment purpose source")
    return model_purpose if rank[model_purpose] >= rank[calibration_purpose] else calibration_purpose


def deployment_flags(model_flags, report_flags):
    """Union every recorded data flag; markers must never be dropped or overridden."""
    require(isinstance(model_flags, list) and isinstance(report_flags, list), "Missing deployment flag sources")
    return sorted(set(model_flags) | set(report_flags))


def deployment_is_test(purpose, flags):
    """Test, diagnostic or synthetic-marked deployments stay restricted."""
    restricted = set(RESTRICTED_PURPOSES)
    return purpose in restricted or bool(restricted & set(flags))


def verify_binding(binding, model, metadata, trigger_config, calibration_identity):
    require(isinstance(binding, dict) and binding.get("schema") == _BINDING_SCHEMA, "Missing calibration binding")
    current = build_binding(model, metadata, trigger_config, calibration_identity)
    # The recorded source hash is provenance, not the identity of a repacked file.
    require(binding == current, "Calibration binding mismatch")
    return True
