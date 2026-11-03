"""Atomic, restricted checkpoints and explicit trusted legacy conversion."""
import hashlib
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch


SCHEMA = "easykws-training-checkpoint/v1"


class TrainingError(ValueError):
    """A training contract or saved state is invalid."""


def require(condition, message):
    if not condition:
        raise TrainingError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def primitive_tree(value):
    """Strip container subclasses; reject objects requiring pickle globals."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        require(all(type(k) in (str, int) for k in value), "Unsupported checkpoint key")
        return {k: primitive_tree(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return type(value)(primitive_tree(v) for v in value)
    raise TrainingError("Unsupported checkpoint object: " + type(value).__name__)


def atomic_save(path, payload):
    path = Path(path)
    require(not path.exists(), "Checkpoint already exists")
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(primitive_tree(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        # A failed or truncated write is never advertised as a checkpoint.
        primitive_tree(torch.load(temporary, map_location="cpu", weights_only=True))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_hash(path)


def restricted_load(path, expected_hash=None):
    path = Path(path)
    require(path.is_file() and path.suffix != ".tmp", "Missing or partial checkpoint")
    if expected_hash is not None:
        require(file_hash(path) == expected_hash, "Checkpoint hash mismatch")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        return primitive_tree(value)
    except Exception as exc:
        raise TrainingError("Cannot load restricted checkpoint: " + str(exc)) from exc


def capture_rng(generator):
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": state[0], "keys": torch.tensor(state[1].astype(np.int64)),
                      "position": int(state[2]), "has_gauss": int(state[3]), "cached_gaussian": float(state[4])},
            "torch_cpu": torch.get_rng_state(), "sampler": generator.get_state()}


def restore_rng(state, generator):
    try:
        require(set(state) == {"python", "numpy", "torch_cpu", "sampler"}, "Missing RNG state")
        numpy = state["numpy"]
        random.setstate(state["python"])
        np.random.set_state((numpy["algorithm"], numpy["keys"].numpy().astype(np.uint32),
                             numpy["position"], numpy["has_gauss"], numpy["cached_gaussian"]))
        torch.set_rng_state(state["torch_cpu"])
        generator.set_state(state["sampler"])
    except Exception as exc:
        raise TrainingError("Invalid RNG state") from exc


def validate_checkpoint(value, identity):
    required = {"schema", "complete", "identity", "model", "optimizer", "scheduler", "rng",
                "completed_epochs", "global_step", "history", "best_val_loss", "best_val_loss_checkpoint"}
    require(isinstance(value, dict) and required <= value.keys(), "Incomplete training checkpoint")
    require(value["schema"] == SCHEMA and value["complete"] is True, "Unsupported or partial training checkpoint")
    require(value["identity"] == identity, "Checkpoint identity mismatch")
    require(type(value["completed_epochs"]) is int and value["completed_epochs"] >= 0
            and type(value["global_step"]) is int and value["global_step"] >= 0, "Invalid checkpoint progress")
    require(isinstance(value["history"], list) and len(value["history"]) == value["completed_epochs"],
            "Incomplete checkpoint history")
    require(all(isinstance(value[k], dict) for k in ("model", "optimizer", "scheduler", "rng")),
            "Missing checkpoint training state")
    return value


def migrate_checkpoint(source, output, *, trust_legacy_input=False):
    """The sole unsafe loader: the caller explicitly trusts executable pickle input."""
    require(trust_legacy_input is True, "Legacy conversion requires --trust-legacy-input")
    source, output = Path(source).resolve(), Path(output).resolve()
    require(source != output and not output.exists(), "Migration output already exists or replaces input")
    try:
        legacy = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrainingError("Cannot read explicitly trusted legacy input") from exc

    def convert(value):
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(convert(v) for v in value)
        return primitive_tree(value)

    payload = {"schema": "easykws-migrated-checkpoint/v1", "source_sha256": file_hash(source),
               "resumable": False, "reason": "Legacy identity and epoch completeness require separate verification",
               "legacy_state": convert(legacy)}
    output.parent.mkdir(parents=True, exist_ok=True)
    return {"schema": payload["schema"], "sha256": atomic_save(output, payload), "resumable": False}
