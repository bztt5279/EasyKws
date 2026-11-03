import copy
import json

import pytest
import torch

from easykws.calibration.binding import (build_binding, execution_contract, load_candidate,
                                            state_digest, verify_binding)
from easykws.training import runner
from easykws.training.checkpoint import SCHEMA, TrainingError, file_hash
from easykws.training.model import build_model
from easykws.preprocess.frontend import FBANK
from easykws.trigger import trigger_config as resolve_trigger_config


@pytest.fixture
def candidate(tmp_path):
    cmvn = {"mean": [0.1] * 40, "istd": [1.2] * 40}
    bundle = {"identity": "a" * 64, "cmvn_identity": "b" * 64, "cmvn": cmvn,
              "purpose": "synthetic-test", "data_flags": ["synthetic-test"],
              "frontend": {"implementation": "torchaudio.compliance.kaldi.fbank", "parameters": dict(FBANK), "waveform_scale": 32768.0, "cmvn_applied": False}, "selection": None}
    config = runner.configure({"model": {"hidden_dim": 4, "num_layers": 1, "kernel_size": 2},
                               "sampler": {"group_weights": {"positive": 1}}})
    identity = runner.identity(bundle, config)
    model = build_model(cmvn, config["model"])
    checkpoint = {"schema": SCHEMA, "complete": True, "identity": identity, "model": model.state_dict(),
                  "optimizer": {}, "scheduler": {}, "rng": {}, "completed_epochs": 1, "global_step": 1,
                  "history": [{}], "best_val_loss": 1., "best_val_loss_checkpoint": "candidate.pt"}
    torch.save(checkpoint, tmp_path / "candidate.pt")
    status = {"schema": runner.RUN_SCHEMA, "state": "completed", "identity": identity, "config": config,
              "purpose": bundle["purpose"], "data_flags": bundle["data_flags"],
              "candidates": [{"path": "candidate.pt", "sha256": file_hash(tmp_path / "candidate.pt"), "epoch": 1}]}
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return tmp_path, bundle, status


def bind(candidate):
    path, bundle, _ = candidate
    model, meta = load_candidate(path, "candidate.pt", bundle)
    trigger = {"threshold": 0.5, "rules": resolve_trigger_config()}
    calibration = {"schema": "easykws-calibration-identity/v1", "manifest_sha256": "c" * 64, "inputs": {"s": "d" * 64}, "protocol": {"reset": True},
                   "grid": [0.5, 0.6], "constraints": {}, "selection_policy": "min", "candidates": ["candidate.pt"], "chunk_samples": 160}
    return model, meta, trigger, calibration, build_binding(model, meta, trigger, calibration)


def test_digest_frame_order_dtype_shape_and_nonfinite():
    x = {"a": torch.tensor([1., 2.]), "b": torch.tensor(3, dtype=torch.int64)}
    assert state_digest(x) == state_digest(dict(reversed(list(x.items()))))
    for value in ({**x, "a": x["a"].reshape(2, 1)}, {**x, "a": x["a"].double()},
                  {**x, "b": torch.tensor(4, dtype=torch.int64)}):
        assert state_digest(x) != state_digest(value)
    with pytest.raises(TrainingError, match="Nonfinite"):
        state_digest({"x": torch.tensor([float("nan")])})
    with pytest.raises(TrainingError, match="Unsupported"):
        state_digest({"x": torch.tensor([1 + 2j])})


def test_binding_rejects_changed_weights_metadata_trigger_execution_and_missing(candidate, monkeypatch):
    model, meta, trigger, calibration, binding = bind(candidate)
    assert verify_binding(binding, model, meta, trigger, calibration)
    with torch.no_grad(): model.classifier.weight[0, 0] += 1
    with pytest.raises(TrainingError, match="mismatch"): verify_binding(binding, model, meta, trigger, calibration)
    model, meta, trigger, calibration, binding = bind(candidate)
    for key in ("cmvn", "frontend"):
        altered = copy.deepcopy(meta)
        if key == "cmvn": altered[key]["mean"][0] += 1
        else: altered[key]["implementation"] = "other"
        with pytest.raises(TrainingError): verify_binding(binding, model, altered, trigger, calibration)
    with pytest.raises(TrainingError): verify_binding(binding, model, meta, {"threshold": 0.6, "rules": trigger["rules"]}, calibration)
    changed = copy.deepcopy(binding); changed["execution"]["torch_threads"] += 1
    with pytest.raises(TrainingError): verify_binding(changed, model, meta, trigger, calibration)
    with pytest.raises(TrainingError, match="Missing calibration"): verify_binding(None, model, meta, trigger, calibration)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(TrainingError, match="Autocast"): execution_contract(model)


def test_reserialized_state_preserves_content_identity(candidate, tmp_path):
    model, meta, trigger, calibration, binding = bind(candidate)
    path = tmp_path / "repacked.pt"
    torch.save({"state": model.state_dict(), "new_container": True}, path)
    restored = build_model(meta["cmvn"], meta["model_config"])
    restored.load_state_dict(torch.load(path, weights_only=True)["state"]); restored.eval()
    assert file_hash(path) != meta["source_checkpoint_sha256"]
    assert verify_binding(binding, restored, meta, trigger, calibration)


def test_candidate_missing_hash_and_tampered_cmvn_rejected(candidate):
    path, bundle, status = candidate
    del status["candidates"][0]["sha256"]
    (path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(TrainingError, match="SHA256"): load_candidate(path, "candidate.pt", bundle)
    status["candidates"][0]["sha256"] = file_hash(path / "candidate.pt")
    (path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    bundle = copy.deepcopy(bundle); bundle["cmvn"]["mean"][0] += 1
    with pytest.raises(TrainingError, match="CMVN"): load_candidate(path, "candidate.pt", bundle)


def test_implementation_identity_includes_streaming_and_trigger(monkeypatch):
    from easykws.calibration import binding as module
    seen = []
    original = module.file_hash
    def record(path):
        seen.append(path.name)
        return original(path)
    monkeypatch.setattr(module, "file_hash", record)
    module.implementation_identity()
    assert "trigger.py" in seen and "streaming.py" in seen


def test_binding_rejects_invalid_frontend_purpose_flags_and_rules(candidate):
    model, meta, trigger, calibration, binding = bind(candidate)
    for key, value in (("frontend", {"implementation": "test"}), ("purpose", "unknown"),
                       ("data_flags", ["synthetic-test", "synthetic-test"])):
        changed = copy.deepcopy(meta); changed[key] = value
        with pytest.raises(TrainingError):
            build_binding(model, changed, trigger, calibration)
    changed = copy.deepcopy(trigger); changed["rules"]["confirm_frames"] += 1
    with pytest.raises(TrainingError):
        verify_binding(binding, model, meta, changed, calibration)
    changed = copy.deepcopy(binding); changed["execution"]["float32_matmul_precision"] = "changed"
    with pytest.raises(TrainingError):
        verify_binding(changed, model, meta, trigger, calibration)
    with pytest.raises(TrainingError):
        build_binding(model, meta, {"threshold": 0.5, "rules": {"comparison": "ge"}}, calibration)
