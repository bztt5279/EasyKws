"""CPU/FP32 training with full-epoch recovery and validation-only candidates."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
from datetime import datetime, timezone

import numpy as np
import torch

from .checkpoint import (SCHEMA, TrainingError, atomic_save, capture_rng, file_hash,
                         require, restricted_load, restore_rng, validate_checkpoint)
from .model import build_model, model_config
from .loss import max_pooling_loss


RUN_SCHEMA = "easykws-training-run/v1"
DEFAULT_CONFIG = {
    "seed": 666, "device": "cpu", "precision": "float32", "model": {},
    "training": {"num_epochs": 15, "batch_size": 16, "num_workers": 0, "min_duration": 0, "grad_clip": 5.0},
    "sampler": {"group_weights": {}, "balance_unit": "sample", "replacement": True, "sample_ratio": 1.0},
    "optimizer": {"type": "adam", "lr": .001, "weight_decay": .0001},
    "scheduler": {"type": "cosine", "min_lr": 1e-6, "patience": 3, "factor": .5, "threshold": .01},
    "spec_augment": {"enabled": True, "num_t_mask": 1, "num_f_mask": 1, "max_t": 20, "max_f": 10},
}


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def configure(config=None):
    if isinstance(config, (str, Path)):
        try:
            config = json.loads(Path(config).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise TrainingError("Training config must be a readable JSON object") from exc
    config = {} if config is None else copy.deepcopy(config)
    require(isinstance(config, dict), "Training config must be an object")
    require(not set(config) - DEFAULT_CONFIG.keys(), "Unknown training config field")
    result = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in config.items():
        if isinstance(result[key], dict):
            require(isinstance(value, dict), "Config section must be an object: " + key)
            allowed = set(result[key]) | ({"draws_per_epoch"} if key == "sampler" else set())
            if key != "model":
                require(not set(value) - allowed, "Unknown config field: " + key)
            if key == "sampler" and "draws_per_epoch" in value:
                require("sample_ratio" not in value, "draws_per_epoch and sample_ratio are mutually exclusive")
                result[key].pop("sample_ratio", None)
            result[key].update(value)
        else:
            result[key] = value
    try:
        result["model"] = model_config(result["model"])
    except ValueError as exc:
        raise TrainingError(str(exc)) from exc
    require(type(result["seed"]) is int and 0 <= result["seed"] < 2**32, "seed must be an unsigned 32-bit integer")
    require(result["device"] == "cpu" and result["precision"] == "float32", "Only CPU/float32 is supported")
    tr, sampler, opt, sched, spec = (result[k] for k in ("training", "sampler", "optimizer", "scheduler", "spec_augment"))
    for key in ("num_epochs", "batch_size"):
        require(type(tr[key]) is int and tr[key] > 0, "training." + key + " must be positive")
    require(type(tr["num_workers"]) is int and tr["num_workers"] == 0, "Only num_workers=0 is supported")
    require(type(tr["min_duration"]) is int and tr["min_duration"] >= 0, "Invalid min_duration")
    require(sampler["balance_unit"] == "sample" and sampler["replacement"] is True, "Only sample/replacement sampling is supported")
    require(isinstance(sampler["group_weights"], dict) and sampler["group_weights"], "sampler.group_weights must explicitly name group weights")
    for key, value in sampler["group_weights"].items():
        require(isinstance(key, str) and key and type(value) in (int, float) and math.isfinite(value) and value >= 0,
                "Invalid sampler group weight")
    if "draws_per_epoch" in sampler:
        require(type(sampler["draws_per_epoch"]) is int and sampler["draws_per_epoch"] > 0, "draws_per_epoch must be positive")
    else:
        require(type(sampler["sample_ratio"]) in (int, float) and math.isfinite(sampler["sample_ratio"])
                and sampler["sample_ratio"] > 0, "sample_ratio must be positive")
    require(opt["type"] == "adam", "Only Adam is supported")
    require(sched["type"] in ("cosine", "plateau"), "scheduler.type must be cosine or plateau")
    for section, key, positive in ((tr, "grad_clip", False), (opt, "lr", True), (opt, "weight_decay", False),
                                    (sched, "min_lr", False), (sched, "threshold", False)):
        value = section[key]
        require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0),
                "Invalid numeric config: " + key)
    require(sched["min_lr"] <= opt["lr"], "min_lr exceeds initial lr")
    require(type(sched["patience"]) is int and sched["patience"] >= 0, "Invalid scheduler patience")
    require(type(sched["factor"]) in (int, float) and math.isfinite(sched["factor"]) and 0 < sched["factor"] < 1,
            "Invalid scheduler factor")
    require(type(spec["enabled"]) is bool, "spec_augment.enabled must be boolean")
    for key in ("num_t_mask", "num_f_mask", "max_t", "max_f"):
        require(type(spec[key]) is int and spec[key] >= 0, "Invalid SpecAugment parameter: " + key)
    require(not spec["enabled"] or (not spec["num_t_mask"] or spec["max_t"] > 0)
            and (not spec["num_f_mask"] or spec["max_f"] > 0), "Enabled mask bounds must be positive")
    return result


def code_identity():
    root = Path(__file__).resolve().parents[1]
    return digest({path.relative_to(root).as_posix(): file_hash(path)
                   for directory in ("training", "prep", "preprocess")
                   for path in sorted((root / directory).glob("*.py"))})


def environment():
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__),
            "platform": sys.platform, "machine": platform.machine(), "device": "cpu", "dtype": "float32",
            "autocast": False, "num_workers": 0, "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled, "default_dtype": str(torch.get_default_dtype())}


def identity(bundle, config):
    return {"schema": RUN_SCHEMA, "config": config, "config_sha256": digest(config),
            "model": config["model"], "data": bundle["identity"], "cmvn": bundle["cmvn_identity"],
            "purpose": bundle["purpose"], "data_flags": bundle["data_flags"], "selection": bundle.get("selection"),
            "frontend": bundle["frontend"], "code_sha256": code_identity(), "environment": environment()}


def _json_write(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encode(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    stream.flush()


def _build(config, cmvn):
    model = build_model(cmvn, config["model"]).to(device="cpu", dtype=torch.float32)
    opt = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"])
    sched = config["scheduler"]
    if sched["type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"], eta_min=sched["min_lr"])
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=sched["factor"],
                    patience=sched["patience"], min_lr=sched["min_lr"], threshold=sched["threshold"])
    return model, optimizer, scheduler


def _payload(status, model, optimizer, scheduler, generator):
    return {"schema": SCHEMA, "complete": True, "identity": status["identity"], "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": capture_rng(generator),
            "completed_epochs": status["last_complete_epoch"], "global_step": status["global_step"],
            "history": status["history"], "best_val_loss": status["best_val_loss"],
            "best_val_loss_checkpoint": status["best_val_loss_checkpoint"]}


def _read_status(output):
    try:
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainingError("Missing or invalid training status") from exc
    require(status.get("schema") == RUN_SCHEMA and status.get("state") in ("running", "completed", "failed", "interrupted"),
            "Unsupported training run status")
    return status


def _checkpoint_path(output, name):
    require(isinstance(name, str) and Path(name).name == name and name.endswith(".pt"), "Unsafe checkpoint path")
    return output / name


def _restore(status, output, model, optimizer, scheduler, generator, expected):
    checkpoint_hash = status.get("last_checkpoint_sha256")
    require(isinstance(checkpoint_hash, str) and re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash) is not None,
            "Missing or invalid last_checkpoint_sha256")
    value = validate_checkpoint(restricted_load(_checkpoint_path(output, status.get("last_checkpoint")),
                                checkpoint_hash), expected)
    require(value["completed_epochs"] == status["last_complete_epoch"]
            and value["global_step"] == status["global_step"] and value["history"] == status["history"]
            and value["best_val_loss"] == status["best_val_loss"]
            and value["best_val_loss_checkpoint"] == status["best_val_loss_checkpoint"], "Checkpoint/status progress mismatch")
    try:
        model.load_state_dict(value["model"], strict=True)
        optimizer.load_state_dict(value["optimizer"])
        scheduler.load_state_dict(value["scheduler"])
        restore_rng(value["rng"], generator)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise TrainingError("Invalid checkpoint training state") from exc


def _execute(features, config, output, *, recovering, stop_after_epoch):
    # Training needs deterministic single-thread CPU execution, but must not
    # leak those global settings into later calibration or runtime loads.
    threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        return _execute_deterministic(features, config, output,
                                      recovering=recovering, stop_after_epoch=stop_after_epoch)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic,
                                           warn_only=previous_warn_only if previous_deterministic else False)
        torch.set_num_threads(threads)


def _execute_deterministic(features, config, output, *, recovering, stop_after_epoch):
    from .augment import SpecAugment
    from .data import FeatureDataset, collate_features, load_bundle
    from .sampling import build_sampler, exposure_report

    config = configure(config)
    output, features = Path(output).resolve(), Path(features).resolve()
    require(not output.is_relative_to(features), "Training output must be outside feature bundle")
    require(stop_after_epoch is None or type(stop_after_epoch) is int and stop_after_epoch > 0, "Invalid stop_after_epoch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require(torch.get_default_dtype() == torch.float32, "Default dtype must be float32")
    try:
        bundle = load_bundle(features)
    except (ValueError, OSError) as exc:
        raise TrainingError(str(exc)) from exc
    require(bundle["rows"]["train"] and bundle["rows"]["val"], "Training requires nonempty train and val")
    expected = identity(bundle, config)
    if recovering:
        status = _read_status(output)
        require(status["identity"] == expected, "Run identity mismatch; use a new run for changed data/config/code/environment")
    else:
        require(not output.exists() or output.is_dir() and not any(output.iterdir()), "New training output must be empty")
        output.mkdir(parents=True, exist_ok=True)
        status = {"schema": RUN_SCHEMA, "state": "running", "identity": expected, "config": config,
                  "purpose": bundle["purpose"], "data_flags": bundle["data_flags"], "selection": bundle.get("selection"),
                  "last_complete_epoch": 0, "global_step": 0, "history": [], "candidates": [],
                  "best_val_loss": None, "best_val_loss_checkpoint": None, "selected_checkpoint": None,
                  "candidate_rule": "minimum sample-mean validation loss; ties keep earliest epoch",
                  "selection_status": "awaiting-validation-calibration", "attempts": [], "created_at": _now()}
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    model, optimizer, scheduler = _build(config, bundle["cmvn"])
    sampler_cfg = config["sampler"]
    sampler, expected_exposure = build_sampler(bundle["rows"]["train"], sampler_cfg["group_weights"], generator=generator,
                    sample_ratio=sampler_cfg.get("sample_ratio"), draws_per_epoch=sampler_cfg.get("draws_per_epoch"))
    status["expected_exposure"] = expected_exposure
    if recovering:
        _restore(status, output, model, optimizer, scheduler, generator, expected)
        if status["state"] == "completed":
            return status
    else:
        status["last_checkpoint"] = "initial.pt"
        status["last_checkpoint_sha256"] = atomic_save(output / "initial.pt", _payload(status, model, optimizer, scheduler, generator))
        _json_write(output / "config.json", config)
    spec_cfg = {k: v for k, v in config["spec_augment"].items() if k != "enabled"}
    train = FeatureDataset(bundle, "train", SpecAugment(**spec_cfg) if config["spec_augment"]["enabled"] else None)
    val = FeatureDataset(bundle, "val")
    # Test features are checked by the bundle reader but never scored or selected here.
    attempt_number = len(status["attempts"]) + 1
    attempt = {"attempt": attempt_number, "state": "running", "started_at": _now(),
               "start_epoch": status["last_complete_epoch"] + 1, "last_complete_epoch": status["last_complete_epoch"],
               "redo_incomplete_epoch": recovering and status["state"] in ("running", "failed", "interrupted")}
    status["attempts"].append(attempt)
    status["state"] = "running"
    _json_write(output / "status.json", status)
    batch_size = config["training"]["batch_size"]
    log_path = output / ("attempt_%04d.jsonl" % attempt_number)
    completed_status = copy.deepcopy(status)
    try:
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            for epoch in range(status["last_complete_epoch"] + 1, config["training"]["num_epochs"] + 1):
                indices = list(sampler)
                _record(log, {"event": "epoch-start", "epoch": epoch, "indices": indices,
                              "sample_ids": [train.rows[i]["sample_id"] for i in indices]})
                model.train()
                train_loss = train_acc = 0.0
                batches = actual_frames = 0
                lr = float(optimizer.param_groups[0]["lr"])
                for offset in range(0, len(indices), batch_size):
                    x, lengths, labels, ids = collate_features([train[i] for i in indices[offset:offset + batch_size]])
                    feature_digest = hashlib.sha256(x.numpy().tobytes()).hexdigest()
                    optimizer.zero_grad(set_to_none=True)
                    probabilities, _ = model(x)
                    loss, accuracy = max_pooling_loss(probabilities, labels, lengths, config["training"]["min_duration"])
                    require(bool(torch.isfinite(loss)), "Nonfinite training loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"] or float("inf"))
                    require(bool(torch.isfinite(norm)), "Nonfinite training gradient")
                    optimizer.step()
                    status["global_step"] += 1
                    batches += 1; actual_frames += int(lengths.sum())
                    train_loss += float(loss.detach()) * len(ids); train_acc += float(accuracy) * len(ids)
                    _record(log, {"event": "batch", "epoch": epoch, "step": status["global_step"],
                                  "sample_ids": ids, "feature_sha256": feature_digest, "frames": int(lengths.sum()),
                                  "loss": float(loss.detach()), "lr": lr})
                model.eval()
                val_loss = val_acc = 0.0
                val_batches = 0
                with torch.no_grad():
                    for offset in range(0, len(val), batch_size):
                        x, lengths, labels, _ = collate_features([val[i] for i in range(offset, min(offset + batch_size, len(val)))])
                        probabilities, _ = model(x)
                        loss, accuracy = max_pooling_loss(probabilities, labels, lengths, config["training"]["min_duration"])
                        require(bool(torch.isfinite(loss)), "Nonfinite validation loss")
                        val_loss += float(loss) * len(labels); val_acc += float(accuracy) * len(labels); val_batches += 1
                val_loss /= len(val)
                if config["scheduler"]["type"] == "cosine":
                    scheduler.step()
                else:
                    scheduler.step(val_loss)
                exposure = exposure_report(train.rows, indices)
                metrics = {"epoch": epoch, "global_step": status["global_step"], "train_loss": train_loss / len(indices),
                           "train_accuracy": train_acc / len(indices), "val_loss": val_loss, "val_accuracy": val_acc / len(val),
                           "batches": batches, "draws": len(indices), "actual_frames": actual_frames,
                           "lr": lr, "next_lr": float(optimizer.param_groups[0]["lr"]), "exposure": exposure}
                name = "epoch_%04d_attempt_%04d.pt" % (epoch, attempt_number)
                status["last_complete_epoch"] = epoch
                status["history"].append(metrics)
                if status["best_val_loss"] is None or val_loss < status["best_val_loss"]:
                    status["best_val_loss"], status["best_val_loss_checkpoint"] = val_loss, name
                require(code_identity() == expected["code_sha256"], "Training code changed during run")
                current_bundle = load_bundle(features)
                require(current_bundle["identity"] == bundle["identity"], "Feature bundle changed during run")
                checkpoint_sha = atomic_save(output / name, _payload(status, model, optimizer, scheduler, generator))
                status["candidates"].append({"path": name, "sha256": checkpoint_sha, "epoch": epoch,
                                             "val_loss": val_loss, "comparison_split": "val"})
                status["last_checkpoint"], status["last_checkpoint_sha256"] = name, checkpoint_sha
                attempt["last_complete_epoch"] = epoch
                _record(log, {"event": "epoch-complete", **metrics, "checkpoint": name, "sha256": checkpoint_sha})
                _json_write(output / "status.json", status)
                completed_status = copy.deepcopy(status)
                if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < config["training"]["num_epochs"]:
                    status["state"] = attempt["state"] = "interrupted"
                    attempt["reason"] = "stop-after-epoch"
                    break
            else:
                status["state"] = attempt["state"] = "completed"
            attempt["finished_at"] = _now()
            _json_write(output / "status.json", status)
            return status
    except BaseException as exc:
        # Partial optimizer steps are audit evidence only; resume uses the last committed epoch.
        state = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        failed = completed_status
        failed["state"] = state
        failed_attempt = failed["attempts"][-1]
        failed_attempt.update(state=state, finished_at=_now(), error=type(exc).__name__, reason=str(exc),
                              failed_epoch=failed["last_complete_epoch"] + 1,
                              attempted_global_step=status["global_step"])
        # Failure state must not depend on a successfully opened/writable log.
        _json_write(output / "status.json", failed)
        raise


def run(features, config, output, *, stop_after_epoch=None):
    return _execute(features, config, output, recovering=False, stop_after_epoch=stop_after_epoch)


def resume(run_dir, features, config=None, *, stop_after_epoch=None):
    output = Path(run_dir).resolve()
    if config is None:
        config = _read_status(output)["config"]
    return _execute(features, config, output, recovering=True, stop_after_epoch=stop_after_epoch)
