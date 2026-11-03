"""Export validation-chain positives and negatives on a real calibration."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from easykws.export import export_package
from easykws.runtime import DeploymentError, load_model

from helpers import inventory, rewrite_manifest_hashes, rewrite_ready

PROTECTED = ("run", "features", "calibration", "session-audio")


def test_export_rejects_existing_and_overlapping_outputs(world):
    with pytest.raises(ValueError, match="already exists"):
        export_package(world.run, world.bundle, world.calibration, world.manifest, world.package)
    for parent in PROTECTED:
        inside = world.root / parent / "nested-export"
        with pytest.raises(ValueError, match="protected"):
            export_package(world.run, world.bundle, world.calibration, world.manifest, inside)
    with pytest.raises(ValueError, match="protected"):
        export_package(world.run, world.bundle, world.calibration, world.manifest, world.root)


def test_export_rejects_incomplete_or_failed_calibration(world, tmp_path):
    broken = tmp_path / "cal-no-ready"
    broken.mkdir()
    for item in world.calibration.iterdir():
        (broken / item.name).write_bytes(item.read_bytes())
    (broken / "READY.json").unlink()
    with pytest.raises(ValueError, match="READY"):
        export_package(world.run, world.bundle, broken, world.manifest, tmp_path / "out1")
    failed = tmp_path / "cal-failed"
    failed.mkdir()
    for item in world.calibration.iterdir():
        (failed / item.name).write_bytes(item.read_bytes())
    (failed / "FAILED.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="failure marker"):
        export_package(world.run, world.bundle, failed, world.manifest, tmp_path / "out2")
    no_selection = tmp_path / "cal-no-selection"
    no_selection.mkdir()
    for item in world.calibration.iterdir():
        (no_selection / item.name).write_bytes(item.read_bytes())
    ready = json.loads((no_selection / "READY.json").read_text(encoding="utf-8"))
    del ready["files"]["selected_checkpoint.json"]
    (no_selection / "READY.json").write_text(json.dumps(ready), encoding="utf-8")
    (no_selection / "selected_checkpoint.json").unlink()
    with pytest.raises(ValueError, match="inventory|selection record"):
        export_package(world.run, world.bundle, no_selection, world.manifest, tmp_path / "out3")


def _repacked_calibration(source, target):
    target.mkdir()
    for item in source.iterdir():
        (target / item.name).write_bytes(item.read_bytes())
    rewrite_ready(target)


def test_export_rejects_selection_not_matching_report(world, tmp_path):
    changed = tmp_path / "cal-selection"
    _repacked_calibration(world.calibration, changed)
    selection = json.loads((changed / "selected_checkpoint.json").read_text(encoding="utf-8"))
    selection["threshold"] = 0.9
    (changed / "selected_checkpoint.json").write_text(json.dumps(selection), encoding="utf-8")
    rewrite_ready(changed)
    with pytest.raises(ValueError, match="exactly one calibration record"):
        export_package(world.run, world.bundle, changed, world.manifest, tmp_path / "out-a")
    foreign = tmp_path / "cal-candidate"
    _repacked_calibration(world.calibration, foreign)
    selection = json.loads((foreign / "selected_checkpoint.json").read_text(encoding="utf-8"))
    selection["candidate"] = "missing.pt"
    (foreign / "selected_checkpoint.json").write_text(json.dumps(selection), encoding="utf-8")
    rewrite_ready(foreign)
    with pytest.raises(ValueError, match="configured candidate"):
        export_package(world.run, world.bundle, foreign, world.manifest, tmp_path / "out-b")


def test_export_rejects_infeasible_report(world, tmp_path):
    changed = tmp_path / "cal-infeasible"
    _repacked_calibration(world.calibration, changed)
    report = json.loads((changed / "report.json").read_text(encoding="utf-8"))
    report["feasible"] = False
    report["selected_checkpoint"] = None
    report["selected_threshold"] = None
    (changed / "report.json").write_text(json.dumps(report), encoding="utf-8")
    rewrite_ready(changed)
    with pytest.raises(ValueError, match="not feasible"):
        export_package(world.run, world.bundle, changed, world.manifest, tmp_path / "out")


def test_export_rejects_selection_that_is_not_rule_winner(world, tmp_path):
    """An internally consistent selection that loses the deterministic rule must fail."""
    changed = tmp_path / "cal-winner"
    _repacked_calibration(world.calibration, changed)
    report = json.loads((changed / "report.json").read_text(encoding="utf-8"))
    other = [r for r in report["records"] if r["threshold"] != report["selected_threshold"]][0]
    report["selected_threshold"] = other["threshold"]
    (changed / "report.json").write_text(json.dumps(report), encoding="utf-8")
    selection = json.loads((changed / "selected_checkpoint.json").read_text(encoding="utf-8"))
    selection["threshold"] = other["threshold"]
    selection["metrics"] = other["metrics"]
    selection["binding"] = other["binding"]
    (changed / "selected_checkpoint.json").write_text(json.dumps(selection), encoding="utf-8")
    rewrite_ready(changed)
    with pytest.raises(ValueError, match="winner"):
        export_package(world.run, world.bundle, changed, world.manifest, tmp_path / "out")


def test_export_rejects_changed_sessions_manifest(world, tmp_path):
    other = tmp_path / "other-sessions"
    other.mkdir()
    raw = world.manifest.read_text(encoding="utf-8")
    value = json.loads(raw)
    value["sessions"] = value["sessions"][:1]
    (other / "sessions.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="inputs changed|protocol changed|identity changed"):
        export_package(world.run, world.bundle, world.calibration, other / "sessions.json",
                       tmp_path / "out", path_root=world.audio_root)


def test_export_rejects_foreign_same_structure_run(world, tmp_path):
    """The same calibration bound to a different-weight model B must be refused."""
    with pytest.raises(ValueError, match="binding mismatch"):
        export_package(world.run_b, world.bundle, world.calibration, world.manifest, tmp_path / "out-b")


def test_export_strict_json_rejects_duplicate_keys_and_nan(world, tmp_path):
    broken = tmp_path / "cal-json"
    broken.mkdir()
    for item in world.calibration.iterdir():
        (broken / item.name).write_bytes(item.read_bytes())
    report_raw = (broken / "report.json").read_text(encoding="utf-8")
    duplicated = report_raw[:-1] + ', "schema": "easykws-calibration-report/v1"}'
    (broken / "report.json").write_text(duplicated, encoding="utf-8")
    rewrite_ready(broken)
    with pytest.raises(ValueError):
        export_package(world.run, world.bundle, broken, world.manifest, tmp_path / "out1")
    nan = tmp_path / "cal-nan"
    nan.mkdir()
    for item in world.calibration.iterdir():
        (nan / item.name).write_bytes(item.read_bytes())
    (nan / "report.json").write_text(report_raw.replace('"feasible": true', '"feasible": NaN'), encoding="utf-8")
    rewrite_ready(nan)
    with pytest.raises(ValueError):
        export_package(world.run, world.bundle, nan, world.manifest, tmp_path / "out2")


def test_export_failure_leaves_no_ready_and_inputs_unchanged(world, tmp_path, monkeypatch):
    inputs = {name: inventory(world.root / name) for name in PROTECTED}
    inputs["manifest"] = inventory(world.manifest.parent)
    import easykws.export.core as core
    monkeypatch.setattr(core, "_replay_scores", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected replay failure")))
    with pytest.raises(RuntimeError, match="injected replay failure"):
        export_package(world.run, world.bundle, world.calibration, world.manifest, tmp_path / "broken-out")
    output = tmp_path / "broken-out"
    assert (output / "FAILED.json").exists()
    assert not (output / "READY.json").exists() and not (output / "manifest.json").exists()
    after = {name: inventory(world.root / name) for name in PROTECTED}
    after["manifest"] = inventory(world.manifest.parent)
    assert after == inputs


def test_package_rejects_tampered_tensor_even_with_fresh_hashes(world, package_copy):
    saved = torch.load(package_copy / "model.pt", weights_only=True)
    saved["state_dict"]["classifier.weight"][0, 0] += 0.5
    torch.save(saved, package_copy / "model.pt")
    rewrite_manifest_hashes(package_copy)
    rewrite_ready(package_copy)
    with pytest.raises(DeploymentError, match="inference content"):
        load_model(package_copy, allow_test_model=True)


def test_package_accepts_repacked_container_with_identical_content(world, package_copy):
    saved = torch.load(package_copy / "model.pt", weights_only=True)
    repacked = {"schema": saved["schema"], "model_config": saved["model_config"],
                "state_dict": dict(reversed(list(saved["state_dict"].items()))), "extra_note": "same tensors"}
    torch.save(repacked, package_copy / "model.pt")
    assert hashlib.sha256((package_copy / "model.pt").read_bytes()).hexdigest() != world.summary["model_sha256"]
    rewrite_manifest_hashes(package_copy)
    rewrite_ready(package_copy)
    loaded = load_model(package_copy, allow_test_model=True)
    assert loaded.state_sha256 == world.summary["state_sha256"]
