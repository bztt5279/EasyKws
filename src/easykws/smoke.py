"""End-to-end synthetic smoke pipeline: generated tones to deployment replay.

This is the single reproducible entry point for users and CI. It generates
fictional tones and metadata only, runs the formal preparation, expansion
import, preprocessing, training, calibration and export implementations, and
replays the exported package in a fresh Python process. Every artifact stays
test-purpose; nothing here demonstrates recognition quality, and no real
recordings, engines or public corpora are involved. All outputs are written
to a new user-specified directory outside the repository.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import wave
from pathlib import Path

SAMPLES = 16000
SESSION_SAMPLES = 48000


def _require_full_dependencies():
    try:
        import numpy  # noqa: F401
        import soundfile  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"the smoke pipeline needs the data extras: "
                         f"pip install '.[preprocess]' (missing: {exc})") from exc


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _wav(path, samples, rate=16000):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return _sha(path)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _inventory(root):
    return {str(path.relative_to(root)): _sha(path) for path in Path(root).rglob("*") if path.is_file()}


def _stage(summary, name, description):
    summary["stages"].append({"stage": name, "description": description, "checks": {}})
    return summary["stages"][-1]


def generate(root):
    """Source recordings, fixture resources, negative-sample metadata, sessions."""
    import numpy as np
    audio = root / "source-audio"
    from .prep import v2
    document = {"schema": v2.SCHEMA, "datasets": [
        {"dataset_id": "smoke-tones", "source": "generated-test-signals", "version": "v1",
         "license": "test-only", "split_policy": "speaker"}], "resources": [],
        "selection": {"kind": "full", "reason": "", "available_count": 6}, "rows": []}
    frequencies = {"positive": 220, "negative": 650}
    for person in range(3):
        for label, group in ((0, "target"), (-1, "background")):
            name = f"speaker{person}_{'target' if label == 0 else 'background'}"
            n = np.arange(64000 if label == 0 else 3200)
            samples = 2500 * np.sin(2 * np.pi * (frequencies["positive" if label == 0 else "negative"]
                                                 + person * 7) * n / 16000)
            sha = _wav(audio / (name + ".wav"), samples)
            document["rows"].append({
                "sample_id": name, "dataset_id": "smoke-tones", "speaker_id": f"person{person}",
                "source_recording_id": f"recording{person}", "audio_path": name + ".wav", "audio_sha256": sha,
                "label": label, "group": group, "expected_phrase": "smoke" if label == 0 else "",
                "review_status": "pass", "origin": "test-fixture", "purpose": "synthetic-test",
                "parent_sample_ids": [], "resource_ids": [],
                "requested_split": ("train", "val", "test")[person], "generation": None})
    rng = np.random.default_rng(7)
    for key, signal in (("noise_a", rng.normal(0, 1500, 8000)),
                        ("noise_b", rng.normal(0, 1600, 6000)),
                        ("impulse", np.r_[np.zeros(200), 12000, np.ones(2800) * 90])):
        sha = _wav(audio / (key + ".wav"), signal)
        document["resources"].append({"resource_id": key, "dataset_id": "smoke-tones",
                                      "audio_path": key + ".wav", "sha256": sha,
                                      "kind": "rir" if key == "impulse" else "noise",
                                      "purpose": "synthetic-test", "split": "external",
                                      "speaker_id": None, "source_sample_id": None})
    manifest = root / "recordings-v2.json"
    _write_json(manifest, document)
    return document, audio, manifest


def vc_candidates(base_manifest, audio):
    """Two fixture 'conversion' outputs over a frozen train target, with QC."""
    source = next(row for row in base_manifest["rows"] if row["requested_split"] == "train" and row["label"] == 0)
    reference = {"resource_id": "smoke-reference", "dataset_id": "smoke-tones",
                 "audio_path": source["audio_path"], "sha256": source["audio_sha256"],
                 "kind": "reference-voice", "purpose": "synthetic-test", "split": "train",
                 "speaker_id": source["speaker_id"], "source_sample_id": source["sample_id"]}
    with wave.open(str(audio / source["audio_path"]), "rb") as stream:
        raw = stream.readframes(stream.getnframes())
    samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    rows, qc = [], {}
    for index in range(2):
        name = f"vc_fixture_{index}"
        scaled = _wav(audio / (name + ".wav"), [int(value * (0.4 + 0.2 * index)) for value in samples])
        row = dict(source)
        row.update(sample_id=name, audio_path=name + ".wav", audio_sha256=scaled, origin="vc",
                   purpose="synthetic-test", parent_sample_ids=[source["sample_id"]],
                   resource_ids=[reference["resource_id"]], requested_split=None,
                   generation={"engine": "interface-fixture", "model": "none", "config_sha256": "0" * 64,
                               "seed": index, "status": "success"})
        rows.append(row)
        qc[name] = {"source_sha256": source["audio_sha256"], "output_sha256": scaled,
                    "resource_sha256": {reference["resource_id"]: reference["sha256"]},
                    "status": "ok", "reasons": []}
    return rows, [reference], qc


def negative_candidates(root):
    """Local fixture metadata through the public-negative adapter; no downloads."""
    import numpy as np

    from .prep import mobvoi
    metadata, negative_audio = root / "negative-metadata", root / "negative-audio"
    metadata.mkdir(parents=True, exist_ok=True)
    included = ("n_train", "n_dev", "n_test")
    for index, name in enumerate(mobvoi.SETS):
        record = {"utt_id": f"negative_{index}", "keyword_id": 7 if index < 3 else 8,
                  "speaker_id": f"neg-speaker-{index}"}
        _write_json(metadata / (name + ".json"), [record])
        if index < 3:
            n = np.arange(4800)
            _wav(negative_audio / (record["utt_id"] + ".wav"),
                 1800 * np.sin(2 * np.pi * (95 + 11 * index) * n / 16000))
    document, report = mobvoi.build(metadata, negative_audio, dataset_id="smoke-negatives",
                                    source="local-fixture-metadata", version="fixture-v1",
                                    license="test-only", keyword_labels={"7": -1, "8": "exclude"},
                                    purpose="synthetic-test")
    assert report["included_uids"] and report["excluded_uids"], "negative fixture must include exclusions"
    return document, report


def expansion(prepared, candidates_path, path_map, output, expected_ids, qc):
    from .prep import expand
    return expand.run(prepared, candidates_path, path_map, output,
                      qc=qc, expected_ids=expected_ids, max_total=None, max_per_source=None)


def preprocess(prepared, output, augmentation):
    from .preprocess import run
    return run(prepared, None, output, config=augmentation,
               path_map={"smoke-tones": prepared.parent.parent / "source-audio",
                         "smoke-negatives": prepared.parent.parent / "negative-audio"})


def train(features, output):
    from .training import runner
    config = {"model": {"hidden_dim": 8, "num_layers": 2, "kernel_size": 3},
              "training": {"num_epochs": 2, "batch_size": 2},
              "sampler": {"group_weights": {"target": 1, "background": 1, "public-negative": 1},
                          "draws_per_epoch": 10}}
    return runner.run(features, config, output)


def calibrate(run_dir, features, manifest, output, config_path):
    from .calibration import calibrate
    return calibrate(run_dir, features, manifest, output, config=str(config_path))


def export(run_dir, features, calibration, manifest, output):
    from .export import export_package
    return export_package(run_dir, features, calibration, manifest, output)


REPLAY = """
import json, sys
import numpy as np
import soundfile as sf
from easykws.runtime import load_model
model = load_model(sys.argv[1], allow_test_model=True)
frames = {}
for name in ("silence", "proxy", "mixed"):
    audio, rate = sf.read(sys.argv[2] + "/" + name + ".wav", dtype="float32")
    assert rate == 16000 and audio.ndim == 1
    rows, events = [], []
    for start in range(0, len(audio), 512):
        result = model.push(np.ascontiguousarray(audio[start:start + 512]))
        rows.extend(result["frames"])
        events.extend(result["events"])
    model.flush()
    model.reset()
    frames[name] = {"frames": rows, "events": events,
                    "samples": len(audio), "last_sec": rows[-1]["t_audio_sec"] if rows else 0.0}
model.flush()
print(json.dumps(frames))
"""


def verify_package(root, replay_results=None):
    """Fresh-process load, push/reset/flush replay and READY verification.

    ``replay_results`` lets tests inject a replay payload (or an invalid one)
    instead of running the subprocess; production callers always replay.
    """
    package = root / "package"
    deployed = root / "deployed-package"
    deployed.mkdir()
    for item in package.iterdir():
        (deployed / item.name).write_bytes(item.read_bytes())
    ready = json.loads((deployed / "READY.json").read_text(encoding="utf-8"))
    mismatched = [name for name, digest in ready["files"].items() if _sha(deployed / name) != digest]
    assert not mismatched, f"READY hash mismatch: {mismatched}"
    session_audio = root / "session-audio"
    replay_audio = root / "replay-audio"
    replay_audio.mkdir()
    for name in ("silence.wav", "proxy.wav", "mixed.wav"):
        (replay_audio / name).write_bytes((session_audio / name).read_bytes())
    if replay_results is None:
        script = root / "replay_smoke.py"
        script.write_text(REPLAY, encoding="utf-8")
        # The replay runs in a fresh process with cwd outside the package. Point
        # it at whatever parent directory provides this easykws (src checkout or
        # site-packages), so the child imports the same installation, no others.
        package_parent = str(Path(__file__).resolve().parents[1])
        existing = os.environ.get("PYTHONPATH")
        env = dict(os.environ, PYTHONPATH=package_parent + (os.pathsep + existing if existing else ""),
                   PYTHONUTF8="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        done = subprocess.run([sys.executable, "-B", "-c", REPLAY, str(deployed), str(replay_audio)],
                              capture_output=True, text=True, encoding="utf-8", cwd=str(root), env=env)
        assert done.returncode == 0, f"fresh-process replay failed: {done.stderr}"
        frames = json.loads(done.stdout.strip().splitlines()[-1])
    else:
        frames = replay_results
    selection = json.loads((root / "calibration" / "selected_checkpoint.json").read_text(encoding="utf-8"))
    scores_path = sorted((root / "calibration").glob("scores_*.json"))
    saved = next(json.loads(path.read_text(encoding="utf-8")) for path in scores_path
                 if json.loads(path.read_text(encoding="utf-8"))["candidate"] == selection["candidate"])
    assert set(frames) == set(saved["sessions"]), \
        f"replay session set differs: {sorted(frames)} vs {sorted(saved['sessions'])}"
    binding = selection["binding"]
    sample_rate = 16000
    reported = {}
    for name, payload in frames.items():
        rows = payload["frames"]
        assert rows == saved["sessions"][name], f"replay frames differ for {name}"
        # Recompute the calibration events over the saved scores with the
        # bound trigger and require the replayed events to match field by
        # field; only the runtime session_id has no calibration counterpart.
        from .trigger import FreshFrameTrigger
        trigger = FreshFrameTrigger(binding["trigger_config"]["threshold"],
                                    binding["trigger_config"]["rules"])
        reference = [dict(event) for row in rows for event in
                     trigger.update(row["score"], row["t_audio_sec"])]
        replayed = [{key: value for key, value in event.items() if key != "session_id"}
                    for event in payload["events"]]
        assert replayed == reference, f"replay events differ for {name}"
        assert payload["samples"] == SESSION_SAMPLES, f"replayed sample count wrong for {name}"
        expected_last = ((len(rows) - 1) * 160 + 400) / sample_rate if rows else 0.0
        assert payload["last_sec"] == expected_last, f"replay clock wrong for {name}"
        reported[name] = {"frames": len(rows), "events": len(replayed),
                          "samples": payload["samples"], "last_sec": payload["last_sec"]}
    return {"ready_files_verified": len(ready["files"]), "fresh_process_frames":
            {name: payload["frames"] for name, payload in reported.items()},
            "events": {name: payload["events"] for name, payload in reported.items()},
            "clocks_sec": {name: payload["last_sec"] for name, payload in reported.items()},
            "selected": selection["candidate"], "threshold": selection["threshold"]}


def sessions_manifest(root):
    """Three independent sessions: silence, proxy tone, silence-signal-silence."""
    import numpy as np
    import soundfile as sf
    audio_root = root / "session-audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    n = SESSION_SAMPLES
    times = np.arange(n, dtype=np.float64) / 16000
    tone = (.1 * np.sin(2 * np.pi * 733 * times[16000:32000])).astype(np.float32)
    mixed = np.zeros(n, np.float32)
    mixed[8000:12000] = (.1 * np.sin(2 * np.pi * 733 * times[8000:12000])).astype(np.float32)
    mixed[20000:28000] = (.1 * np.sin(2 * np.pi * 733 * times[20000:28000])).astype(np.float32)
    definitions = (("silence", np.zeros(n, np.float32), [],
                    [{"start_sec": 0., "end_sec": 3.}]),
                   ("proxy", np.pad(tone, (16000, n - 16000 - tone.size)), [],
                    [{"start_sec": 0., "end_sec": 1.}, {"start_sec": 2., "end_sec": 3.}]),
                   ("mixed", mixed, [{"id": "window-1", "start_sec": 1.2, "end_sec": 1.9}],
                    [{"start_sec": 0., "end_sec": 1.2}, {"start_sec": 1.9, "end_sec": 2.4},
                     {"start_sec": 2.8, "end_sec": 3.}]))
    sessions = []
    for name, audio, positives, negatives in definitions:
        path = audio_root / (name + ".wav")
        sf.write(path, audio, 16000, subtype="PCM_16")
        sessions.append({"id": name, "path": path.name, "sha256": _sha(path), "duration_samples": n,
                         "recording_kind": "synthetic-concatenation", "speaker_ids": [],
                         "positive_windows": positives, "negative_intervals": negatives})
    manifest = {"schema": "easykws-calibration-sessions/v1", "purpose": "synthetic-test",
                "split": "calibration", "sample_rate": 16000, "channels": 1, "clock": "samples",
                "boundary_policy": "reset-each-session", "gap_policy": "split-session",
                "provenance": {"source_id": "easykws-smoke",
                               "description": "generated silence and tones; interface verification only"},
                "sessions": sessions}
    path = audio_root / "sessions.json"
    _write_json(path, manifest)
    calibration_config = {"schema": "easykws-calibration-config/v1",
                          "candidates": ["epoch_0001_attempt_0001.pt", "epoch_0002_attempt_0001.pt"],
                          "threshold_grid": [.1, .5, .9], "min_recall": 0., "max_fah": None,
                          "trigger": {}, "chunk_samples": 512,
                          "selection_rule": "recall-desc/fah-asc/threshold-desc/candidate-asc"}
    config_path = root / "calibration-config.json"
    _write_json(config_path, calibration_config)
    return path, config_path


def environment_info():
    from . import __version__
    try:
        import tomllib
        pyproject = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8"))
        version = pyproject["project"]["version"]
    except (OSError, KeyError):
        version = __version__
    info = {"package_version": version, "python": platform.python_version(),
            "executable": sys.executable, "platform": sys.platform}
    try:
        done = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                              cwd=str(Path(__file__).resolve().parents[2]))
        if done.returncode == 0:
            head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                  cwd=done.stdout.strip())
            dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                   cwd=done.stdout.strip())
            info["git_head"] = head.stdout.strip()
            info["git_dirty"] = bool(dirty.stdout.strip())
        else:
            info["git_head"] = None
            info["git_dirty"] = None
    except OSError:
        info["git_head"] = None
        info["git_dirty"] = None
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["execution_device"] = "cpu"
    return info


def run_pipeline(root):
    root = Path(root).resolve()
    # One smoke-wide execution profile: every stage binds and replays the same
    # thread setting, independent of the caller's OMP/MKL environment. Without
    # this, a default 8-thread shell would calibrate at 8 while the replay
    # subprocess pins 1, and the runtime would correctly refuse the mismatch.
    import torch
    torch.set_num_threads(1)
    summary = {"pipeline": "easykws-smoke", "budget": {"source_recordings": 6, "resources": 3,
               "vc_fixtures": 2, "negative_fixture_rows": 6, "train_budget_draws": 20,
               "calibration_sessions": 3, "calibration_thresholds": 3, "candidates": 2},
               "stages": [], "environment": environment_info()}
    # Stage 1: source generation.
    stage = _stage(summary, "generate", "source tones, fixture noise/RIR resources")
    base_manifest, audio, manifest_path = generate(root)
    stage["checks"] = {"source_rows": len(base_manifest["rows"]), "resources": len(base_manifest["resources"]),
                       "manifest": manifest_path.name}
    # Stage 2: formal v2 preparation with speaker splits.
    stage = _stage(summary, "prepare", "v2.prepare speaker split of generated recordings")
    from .prep import v2
    prepared = root / "prepared"
    v2.prepare(manifest_path, {"smoke-tones": audio}, prepared)
    stage["checks"] = {"prepared_ready_sha256": _sha(prepared / "READY.json")}
    # Stage 3: fixture VC conversion + public-negative adapter import with QC.
    stage = _stage(summary, "expand", "fixture VC candidates with QC and local negative metadata via adapter")
    vc_rows, resources, qc = vc_candidates(base_manifest, audio)
    negative_document, negative_report = negative_candidates(root)
    source_dataset = copy.deepcopy(base_manifest["datasets"][0])
    candidates = {"schema": v2.SCHEMA, "datasets": [source_dataset, *negative_document["datasets"]],
                  "resources": resources, "selection": {
                      "kind": "full", "reason": "",
                      "available_count": len(vc_rows) + len(negative_document["rows"])},
                  "rows": sorted(vc_rows + negative_document["rows"], key=lambda row: row["sample_id"])}
    candidates_path = root / "candidates.json"
    _write_json(candidates_path, candidates)
    expanded = root / "expanded"
    report = expansion(prepared, candidates_path, {"smoke-tones": audio, "smoke-negatives": root / "negative-audio"},
                       expanded, expected_ids=[row["sample_id"] for row in candidates["rows"]], qc=qc)
    stage["checks"] = {"accepted": report["counts"].get("accepted"), "all_counts": report["counts"],
                       "purpose": report["purpose"], "data_flags": report["data_flags"],
                       "excluded_uids": len(negative_report["excluded_uids"])}
    assert report["purpose"] == "synthetic-test", "expanded preparation must remain synthetic-test"
    inputs = {"source-audio": _inventory(audio), "negative-audio": _inventory(root / "negative-audio")}
    # Stage 4: offline augmentation gain/speed/noise/RIR/filter + fbank/CMVN.
    stage = _stage(summary, "preprocess", "offline augmentation over five kinds, fbank, train-only CMVN")
    augmentation = {"augmentation": {"seed": 11,
                                    "selectors": {"groups": ["target", "background", "public-negative"],
                                                  "labels": [0, -1], "origins": ["test-fixture", "vc", "public"]},
                                    "variants": {"regular": 1, "challenge": 1},
                                    "kinds": ["gain", "speed", "noise", "rir", "filter"],
                                    "speed": [0.9, 1.1],
                                    "noise": {"resource_ids": ["noise_a", "noise_b"], "snr_db": [10, 10]},
                                    "rir": {"resource_ids": ["impulse"], "channel": 0}}}
    features = root / "features"
    result = preprocess(expanded / "prepared", features, augmentation)
    stage["checks"] = {"splits": result["splits"], "included_count": result["included_count"],
                       "generated": result["augmentation"]["generated_count"],
                       "cmvn_fit_split": json.loads((features / "cmvn.json").read_text(encoding="utf-8"))["fit_split"],
                       "purpose": result["purpose"]}
    assert result["purpose"] == "synthetic-test"
    train_rows = [json.loads(line) for line in (features / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(train_rows) <= 100, "smoke budget exceeded"
    # Stage 5: limited CPU training.
    stage = _stage(summary, "train", "two epochs, ten draws each, small causal model")
    status = train(features, root / "run")
    stage["checks"] = {"epochs": status["last_complete_epoch"], "global_steps": status["global_step"],
                       "candidates": [c["path"] for c in status["candidates"]],
                       "purpose": status["purpose"], "selection_status": status["selection_status"]}
    assert status["global_step"] <= 20 and status["state"] == "completed"
    assert status["selected_checkpoint"] is None, "training must not select; calibration owns the choice"
    # Stage 6: streaming calibration over three sessions.
    stage = _stage(summary, "calibrate", "three generated sessions, two candidates, three thresholds")
    manifest_path, config_path = sessions_manifest(root)
    inputs["session-audio"] = _inventory(root / "session-audio")
    report = calibrate(root / "run", features, manifest_path, root / "calibration", config_path)
    assert report["feasible"] and report["selected_checkpoint"], "smoke calibration must be feasible"
    stage["checks"] = {"selected": report["selected_checkpoint"], "threshold": report["selected_threshold"],
                       "records": len(report["records"]), "purpose": report["purpose"],
                       "data_flags": report["data_flags"]}
    # Stage 7: export.
    stage = _stage(summary, "export", "verified export of the selected calibration record")
    exported = export(root / "run", features, root / "calibration", manifest_path, root / "package")
    stage["checks"] = {"test_model": exported["test_model"], "purpose": exported["purpose"],
                       "state_sha256": exported["state_sha256"],
                       "parameter_count": exported["parameter_count"]}
    # Stage 8: fresh-process load and replay.
    stage = _stage(summary, "replay", "new process loads moved package, push/reset/flush, compares calibration frames")
    stage["checks"] = verify_package(root)
    # Generated inputs unchanged across the pipeline.
    after = {"source-audio": _inventory(root / "source-audio"),
             "negative-audio": _inventory(root / "negative-audio"),
             "session-audio": _inventory(root / "session-audio")}
    assert after == {**inputs, "session-audio": inputs["session-audio"]}, "generated inputs changed"
    summary["source_inputs_unchanged"] = True
    summary["conclusion"] = ("All stages executed on generated fixtures; every artifact is synthetic-test; "
                             "this verifies interfaces and reproducibility, not recognition quality.")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="easykws-smoke", description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new directory for all generated data, models and packages; must not exist")
    args = parser.parse_args(argv)
    _require_full_dependencies()
    root = args.output.resolve()
    if root.exists():
        parser.error(f"output directory already exists: {root}")
    root.mkdir(parents=True)
    try:
        summary = run_pipeline(root)
    except BaseException as exc:
        _write_json(root / "FAILED.json", {"error": type(exc).__name__, "message": str(exc)})
        raise
    _write_json(root / "summary.json", summary)
    replay_checks = summary["stages"][-1]["checks"]
    print(json.dumps({"summary": str(root / "summary.json"),
                      "selected": replay_checks["selected"], "threshold": replay_checks["threshold"],
                      "fresh_process_frames": replay_checks["fresh_process_frames"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
