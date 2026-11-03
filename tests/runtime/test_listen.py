"""Microphone entry acceptance with injected fake devices; no real recording."""
import json
import os
import queue as queue_module
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from easykws.runtime import listen
from easykws.runtime.listen import AudioFailure, main, run_listener

from helpers import SRC

BLOCK = 1600


class FakeHandle:
    def __init__(self):
        self.started = self.stopped = self.closed = False
        self.active_value = True

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        self.active_value = False

    def close(self):
        self.closed = True

    @property
    def active(self):
        return self.active_value


class FakeBackend:
    """Deterministic stand-in for the sounddevice backend."""

    def __init__(self, fail_open=None, devices=None, device_names=("Fake Mic",)):
        self.fail_open = fail_open
        self._devices = devices
        self.device_names = list(device_names)
        self.handle = None

    def input_devices(self):
        if self._devices is not None:
            return self._devices
        return [{"index": i, "name": name, "inputs": 1, "default_samplerate": 16000.0}
                for i, name in enumerate(self.device_names)]

    def resolve_input_device(self, device):
        devices = self.input_devices()
        if isinstance(device, int) and not isinstance(device, bool):
            if not any(entry["index"] == device for entry in devices):
                raise ValueError(f"no input device with index {device}")
            return device
        matches = [entry["index"] for entry in devices if entry["name"] == str(device)]
        if len(matches) != 1:
            raise ValueError(f"input device {device!r} did not match exactly one device")
        return matches[0]

    def open_input(self, device_index, *, on_block, on_failure):
        if self.fail_open is not None:
            raise self.fail_open
        handle = FakeHandle()
        self.handle = handle
        self.on_block = on_block
        self.on_failure = on_failure
        return handle

    def feed(self, block):
        self.on_block(block)

    def fail(self, message):
        self.on_failure(message)

    def disconnect(self):
        self.handle.active_value = False


@pytest.fixture
def loaded(world):
    from easykws.runtime import load_model
    return load_model(world.package, allow_test_model=True)


def block_list(audio, count):
    return [np.ascontiguousarray(audio[start:start + BLOCK]) for start in range(0, BLOCK * count, BLOCK)]


def feeder(backend, blocks, loaded, failure_at=None):
    """stop_when callback: feeds one block per loop pass, ends when all consumed."""
    state = {"fed": 0, "total": len(blocks)}

    def drive():
        if state["fed"] < state["total"]:
            if failure_at is not None and state["fed"] == failure_at:
                backend.fail("capture queue overflow; audio continuity is lost")
                return False
            backend.feed(blocks[state["fed"]])
            state["fed"] += 1
        return state["fed"] == state["total"] and loaded.session_samples == state["total"] * BLOCK

    return drive, state


def test_successful_listening_reports_events_and_releases_device(world, loaded, tmp_path):
    backend = FakeBackend()
    blocks = block_list(world.proxy_samples, 30)
    events = []
    drive, state = feeder(backend, blocks, loaded)
    summary = run_listener(loaded, "Fake Mic", backend=backend, emit=events.append,
                           stop_when=drive, idle_timeout=0.01)
    assert summary["stopped"] == "stop-condition"
    assert summary["samples"] == 30 * BLOCK
    assert summary["events"] == len(events)
    assert backend.handle.started and backend.handle.stopped and backend.handle.closed
    for event in events:
        assert event["session_id"] == 0
    assert loaded.recent_frames and loaded.session_id == 0
    assert not list(tmp_path.rglob("*.wav")) and not list(tmp_path.rglob("*.pcm"))


def test_queue_overflow_stops_listening_and_releases_device(world, loaded):
    backend = FakeBackend()
    blocks = block_list(world.proxy_samples, 10)
    drive, state = feeder(backend, blocks, loaded, failure_at=3)
    with pytest.raises(AudioFailure, match="overflow"):
        run_listener(loaded, "Fake Mic", backend=backend, emit=lambda e: None,
                     stop_when=drive, idle_timeout=0.01)
    assert backend.handle.started and backend.handle.stopped and backend.handle.closed


def test_device_disconnect_marks_continuity_lost(world, loaded):
    backend = FakeBackend()
    blocks = block_list(world.proxy_samples, 10)
    state = {"fed": 0}

    def drive():
        if state["fed"] < len(blocks):
            backend.feed(blocks[state["fed"]])
            state["fed"] += 1
            if state["fed"] == 5:
                backend.disconnect()
        return False

    with pytest.raises(AudioFailure, match="stopped unexpectedly"):
        run_listener(loaded, "Fake Mic", backend=backend, emit=lambda e: None,
                     stop_when=drive, idle_timeout=0.01)
    assert backend.handle.stopped and backend.handle.closed


def test_permission_error_and_unknown_device(world, loaded):
    backend = FakeBackend(fail_open=PermissionError("microphone access denied"))
    with pytest.raises(PermissionError):
        run_listener(loaded, "Fake Mic", backend=backend, emit=lambda e: None, idle_timeout=0.01)
    assert backend.handle is None
    empty = FakeBackend(device_names=())
    with pytest.raises(ValueError, match="exactly one device"):
        run_listener(loaded, "Fake Mic", backend=empty, emit=lambda e: None, idle_timeout=0.01)


def test_consumer_error_still_releases_device(world, loaded):
    backend = FakeBackend()
    blocks = block_list(world.proxy_samples, 4)
    state = {"fed": 0, "calls": 0}

    class FailingConsumer:
        """Simulates a scoring/consumer failure outside the audio callback."""

        def __getattr__(self, name):
            return getattr(loaded, name)

        def push(self, samples):
            state["calls"] += 1
            if state["calls"] >= 3:
                raise RuntimeError("injected consumer failure")
            return loaded.push(samples)

    def drive():
        if state["fed"] < len(blocks):
            backend.feed(blocks[state["fed"]])
            state["fed"] += 1
        return False

    with pytest.raises(RuntimeError, match="injected consumer failure"):
        run_listener(FailingConsumer(), "Fake Mic", backend=backend, emit=lambda e: None,
                     stop_when=drive, idle_timeout=0.01)
    assert backend.handle.stopped and backend.handle.closed


def test_keyboard_interrupt_is_graceful(world, loaded):
    backend = FakeBackend()
    blocks = block_list(world.proxy_samples, 4)
    state = {"fed": 0}

    def drive():
        if state["fed"] < len(blocks):
            backend.feed(blocks[state["fed"]])
            state["fed"] += 1
            return False
        raise KeyboardInterrupt

    summary = run_listener(loaded, "Fake Mic", backend=backend, emit=lambda e: None,
                           stop_when=drive, idle_timeout=0.01)
    assert summary["stopped"] == "interrupt"
    assert summary["samples"] == loaded.session_samples
    assert backend.handle.stopped and backend.handle.closed


def test_cli_modes_usage_errors_check_and_fake_device(world, monkeypatch, capsys, tmp_path):
    assert main([]) == 2  # help only, no device opened
    for usage_args in (["--model", str(world.package), "--check", "--device", "0"],
                       ["--list-devices", "--model", str(world.package)], ["--device", "0"]):
        with pytest.raises(SystemExit) as excinfo:
            main(usage_args)
        assert excinfo.value.code == 2
    capsys.readouterr()
    assert main(["--model", str(world.package), "--check", "--allow-test-model"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["test_model"] is True and summary["test_model_allowed"] is True

    backend = FakeBackend()
    monkeypatch.setattr(listen, "SoundDeviceBackend", lambda: backend)
    blocks = block_list(world.proxy_samples, 6)
    state = {"fed": 0}

    class InterruptingQueue(queue_module.Queue):
        def get(self, *args, **kwargs):
            if state["fed"] < len(blocks):
                backend.feed(blocks[state["fed"]])
                state["fed"] += 1
            if state["fed"] >= len(blocks):
                raise KeyboardInterrupt
            return super().get(*args, **kwargs)

    monkeypatch.setattr(listen.queue, "Queue", InterruptingQueue)
    code = main(["--model", str(world.package), "--device", "Fake Mic", "--allow-test-model"])
    assert code == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out.strip().splitlines()[-1])
    assert result["listening_stopped"] == "interrupt"
    assert backend.handle.stopped and backend.handle.closed
    assert captured.err.count("test/diagnostic model") == 1


def test_cli_overflow_exit_code_is_three(world, monkeypatch, capsys):
    backend = FakeBackend()
    state = {"failed": False}

    class PoisonedQueue(queue_module.Queue):
        def get(self, *args, **kwargs):
            if not state["failed"]:
                state["failed"] = True
                backend.fail("capture queue overflow; audio continuity is lost")
                raise queue_module.Empty
            return super().get(*args, **kwargs)

    monkeypatch.setattr(listen.queue, "Queue", PoisonedQueue)
    monkeypatch.setattr(listen, "SoundDeviceBackend", lambda: backend)
    code = main(["--model", str(world.package), "--device", "Fake Mic", "--allow-test-model"])
    assert code == 3
    assert "continuity" in capsys.readouterr().err
    assert backend.handle.stopped and backend.handle.closed


def test_check_and_list_devices_never_import_sounddevice(world, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "sounddevice.py").write_text("raise ImportError('blocked for test')\n", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(blocker) + os.pathsep + str(SRC), PYTHONUTF8="1",
               PYTHONDONTWRITEBYTECODE="1")
    check = subprocess.run([sys.executable, "-B", "-m", "easykws.runtime", "--model", str(world.package),
                            "--check", "--allow-test-model"], capture_output=True, text=True, env=env,
                           cwd=str(tmp_path))
    assert check.returncode == 0, check.stderr
    assert json.loads(check.stdout)["test_model"] is True
    listing = subprocess.run([sys.executable, "-B", "-m", "easykws.runtime", "--list-devices"],
                             capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert listing.returncode == 2
    assert "easykws[listen]" in listing.stderr


def test_real_backend_resolves_numeric_string_index_and_exact_name(monkeypatch):
    """P1: CLI passes strings; the real backend must accept '0' as an index."""
    import sounddevice
    monkeypatch.setattr(sounddevice, "query_devices",
                        lambda: [{"name": "Example Mic", "max_input_channels": 1,
                                  "default_samplerate": 16000.0}])
    backend = listen.SoundDeviceBackend()  # real class; only the low-level query is faked
    assert backend.resolve_input_device(0) == 0
    assert backend.resolve_input_device("0") == 0
    assert backend.resolve_input_device("Example Mic") == 0
    for bad in ("9", "-1", "Nope", "Example Mic 2"):
        with pytest.raises(ValueError):
            backend.resolve_input_device(bad)


def test_cli_numeric_device_string_reaches_real_resolution(world, monkeypatch, capsys):
    """P1: main -> argparse -> real SoundDeviceBackend.resolve_input_device('0')."""
    import sounddevice
    monkeypatch.setattr(sounddevice, "query_devices",
                        lambda: [{"name": "Example Mic", "max_input_channels": 1,
                                  "default_samplerate": 16000.0}])

    def refusing_open(self, device_index, *, on_block, on_failure):
        raise RuntimeError("injected-open:" + str(device_index))

    monkeypatch.setattr(listen.SoundDeviceBackend, "open_input", refusing_open)
    # main reports failures on stderr with a non-zero exit; the message proves
    # resolution succeeded and the injected open stub was reached.
    assert main(["--model", str(world.package), "--device", "0", "--allow-test-model"]) == 2
    assert "RuntimeError: injected-open:0" in capsys.readouterr().err
    assert main(["--model", str(world.package), "--device", "Example Mic", "--allow-test-model"]) == 2
    assert "RuntimeError: injected-open:0" in capsys.readouterr().err
    assert main(["--model", str(world.package), "--device", "Nope", "--allow-test-model"]) == 2
    assert "did not match exactly one device" in capsys.readouterr().err


def test_listen_extra_declares_runtime_third_party_imports():
    """P1: the listen extra must cover everything the CLI import chain needs."""
    import ast
    import re
    import sys
    import tomllib
    pyproject = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {re.split(r"[<>=!~\[;]", requirement)[0].strip().lower().replace("_", "-")
                for requirement in pyproject["project"]["optional-dependencies"]["listen"]}
    package_root = Path(__file__).resolve().parents[2] / "src" / "easykws"
    sources = [package_root / name for name in ("runtime", "export", "calibration", "training", "preprocess")]
    sources += [package_root / "streaming.py", package_root / "trigger.py"]
    third_party = set()
    for root in sources:
        for path in (sorted(root.rglob("*.py")) if root.is_dir() else [root]):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    third_party.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    third_party.add(node.module.split(".")[0])
    third_party -= {"easykws"} | set(sys.stdlib_module_names)
    missing = {name for name in third_party if name.lower() not in declared}
    assert not missing, f"listen extra is missing declared dependencies: {sorted(missing)}"
