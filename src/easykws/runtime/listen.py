"""Interactive microphone entry point for a verified deployment package.

This version is accepted against simulated devices only; real-hardware
listening is explicitly unverified. The audio callback only copies blocks
into a bounded queue; scoring and event output happen outside the callback.
Overflow or device failure stops the session with a non-zero status instead
of pretending continuity. No audio is ever written to disk.
"""
import argparse
import json
import queue
import sys
from pathlib import Path

from .api import DeploymentError
from .package import load_model

BLOCKSIZE = 1600
QUEUE_BLOCKS = 100
SAMPLE_RATE = 16000
EXIT_USAGE = 2
EXIT_AUDIO = 3


class AudioFailure(RuntimeError):
    """Real device failure or overflow; the observed stream is not continuous."""


class SoundDeviceBackend:
    """Default PortAudio backend; imported only when a device is actually used."""

    def __init__(self):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError(
                "audio input requires the optional sounddevice dependency; "
                "install it with: pip install 'easykws[listen]'") from exc
        self._sd = sd

    def input_devices(self):
        return [{"index": index, "name": device["name"], "inputs": device["max_input_channels"],
                 "default_samplerate": device["default_samplerate"]}
                for index, device in enumerate(self._sd.query_devices())
                if device["max_input_channels"] > 0]

    def resolve_input_device(self, device):
        """Resolve an index (int or digit string) or an exact device name.

        CLI arguments always arrive as strings; digit strings are parsed as
        PortAudio indexes. A device whose name is a digit string cannot be
        selected by name; pass its index instead.
        """
        devices = self.input_devices()
        if isinstance(device, bool):
            raise ValueError("device must be an input device index or an exact device name")
        if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            index = int(device)
            if not any(entry["index"] == index for entry in devices):
                raise ValueError(f"no input device with index {index}")
            return index
        name = str(device)
        matches = [entry["index"] for entry in devices if entry["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"input device {name!r} did not match exactly one device")
        return matches[0]

    def open_input(self, device_index, *, on_block, on_failure):
        holder = StreamHandle(self._sd)
        stream = self._sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCKSIZE,
            device=device_index, callback=holder.make_callback(on_block, on_failure),
            finished_callback=lambda: holder.mark_finished())
        holder.attach(stream)
        return holder


class StreamHandle:
    """Started input stream wrapper; callback work stays copy-and-enqueue only."""

    def __init__(self, sd):
        self._sd = sd
        self._stream = None
        self.finished = False

    def attach(self, stream):
        self._stream = stream

    def make_callback(self, on_block, on_failure):
        def callback(indata, frames, time_info, status):
            if status:
                on_failure("device reported: " + str(status))
                return
            try:
                on_block(indata.copy())
            except queue.Full:
                on_failure("capture queue overflow; audio continuity is lost")
        return callback

    def mark_finished(self):
        self.finished = True

    @property
    def active(self):
        return self._stream is not None and self._stream.active

    def start(self):
        self._stream.start()

    def stop(self):
        try:
            self._stream.stop()
        except self._sd.PortAudioError:
            pass

    def close(self):
        try:
            self._stream.close()
        except self._sd.PortAudioError:
            pass


def run_listener(model, device, *, allow_test_model=False, backend=None, emit=None,
                 stop_when=None, idle_timeout=0.5):
    """Load checks first, then open the input stream and report events.

    Returns a summary dictionary. AudioFailure signals overflow or device
    loss; KeyboardInterrupt is a graceful stop by the user.
    """
    emit = emit or (lambda event: None)
    if backend is None:
        backend = SoundDeviceBackend()
    index = backend.resolve_input_device(device)
    blocks = queue.Queue(maxsize=QUEUE_BLOCKS)

    def on_block(block):
        blocks.put_nowait(block)

    def on_failure(message):
        failures.append(message)

    failures = []
    handle = backend.open_input(index, on_block=on_block, on_failure=on_failure)
    started = False
    try:
        handle.start()
        started = True
        events = 0
        session_id = model.session_id
        try:
            while True:
                if failures:
                    raise AudioFailure(failures[0])
                if stop_when is not None and stop_when():
                    break
                if not handle.active:
                    raise AudioFailure("audio device stream stopped unexpectedly")
                try:
                    block = blocks.get(timeout=idle_timeout)
                except queue.Empty:
                    continue
                result = model.push(block.reshape(-1))
                for event in result["events"]:
                    emit(event)
                    events += 1
        except KeyboardInterrupt:
            return {"stopped": "interrupt", "session_id": session_id, "events": events,
                    "samples": model.session_samples, "sec": model.session_sec, "released": True}
        return {"stopped": "stop-condition", "session_id": session_id, "events": events,
                "samples": model.session_samples, "sec": model.session_sec, "released": True}
    finally:
        if started:
            handle.stop()
        handle.close()


def _print_event(event):
    print(json.dumps(event, ensure_ascii=False, sort_keys=True))


def _cmd_check(model_path, allow_test_model):
    loaded = load_model(model_path, allow_test_model=allow_test_model)
    print(json.dumps(loaded.summary(), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
    return 0


def _cmd_list_devices():
    backend = SoundDeviceBackend()
    for entry in backend.input_devices():
        print(json.dumps(entry, ensure_ascii=False, sort_keys=True))
    return 0


def _cmd_listen(model_path, device, allow_test_model):
    loaded = load_model(model_path, allow_test_model=allow_test_model)
    if loaded.is_test_model:
        print("test/diagnostic model in use; purpose=" + loaded.purpose +
              " flags=" + ",".join(loaded.data_flags), file=sys.stderr)
    summary = run_listener(loaded, device, emit=_print_event)
    print(json.dumps({"listening_stopped": summary["stopped"], "session_id": summary["session_id"],
                      "events": summary["events"], "samples": summary["samples"],
                      "sec": summary["sec"]}, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="easykws-listen",
        description="Load a verified deployment package and, only on explicit request, "
                    "listen on a chosen 16 kHz mono input device. Simulated-device acceptance "
                    "only in this version; real hardware behavior is unverified.")
    parser.add_argument("--model", help="deployment package directory")
    parser.add_argument("--check", action="store_true",
                        help="load and verify the package only; never opens a device")
    parser.add_argument("--device", metavar="DEVICE",
                        help="input device index or exact name; listening requires it explicitly")
    parser.add_argument("--list-devices", action="store_true",
                        help="list input devices; never starts recording")
    parser.add_argument("--allow-test-model", action="store_true",
                        help="allow test/diagnostic packages; shows their markers and lifts only the purpose restriction")
    args = parser.parse_args(argv)
    try:
        if args.list_devices:
            if args.model or args.check or args.device is not None:
                parser.error("--list-devices cannot be combined with --model/--check/--device")
            return _cmd_list_devices()
        if args.check:
            if not args.model or args.device is not None:
                parser.error("--check requires --model and cannot be combined with --device")
            return _cmd_check(args.model, args.allow_test_model)
        if args.device is not None:
            if not args.model:
                parser.error("--device requires --model")
            return _cmd_listen(args.model, args.device, args.allow_test_model)
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    except AudioFailure as exc:
        print("listening stopped: " + str(exc), file=sys.stderr)
        print("audio continuity was lost; run the command again for a new session", file=sys.stderr)
        return EXIT_AUDIO
    except (DeploymentError, ValueError, OSError, RuntimeError) as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 0
