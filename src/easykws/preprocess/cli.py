"""CLI for verified, explicitly configured feature preprocessing."""
import argparse
import json
from pathlib import Path
import sys

from ..prep.core import load_json
from .core import plan, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify prepared recordings and extract raw KWS features")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--prepared", type=Path, required=True)
        roots = command.add_mutually_exclusive_group(required=True)
        roots.add_argument("--audio-root", type=Path)
        roots.add_argument("--path-map", type=Path)
        command.add_argument("--config", type=Path)
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_json(args.config.read_text(encoding="utf-8-sig")) if args.config else None
        path_map = load_json(args.path_map.read_text(encoding="utf-8-sig")) if args.path_map else None
        if args.command == "run":
            result = run(args.prepared, args.audio_root, args.output, config, path_map=path_map)
        else:
            result = plan(args.prepared, args.audio_root, config, path_map=path_map)
            result = {key: value for key, value in result.items() if key not in {"rows", "config"}}
    except ImportError:
        print('Missing preprocessing dependencies; install "easykws[preprocess]".', file=sys.stderr)
        return 2
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Preprocessing failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
