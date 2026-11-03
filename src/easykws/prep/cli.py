"""Command-line entry point for preparation."""
import argparse
import json
from pathlib import Path
import sys

from .core import PrepError, load_json, plan, prepare
from . import v2


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate and split your recording manifest")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("expand-plan", "expand-run"):
        command = commands.add_parser(name)
        command.add_argument("--prepared", type=Path, required=True)
        command.add_argument("--candidates", type=Path, required=True)
        command.add_argument("--path-map", type=Path, required=True)
        command.add_argument("--qc", type=Path)
        command.add_argument("--expected-ids", type=Path)
        command.add_argument("--max-total", type=int)
        command.add_argument("--max-per-source", type=int)
        command.add_argument("--seed", type=int, default=0)
        if name == "expand-run":
            command.add_argument("--output", type=Path, required=True)
    public = commands.add_parser("import-mobvoi")
    for name in ("metadata", "audio-root", "keyword-labels", "output"):
        public.add_argument("--" + name, type=Path, required=True)
    for name in ("dataset-id", "source", "version", "license"):
        public.add_argument("--" + name, required=True)
    public.add_argument("--purpose", choices=("training-data", "synthetic-test"), default="training-data")
    for name in ("plan", "run", "plan-v2", "run-v2", "upgrade-v1"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name.endswith("-v2"):
            command.add_argument("--path-map", type=Path, required=True)
        else:
            command.add_argument("--audio-root", type=Path, required=True)
        command.add_argument("--config", type=Path)
        if name in ("run", "run-v2", "upgrade-v1"):
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in ("expand-plan", "expand-run"):
            from . import expand
            paths = load_json(args.path_map.read_text(encoding="utf-8-sig"))
            options = {"qc": load_json(args.qc.read_text(encoding="utf-8-sig")) if args.qc else None,
                       "expected_ids": load_json(args.expected_ids.read_text(encoding="utf-8-sig")) if args.expected_ids else None,
                       "max_total": args.max_total, "max_per_source": args.max_per_source, "seed": args.seed}
            if args.command == "expand-run":
                result = expand.run(args.prepared, args.candidates, paths, args.output, **options)
            else:
                result = expand.plan(args.prepared, args.candidates, paths, **options)["report"]
            summary = {key: result[key] for key in ("counts", "coverage", "purpose", "data_flags")}
            summary.update(candidate_count=len(result["candidate_ids"]), accepted_count=len(result["accepted_ids"]))
            print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.command == "import-mobvoi":
            from .mobvoi import build
            output = args.output.resolve()
            if output.exists() or output.is_relative_to(args.audio_root.resolve()) or output.is_relative_to(args.metadata.resolve()):
                raise PrepError("Output must be new and outside input directories")
            mapping = load_json(args.keyword_labels.read_text(encoding="utf-8-sig"))
            document, report = build(args.metadata, args.audio_root, dataset_id=args.dataset_id, source=args.source,
                                     version=args.version, license=args.license, keyword_labels=mapping, purpose=args.purpose)
            output.mkdir(parents=True, exist_ok=False)
            try:
                for name, value in (("manifest.json", document), ("report.json", report)):
                    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
                    (output / name).write_bytes(payload)
                    if (output / name).read_bytes() != payload:
                        raise PrepError("Public import readback mismatch")
            except BaseException as exc:
                (output / "FAILED.json").write_text(json.dumps({"error": type(exc).__name__}), encoding="utf-8")
                raise
            print(json.dumps({"output": str(output), "rows": len(document["rows"])}, ensure_ascii=False))
            return 0
        config = load_json(args.config.read_text(encoding="utf-8-sig")) if args.config else None
        if args.command.endswith("-v2"):
            path_map = load_json(args.path_map.read_text(encoding="utf-8-sig"))
            if args.command == "run-v2":
                result = v2.prepare(args.manifest, path_map, args.output, config)
            else:
                result = v2.plan(args.manifest, path_map, config)
                result = {k: v for k, v in result.items() if k not in {"rows", "config"}}
        elif args.command == "upgrade-v1":
            result = v2.upgrade_v1(args.manifest, args.audio_root, args.output, config)
        elif args.command == "run":
            result = prepare(args.manifest, args.audio_root, args.output, config)
        else:
            result = plan(args.manifest, args.audio_root, config)
            result = {k: v for k, v in result.items() if k not in {"rows", "config"}}
    except ImportError:
        print('VC output checks require "easykws[preprocess]".', file=sys.stderr)
        return 2
    except (PrepError, OSError, UnicodeError, ValueError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
