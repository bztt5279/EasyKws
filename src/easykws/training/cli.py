"""Training commands; JSON configs contain no input or output paths."""
import argparse
import json
import sys

from .checkpoint import TrainingError, migrate_checkpoint
from .runner import resume, run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="easykws-train")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("run")
    start.add_argument("--features", required=True)
    start.add_argument("--config", required=True)
    start.add_argument("--output", required=True)
    start.add_argument("--stop-after-epoch", type=int)
    recovery = commands.add_parser("resume")
    recovery.add_argument("--run", required=True)
    recovery.add_argument("--features", required=True)
    recovery.add_argument("--config")
    recovery.add_argument("--stop-after-epoch", type=int)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--run", required=True)
    calibration.add_argument("--features", required=True)
    calibration.add_argument("--manifest", required=True)
    calibration.add_argument("--path-root")
    calibration.add_argument("--config", required=True)
    calibration.add_argument("--output", required=True)
    export = commands.add_parser("export")
    export.add_argument("--run", required=True)
    export.add_argument("--features", required=True)
    export.add_argument("--calibration", required=True)
    export.add_argument("--manifest", required=True)
    export.add_argument("--path-root")
    export.add_argument("--output", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--features", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--candidates", nargs="+")
    evaluation.add_argument("--thresholds", nargs="+", type=float)
    migration = commands.add_parser("migrate-checkpoint", description="WARNING: trusted legacy pickle input may execute code")
    migration.add_argument("--input", required=True)
    migration.add_argument("--output", required=True)
    migration.add_argument("--trust-legacy-input", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run(args.features, args.config, args.output, stop_after_epoch=args.stop_after_epoch)
        elif args.command == "resume":
            result = resume(args.run, args.features, args.config, stop_after_epoch=args.stop_after_epoch)
        elif args.command == "calibrate":
            from ..calibration.core import calibrate
            result = calibrate(args.run, args.features, args.manifest, args.output,
                               config=args.config, path_root=args.path_root)
        elif args.command == "export":
            from ..export import export_package
            result = export_package(args.run, args.features, args.calibration, args.manifest,
                                    args.output, path_root=args.path_root)
        elif args.command == "evaluate":
            from ..calibration.core import evaluate
            result = evaluate(args.run, args.features, args.output,
                              candidates=args.candidates, thresholds=args.thresholds)
        else:
            result = migrate_checkpoint(args.input, args.output, trust_legacy_input=args.trust_legacy_input)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (TrainingError, OSError, ValueError) as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 2
