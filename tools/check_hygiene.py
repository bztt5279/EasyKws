"""Heuristic release-file checks; never a complete anonymization guarantee."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess

BLOCKED = set(".wav .flac .mp3 .ogg .m4a .aac .pcm .aiff .aif .wma .opus .pt .pth .onnx .ckpt .safetensors .h5 .hdf5 .tflite .pb .bin .zip .tar .gz .bz2 .xz .7z .rar .tgz".split())
USER_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]Users[\\/]|/(?:home|Users)/)([^\\/\s\"'<>]+)")
PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")
TOKEN = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{20,})")
CREDENTIAL = re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret[_-]?key)\s*[:=]\s*[\"']([^\"'\r\n]{8,})[\"']")


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def selected_git(root: Path, staged: bool) -> list[tuple[str, str]]:
    if staged:
        paths = git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z").decode().split("\0")
    else:
        paths = git(root, "ls-files", "-z").decode().split("\0")
    entries = {}
    for entry in git(root, "ls-files", "--stage", "-z").decode().split("\0"):
        if entry:
            metadata, name = entry.split("\t", 1)
            mode, oid, stage = metadata.split()
            if stage != "0":
                raise ValueError(f"unmerged index entry: {name}")
            entries[name] = (mode, oid)
    return [(name, entries[name][0]) for name in paths if name]


def check_text(name: str, text: str) -> list[str]:
    issues = []
    for number, line in enumerate(text.splitlines(), 1):
        for match in USER_PATH.finditer(line):
            synthetic = (PurePosixPath(name).parts[0] == "tests" and
                         "SYNTHETIC_EXAMPLE" in line and
                         match.group(1) in {"synthetic-user", "example-user"})
            if not synthetic:
                issues.append(f"{name}:{number}: absolute user directory")
        if PRIVATE_KEY.search(line):
            issues.append(f"{name}:{number}: private key")
        if TOKEN.search(line) or CREDENTIAL.search(line):
            issues.append(f"{name}:{number}: possible credential")
    return issues


def check(root: Path, files: list[tuple[str, str]], staged: bool = False) -> list[str]:
    root = root.resolve()
    issues = []
    for name, mode in files:
        relative = PurePosixPath(name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or re.match(r"^[A-Za-z]:", name):
            issues.append(f"{name}: path escapes release root")
            continue
        path = root.joinpath(*relative.parts)
        if not relative.parts or not path.resolve().is_relative_to(root):
            issues.append(f"{name}: path escapes release root")
            continue
        if mode in {"120000", "160000"} or any(p.is_symlink() for p in [path, *path.parents] if p != root):
            issues.append(f"{name}: symlink or submodule is not allowed")
            continue
        if any(suffix.lower() in BLOCKED for suffix in relative.suffixes):
            issues.append(f"{name}: audio, weights, or archive extension is not allowed")
            continue
        try:
            data = git(root, "show", f":{name}") if staged else path.read_bytes()
            if b"\x00" in data:
                issues.append(f"{name}: binary content requires manual review")
                continue
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            issues.append(f"{name}: non-UTF-8 content requires manual review")
            continue
        except (OSError, subprocess.CalledProcessError):
            issues.append(f"{name}: cannot read selected file")
            continue
        issues.extend(check_text(relative.as_posix(), text))
    return issues


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--files", nargs="+", metavar="RELATIVE_PATH")
    selection.add_argument("--tracked", action="store_true")
    selection.add_argument("--staged", action="store_true")
    args = parser.parse_args(argv)
    try:
        files = ([(name, "") for name in args.files] if args.files is not None
                 else selected_git(args.root, args.staged))
        issues = check(args.root, files, args.staged)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Selection failed: {type(exc).__name__}")
        return 2
    for issue in issues:
        print(issue)
    print(f"Checked {len(files)} selected files; {len(issues)} findings.")
    print("Heuristic checks cannot guarantee complete anonymization. Review every release file; .gitignore does not replace review.")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
