"""Public full-history scan: run the shared hygiene rules over every blob.

Covers all commit trees reachable from refs so a pre-release review cannot
stop at the newest tree. Rule-based only, never a complete guarantee; private
scans continue to run outside the repository.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_hygiene import BLOCKED, check_text


def git(root, *args, data=None):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE, input=data)


def scan(root):
    root = Path(root)
    entries = git(root, "rev-list", "--objects", "--all").decode("utf-8", "surrogateescape").split("\n")
    blobs = {}
    for entry in entries:
        if not entry:
            continue
        digest, _, name = entry.partition(" ")
        if digest and name:
            blobs[digest] = name
    checked, issues, skipped = 0, [], 0
    types = git(root, "cat-file", "--batch-check=%(objectname) %(objecttype)",
                data="\n".join(blobs).encode()).decode().split("\n")
    blob_ids = [line.split()[0] for line in types if line.endswith(" blob")]
    for digest in blob_ids:
        name = blobs.get(digest, digest)
        if PurePosixPath(name.replace("\\", "/")).suffix.lower() in BLOCKED:
            issues.append(f"blocked file type exists in history: {name}")
            continue
        try:
            content = git(root, "cat-file", "blob", digest).decode("utf-8")
        except UnicodeError:
            skipped += 1
            issues.append(f"non-utf8 blob needs manual review: {name}")
            continue
        except subprocess.CalledProcessError:
            skipped += 1
            continue
        checked += 1
        issues.extend(f"{name}: {issue}" for issue in check_text(name, content))
    # Commit metadata is part of the publishable record.
    log = git(root, "log", "--format=%H%x09%an%x09%ae%x09%s").decode("utf-8", "surrogateescape")
    issues.extend(f"commit metadata: {issue}" for issue in check_text("git log", log))
    tags = git(root, "tag", "--list").decode("utf-8", "surrogateescape").strip()
    if tags:
        issues.append(f"tags exist; release tagging is a later decision: {tags}")
    return {"passed": not issues, "commits_scanned": len(
        git(root, "rev-list", "--all").decode().split()), "blobs_checked": checked,
        "blobs_skipped": skipped, "issues": issues}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result = scan(args.root)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False)[:2000])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
