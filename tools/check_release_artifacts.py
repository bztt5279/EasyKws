"""Release artifact review: member lists, metadata and content of dist files.

Public, rule-based checks only; passing them is not a complete anonymization
guarantee. Deeper private scans stay outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from check_hygiene import BLOCKED, check_text

FORBIDDEN_SUFFIXES = BLOCKED | {".npy", ".npz", ".pkl", ".pickle", ".log", ".pyc", ".pyo"}
FORBIDDEN_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}
ENTRY_POINTS = ("easykws-prep", "easykws-preprocess", "easykws-train", "easykws-listen", "easykws-smoke")
SDIST_REQUIRED = ("PKG-INFO", "pyproject.toml", "LICENSE", "NOTICES.md", "setup.py",
                  "src/easykws/smoke.py", "configs/cpu-constraints.txt", "configs/prepare.example.json",
                  "docs/install.md", "docs/DATA_PROVENANCE.md", "examples/make_synthetic_recordings.py")


def _safe_members(names):
    problems = []
    for name in names:
        pure = PurePosixPath(name.replace("\\", "/"))
        if pure.is_absolute() or not name:
            problems.append(f"absolute member path: {name}")
            continue
        parts = set(pure.parts)
        if ".." in parts:
            problems.append(f"path traversal member: {name}")
        if parts & FORBIDDEN_PARTS:
            problems.append(f"forbidden directory in member: {name}")
        if pure.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden file type in member: {name}")
    return problems


def _extract(archive, target):
    """Extract strictly inside target; refuse links and escaping members."""
    if isinstance(archive, zipfile.ZipFile):
        for info in archive.infolist():
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise SystemExit(f"symlink member refused: {info.filename}")
        archive.extractall(target)
    else:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise SystemExit(f"link member refused: {member.name}")
        archive.extractall(target, filter="data")


def _contained(root, name):
    """True when resolving the extracted member stays inside the root."""
    candidate = (root / name).resolve()
    try:
        return candidate.is_relative_to(root.resolve())
    except OSError:
        return False


def _read_record(root, record_name):
    """Parse RECORD rows per the wheel spec; return rows and integrity issues.

    Each row is (path, sha256_hex_or_None, size_or_None). RECORD may list
    itself with empty digest fields; every other entry must carry both.
    """
    import base64
    import csv
    import io
    issues, rows, seen = [], [], set()
    text = (root / record_name).read_text(encoding="utf-8-sig")
    for row_number, row in enumerate(csv.reader(io.StringIO(text)), 1):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if len(row) != 3:
            issues.append(f"RECORD row {row_number} does not have 3 fields")
            continue
        name, digest, size = (field.strip() for field in row)
        pure = PurePosixPath(name.replace("\\", "/"))
        if pure.is_absolute() or ".." in pure.parts or not name:
            issues.append(f"RECORD row {row_number} has an unsafe path: {name}")
            continue
        if name in seen:
            issues.append(f"RECORD lists {name} twice")
        seen.add(name)
        declared_hash, declared_size = None, None
        if digest:
            scheme, _, value = digest.partition("=")
            if scheme != "sha256":
                issues.append(f"RECORD row {row_number} uses unsupported digest {scheme!r}")
            else:
                try:
                    # RECORD digests are URL-safe base64 without padding.
                    padded = value + "=" * (-len(value) % 4)
                    declared_hash = base64.urlsafe_b64decode(padded.encode("ascii"))
                    if len(declared_hash) != 32:
                        raise ValueError
                except (ValueError, UnicodeError):
                    issues.append(f"RECORD row {row_number} has a malformed sha256 value")
                    declared_hash = None
        elif name != record_name:
            issues.append(f"RECORD row {row_number} lacks a digest: {name}")
        if size:
            if not size.isdigit():
                issues.append(f"RECORD row {row_number} has a malformed size: {size}")
            else:
                declared_size = int(size)
        elif name != record_name:
            issues.append(f"RECORD row {row_number} lacks a size: {name}")
        if not _contained(root, name):
            issues.append(f"RECORD entry escapes the package root: {name}")
            continue
        member = root / name
        if not member.is_file():
            issues.append(f"RECORD lists a missing member: {name}")
            continue
        payload = member.read_bytes()
        if declared_hash is not None and hashlib.sha256(payload).digest() != declared_hash:
            issues.append(f"RECORD digest mismatch for {name}")
        if declared_size is not None and len(payload) != declared_size:
            issues.append(f"RECORD size mismatch for {name}")
        rows.append((name, declared_hash, declared_size))
    return rows, issues


def _content_issues(root):
    """Run the shared hygiene rules over every extracted text member."""
    from check_hygiene import check_text as full_check
    issues = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeError, ValueError):
            issues.append(f"non-utf8 content needs manual review: {path.name}")
            continue
        issues.extend(f"{path.relative_to(root).as_posix()}: {issue}"
                      for issue in full_check(path.name, text))
    return issues


def review(path):
    path = Path(path)
    issues, members, digests = [], [], {}
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary)
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                issues += _safe_members(names)
                if issues:
                    # Refuse unsafe members before touching the filesystem.
                    return {"artifact": path.name, "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "members": sorted(names), "member_count": len(names),
                            "issues": issues}
                _extract(archive, target)
        elif path.suffix == ".gz":
            with tarfile.open(path, "r:gz") as archive:
                names = archive.getnames()
                issues += _safe_members(names)
                if issues:
                    return {"artifact": path.name, "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "members": sorted(names), "member_count": len(names),
                            "issues": issues}
                _extract(archive, target)
        else:
            raise SystemExit(f"unsupported artifact type: {path.name}")
        members = sorted(names)
        for name in members:
            source = target / name
            if source.is_file() and _contained(target, name):
                digests[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        # Member names and digests go through the shared rules as well.
        issues += check_text(path.name + " members", "\n".join(
            f"{name}: {digest}" for name, digest in sorted(digests.items())))
        issues += _content_issues(target)
        if path.suffix == ".whl":
            dist_info = next((name for name in members if name.endswith(".dist-info/METADATA")), None)
            if dist_info is None:
                issues.append("wheel has no dist-info/METADATA")
            else:
                metadata = (target / dist_info).read_text(encoding="utf-8")
                for field in ("Name: easykws", "Version:", "Requires-Python: >=3.10",
                              "License-File: LICENSE"):
                    if field not in metadata:
                        issues.append(f"METADATA missing: {field}")
            entries = next((name for name in members if name.endswith(".dist-info/entry_points.txt")), None)
            entry_text = (target / entries).read_text(encoding="utf-8") if entries else ""
            for script in ENTRY_POINTS:
                if script not in entry_text:
                    issues.append(f"entry point missing: {script}")
            record = next((name for name in members if name.endswith(".dist-info/RECORD")), None)
            if record is None:
                issues.append("wheel has no RECORD")
            else:
                rows, record_issues = _read_record(target, record)
                issues += record_issues
                listed = {row[0] for row in rows}
                if listed - {record} != set(members) - {record}:
                    issues.append("RECORD does not match the member list")
            licenses = {PurePosixPath(name).name for name in members if ".dist-info/licenses/" in name}
            for required in ("LICENSE", "NOTICES.md"):
                if required not in licenses:
                    issues.append(f"wheel does not carry {required} as a license file")
        else:
            top = {PurePosixPath(name).parts[0] for name in members if name}
            inner = {"/".join(PurePosixPath(name).parts[1:]) for name in members
                     if len(PurePosixPath(name).parts) > 1}
            for required in SDIST_REQUIRED:
                if required not in inner:
                    issues.append(f"sdist missing: {required}")
            if len(top) != 1:
                issues.append("sdist must unpack to exactly one top-level directory")
    return {"artifact": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "members": members, "member_count": len(members), "issues": issues}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)
    results = [review(path) for path in args.artifacts]
    failed = any(result["issues"] for result in results)
    report = {"passed": not failed, "artifacts": results}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"],
                      "artifacts": {result["artifact"]: {"members": result["member_count"],
                                                         "sha256": result["sha256"][:16],
                                                         "issues": result["issues"]}
                                     for result in results}}, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
