"""Wheel-review checks over synthetic dist archives, including tamper cases."""
import base64
import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from check_release_artifacts import review  # noqa: E402

DIST_INFO = "easykws-0.1.0.dist-info"
METADATA = "\n".join([
    "Metadata-Version: 2.4", "Name: easykws", "Version: 0.1.0",
    "Requires-Python: >=3.10", "License-File: LICENSE", "License-File: NOTICES.md", ""])
WHEEL_FILE = "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
ENTRY_POINTS = "\n".join(
    f"{name} = easykws.{module}:main" for name, module in
    [("easykws-prep", "prep.cli"), ("easykws-preprocess", "preprocess.cli"), ("easykws-train", "training.cli"),
     ("easykws-listen", "runtime.listen"), ("easykws-smoke", "smoke")]) + "\n"
LICENSE = "Apache License 2.0 text\n"
NOTICES = "Third-party notices\n"
MODULE = "print('package module')\n"


def _record_line(name, payload, *, digest=True, size=True):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    fields = [name, "", ""]
    if digest:
        raw = hashlib.sha256(payload).digest()
        fields[1] = "sha256=" + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    if size:
        fields[2] = str(len(payload))
    return ",".join(fields)


def build_wheel(tmp_path, *, tamper=None, drop_digest=False, wrong_size=False):
    members = {
        "easykws/__init__.py": MODULE,
        f"{DIST_INFO}/METADATA": METADATA,
        f"{DIST_INFO}/WHEEL": WHEEL_FILE,
        f"{DIST_INFO}/entry_points.txt": ENTRY_POINTS,
        f"{DIST_INFO}/licenses/LICENSE": LICENSE,
        f"{DIST_INFO}/licenses/NOTICES.md": NOTICES,
    }
    record_rows = [_record_line(name, payload) for name, payload in sorted(members.items())]
    record_name = f"{DIST_INFO}/RECORD"
    if drop_digest:
        # A non-RECORD row without digest and size is invalid in both fields.
        first = sorted(members)[0]
        record_rows[record_rows.index(_record_line(first, members[first]))] = f"{first},,"
    if wrong_size:
        first = sorted(members)[0]
        line = _record_line(first, members[first])
        name, digest, _ = line.split(",")
        record_rows[record_rows.index(line)] = f"{name},{digest},999999"
    record_payload = (",".join([record_name, "", ""]) + "\n" + "\n".join(record_rows) + "\n").encode("utf-8")
    members[record_name] = record_payload
    if tamper == "content":
        members["easykws/__init__.py"] = MODULE + "# injected comment\n"
    path = tmp_path / "easykws-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def test_review_accepts_well_formed_wheel(tmp_path):
    result = review(build_wheel(tmp_path))
    assert result["issues"] == [], result["issues"]
    assert result["member_count"] == 7


def test_review_rejects_tampered_member_with_stale_record(tmp_path):
    """The review-probe case: a changed member with the old RECORD must fail."""
    result = review(build_wheel(tmp_path, tamper="content"))
    assert any("RECORD digest mismatch" in issue for issue in result["issues"]), result["issues"]
    assert not any("non-utf8" in issue for issue in result["issues"])


def test_review_rejects_wrong_recorded_size(tmp_path):
    result = review(build_wheel(tmp_path, wrong_size=True))
    assert any("RECORD size mismatch" in issue for issue in result["issues"]), result["issues"]


def test_review_rejects_missing_digest_and_size(tmp_path):
    result = review(build_wheel(tmp_path, drop_digest=True))
    issues = "\n".join(result["issues"])
    assert "lacks a digest" in issues
    assert "lacks a size" in issues


def test_review_refuses_symlink_member(tmp_path):
    path = build_wheel(tmp_path)
    with zipfile.ZipFile(path, "a") as archive:
        info = zipfile.ZipInfo("easykws/link.py")
        info.external_attr = (0o120777 << 16)
        archive.writestr(info, "../escape")
    with pytest.raises(SystemExit, match="symlink"):
        review(path)


def test_review_rejects_path_traversal_without_extracting(tmp_path):
    path = build_wheel(tmp_path)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("../escape.py", "payload")
    result = review(path)
    assert any("path traversal" in issue for issue in result["issues"])
    assert not (tmp_path / "escape.py").exists()
