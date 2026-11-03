import importlib.util
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location("check_hygiene", Path(__file__).resolve().parents[1] / "tools" / "check_hygiene.py")
hygiene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hygiene)


def test_explicit_files_and_blocked_extensions(tmp_path):
    (tmp_path / "safe.md").write_text("public example", encoding="utf-8")
    assert hygiene.main(["--root", str(tmp_path), "--files", "safe.md"]) == 0
    for name in ["sample.wav", "model.onnx", "bundle.tar.gz"]:
        assert hygiene.check(tmp_path, [(name, "")])
    assert hygiene.check(tmp_path, [("../outside.txt", "")])
    assert hygiene.check(tmp_path, [("linked", "120000")])
    assert hygiene.check(tmp_path, [("module", "160000")])


def test_user_path_exception_is_per_match():
    example = "C:/" + "Users/example-user/project"
    other = "/" + "home/ordinary-person/private"
    assert not hygiene.check_text("tests/example.py", "SYNTHETIC_EXAMPLE " + example)
    assert hygiene.check_text("docs/example.md", "SYNTHETIC_EXAMPLE " + example)
    assert hygiene.check_text("tests/example.py", example)
    assert hygiene.check_text("tests/example.py", "SYNTHETIC_EXAMPLE " + example + " " + other)


def test_secrets_are_detected_without_echoing_values():
    samples = ["-----BEGIN " + "RSA PRIVATE KEY-----", "gh" + "p_" + "a" * 30,
               "api" + "_key = '" + "fictional" * 3 + "'"]
    for sample in samples:
        findings = hygiene.check_text("tests/example.py", "SYNTHETIC_EXAMPLE " + sample)
        assert findings
        assert sample not in " ".join(findings)


def test_binary_requires_review(tmp_path):
    (tmp_path / "opaque.dat").write_bytes(bytes([255, 254]))
    assert hygiene.check(tmp_path, [("opaque.dat", "")])
    (tmp_path / "disguised.md").write_bytes(b"text\x00binary")
    assert hygiene.check(tmp_path, [("disguised.md", "")])


def test_staged_reads_index_and_tracked_reads_worktree(tmp_path):
    def run(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    run("init")
    (tmp_path / "example.txt").write_text("password = '" + "fictional" * 3 + "'", encoding="utf-8")
    run("add", "example.txt")
    (tmp_path / "example.txt").write_text("safe", encoding="utf-8")
    assert hygiene.main(["--root", str(tmp_path), "--staged"]) == 1
    assert hygiene.main(["--root", str(tmp_path), "--tracked"]) == 0


def test_selection_is_required():
    with pytest.raises(SystemExit):
        hygiene.main([])
