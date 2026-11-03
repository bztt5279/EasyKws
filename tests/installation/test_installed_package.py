"""Black-box acceptance of an installed wheel; skipped in source checkouts.

The CI packaging job installs the built wheel into a clean virtual
environment and runs this module there. Skipping it in source-tree pytest
runs is deliberate: source tests import via the checkout and must not be
counted as installation verification.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import easykws

INSTALLED = "site-packages" in Path(easykws.__file__).parts

pytestmark = pytest.mark.skipif(not INSTALLED, reason="easykws is not installed from site-packages")


def _script(name):
    directory = Path(sys.executable).parent
    candidates = [directory / (name + ".exe"), directory / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise AssertionError(f"console script not found next to {sys.executable}: {name}")


def _run(argv, cwd=None, *, strip_threads=False):
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "PYTHONHOME")}
    if strip_threads:
        # Reproduce the reviewer scenario: default thread environment, no
        # OMP/MKL pinning anywhere in the chain.
        env.pop("OMP_NUM_THREADS", None)
        env.pop("MKL_NUM_THREADS", None)
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                          cwd=cwd, env=env)


def test_package_imports_from_site_packages():
    assert "site-packages" in Path(easykws.__file__).parts, easykws.__file__


def test_pip_check_is_clean():
    done = _run([sys.executable, "-m", "pip", "check"])
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize("script", ["easykws-train", "easykws-prep", "easykws-preprocess",
                                    "easykws-listen", "easykws-smoke"])
def test_console_scripts_help(script):
    done = _run([_script(script), "--help"])
    assert done.returncode == 0, done.stderr
    assert "usage" in done.stdout.lower()


def test_listen_check_runs_without_device_and_without_sounddevice(tmp_path):
    """--check must work with the installed wheel and never touch a device."""
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "sounddevice.py").write_text("raise ImportError('blocked')\n", encoding="utf-8")
    package = tmp_path / "package"
    package.mkdir()
    # An empty directory must fail cleanly (missing files), which already
    # proves the installed runtime imports and reports without sounddevice.
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "PYTHONHOME")}
    env["PYTHONPATH"] = str(blocker)
    done = subprocess.run([_script("easykws-listen"), "--model", str(package), "--check"],
                          capture_output=True, text=True, encoding="utf-8", env=env)
    assert done.returncode == 2 and "missing" in done.stderr.lower(), done.stderr


def test_full_smoke_pipeline_on_installed_wheel(tmp_path):
    """The installed wheel alone reproduces the complete synthetic pipeline.

    Runs with OMP/MKL variables stripped: the smoke pipeline must establish
    and propagate one execution profile itself instead of relying on the
    caller's thread environment.
    """
    output = tmp_path / "smoke-run"
    done = _run([_script("easykws-smoke"), "--output", str(output)], strip_threads=True)
    assert done.returncode == 0, done.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["conclusion"].startswith("All stages executed")
    replay = summary["stages"][-1]["checks"]
    assert replay["ready_files_verified"] == 5
    assert set(replay["fresh_process_frames"].values()) == {298}
