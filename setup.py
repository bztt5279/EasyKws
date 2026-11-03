"""Setuptools shim: extend the sdist with repository materials users need.

pyproject.toml keeps all project metadata. This file only customises the
sdist command so that configs, examples and docs travel with the source
distribution, as docs/install.md promises. Wheels stay minimal.
"""
from setuptools import setup
from setuptools.command.sdist import sdist as _sdist

SDIST_DIRECTORIES = ("configs", "examples", "docs")


class sdist_with_materials(_sdist):
    """Standard sdist plus configs/, examples/ and docs/."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        import shutil
        from pathlib import Path
        destination = Path(base_dir)
        for directory in SDIST_DIRECTORIES:
            source = Path(directory)
            for item in sorted(source.rglob("*")):
                if item.is_dir() or "__pycache__" in item.parts:
                    continue
                relative = item.relative_to(Path("."))
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item, target)


setup(cmdclass={"sdist": sdist_with_materials})
