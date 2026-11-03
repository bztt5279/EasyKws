"""Validated, portable feature preparation for custom wake-word recordings."""
from .core import plan, run
from .errors import PreprocessError

__all__ = ["PreprocessError", "plan", "run"]
