"""CPU training and explicit epoch-boundary recovery."""
from .checkpoint import TrainingError
from .runner import configure, resume, run

__all__ = ["TrainingError", "configure", "resume", "run"]
