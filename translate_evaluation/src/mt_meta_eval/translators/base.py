from abc import ABC, abstractmethod

import torch


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device in ("mps", "cuda") else torch.float32


class Translator(ABC):
    """A translation model wrapper. `key` names the output directory."""

    key: str

    @abstractmethod
    def translate_batch(self, sources: list[str], pair: str) -> list[str]:
        """Translate a batch of source segments for a pair like 'en-de_DE' or 'es-fr'."""
