"""OMR engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class OmrError(RuntimeError):
    """Raised when optical music recognition fails."""


class OmrEngine(ABC):
    name: str = "base"

    @abstractmethod
    def recognize(self, pdf: Path, out_dir: Path) -> list[Path]:
        """Run OMR on ``pdf``; return produced MusicXML files (``.mxl``/``.musicxml``)."""
