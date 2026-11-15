"""Audiveris OMR engine (runs Audiveris in batch mode)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable

from .base import OmrEngine, OmrError

log = logging.getLogger(__name__)


class AudiverisEngine(OmrEngine):
    name = "audiveris"

    def __init__(self, exe: Path, timeout_sec: int = 1800,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
        self.exe = Path(exe)
        self.timeout_sec = timeout_sec
        self._run = runner

    def build_command(self, pdf: Path, out_dir: Path) -> list[str]:
        return [str(self.exe), "-batch", "-export", "-output", str(out_dir), "--", str(pdf)]

    def recognize(self, pdf: Path, out_dir: Path) -> list[Path]:
        pdf, out_dir = Path(pdf), Path(out_dir)
        if not pdf.is_file():
            raise OmrError(f"Input PDF not found: {pdf}")
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_command(pdf, out_dir)
        log.info("Running Audiveris: %s", " ".join(cmd))
        try:
            res = self._run(cmd, capture_output=True, text=True, timeout=self.timeout_sec)
        except subprocess.TimeoutExpired as exc:
            raise OmrError(f"Audiveris timed out after {self.timeout_sec}s") from exc
        except OSError as exc:
            raise OmrError(f"Failed to start Audiveris: {exc}") from exc

        for line in (res.stdout or "").splitlines():
            log.debug("[audiveris] %s", line)
        for line in (res.stderr or "").splitlines():
            log.debug("[audiveris:err] %s", line)

        if res.returncode != 0:
            raise OmrError(f"Audiveris exited with code {res.returncode}: {(res.stderr or '').strip()[-500:]}")

        outputs = sorted(p for p in out_dir.rglob("*") if p.suffix.lower() in (".mxl", ".musicxml"))
        if not outputs:
            raise OmrError("Audiveris produced no MusicXML output (score may be unreadable)")
        log.info("Audiveris produced %d file(s): %s", len(outputs), [p.name for p in outputs])
        return outputs
