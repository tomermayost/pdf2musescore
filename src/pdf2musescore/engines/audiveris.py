"""Audiveris OMR engine (runs Audiveris in batch mode, with partial-sheet recovery)."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence

from .base import OmrEngine, OmrError

log = logging.getLogger(__name__)

_SHEET_STORED_RE = re.compile(r"Stored\s+/sheet#(\d+)/sheet#\d+\.xml")
_SHEET_FAILED_RE = re.compile(
    r"\[[^\]]*#(\d+)\][^\n]*(?:Error processing stub|flagged as invalid|NullPointerException|OutOfBoundsException)"
)


def parse_valid_sheets(output_text: str) -> list[int]:
    """Identify sheet numbers that stored XML without fatal processing errors."""
    stored = {int(m.group(1)) for m in _SHEET_STORED_RE.finditer(output_text)}
    failed = {int(m.group(1)) for m in _SHEET_FAILED_RE.finditer(output_text)}
    return sorted(stored - failed)


class AudiverisEngine(OmrEngine):
    name = "audiveris"

    def __init__(self, exe: Path, timeout_sec: int = 1800,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
        self.exe = Path(exe)
        self.timeout_sec = timeout_sec
        self._run = runner

    def build_command(self, pdf: Path, out_dir: Path,
                      sheets: Optional[Sequence[int]] = None) -> list[str]:
        cmd = [str(self.exe), "-batch", "-export"]
        if sheets:
            cmd.extend(["-sheets", ",".join(str(s) for s in sheets)])
        cmd.extend(["-output", str(out_dir), "--", str(pdf)])
        return cmd

    def _invoke(self, cmd: list[str]) -> subprocess.CompletedProcess:
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
        return res

    @staticmethod
    def _collect_outputs(out_dir: Path) -> list[Path]:
        return sorted(p for p in out_dir.rglob("*") if p.suffix.lower() in (".mxl", ".musicxml"))

    def recognize(self, pdf: Path, out_dir: Path) -> list[Path]:
        pdf, out_dir = Path(pdf), Path(out_dir)
        if not pdf.is_file():
            raise OmrError(f"Input PDF not found: {pdf}")
        out_dir.mkdir(parents=True, exist_ok=True)

        res = self._invoke(self.build_command(pdf, out_dir))
        outputs = self._collect_outputs(out_dir)

        if (res.returncode != 0 or not outputs):
            combined_out = "\n".join(filter(None, [(res.stderr or "").strip(), (res.stdout or "").strip()]))
            valid_sheets = parse_valid_sheets(combined_out)
            if valid_sheets:
                log.warning(
                    "Some page(s) failed in Audiveris; retrying export with valid sheets %s",
                    valid_sheets,
                )
                shutil.rmtree(out_dir, ignore_errors=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                res = self._invoke(self.build_command(pdf, out_dir, sheets=valid_sheets))
                outputs = self._collect_outputs(out_dir)

        if res.returncode != 0 and not outputs:
            combined_out = "\n".join(filter(None, [(res.stderr or "").strip(), (res.stdout or "").strip()]))
            raise OmrError(f"Audiveris exited with code {res.returncode}: {combined_out[-500:]}")

        if not outputs:
            raise OmrError("Audiveris produced no MusicXML output (score may be unreadable)")
        log.info("Audiveris produced %d file(s): %s", len(outputs), [p.name for p in outputs])
        return outputs
