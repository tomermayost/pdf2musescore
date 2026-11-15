"""Export MusicXML to MuseScore (.mscz) using the MuseScore command line."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


class ExportError(RuntimeError):
    pass


class MuseScoreExporter:
    def __init__(self, exe: Path, timeout_sec: int = 300,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
        self.exe = Path(exe)
        self.timeout_sec = timeout_sec
        self._run = runner

    def export(self, src: Path, dst: Path) -> Path:
        src, dst = Path(src), Path(dst)
        if not src.is_file():
            raise ExportError(f"Source not found: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(self.exe), "-o", str(dst), str(src)]
        log.info("Running MuseScore: %s", " ".join(cmd))
        try:
            res = self._run(cmd, capture_output=True, text=True, timeout=self.timeout_sec)
        except subprocess.TimeoutExpired as exc:
            raise ExportError(f"MuseScore timed out after {self.timeout_sec}s") from exc
        except OSError as exc:
            raise ExportError(f"Failed to start MuseScore: {exc}") from exc
        for line in ((res.stdout or "") + (res.stderr or "")).splitlines():
            log.debug("[musescore] %s", line)
        if res.returncode != 0 or not dst.is_file():
            raise ExportError(f"MuseScore export failed (code {res.returncode})")
        log.info("Wrote %s", dst)
        return dst
