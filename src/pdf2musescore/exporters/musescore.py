"""Export MusicXML to MuseScore (.mscz) using the MuseScore command line."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _read_recent_musescore_error(since_ts: float, log_dir: Optional[Path] = None) -> Optional[str]:
    """Extract the most specific ERROR message from a MuseScore 4 log written after ``since_ts``."""
    try:
        if log_dir is not None:
            target_dir = Path(log_dir)
        elif sys.platform == "win32":
            local_app = os.environ.get("LOCALAPPDATA", "").strip()
            if not local_app:
                return None
            target_dir = Path(local_app) / "MuseScore" / "MuseScore4" / "logs"
        else:
            return None

        if not target_dir.is_dir():
            return None

        candidates = sorted(
            target_dir.glob("MuseScore_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        best_file = next((p for p in candidates if p.stat().st_mtime >= since_ts - 0.5), None)
        if best_file is None:
            return None

        lines = best_file.read_text(encoding="utf-8", errors="replace").splitlines()
        generic_err: Optional[str] = None
        for line in reversed(lines):
            if "| ERROR" not in line:
                continue
            cleaned = _HTML_TAG_RE.sub("", line).strip()
            parts = [p.strip() for p in cleaned.split("|")]
            msg = parts[-1] if parts else cleaned
            if "failed load notation" in cleaned.lower() or "err:" in cleaned.lower():
                return msg[:220]
            if generic_err is None:
                generic_err = msg[:220]
        return generic_err
    except Exception:
        return None


class ExportError(RuntimeError):
    """Raised when MuseScore CLI export fails."""

    def __init__(self, message: str, *, musescore_detail: str = "") -> None:
        super().__init__(message)
        self.musescore_detail = musescore_detail


class MuseScoreExporter:
    def __init__(
        self,
        exe: Path,
        timeout_sec: int = 300,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        musescore_log_dir: Optional[Path] = None,
    ) -> None:
        self.exe = Path(exe)
        self.timeout_sec = timeout_sec
        self._run = runner
        self.musescore_log_dir = musescore_log_dir

    def export(self, src: Path, dst: Path) -> Path:
        src, dst = Path(src), Path(dst)
        if not src.is_file():
            raise ExportError(f"Source not found: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(self.exe), "-o", str(dst), str(src)]
        log.info("Running MuseScore: %s", " ".join(cmd))
        started = time.time()
        try:
            res = self._run(cmd, capture_output=True, text=True, timeout=self.timeout_sec)
        except subprocess.TimeoutExpired as exc:
            raise ExportError(f"MuseScore timed out after {self.timeout_sec}s") from exc
        except OSError as exc:
            raise ExportError(f"Failed to start MuseScore: {exc}") from exc

        out_lines = [ln for ln in ((res.stdout or "") + "\n" + (res.stderr or "")).splitlines() if ln.strip()]
        if res.returncode != 0 or not dst.is_file():
            for line in out_lines:
                log.error("[musescore] %s", line)
            detail = _read_recent_musescore_error(started, self.musescore_log_dir)
            if detail:
                log.error("MuseScore log detail: %s", detail)
                raise ExportError(
                    f"MuseScore export failed (code {res.returncode}): {detail}",
                    musescore_detail=detail,
                )
            raise ExportError(f"MuseScore export failed (code {res.returncode})")

        for line in out_lines:
            log.debug("[musescore] %s", line)
        log.info("Wrote %s", dst)
        return dst
