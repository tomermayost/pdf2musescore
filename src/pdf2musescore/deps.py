"""Detection and installation of MuseScore."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    exe_names: Sequence[str]
    install_dir_globs: Sequence[str]
    winget_id: str


def _program_dirs() -> list[Path]:
    dirs = []
    for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        val = os.environ.get(var)
        if val:
            dirs.append(Path(val))
    return dirs


MUSESCORE = ToolSpec(
    name="MuseScore",
    exe_names=("MuseScore4.exe", "MuseScore3.exe"),
    install_dir_globs=("MuseScore*",),
    winget_id="Musescore.Musescore",
)


class ToolNotFoundError(RuntimeError):
    pass


class InstallError(RuntimeError):
    pass


def find_tool(spec: ToolSpec, configured: Optional[Path] = None,
              search_dirs: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """Locate a tool: configured path > install folders > PATH."""
    if configured:
        configured = Path(configured)
        if configured.is_file():
            log.info("%s: using configured path %s", spec.name, configured)
            return configured
        log.warning("%s: configured path %s does not exist - falling back to auto-detect", spec.name, configured)

    for base in (search_dirs if search_dirs is not None else _program_dirs()):
        for pattern in spec.install_dir_globs:
            for folder in sorted(Path(base).glob(pattern)):
                for exe in spec.exe_names:
                    for candidate in (folder / exe, folder / "bin" / exe):
                        log.debug("%s: probing %s", spec.name, candidate)
                        if candidate.is_file():
                            log.info("%s: found at %s", spec.name, candidate)
                            return candidate

    for exe in spec.exe_names:
        found = shutil.which(exe)
        if found:
            log.info("%s: found on PATH at %s", spec.name, found)
            return Path(found)

    log.info("%s: not found", spec.name)
    return None


Runner = Callable[..., subprocess.CompletedProcess]


def install_with_winget(spec: ToolSpec, runner: Runner = subprocess.run) -> bool:
    cmd = ["winget", "install", "--id", spec.winget_id, "-e", "--silent",
           "--accept-package-agreements", "--accept-source-agreements"]
    log.info("%s: installing via winget: %s", spec.name, " ".join(cmd))
    try:
        res = runner(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        log.warning("winget not available")
        return False
    log.debug("winget stdout:\n%s", res.stdout)
    if res.returncode != 0:
        log.warning("%s: winget failed (code %s): %s", spec.name, res.returncode, (res.stderr or "").strip())
        return False
    return True


def ensure_tool(spec: ToolSpec, configured: Optional[Path] = None,
                confirm: Callable[[str], bool] = lambda msg: False,
                runner: Runner = subprocess.run, search_dirs: Optional[Iterable[Path]] = None) -> Path:
    """Return tool path, installing (after user confirmation) if missing."""
    path = find_tool(spec, configured, search_dirs)
    if path:
        return path
    if not confirm(f"{spec.name} is not installed. Install it now?"):
        raise ToolNotFoundError(f"{spec.name} is required but not installed")
    if not install_with_winget(spec, runner):
        raise InstallError(f"Could not install {spec.name}")
    path = find_tool(spec, configured, search_dirs)
    if not path:
        raise InstallError(f"{spec.name} installed but executable not found")
    return path
