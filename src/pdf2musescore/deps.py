"""Detection and installation of external tools (Audiveris, MuseScore)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
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
    github_repo: Optional[str] = None  # fallback installer source ("owner/repo")


def _program_dirs() -> list[Path]:
    dirs = []
    for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        val = os.environ.get(var)
        if val:
            dirs.append(Path(val))
    return dirs


AUDIVERIS = ToolSpec(
    name="Audiveris",
    exe_names=("Audiveris.exe",),
    install_dir_globs=("Audiveris*",),
    winget_id="audiveris.org.Audiveris",
    github_repo="Audiveris/audiveris",
)

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


def _latest_github_installer(repo: str, opener=urllib.request.urlopen) -> str:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    log.debug("Querying %s", url)
    with opener(url, timeout=30) as resp:
        data = json.load(resp)
    for asset in data.get("assets", []):
        name = asset.get("name", "").lower()
        if name.endswith((".msi", ".exe")) and "win" in name:
            return asset["browser_download_url"]
    raise InstallError(f"No Windows installer found in latest release of {repo}")


def install_from_github(spec: ToolSpec, runner: Runner = subprocess.run,
                        opener=urllib.request.urlopen, download=urllib.request.urlretrieve) -> bool:
    if not spec.github_repo:
        return False
    try:
        url = _latest_github_installer(spec.github_repo, opener)
        target = Path(tempfile.gettempdir()) / url.rsplit("/", 1)[-1]
        log.info("%s: downloading %s -> %s", spec.name, url, target)
        download(url, target)
    except Exception as exc:  # network errors of many kinds
        log.error("%s: GitHub download failed: %s", spec.name, exc)
        return False
    cmd = (["msiexec", "/i", str(target), "/passive"] if target.suffix == ".msi" else [str(target)])
    log.info("%s: running installer %s", spec.name, cmd)
    res = runner(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log.error("%s: installer failed (code %s)", spec.name, res.returncode)
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
    if not install_with_winget(spec, runner) and not install_from_github(spec, runner):
        raise InstallError(f"Could not install {spec.name}")
    path = find_tool(spec, configured, search_dirs)
    if not path:
        raise InstallError(f"{spec.name} installed but executable not found")
    return path
