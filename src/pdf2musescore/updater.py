"""Silent GitHub Releases update check and installer downloader."""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

GITHUB_LATEST_RELEASE_API = "https://api.github.com/repos/tomermayost/pdf2musescore/releases/latest"
RELEASES_FALLBACK_URL = "https://github.com/tomermayost/pdf2musescore/releases/latest"
INSTALLER_ASSET_NAME = "pdf2musescore-setup.exe"


class UpdateError(RuntimeError):
    """Raised when downloading an update installer fails."""


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    download_url: str
    release_url: str
    notes: str = ""


def parse_version(tag: str) -> tuple[int, ...]:
    """Parse a release tag like ``'v0.3.1'`` or ``'0.3.1-beta'`` into an integer tuple."""
    cleaned = (tag or "").strip().lstrip("vV")
    nums: list[int] = []
    for seg in cleaned.split("."):
        m = re.match(r"^(\d+)", seg.strip())
        if not m:
            break
        nums.append(int(m.group(1)))
    return tuple(nums) if nums else (0,)


def is_newer_version(latest: str, current: str) -> bool:
    """Return True if ``latest`` represents a strictly newer version than ``current``."""
    a = parse_version(latest)
    b = parse_version(current)
    width = max(len(a), len(b))
    a_pad = a + (0,) * (width - len(a))
    b_pad = b + (0,) * (width - len(b))
    return a_pad > b_pad


def _select_installer_url(assets: Any, version_clean: str, release_url: str) -> str:
    """Pick the best Windows Setup .exe URL from GitHub Release ``assets``."""
    preferred_names = {
        f"pdf2musescore-{version_clean}-setup-x64.exe".lower(),
        INSTALLER_ASSET_NAME,
    }
    exe_assets: list[tuple[str, str]] = []
    for asset in assets or ():
        name = str(asset.get("name") or "").strip()
        url = str(asset.get("browser_download_url") or "")
        lower = name.lower()
        if lower in preferred_names and url:
            return url
        if lower.endswith(".exe") and url:
            exe_assets.append((lower, url))
    for lower, url in exe_assets:
        if "setup" in lower:
            return url
    return exe_assets[0][1] if exe_assets else release_url


def check_for_update(
    current_version: str,
    api_url: str = GITHUB_LATEST_RELEASE_API,
    timeout: float = 5.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Optional[UpdateInfo]:
    """Check GitHub Releases silently; log warning and return ``None`` on error or when up to date."""
    req = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": f"pdf2musescore/{current_version}",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with opener(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.warning("Update check failed (continuing offline): %s", exc)
        return None

    tag = str(data.get("tag_name") or "").strip()
    if not tag or not is_newer_version(tag, current_version):
        log.debug("Update check: current v%s is up to date (latest tag=%r)", current_version, tag)
        return None

    version_clean = tag.lstrip("vV")
    release_url = str(data.get("html_url") or RELEASES_FALLBACK_URL)
    download_url = _select_installer_url(data.get("assets"), version_clean, release_url)

    log.info("Update available: v%s (current v%s) -> %s", version_clean, current_version, download_url)
    return UpdateInfo(
        version=version_clean,
        download_url=download_url,
        release_url=release_url,
        notes=str(data.get("body") or ""),
    )


def download_installer(
    update: UpdateInfo,
    dest_dir: Path,
    progress: Optional[Callable[[int, str], None]] = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout: float = 60.0,
) -> Path:
    """Download ``update.download_url`` to ``dest_dir`` cleanly via a ``.part`` temp file."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_ver = re.sub(r"[^0-9A-Za-z._-]", "_", update.version or "latest")
    target = dest_dir / f"pdf2musescore-setup-{safe_ver}.exe"
    part = dest_dir / f"pdf2musescore-setup-{safe_ver}.exe.part"

    req = urllib.request.Request(
        update.download_url,
        headers={"User-Agent": f"pdf2musescore/{safe_ver}"},
    )
    try:
        downloaded = 0
        with opener(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length") or 0) if getattr(resp, "headers", None) else 0
            with part.open("wb") as fh:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        pct = (
                            min(99, int(downloaded * 100 / total))
                            if total > 0
                            else min(95, 10 + downloaded // (2 * 1048576))
                        )
                        progress(
                            pct,
                            f"Downloading update v{update.version} ({downloaded / 1048576:.1f} MB)...",
                        )
        if downloaded == 0:
            raise UpdateError("Downloaded installer was empty.")
        part.replace(target)
        log.info("Downloaded update installer to %s (%d bytes)", target, downloaded)
        return target
    except UpdateError:
        part.unlink(missing_ok=True)
        raise
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise UpdateError(f"Failed to download installer: {exc}") from exc
