"""Read-only detection of side-by-side Microsoft Store (MSIX) and Standalone (.exe) installs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import sys
from typing import Callable, Literal, Optional, Sequence

try:  # pragma: no cover - Windows standard library module
    import winreg  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - non-Windows hosts
    winreg = None  # type: ignore[assignment]

from .config import is_frozen
from .updater import is_newer_version

log = logging.getLogger(__name__)

MSIX_PACKAGE_NAME = "TomerMayost.PDF2MuseScore"
MSIX_PACKAGE_PREFIX = f"{MSIX_PACKAGE_NAME}_"
MSIX_STORE_PRODUCT_ID = "9N6NCBWM3BHT"
MSIX_STORE_URI = f"ms-windows-store://pdp/?ProductId={MSIX_STORE_PRODUCT_ID}"
MSIX_STORE_WEB_URL = f"https://apps.microsoft.com/detail/{MSIX_STORE_PRODUCT_ID}"
WINDOWS_INSTALLED_APPS_URI = "ms-settings:appsfeatures"

INNO_UNINSTALL_SUBKEY = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{8D4F61C0-5A2E-4E91-9A23-7B8C1F42E101}_is1"
)
APPMODEL_PACKAGES_SUBKEY = (
    r"Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppModel\Repository\Packages"
)

EditionKind = Literal["store", "standalone"]
VersionComparison = Literal["other_newer", "same", "current_newer"]


@dataclass(frozen=True)
class InstalledEdition:
    """Represents a detected installation of PDF2MuseScore."""

    kind: EditionKind
    version: str
    location: str = ""
    package_full_name: str = ""


@dataclass(frozen=True)
class DualInstallNotice:
    """Describes an active side-by-side Store/Standalone installation conflict or mismatch."""

    current_kind: EditionKind
    current_version: str
    other: InstalledEdition
    comparison: VersionComparison
    message: str
    show_store_update: bool
    dismiss_key: str


def normalize_version_display(raw: str) -> str:
    """Normalize a version string such as ``'v0.4.3.0'`` into ``'0.4.3'``."""
    text = (raw or "").strip().lstrip("vV")
    if not text:
        return "0.0.0"
    parts = text.split(".")
    if len(parts) == 4 and parts[3] == "0":
        parts = parts[:3]
    return ".".join(parts)


def parse_msix_package_version(
    package_full_name: str,
    prefix: str = MSIX_PACKAGE_PREFIX,
) -> Optional[str]:
    """Extract normalized version from a Windows PackageFullName matching ``prefix``."""
    if not package_full_name or len(package_full_name) <= len(prefix):
        return None
    if package_full_name[: len(prefix)].lower() != prefix.lower():
        return None
    remainder = package_full_name[len(prefix) :]
    ver_segment = remainder.split("_", 1)[0].strip()
    if not re.match(r"^\d+(?:\.\d+)*$", ver_segment):
        return None
    return normalize_version_display(ver_segment)


def compare_versions(ver_a: str, ver_b: str) -> int:
    """Compare two version strings cleanly: ``1`` if a > b, ``-1`` if a < b, ``0`` if equal."""
    if is_newer_version(ver_a, ver_b):
        return 1
    if is_newer_version(ver_b, ver_a):
        return -1
    return 0


def _default_enum_hkcu_subkeys(subkey: str) -> Sequence[str]:
    if sys.platform != "win32" or winreg is None:
        return ()
    names: list[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, subkey, 0, winreg.KEY_READ) as key:
            idx = 0
            while True:
                try:
                    names.append(winreg.EnumKey(key, idx))
                    idx += 1
                except OSError:
                    break
    except OSError as exc:
        log.debug("Registry key HKCU\\%s unreadable: %s", subkey, exc)
        return ()
    return names


def _default_query_uninstall_entry(root_name: str, subkey: str) -> Optional[tuple[str, str]]:
    if sys.platform != "win32" or winreg is None:
        return None
    hive = winreg.HKEY_CURRENT_USER if root_name == "HKCU" else winreg.HKEY_LOCAL_MACHINE
    try:
        with winreg.OpenKey(hive, subkey, 0, winreg.KEY_READ) as key:
            version, _ = winreg.QueryValueEx(key, "DisplayVersion")
            try:
                location, _ = winreg.QueryValueEx(key, "InstallLocation")
            except OSError:
                location = ""
            return str(version or "").strip(), str(location or "").strip()
    except OSError as exc:
        log.debug("Registry uninstall key %s\\%s unreadable: %s", root_name, subkey, exc)
        return None


def detect_store_install(
    *,
    enum_subkeys: Optional[Callable[[str], Sequence[str]]] = None,
) -> Optional[InstalledEdition]:
    """Detect whether the Microsoft Store MSIX edition is installed for the current user."""
    reader = _default_enum_hkcu_subkeys if enum_subkeys is None else enum_subkeys
    try:
        names = reader(APPMODEL_PACKAGES_SUBKEY)
    except Exception as exc:
        log.debug("Store install probe failed: %s", exc)
        return None

    best: Optional[InstalledEdition] = None
    for full_name in names:
        ver = parse_msix_package_version(str(full_name))
        if ver is None:
            continue
        candidate = InstalledEdition(
            kind="store",
            version=ver,
            package_full_name=str(full_name),
        )
        if best is None or compare_versions(candidate.version, best.version) > 0:
            best = candidate
    return best


def detect_standalone_install(
    *,
    query_uninstall: Optional[Callable[[str, str], Optional[tuple[str, str]]]] = None,
) -> Optional[InstalledEdition]:
    """Detect whether the standalone Inno Setup (.exe) edition is installed."""
    reader = _default_query_uninstall_entry if query_uninstall is None else query_uninstall
    for root_name in ("HKCU", "HKLM"):
        try:
            hit = reader(root_name, INNO_UNINSTALL_SUBKEY)
        except Exception as exc:
            log.debug("Standalone install probe failed for %s: %s", root_name, exc)
            continue
        if hit is None:
            continue
        raw_ver, location = hit
        cleaned_ver = (raw_ver or "").strip()
        if not cleaned_ver:
            continue
        return InstalledEdition(
            kind="standalone",
            version=normalize_version_display(cleaned_ver),
            location=(location or "").strip(),
        )
    return None


def detect_other_edition(
    *,
    store_mode: bool,
    store_detector: Optional[Callable[[], Optional[InstalledEdition]]] = None,
    standalone_detector: Optional[Callable[[], Optional[InstalledEdition]]] = None,
) -> Optional[InstalledEdition]:
    """Probe for the sibling installation kind opposite ``store_mode``."""
    if store_mode:
        probe = detect_standalone_install if standalone_detector is None else standalone_detector
    else:
        probe = detect_store_install if store_detector is None else store_detector
    try:
        return probe()
    except Exception as exc:
        log.debug("Sibling edition detection failed: %s", exc)
        return None


_NOTICE_MESSAGES: dict[tuple[EditionKind, VersionComparison], str] = {
    (
        "store",
        "other_newer",
    ): "Running Microsoft Store v{cur}, while Standalone .exe v{other} is also installed. Open Microsoft Store to update, or remove the duplicate in Installed Apps.",
    (
        "store",
        "same",
    ): "Both Microsoft Store (v{cur}) and Standalone .exe (v{other}) are installed. Remove the Standalone copy in Installed Apps to avoid duplicate Start Menu entries.",
    (
        "store",
        "current_newer",
    ): "Running newer Microsoft Store v{cur}, while older Standalone .exe v{other} is still installed. You can remove the old Standalone copy in Installed Apps.",
    (
        "standalone",
        "other_newer",
    ): "Running Standalone .exe v{cur}, while newer Microsoft Store v{other} is installed. Remove the Standalone copy in Installed Apps to keep the newer Store edition.",
    (
        "standalone",
        "same",
    ): "Both Standalone .exe (v{cur}) and Microsoft Store (v{other}) are installed. Remove one copy in Installed Apps to avoid duplicate Start Menu entries.",
    (
        "standalone",
        "current_newer",
    ): "Running Standalone .exe v{cur}, while older Microsoft Store v{other} is also installed. Open Microsoft Store to update it, or keep just one copy in Installed Apps.",
}


def check_dual_install(
    current_version: str,
    *,
    store_mode: bool,
    frozen: Optional[bool] = None,
    store_detector: Optional[Callable[[], Optional[InstalledEdition]]] = None,
    standalone_detector: Optional[Callable[[], Optional[InstalledEdition]]] = None,
) -> Optional[DualInstallNotice]:
    """Return a non-intrusive notice if both Store and Standalone editions are installed."""
    is_app_frozen = is_frozen() if frozen is None else bool(frozen)
    if not is_app_frozen:
        return None

    other = detect_other_edition(
        store_mode=store_mode,
        store_detector=store_detector,
        standalone_detector=standalone_detector,
    )
    if other is None:
        return None

    cur = normalize_version_display(current_version)
    current_kind: EditionKind = "store" if store_mode else "standalone"
    cmp_val = compare_versions(other.version, cur)
    if cmp_val > 0:
        comparison: VersionComparison = "other_newer"
    elif cmp_val < 0:
        comparison = "current_newer"
    else:
        comparison = "same"

    show_store_update = (current_kind == "store" and comparison == "other_newer") or (
        current_kind == "standalone" and comparison == "current_newer"
    )
    message = _NOTICE_MESSAGES[(current_kind, comparison)].format(cur=cur, other=other.version)
    dismiss_key = f"{current_kind}:{cur}|{other.kind}:{other.version}"
    log.info(
        "Dual installation detected: running %s v%s alongside %s v%s (%s)",
        current_kind,
        cur,
        other.kind,
        other.version,
        comparison,
    )
    return DualInstallNotice(
        current_kind=current_kind,
        current_version=cur,
        other=other,
        comparison=comparison,
        message=message,
        show_store_update=show_store_update,
        dismiss_key=dismiss_key,
    )
