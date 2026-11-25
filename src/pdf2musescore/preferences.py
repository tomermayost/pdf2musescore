"""Persistent GUI user preferences stored cleanly as atomic JSON (independent of config.toml)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from . import APP_NAME
from .config import is_frozen
from .options import (
    CLEF_CHOICES,
    FULL_SCORE_DEFAULT_WIDTH,
    FULL_SCORE_MAX_WIDTH,
    FULL_SCORE_MIN_WIDTH,
    PREPROCESS_MODES,
)

log = logging.getLogger(__name__)

VALID_THEMES = ("dark", "light")


def default_preferences_path() -> Path:
    """Return ``%APPDATA%/pdf2musescore/preferences.json`` when packaged, or ``tmp/preferences.json`` in dev."""
    if is_frozen():
        base = os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / APP_NAME / "preferences.json"
    return Path("tmp") / "preferences.json"


@dataclass(frozen=True)
class UserPreferences:
    theme: str = "dark"
    advanced_expanded: bool = False
    log_expanded: bool = True
    render_dpi: int = 300
    preprocess_mode: str = "auto"
    deskew: bool = False
    segnet_step_size: int = 320
    min_notehead_size: int = 4
    transpose_semitones: int = 0
    clef_override: str = "auto"
    instrument_name: str = "auto"
    keep_line_breaks: bool = False
    full_score: bool = False
    full_score_target_width: int = FULL_SCORE_DEFAULT_WIDTH
    last_dir: str = ""
    dismissed_dual_install: str = ""
    dismissed_crash: str = ""


def _clamp_default_dpi(default_dpi: int) -> int:
    try:
        dpi = int(default_dpi)
    except (TypeError, ValueError):
        return 300
    return dpi if 100 <= dpi <= 600 else 300


def _coerce_bool(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in raw:
        return default
    val = raw[key]
    if isinstance(val, bool):
        return val
    log.warning("Invalid preference %s=%r (expected boolean) - using %r", key, val, default)
    return default


def _coerce_int_range(raw: Mapping[str, Any], key: str, lo: int, hi: int, default: int) -> int:
    if key not in raw:
        return default
    val = raw[key]
    if isinstance(val, bool):
        log.warning("Invalid preference %s=%r (expected integer) - using %d", key, val, default)
        return default
    try:
        parsed = int(val)
    except (TypeError, ValueError):
        log.warning("Invalid preference %s=%r (expected integer) - using %d", key, val, default)
        return default
    if not (lo <= parsed <= hi):
        log.warning("Out-of-range preference %s=%d (valid %d..%d) - using %d", key, parsed, lo, hi, default)
        return default
    return parsed


def _coerce_choice(raw: Mapping[str, Any], key: str, valid: tuple[str, ...], default: str) -> str:
    if key not in raw:
        return default
    val = raw[key]
    cleaned = str(val).strip().lower() if val is not None else ""
    if cleaned in valid:
        return cleaned
    log.warning("Invalid preference %s=%r (expected one of %s) - using %r", key, val, valid, default)
    return default


def _coerce_str(raw: Mapping[str, Any], key: str, default: str = "") -> str:
    if key not in raw:
        return default
    val = raw[key]
    if isinstance(val, str):
        return val.strip()
    log.warning("Invalid preference %s=%r - ignoring", key, val)
    return default


def _sanitize_preferences(raw: Mapping[str, Any], default_dpi: int = 300) -> UserPreferences:
    """Validate and normalize every preference field independently without raising."""
    safe_dpi = _clamp_default_dpi(default_dpi)

    inst_raw = raw.get("instrument_name", "auto")
    if isinstance(inst_raw, str):
        inst = inst_raw.strip() or "auto"
    else:
        log.warning("Invalid preference instrument_name=%r - using 'auto'", inst_raw)
        inst = "auto"

    return UserPreferences(
        theme=_coerce_choice(raw, "theme", VALID_THEMES, "dark"),
        advanced_expanded=_coerce_bool(raw, "advanced_expanded", False),
        log_expanded=_coerce_bool(raw, "log_expanded", True),
        render_dpi=_coerce_int_range(raw, "render_dpi", 100, 600, safe_dpi),
        preprocess_mode=_coerce_choice(raw, "preprocess_mode", PREPROCESS_MODES, "auto"),
        deskew=_coerce_bool(raw, "deskew", False),
        segnet_step_size=_coerce_int_range(raw, "segnet_step_size", 80, 320, 320),
        min_notehead_size=_coerce_int_range(raw, "min_notehead_size", 2, 12, 4),
        transpose_semitones=_coerce_int_range(raw, "transpose_semitones", -24, 24, 0),
        clef_override=_coerce_choice(raw, "clef_override", CLEF_CHOICES, "auto"),
        instrument_name=inst,
        keep_line_breaks=_coerce_bool(raw, "keep_line_breaks", False),
        full_score=_coerce_bool(raw, "full_score", False),
        full_score_target_width=_coerce_int_range(
            raw, "full_score_target_width", FULL_SCORE_MIN_WIDTH, FULL_SCORE_MAX_WIDTH, FULL_SCORE_DEFAULT_WIDTH
        ),
        last_dir=_coerce_str(raw, "last_dir"),
        dismissed_dual_install=_coerce_str(raw, "dismissed_dual_install"),
        dismissed_crash=_coerce_str(raw, "dismissed_crash"),
    )


def load_preferences(path: Optional[Path] = None, default_dpi: int = 300) -> UserPreferences:
    """Load saved user preferences from JSON, falling back safely on missing or corrupt files."""
    target = Path(path) if path is not None else default_preferences_path()
    safe_dpi = _clamp_default_dpi(default_dpi)
    if not target.is_file():
        log.debug("No user preferences file at %s - using defaults", target)
        return UserPreferences(render_dpi=safe_dpi)

    try:
        raw_text = target.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.warning("Could not read preferences from %s (%s) - using defaults", target, exc)
        return UserPreferences(render_dpi=safe_dpi)

    if not isinstance(data, dict):
        log.warning("Invalid preferences root type in %s (%s) - using defaults", target, type(data).__name__)
        return UserPreferences(render_dpi=safe_dpi)

    log.info("Loaded user preferences from %s", target)
    return _sanitize_preferences(data, default_dpi=safe_dpi)


def save_preferences(prefs: UserPreferences, path: Optional[Path] = None) -> bool:
    """Atomically write ``prefs`` to JSON. Logs warning and returns False on I/O errors."""
    target = Path(path) if path is not None else default_preferences_path()
    tmp_path: Optional[Path] = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(prefs), indent=2, ensure_ascii=False) + "\n"
        fd, tmp_name = tempfile.mkstemp(prefix=".preferences.", suffix=".tmp", dir=str(target.parent))
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_name, target)
        log.debug("Saved user preferences to %s (theme=%s)", target, prefs.theme)
        return True
    except OSError as exc:
        log.warning("Failed to save user preferences to %s: %s", target, exc)
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return False
