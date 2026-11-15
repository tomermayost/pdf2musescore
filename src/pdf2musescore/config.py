"""Application configuration.

Precedence (highest first):
    1. Environment variables (``PDF2MSCZ_*``)
    2. ``config.toml`` (path given explicitly, or ``%APPDATA%/pdf2musescore/config.toml``)
    3. Built-in defaults
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, Optional

if sys.version_info >= (3, 11):  # pragma: no cover - depends on interpreter
    import tomllib
else:  # pragma: no cover - depends on interpreter
    import tomli as tomllib

from . import APP_NAME

log = logging.getLogger(__name__)

ENV_PREFIX = "PDF2MSCZ_"
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


def is_frozen() -> bool:
    """True when running as a packaged (PyInstaller) executable."""
    return bool(getattr(sys, "frozen", False))


def ascii_safe_base_dir(preferred: Path) -> Path:
    """Return ``preferred`` if ASCII-only, otherwise an ASCII-safe fallback directory."""
    if str(preferred).isascii():
        return preferred
    fallback = Path(os.environ.get("PROGRAMDATA") or r"C:\Users\Public") / APP_NAME
    log.info("Path %s contains non-ASCII characters; using ASCII-safe %s", preferred, fallback)
    return fallback


def default_data_dir() -> Path:
    """Writable base dir: ``tmp/`` in development, ``%LOCALAPPDATA%/pdf2musescore`` when packaged."""
    if is_frozen():
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return ascii_safe_base_dir(Path(base) / APP_NAME)
    return Path("tmp")


def default_config_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / APP_NAME / "config.toml"


@dataclass(frozen=True)
class AppConfig:
    log_level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: default_data_dir() / "logs")
    work_dir: Path = field(default_factory=lambda: default_data_dir() / "work")
    musescore_path: Optional[Path] = None
    render_dpi: int = 300

    def __post_init__(self) -> None:
        level = str(self.log_level).upper()
        if level not in VALID_LOG_LEVELS:
            raise ConfigError(f"Invalid log_level {self.log_level!r}; expected one of {VALID_LOG_LEVELS}")
        object.__setattr__(self, "log_level", level)
        if int(self.render_dpi) < 72:
            raise ConfigError("render_dpi must be >= 72")


_PATH_FIELDS = {"log_dir", "work_dir", "musescore_path"}
_INT_FIELDS = {"render_dpi"}


def _coerce(name: str, value: Any) -> Any:
    if value is None or value == "":
        return None
    if name in _PATH_FIELDS:
        return Path(value)
    if name in _INT_FIELDS:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name} must be an integer, got {value!r}") from exc
    return str(value)


def _from_mapping(base: AppConfig, data: Mapping[str, Any], source: str) -> AppConfig:
    known = {f.name for f in fields(AppConfig)}
    updates: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            log.warning("Unknown config key %r in %r - ignored", key, source)
            continue
        updates[key] = _coerce(key, value)
        log.debug("Config %s=%r (from %s)", key, updates[key], source)
    return replace(base, **updates)


def load_config(path: Optional[Path] = None, environ: Optional[Mapping[str, str]] = None) -> AppConfig:
    """Load configuration from defaults, file and environment."""
    environ = os.environ if environ is None else environ
    cfg = AppConfig()

    cfg_path = Path(path) if path else default_config_path()
    if cfg_path.is_file():
        log.info("Loading config file %s", cfg_path)
        try:
            with cfg_path.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Invalid TOML in {cfg_path}: {exc}") from exc
        cfg = _from_mapping(cfg, data, str(cfg_path))
    else:
        log.debug("No config file at %s - using defaults", cfg_path)

    env_data = {
        k[len(ENV_PREFIX):].lower(): v for k, v in environ.items() if k.startswith(ENV_PREFIX)
    }
    if env_data:
        cfg = _from_mapping(cfg, env_data, "environment")
    return cfg
