"""Logging configuration: console + rotating file, configurable level."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from . import APP_NAME

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_HANDLER_TAG = "_pdf2musescore_handler"


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> logging.Logger:
    """Configure the package logger. Idempotent: replaces handlers it previously added."""
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False

    for h in list(logger.handlers):
        if getattr(h, _HANDLER_TAG, False):
            logger.removeHandler(h)
            h.close()

    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    setattr(console, _HANDLER_TAG, True)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / f"{APP_NAME}.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        setattr(fh, _HANDLER_TAG, True)
        logger.addHandler(fh)

    logger.debug("Logging initialised (level=%s, log_dir=%s)", level, log_dir)
    return logger
