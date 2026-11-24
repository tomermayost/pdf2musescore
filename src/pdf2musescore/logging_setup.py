"""Logging configuration: console + rotating file, configurable level."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from . import APP_NAME

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_HANDLER_TAG = "_pdf2musescore_handler"


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    third_party_level: str = "INFO",
) -> logging.Logger:
    """Configure package and root loggers (plus Python warnings). Idempotent."""
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False

    root_logger = logging.getLogger()
    root_logger.setLevel(third_party_level.upper())

    removed: list[logging.Handler] = []
    for target in (logger, root_logger):
        for h in list(target.handlers):
            if getattr(h, _HANDLER_TAG, False):
                target.removeHandler(h)
                if h not in removed:
                    removed.append(h)
    for h in removed:
        h.close()

    formatter = logging.Formatter(LOG_FORMAT)
    new_handlers: list[logging.Handler] = []

    console_stream = getattr(sys.stderr, "passthrough", sys.stderr)
    if console_stream is not None:
        console = logging.StreamHandler(console_stream)
        console.setFormatter(formatter)
        setattr(console, _HANDLER_TAG, True)
        new_handlers.append(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / f"{APP_NAME}.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        setattr(fh, _HANDLER_TAG, True)
        new_handlers.append(fh)

    for h in new_handlers:
        logger.addHandler(h)
        root_logger.addHandler(h)

    logging.captureWarnings(True)
    logger.debug(
        "Logging initialised (level=%s, third_party_level=%s, log_dir=%s)",
        level,
        third_party_level,
        log_dir,
    )
    return logger
