"""Application entry point."""

from __future__ import annotations

import logging
import sys

from . import __version__
from .config import ConfigError, load_config
from .logging_setup import setup_logging


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        setup_logging("INFO")
        logging.getLogger("pdf2musescore").error("Invalid configuration: %s", exc)
        _show_fatal(f"Invalid configuration:\n{exc}")
        return 2
    log = setup_logging(config.log_level, config.log_dir)
    log.info("pdf2musescore %s starting (python %s, frozen=%s)", __version__, sys.version.split()[0],
             getattr(sys, "frozen", False))
    from .gui import run_gui
    try:
        run_gui(config)
    except Exception as exc:
        log.exception("Fatal error")
        _show_fatal(str(exc))
        return 1
    return 0


def _show_fatal(message: str) -> None:  # pragma: no cover - UI popup
    try:
        from tkinter import messagebox
        messagebox.showerror("PDF to MuseScore", message)
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
