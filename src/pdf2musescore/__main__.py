"""Application entry point."""

from __future__ import annotations

import logging
import sys

from pdf2musescore import AUTHOR_CREDIT, __version__
from pdf2musescore.config import ConfigError, is_msix_package, load_config
from pdf2musescore.engines.homr_engine import get_engine_version, get_rapidocr_version
from pdf2musescore.logging_setup import setup_logging


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if "--version" in args:
        if sys.stdout is not None:
            print(f"pdf2musescore {__version__} (homr {get_engine_version()}) ({AUTHOR_CREDIT})")
        return 0
    if "--self-test" in args:
        import cv2  # noqa: F401
        import homr.main  # noqa: F401
        import onnxruntime as ort  # noqa: F401
        import pypdfium2  # noqa: F401
        import rapidocr  # noqa: F401
        from PIL import ImageTk  # noqa: F401

        from pdf2musescore.ui_theme import get_assets_dir

        assets = get_assets_dir()
        assert (assets / "icon.ico").is_file()
        assert (assets / "icon.png").is_file()
        assert get_engine_version() != "unknown"
        assert get_rapidocr_version() != "unknown"
        assert ort.get_available_providers()
        return 0
    try:
        config = load_config()
    except ConfigError as exc:
        setup_logging("INFO")
        logging.getLogger("pdf2musescore").error("Invalid configuration: %s", exc)
        _show_fatal(f"Invalid configuration:\n{exc}")
        return 2
    log = setup_logging(config.log_level, config.log_dir)
    log.info(
        "pdf2musescore %s (homr %s) starting (python %s, frozen=%s, msix=%s)",
        __version__,
        get_engine_version(),
        sys.version.split()[0],
        getattr(sys, "frozen", False),
        is_msix_package(),
    )
    from pdf2musescore.gui import run_gui
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
    code = main()
    import os
    os._exit(code)
