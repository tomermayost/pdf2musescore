import logging
import sys

from pdf2musescore import diagnostics
from pdf2musescore.logging_setup import _HANDLER_TAG, setup_logging


def _cleanup_tagged_handlers():
    for target in (logging.getLogger("pdf2musescore"), logging.getLogger()):
        for h in list(target.handlers):
            if getattr(h, _HANDLER_TAG, False):
                target.removeHandler(h)
                h.close()


def test_console_only():
    try:
        logger = setup_logging("WARNING")
        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 1
    finally:
        _cleanup_tagged_handlers()


def test_file_handler_writes_package_and_third_party(tmp_path):
    try:
        logger = setup_logging("debug", tmp_path / "logs", third_party_level="INFO")
        logger.debug("hello-file")
        logging.getLogger("homr.fake").info("third-party-info")
        logging.getLogger("homr.fake").debug("third-party-debug-hidden")
        for h in logger.handlers:
            h.flush()
        content = (tmp_path / "logs" / "pdf2musescore.log").read_text(encoding="utf-8")
        assert "hello-file" in content
        assert "third-party-info" in content
        assert "third-party-debug-hidden" not in content
    finally:
        _cleanup_tagged_handlers()


def test_idempotent_and_stdio_passthrough(tmp_path):
    orig_out, orig_err = sys.stdout, sys.stderr
    restore = diagnostics.install_stdio_capture()
    try:
        setup_logging("INFO", tmp_path)
        logger = setup_logging("INFO", tmp_path)
        own = [h for h in logger.handlers if getattr(h, _HANDLER_TAG, False)]
        assert len(own) == 2
        root_own = [h for h in logging.getLogger().handlers if getattr(h, _HANDLER_TAG, False)]
        assert len(root_own) == 2
        stream_handlers = [h for h in own if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        if orig_err is not None:
            assert stream_handlers[0].stream is orig_err
    finally:
        restore()
        sys.stdout = orig_out
        sys.stderr = orig_err
        _cleanup_tagged_handlers()
