import logging

from pdf2musescore.logging_setup import setup_logging


def test_console_only():
    logger = setup_logging("WARNING")
    assert logger.level == logging.WARNING
    assert len(logger.handlers) == 1


def test_file_handler_writes(tmp_path):
    logger = setup_logging("debug", tmp_path / "logs")
    logger.debug("hello-file")
    for h in logger.handlers:
        h.flush()
    content = (tmp_path / "logs" / "pdf2musescore.log").read_text(encoding="utf-8")
    assert "hello-file" in content


def test_idempotent(tmp_path):
    setup_logging("INFO", tmp_path)
    logger = setup_logging("INFO", tmp_path)
    own = [h for h in logger.handlers if getattr(h, "_pdf2musescore_handler", False)]
    assert len(own) == 2
    for h in own:
        h.close()
