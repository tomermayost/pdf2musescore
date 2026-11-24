import logging
from pathlib import Path

from pdf2musescore import __main__ as entry
from pdf2musescore import config as cfgmod
from pdf2musescore import gui
from pdf2musescore.logging_setup import _HANDLER_TAG


def _cleanup_handlers():
    for target in (logging.getLogger("pdf2musescore"), logging.getLogger()):
        for h in list(target.handlers):
            if getattr(h, _HANDLER_TAG, False):
                target.removeHandler(h)
                h.close()


def test_main_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PDF2MSCZ_LOG_DIR", str(tmp_path / "logs"))
    seen = []
    monkeypatch.setattr(gui, "run_gui", lambda c: seen.append(c))
    try:
        assert entry.main([]) == 0 and seen
    finally:
        _cleanup_handlers()
    log_text = (tmp_path / "logs" / "pdf2musescore.log").read_text(encoding="utf-8")
    assert "Environment diagnostic info:" in log_text
    assert (tmp_path / "logs" / "crash.log").is_file()


def test_main_version(capsys):
    assert entry.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "By Tomer Mayost" in out
    assert "(homr " in out


def test_main_self_test():
    assert entry.main(["--self-test"]) == 0


def test_main_bad_config(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PDF2MSCZ_LOG_LEVEL", "LOUD")
    monkeypatch.setattr(entry, "_show_fatal", lambda m: None)
    try:
        assert entry.main() == 2
    finally:
        _cleanup_handlers()


def test_main_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PDF2MSCZ_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(entry, "_show_fatal", lambda m: None)

    def boom(c):
        raise RuntimeError("x")
    monkeypatch.setattr(gui, "run_gui", boom)
    try:
        assert entry.main() == 1
    finally:
        _cleanup_handlers()


def test_frozen_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(cfgmod.sys, "frozen", True, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert cfgmod.default_data_dir() == tmp_path / "pdf2musescore"
    assert cfgmod.AppConfig().log_dir == tmp_path / "pdf2musescore" / "logs"


def test_dev_data_dir():
    assert cfgmod.default_data_dir() == Path("tmp")


def test_ascii_safe_base_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path))
    assert cfgmod.ascii_safe_base_dir(Path(r"C:\Users\תומר\AppData")) == tmp_path / "pdf2musescore"
    assert cfgmod.ascii_safe_base_dir(Path(r"C:\Users\tmayo\AppData")) == Path(r"C:\Users\tmayo\AppData")
