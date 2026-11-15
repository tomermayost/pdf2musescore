from pathlib import Path

from pdf2musescore import __main__ as entry
from pdf2musescore import config as cfgmod
from pdf2musescore import gui


def test_main_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PDF2MSCZ_LOG_DIR", str(tmp_path / "logs"))
    seen = []
    monkeypatch.setattr(gui, "run_gui", lambda c: seen.append(c))
    assert entry.main() == 0 and seen


def test_main_bad_config(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PDF2MSCZ_LOG_LEVEL", "LOUD")
    monkeypatch.setattr(entry, "_show_fatal", lambda m: None)
    assert entry.main() == 2


def test_main_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PDF2MSCZ_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(entry, "_show_fatal", lambda m: None)

    def boom(c):
        raise RuntimeError("x")
    monkeypatch.setattr(gui, "run_gui", boom)
    assert entry.main() == 1


def test_frozen_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(cfgmod.sys, "frozen", True, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert cfgmod.default_data_dir() == tmp_path / "pdf2musescore"
    assert cfgmod.AppConfig().log_dir == tmp_path / "pdf2musescore" / "logs"


def test_dev_data_dir():
    assert cfgmod.default_data_dir() == Path("tmp")
