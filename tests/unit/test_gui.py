import logging
import time
import tkinter as tk
from pathlib import Path

import pytest

from pdf2musescore.config import AppConfig
from pdf2musescore.gui import App
from pdf2musescore.pipeline import ConversionError


class FakePipeline:
    behaviour = "ok"

    def __init__(self, config, confirm, progress):
        self.confirm, self.progress = confirm, progress

    def convert(self, pdf, out):
        self.progress(50, "half")
        logging.getLogger("pdf2musescore.test").info("working")
        if self.behaviour == "confirm":
            if not self.confirm("Install?"):
                raise ConversionError("declined")
        elif self.behaviour == "error":
            raise ConversionError("friendly message")
        elif self.behaviour == "crash":
            raise RuntimeError("boom")
        return out


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("No display")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def calls():
    return {"error": [], "info": [], "yesno": [], "open": []}


def make_app(root, calls, tmp_path, open_path="C:/x/song.pdf", save_path="C:/x/song.mscz",
             yes=True, musescore=True):
    logging.getLogger("pdf2musescore").setLevel("DEBUG")
    return App(root, AppConfig(log_dir=tmp_path / "logs"), pipeline_factory=FakePipeline,
               musescore_available=lambda: musescore,
               ask_open=lambda **k: open_path,
               ask_save=lambda **k: calls.setdefault("save_kwargs", k) and save_path,
               ask_yes_no=lambda t, m: calls["yesno"].append(m) or yes,
               show_error=lambda t, m: calls["error"].append(m),
               show_info=lambda t, m: calls["info"].append(m),
               open_folder=lambda p: calls["open"].append(p))


def pump(app, root, until, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        app.process_queue()
        root.update()
        if until():
            return
        time.sleep(0.02)
    raise AssertionError("timeout")


def test_initial_state(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path)
    assert str(app.convert_btn["state"]) == "disabled"
    app.on_convert()  # no pdf -> no-op
    app.destroy()


def test_load_and_convert_success(root, calls, tmp_path):
    FakePipeline.behaviour = "ok"
    app = make_app(root, calls, tmp_path)
    app.on_load()
    assert app.file_var.get() == "song.pdf"
    assert str(app.convert_btn["state"]) == "normal"
    app.on_convert()
    assert calls["save_kwargs"]["defaultextension"] == ".mscz"
    pump(app, root, lambda: calls["info"])
    assert "song.mscz" in calls["info"][0]
    assert "working" in app.log_text.get("1.0", "end")
    assert app.progress["value"] == 50
    app.destroy()


def test_xml_default_when_no_musescore(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path, save_path="", musescore=False)
    app.on_load()
    app.on_convert()  # cancelled save
    assert calls["save_kwargs"]["defaultextension"] == ".musicxml"
    assert app.worker is None
    app.destroy()


def test_load_cancelled(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path, open_path="")
    app.on_load()
    assert app.pdf is None
    app.destroy()


@pytest.mark.parametrize("behaviour,yes,expected", [
    ("error", True, "friendly message"),
    ("crash", True, "Unexpected error: boom"),
    ("confirm", False, "declined"),
])
def test_errors(root, calls, tmp_path, behaviour, yes, expected):
    FakePipeline.behaviour = behaviour
    app = make_app(root, calls, tmp_path, yes=yes)
    app.on_load()
    app.on_convert()
    pump(app, root, lambda: calls["error"])
    assert calls["error"] == [expected]
    assert "failed" in app.status_var.get()
    app.destroy()


def test_confirm_yes(root, calls, tmp_path):
    FakePipeline.behaviour = "confirm"
    app = make_app(root, calls, tmp_path, yes=True)
    app.on_load()
    app.on_convert()
    pump(app, root, lambda: calls["info"])
    assert "permission" in calls["yesno"][0]
    app.destroy()


def test_open_logs(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path)
    app.on_open_logs()
    assert Path(calls["open"][0]).is_dir()
    app._poll()
    app.destroy()


def test_default_musescore_check(root, tmp_path):
    app = App(root, AppConfig(musescore_path=tmp_path / "none.exe"))
    assert isinstance(app.musescore_available(), bool)
    app.destroy()
