import logging
import time
import tkinter as tk
from pathlib import Path

import pytest

from pdf2musescore.config import AppConfig
from pdf2musescore.engines.homr_engine import CancelledError
from pdf2musescore.gui import App
from pdf2musescore.options import ConversionOptions
from pdf2musescore.pipeline import ConversionError


class FakePipeline:
    behaviour = "ok"
    last_pages_spec = None
    last_options = None

    def __init__(self, config, confirm, progress, is_cancelled=lambda: False):
        self.confirm, self.progress, self.is_cancelled = confirm, progress, is_cancelled

    def convert(self, pdf, out, pages_spec=None, options=None):
        FakePipeline.last_pages_spec = pages_spec
        FakePipeline.last_options = options
        self.progress(50, "half")
        logging.getLogger("pdf2musescore.test").info("working")
        if self.behaviour == "abort" or self.is_cancelled():
            raise CancelledError("aborted")
        if self.behaviour == "confirm":
            if not self.confirm("Install?"):
                raise ConversionError("declined")
        elif self.behaviour == "error":
            raise ConversionError("friendly message")
        elif self.behaviour == "crash":
            raise RuntimeError("boom")
        return out


@pytest.fixture(scope="module")
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
             yes=True, musescore=True, page_counter=lambda p: 4):
    logging.getLogger("pdf2musescore").setLevel("DEBUG")
    return App(root, AppConfig(log_dir=tmp_path / "logs"), pipeline_factory=FakePipeline,
               musescore_available=lambda: musescore,
               page_counter=page_counter,
               ask_open=lambda **k: open_path,
               ask_save=lambda **k: (calls.__setitem__("save_kwargs", k) or save_path),
               ask_yes_no=lambda t, m: (calls["yesno"].append(m) or yes),
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
    assert str(app.abort_btn["state"]) == "disabled"
    assert app.credit_label.cget("text") == "By Tomer Mayost"
    assert "By Tomer Mayost" in root.title()
    app.on_convert()  # no pdf -> no-op
    app.destroy()


def test_options_guide_reset_and_validation(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path)
    app.on_load()
    app.on_show_parameter_guide()
    assert any("2. PDF Rendering & AI Engine" in msg for msg in calls["info"])
    calls["info"].clear()

    # Set custom values then reset
    app.dpi_var.set("400")
    app.preprocess_var.set("faded")
    app.deskew_var.set(True)
    app.step_size_var.set("240")
    app.min_notehead_var.set("3")
    app.transpose_var.set("+2")
    app.clef_var.set("bass")
    app.instrument_var.set("Trombone")
    app.title_var.set("Sonata")
    opts = app.collect_options()
    assert opts == ConversionOptions(
        render_dpi=400,
        preprocess_mode="faded",
        deskew=True,
        segnet_step_size=240,
        min_notehead_size=3,
        transpose_semitones=2,
        clef_override="bass",
        instrument_name="Trombone",
        title_override="Sonata",
    )

    app.on_reset_options()
    assert app.collect_options() == ConversionOptions()

    # Invalid integer fields trigger friendly error popup on_convert
    for var, bad in (
        (app.dpi_var, "abc"),
        (app.step_size_var, "xyz"),
        (app.min_notehead_var, "bad"),
        (app.transpose_var, "notint"),
        (app.dpi_var, "999"),
    ):
        app.on_reset_options()
        var.set(bad)
        app.on_convert()
        assert calls["error"]
        calls["error"].clear()
    app.destroy()


def test_load_and_convert_with_pages(root, calls, tmp_path):
    FakePipeline.behaviour = "ok"
    app = make_app(root, calls, tmp_path)
    app.on_load()
    assert app.file_var.get() == "song.pdf (4 pages)"
    assert str(app.convert_btn["state"]) == "normal"
    app.pages_var.set("2-3")
    app.transpose_var.set("2")
    app.on_convert()
    assert calls["save_kwargs"]["defaultextension"] == ".mscz"
    pump(app, root, lambda: calls["info"])
    assert FakePipeline.last_pages_spec == "2-3"
    assert FakePipeline.last_options.transpose_semitones == 2
    assert "song.mscz" in calls["info"][0]
    assert "working" in app.log_text.get("1.0", "end")
    assert app.progress["value"] == 50
    app.destroy()


def test_load_page_counter_error_fallback(root, calls, tmp_path):
    def bad_counter(p):
        raise ValueError("bad pdf")
    app = make_app(root, calls, tmp_path, page_counter=bad_counter)
    app.on_load()
    assert app.file_var.get() == "song.pdf"
    app.destroy()


def test_abort_conversion(root, calls, tmp_path):
    FakePipeline.behaviour = "abort"
    app = make_app(root, calls, tmp_path)
    app.on_load()
    app.on_abort()
    assert app._cancel_event.is_set()
    app.on_convert()
    pump(app, root, lambda: "aborted by user" in app.status_var.get())
    assert calls["error"] == []
    assert app.progress["value"] == 0
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
