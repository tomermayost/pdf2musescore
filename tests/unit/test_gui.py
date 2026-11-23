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
from pdf2musescore.updater import UpdateError, UpdateInfo


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


def close_app(app):
    if app.worker is not None and app.worker.is_alive():
        app.worker.join(timeout=2)
    if getattr(app, "update_thread", None) is not None and app.update_thread.is_alive():
        app.update_thread.join(timeout=2)
    app.destroy()
    for name in list(app.__dict__):
        if name.endswith("_var"):
            delattr(app, name)
    import gc
    gc.collect()


@pytest.fixture
def calls():
    return {"error": [], "info": [], "yesno": [], "open": [], "launch": [], "url": [], "exit": []}


def make_app(root, calls, tmp_path, open_path="C:/x/song.pdf", save_path="C:/x/song.mscz",
             yes=True, musescore=True, page_counter=lambda p: 4, check_updates=False,
             engine_version="0.7.1", update_checker=None, installer_downloader=None):
    logging.getLogger("pdf2musescore").setLevel("DEBUG")
    kwargs = {}
    if installer_downloader is not None:
        kwargs["installer_downloader"] = installer_downloader
    return App(root, AppConfig(log_dir=tmp_path / "logs", work_dir=tmp_path / "work", check_updates=check_updates),
               pipeline_factory=FakePipeline,
               musescore_available=lambda: musescore,
               page_counter=page_counter,
               ask_open=lambda **k: open_path,
               ask_save=lambda **k: (calls.__setitem__("save_kwargs", k) or save_path),
               ask_yes_no=lambda t, m: (calls["yesno"].append((t, m)) or (yes.pop(0) if isinstance(yes, list) else yes)),
               show_error=lambda t, m: calls["error"].append(m),
               show_info=lambda t, m: calls["info"].append(m),
               open_folder=lambda p: calls["open"].append(p),
               engine_version=engine_version,
               update_checker=update_checker,
               launch_installer=lambda p: calls["launch"].append(p),
               open_url=lambda u: calls["url"].append(u) or True,
               on_exit=lambda: calls["exit"].append(True),
               **kwargs)


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
    app = make_app(root, calls, tmp_path, engine_version="0.7.1")
    assert str(app.convert_btn["state"]) == "disabled"
    assert str(app.abort_btn["state"]) == "disabled"
    assert app.credit_label.cget("text") == "By Tomer Mayost"
    assert "(homr 0.7.1)" in root.title()
    assert "By Tomer Mayost" in root.title()
    assert not hasattr(app, "title_var")
    app.on_convert()  # no pdf -> no-op
    close_app(app)


def test_options_guide_reset_and_validation(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path)
    app.on_load()
    # Verify hover tooltips are attached to all parameters and show/hide cleanly
    assert "render_dpi" in app.tooltips and "pages" in app.tooltips and "keep_line_breaks" in app.tooltips
    assert "title_override" not in app.tooltips
    dpi_tip = app.tooltips["render_dpi"]
    dpi_tip.show()
    dpi_tip.show()  # idempotent when already shown
    assert dpi_tip.tip_window is not None
    dpi_tip.hide()
    assert dpi_tip.tip_window is None

    # Set custom values then reset
    app.dpi_var.set("400")
    app.preprocess_var.set("faded")
    app.deskew_var.set(True)
    app.step_size_var.set("240")
    app.min_notehead_var.set("3")
    app.keep_breaks_var.set(True)
    app.transpose_var.set("+2")
    app.clef_var.set("bass")
    app.instrument_var.set("Trombone")
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
        keep_line_breaks=True,
    )

    app.on_reset_options()
    assert app.keep_breaks_var.get() is False
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
    close_app(app)


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
    close_app(app)


def test_load_page_counter_error_fallback(root, calls, tmp_path):
    def bad_counter(p):
        raise ValueError("bad pdf")
    app = make_app(root, calls, tmp_path, page_counter=bad_counter)
    app.on_load()
    assert app.file_var.get() == "song.pdf"
    close_app(app)


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
    close_app(app)


def test_xml_default_when_no_musescore(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path, save_path="", musescore=False)
    app.on_load()
    app.on_convert()  # cancelled save
    assert calls["save_kwargs"]["defaultextension"] == ".musicxml"
    assert app.worker is None
    close_app(app)


def test_load_cancelled(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path, open_path="")
    app.on_load()
    assert app.pdf is None
    close_app(app)


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
    close_app(app)


def test_confirm_yes(root, calls, tmp_path):
    FakePipeline.behaviour = "confirm"
    app = make_app(root, calls, tmp_path, yes=True)
    app.on_load()
    app.on_convert()
    pump(app, root, lambda: calls["info"])
    assert "permission" in calls["yesno"][0][1]
    close_app(app)


def test_update_check_offline_quiet(root, calls, tmp_path, caplog):
    def failing_checker(_ver):
        raise OSError("offline")

    with caplog.at_level("WARNING"):
        app = make_app(root, calls, tmp_path, check_updates=True, update_checker=failing_checker)
        pump(app, root, lambda: not app.update_thread.is_alive())
        app.process_queue()
    assert "continuing offline" in caplog.text
    assert calls["error"] == []
    assert calls["yesno"] == []
    assert not app.update_btn.winfo_manager()
    app.on_load()
    assert app.pdf is not None
    close_app(app)


def test_update_available_no_then_yes_downloads_and_launches(root, calls, tmp_path):
    info = UpdateInfo("9.9.9", "https://example.com/pdf2musescore-setup.exe", "https://example.com/rel")
    fake_installer = tmp_path / "pdf2musescore-setup-9.9.9.exe"

    def fake_download(upd, dest_dir, progress=None):
        if progress:
            progress(50, "Downloading...")
        fake_installer.write_bytes(b"MZ")
        return fake_installer

    # First prompt -> No (defers); click footer button -> Yes (downloads and launches)
    app = make_app(
        root, calls, tmp_path, check_updates=True, yes=[False, True],
        update_checker=lambda _v: info, installer_downloader=fake_download,
    )
    pump(app, root, lambda: len(calls["yesno"]) == 1)
    assert app.update_btn.winfo_manager()
    assert "v9.9.9" in app.update_btn.cget("text")
    assert calls["launch"] == []

    app.on_update_clicked()
    pump(app, root, lambda: bool(calls["exit"]))
    assert calls["launch"] == [str(fake_installer)]
    close_app(app)


def test_update_download_error_and_non_exe_url(root, calls, tmp_path):
    info_err = UpdateInfo("9.9.9", "https://example.com/setup.exe", "https://example.com/rel")

    def bad_download(_upd, _dest, progress=None):
        raise UpdateError("connection reset")

    app = make_app(
        root, calls, tmp_path, check_updates=True, yes=True,
        update_checker=lambda _v: info_err, installer_downloader=bad_download,
    )
    pump(app, root, lambda: calls["error"])
    assert calls["error"] == ["connection reset"]
    assert "Update failed" in app.status_var.get()
    close_app(app)

    # Non-exe download URL opens browser page
    calls["error"].clear()
    calls["yesno"].clear()
    info_web = UpdateInfo("9.9.9", "https://example.com/rel", "https://example.com/rel")
    app2 = make_app(root, calls, tmp_path, check_updates=True, yes=True, update_checker=lambda _v: info_web)
    pump(app2, root, lambda: calls["url"])
    assert calls["url"] == ["https://example.com/rel"]
    close_app(app2)


def test_open_logs(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path)
    app.on_open_logs()
    assert Path(calls["open"][0]).is_dir()
    app._poll()
    close_app(app)


def test_default_musescore_check(root, tmp_path):
    app = App(root, AppConfig(musescore_path=tmp_path / "none.exe", check_updates=False))
    assert isinstance(app.musescore_available(), bool)
    close_app(app)
