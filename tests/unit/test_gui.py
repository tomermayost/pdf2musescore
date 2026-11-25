import gc
import logging
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pytest
from PIL import Image

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

    def __init__(self, config, confirm, progress, is_cancelled=lambda: False, model_manager=None):
        self.confirm, self.progress, self.is_cancelled = confirm, progress, is_cancelled
        self.model_manager = model_manager

    def convert(self, pdf, out, pages_spec=None, options=None):
        FakePipeline.last_pages_spec = pages_spec
        FakePipeline.last_options = options
        self.progress(50, "half")
        logging.getLogger("pdf2musescore.test").info("working")
        logging.getLogger("pdf2musescore.test").debug("trace detail")
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


def close_app(app):
    if app.worker is not None and app.worker.is_alive():
        app.worker.join(timeout=2)
    if getattr(app, "update_thread", None) is not None and app.update_thread.is_alive():
        app.update_thread.join(timeout=2)
    if getattr(app, "model_thread", None) is not None and app.model_thread.is_alive():
        app.model_thread.join(timeout=2)
    app.destroy()
    for name in list(app.__dict__):
        if name.endswith("_var"):
            delattr(app, name)
    gc.collect()


@pytest.fixture
def calls():
    return {
        "error": [],
        "info": [],
        "yesno": [],
        "open": [],
        "open_file": [],
        "reveal": [],
        "launch": [],
        "url": [],
        "exit": [],
    }


def make_app(
    root,
    calls,
    tmp_path,
    open_path="C:/x/song.pdf",
    save_path="C:/x/song.mscz",
    yes=True,
    musescore=True,
    page_counter=lambda p: 4,
    check_updates=False,
    engine_version="0.7.1",
    update_checker=None,
    installer_downloader=None,
    theme=None,
    prefs_path=None,
):
    logging.getLogger("pdf2musescore").setLevel("DEBUG")
    kwargs = {}
    if installer_downloader is not None:
        kwargs["installer_downloader"] = installer_downloader
    return App(
        root,
        AppConfig(log_dir=tmp_path / "logs", work_dir=tmp_path / "work", check_updates=check_updates),
        pipeline_factory=FakePipeline,
        musescore_available=lambda: musescore,
        page_counter=page_counter,
        page_renderer=lambda p, pg, sz: Image.new("RGB", (100, 140), (255, 255, 255)),
        ask_open=lambda **k: (calls.__setitem__("open_kwargs", k) or open_path),
        ask_save=lambda **k: (calls.__setitem__("save_kwargs", k) or save_path),
        ask_yes_no=lambda t, m: (calls["yesno"].append((t, m)) or (yes.pop(0) if isinstance(yes, list) else yes)),
        show_error=lambda t, m: calls["error"].append(m),
        show_info=lambda t, m: calls["info"].append(m),
        open_folder=lambda p: calls["open"].append(p),
        open_file=lambda p: calls["open_file"].append(p),
        reveal_file=lambda p: calls["reveal"].append(p),
        engine_version=engine_version,
        update_checker=update_checker,
        launch_installer=lambda p: calls["launch"].append(p),
        open_url=lambda u: calls["url"].append(u) or True,
        on_exit=lambda: calls["exit"].append(True),
        theme=theme,
        prefs_path=prefs_path or (tmp_path / "preferences.json"),
        **kwargs,
    )


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
    probe_calls = []
    app = App(
        root,
        AppConfig(log_dir=tmp_path / "logs", work_dir=tmp_path / "work", check_updates=False),
        musescore_available=lambda: probe_calls.append(1) or True,
        engine_version="0.7.1",
        prefs_path=tmp_path / "preferences.json",
    )
    assert app.theme_name == "dark"
    assert len(probe_calls) == 1
    assert str(app.convert_btn["state"]) == "disabled"
    assert str(app.abort_btn["state"]) == "disabled"
    assert app.credit_label.cget("text") == "By Tomer Mayost"
    assert "MuseScore detected" in app.ms_badge_lbl.cget("text")
    assert "(homr 0.7.1)" in root.title()
    assert "By Tomer Mayost" not in root.title()
    assert not hasattr(app, "title_var")
    assert app.advanced_section.is_expanded is False
    assert app.log_section.is_expanded is True
    assert not app.result_banner.winfo_manager()
    app.on_convert()  # no pdf -> no-op
    app.on_open_result()  # no output yet -> no-op
    app.on_reveal_result()  # no output yet -> no-op
    close_app(app)


def test_options_presets_stepper_theme_and_validation(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path)
    app.on_load()

    # Hover tooltips exist and show/hide cleanly
    assert "render_dpi" in app.tooltips and "pages" in app.tooltips and "keep_line_breaks" in app.tooltips
    assert "title_override" not in app.tooltips
    dpi_tip = app.tooltips["render_dpi"]
    dpi_tip.show()
    dpi_tip.show()
    assert dpi_tip.tip_window is not None
    dpi_tip.hide()
    assert dpi_tip.tip_window is None

    # One-click scan presets update engine vars cleanly without duplicating preset name in badge
    app.apply_preset("faded")
    assert app.preset_badge_var.get() == ""
    assert str(app.preset_buttons["faded"].cget("style")) == "PresetActive.TButton"
    assert app.dpi_var.get() == "350"
    assert app.preprocess_var.get() == "faded"
    assert app.deskew_var.get() is True
    assert app.advanced_section.badge_lbl.cget("text") == "Customized"

    app.apply_preset("photo")
    assert app.preset_badge_var.get() == ""
    assert str(app.preset_buttons["photo"].cget("style")) == "PresetActive.TButton"
    assert app.preprocess_var.get() == "binarize"

    # Manual tweak switches preset badge to Custom
    app.dpi_var.set("420")
    assert app.preset_badge_var.get() == "Custom"

    # Transpose stepper + live label
    app.step_transpose(1)
    assert app.transpose_var.get() == "+1"
    assert "+1" in app.transpose_hint_var.get()
    app.transpose_var.set("bad")
    assert "Enter integer" in app.transpose_hint_var.get()
    app.step_transpose(-2)
    assert app.transpose_var.get() == "-2"

    # Reset restores Clean PDF preset & removes Customized badge
    app.on_reset_options()
    assert app.keep_breaks_var.get() is False
    assert app.preset_badge_var.get() == ""
    assert str(app.preset_buttons["clean"].cget("style")) == "PresetActive.TButton"
    assert not app.advanced_section.badge_lbl.winfo_manager()
    assert app.collect_options() == ConversionOptions()

    # Custom full options collection
    app.dpi_var.set("400")
    app.preprocess_var.set("faded")
    app.deskew_var.set(True)
    app.step_size_var.set("240")
    app.min_notehead_var.set("3")
    app.keep_breaks_var.set(True)
    app.transpose_var.set("+2")
    app.clef_var.set("bass")
    app.instrument_var.set("Trombone")
    assert app.collect_options() == ConversionOptions(
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

    # Dark (default) <-> Light theme switch
    assert app.theme_name == "dark"
    assert "Light" in app.theme_btn.cget("text")
    app.toggle_theme()
    assert app.theme_name == "light"
    assert "Dark" in app.theme_btn.cget("text")
    app.toggle_theme()
    assert app.theme_name == "dark"

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


def test_load_preview_pages_and_convert_with_inline_banner(root, calls, tmp_path):
    FakePipeline.behaviour = "ok"
    app = make_app(root, calls, tmp_path)
    app.on_load()
    assert app.file_var.get() == "song.pdf (4 pages)"
    assert str(app.convert_btn["state"]) == "normal"
    assert "Converting all 4 pages" in app.pages_feedback_var.get()

    # Clicking page picker adds page 1, navigating & picking page 2 builds range "1-2"
    app._on_toggle_preview_page(1)
    assert app.pages_var.get() == "1"
    assert "Converting 1 of 4" in app.pages_feedback_var.get()
    app.preview_pane.go_next()
    assert app.preview_pane.current_page == 2
    app._on_toggle_preview_page(2)
    assert app.pages_var.get() == "1-2"

    # Typing an invalid page range shows inline validation error cleanly
    app.pages_var.set("99")
    assert "out of range" in app.pages_feedback_var.get()

    app.pages_var.set("2-3")
    app.transpose_var.set("2")
    app.on_convert()
    assert calls["save_kwargs"]["defaultextension"] == ".mscz"

    # Conversion completes with inline success banner (no modal show_info popup)
    pump(app, root, lambda: bool(app.result_banner.winfo_manager()))
    assert calls["info"] == []
    assert app._last_output == Path("C:/x/song.mscz")
    assert "song.mscz" in app.result_msg_var.get()
    assert app.progress["value"] == 100
    assert app.pct_var.get() == "100%"
    assert FakePipeline.last_pages_spec == "2-3"
    assert FakePipeline.last_options.transpose_semitones == 2
    assert "working" in app.log_text.get("1.0", "end")

    # Quick action buttons trigger open_file / reveal_file
    app.on_open_result()
    app.on_reveal_result()
    assert calls["open_file"] == [str(Path("C:/x/song.mscz"))]
    assert calls["reveal"] == [Path("C:/x/song.mscz")]
    close_app(app)


def test_shortcuts_and_log_badges(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path, open_path="", save_path="")
    fake_evt = type("E", (), {"widget": app})()
    app._on_shortcut_load(fake_evt)  # open_path="" cancels load cleanly

    # Simulate loaded PDF -> Right/Left arrow shortcuts flip preview pages when focus is not an Entry
    app.pdf = tmp_path / "score.pdf"
    app.total_pages = 3
    app.preview_pane.set_pdf(app.pdf, 3)
    assert app._on_shortcut_page(fake_evt, +1) == "break"
    assert app.preview_pane.current_page == 2
    assert app._on_shortcut_page(fake_evt, -1) == "break"
    assert app.preview_pane.current_page == 1

    # Arrow keys are ignored when inside an Entry/Combobox so text cursor movement works
    entry_evt = type("E", (), {"widget": app.pages_entry})()
    assert app._on_shortcut_page(entry_evt, +1) is None

    # Warning/error logs update log drawer badge and color tags
    app.q.put(("log", (logging.WARNING, "12:00:00 WARNING: mild issue")))
    app.process_queue()
    assert "1 warning" in app.log_section.badge_lbl.cget("text")
    app.q.put(("log", (logging.ERROR, "12:00:01 ERROR: serious issue")))
    app.process_queue()
    assert "1 error" in app.log_section.badge_lbl.cget("text")
    app.q.put(("log", "plain string message"))
    app.process_queue()
    assert "plain string message" in app.log_text.get("1.0", "end")

    # Busy guard prevents shortcuts/actions from re-entering
    app._busy = True
    assert app._on_shortcut_load(fake_evt) is None
    assert app._on_shortcut_convert(fake_evt) is None
    assert app._on_shortcut_abort(fake_evt) == "break"
    app._busy = False
    assert app._on_shortcut_abort(fake_evt) is None
    assert app._on_shortcut_convert(fake_evt) == "break"
    close_app(app)


def test_load_page_counter_error_fallback(root, calls, tmp_path, caplog):
    def bad_counter(p):
        raise ValueError("bad pdf")

    app = make_app(root, calls, tmp_path, page_counter=bad_counter)
    with caplog.at_level("WARNING", logger="pdf2musescore.gui"):
        app.on_load()
    assert app.file_var.get() == "song.pdf"
    assert app.preview_pane.total_pages == 1
    assert "Could not count pages" in caplog.text
    assert "ValueError: bad pdf" in caplog.text
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
    assert "MusicXML" in app.convert_btn.cget("text")
    assert "MusicXML export mode" in app.ms_badge_lbl.cget("text")
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


@pytest.mark.parametrize(
    "behaviour,yes,expected_parts",
    [
        ("error", True, ["friendly message", "Full technical log:", "pdf2musescore.log"]),
        ("crash", True, ["Unexpected error.", "Details: RuntimeError: boom", "pdf2musescore.log"]),
        ("confirm", False, ["declined", "Full technical log:", "pdf2musescore.log"]),
    ],
)
def test_errors_auto_expand_log(root, calls, tmp_path, behaviour, yes, expected_parts):
    FakePipeline.behaviour = behaviour
    app = make_app(root, calls, tmp_path, yes=yes)
    app.on_load()
    app.log_section.set_expanded(False)
    app.on_convert()
    pump(app, root, lambda: calls["error"])
    assert len(calls["error"]) == 1
    popup = calls["error"][0]
    for part in expected_parts:
        assert part in popup
    assert "failed" in app.status_var.get()
    assert app.log_section.is_expanded is True
    close_app(app)


def test_gui_error_truncation_tk_exception_hook_and_drop_error(root, calls, tmp_path, caplog):
    orig_hook = root.report_callback_exception
    app = make_app(root, calls, tmp_path)
    try:
        # Long technical details are truncated cleanly to stay readable in messagebox
        formatted = app._format_error("Scan failed", "X" * 1000)
        assert ("Details: " + "X" * 797 + "...") in formatted
        assert "X" * 800 not in formatted
        assert "pdf2musescore.log" in formatted

        # Unhandled UI callback exceptions are routed to logger and expand the log drawer
        app.log_section.set_expanded(False)
        with caplog.at_level("ERROR", logger="pdf2musescore.gui"):
            try:
                raise RuntimeError("tk callback boom")
            except RuntimeError:
                import sys

                root.report_callback_exception(*sys.exc_info())
        assert "Unhandled error in UI callback" in caplog.text
        assert "tk callback boom" in caplog.text
        assert app.log_section.is_expanded is True

        # Unexpected exception inside on_drop_files is caught, logged, and reported
        app.load_source = lambda _p: (_ for _ in ()).throw(OSError("unreadable file"))
        bad_pdf = tmp_path / "broken.pdf"
        bad_pdf.write_bytes(b"%PDF")
        with caplog.at_level("ERROR", logger="pdf2musescore.gui"):
            app.on_drop_files([bad_pdf])
        assert "Failed to open dropped file(s)" in caplog.text
        assert calls["error"] and "OSError: unreadable file" in calls["error"][-1]
    finally:
        close_app(app)
    assert root.report_callback_exception == orig_hook


def test_gui_model_prefetch_progress_ready_and_error(root, calls, tmp_path):
    from pdf2musescore.model_store import ModelDownloadError, ModelStatus

    class _FakePrefetchManager:
        def __init__(self, should_fail=False):
            self.should_fail = should_fail
            self.calls = 0

        def status(self):
            return ModelStatus(ready=False, missing=("homr",), homr_dir=tmp_path, ocr_dir=tmp_path)

        def ensure(self, progress=None, is_cancelled=lambda: False):
            self.calls += 1
            if progress:
                progress(90_000_000, 180_000_000)
            if self.should_fail:
                raise ModelDownloadError("offline")
            return ModelStatus(ready=True, missing=(), homr_dir=tmp_path, ocr_dir=tmp_path)

    # Ready flow
    mgr_ok = _FakePrefetchManager(should_fail=False)
    app = App(
        root,
        AppConfig(log_dir=tmp_path / "logs", work_dir=tmp_path / "work", check_updates=False),
        musescore_available=lambda: True,
        prefs_path=tmp_path / "prefs_ok.json",
        model_manager=mgr_ok,
        prefetch_models=True,
    )
    try:
        pump(app, root, lambda: mgr_ok.calls == 1 and not app.model_thread.is_alive())
        app.process_queue()
        assert "AI models ready" in app.status_var.get()

        # When PDF is loaded, models_ready restores normal Ready text; when busy, updates are ignored
        app.pdf = tmp_path / "x.pdf"
        app.q.put(("models_progress", (50_000_000, 180_000_000)))
        app.process_queue()
        assert "Downloading AI models" in app.status_var.get()
        app.q.put(("models_ready", None))
        app.process_queue()
        assert "Pick a preset" in app.status_var.get()

        app._busy = True
        app.status_var.set("Converting...")
        app.q.put(("models_progress", (10_000_000, 180_000_000)))
        app.q.put(("models_ready", None))
        app.q.put(("models_error", "ignored"))
        app.process_queue()
        assert app.status_var.get() == "Converting..."
        app._busy = False
    finally:
        close_app(app)

    # Error flow shows non-blocking hint in status bar without error modal
    mgr_err = _FakePrefetchManager(should_fail=True)
    app2 = App(
        root,
        AppConfig(log_dir=tmp_path / "logs", work_dir=tmp_path / "work", check_updates=False),
        musescore_available=lambda: True,
        prefs_path=tmp_path / "prefs_err.json",
        model_manager=mgr_err,
        prefetch_models=True,
        show_error=lambda t, m: calls["error"].append(m),
    )
    try:
        pump(app2, root, lambda: mgr_err.calls == 1 and not app2.model_thread.is_alive())
        app2.process_queue()
        assert "could not be downloaded" in app2.status_var.get()
        assert calls["error"] == []
    finally:
        close_app(app2)


def test_confirm_yes(root, calls, tmp_path):
    FakePipeline.behaviour = "confirm"
    app = make_app(root, calls, tmp_path, yes=True)
    app.on_load()
    app.on_convert()
    pump(app, root, lambda: bool(app.result_banner.winfo_manager()))
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

    app = make_app(
        root,
        calls,
        tmp_path,
        check_updates=True,
        yes=[False, True],
        update_checker=lambda _v: info,
        installer_downloader=fake_download,
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
        root,
        calls,
        tmp_path,
        check_updates=True,
        yes=True,
        update_checker=lambda _v: info_err,
        installer_downloader=bad_download,
    )
    pump(app, root, lambda: calls["error"])
    assert calls["error"] == ["connection reset"]
    assert "Update failed" in app.status_var.get()
    assert app.log_section.is_expanded is True
    close_app(app)

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
    app = App(
        root,
        AppConfig(musescore_path=tmp_path / "none.exe", check_updates=False),
        prefs_path=tmp_path / "preferences.json",
    )
    assert isinstance(app.musescore_available(), bool)
    close_app(app)


def test_preferences_persist_across_restarts(root, calls, tmp_path):
    prefs_file = tmp_path / "prefs" / "preferences.json"
    pdf_dir = tmp_path / "scores"
    pdf_dir.mkdir()
    sample_pdf = pdf_dir / "sonata.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4")

    # Session 1: starts in dark mode, log open, advanced closed -> change everything & close
    app1 = make_app(root, calls, tmp_path, open_path=str(sample_pdf), prefs_path=prefs_file)
    assert app1.theme_name == "dark"
    assert app1.advanced_section.is_expanded is False
    assert app1.log_section.is_expanded is True

    app1.toggle_theme()
    assert app1.theme_name == "light"
    app1.advanced_section.toggle()
    app1.log_section.toggle()
    app1.apply_preset("faded")
    app1.step_transpose(2)
    app1.clef_var.set("bass")
    app1.instrument_var.set("Cello")
    app1.keep_breaks_var.set(True)
    app1.on_load()

    # Mid-edit invalid values do not crash save_preferences and keep prior valid DPI (350)
    app1.dpi_var.set("abc")
    app1.preprocess_var.set("bogus")
    app1.clef_var.set("bogus")
    app1.save_preferences()
    # Restore valid strings and test close save path
    app1.dpi_var.set("350")
    app1.preprocess_var.set("faded")
    app1.clef_var.set("bass")
    close_app(app1)

    # Session 2: reopen pointing at same prefs file -> light mode, drawers, options & last_dir restored
    app2 = make_app(root, calls, tmp_path, open_path="", prefs_path=prefs_file)
    assert app2.theme_name == "light"
    assert "Dark" in app2.theme_btn.cget("text")
    assert app2.advanced_section.is_expanded is True
    assert app2.log_section.is_expanded is False
    assert app2._last_dir == str(pdf_dir)
    assert app2.collect_options() == ConversionOptions(
        render_dpi=350,
        preprocess_mode="faded",
        deskew=True,
        segnet_step_size=240,
        min_notehead_size=4,
        transpose_semitones=2,
        clef_override="bass",
        instrument_name="Cello",
        keep_line_breaks=True,
    )
    app2.on_load()
    assert calls["open_kwargs"]["initialdir"] == str(pdf_dir)

    # Reset options reverts conversion params while preserving chosen theme ("light")
    app2.on_reset_options()
    close_app(app2)

    app3 = make_app(root, calls, tmp_path, prefs_path=prefs_file)
    assert app3.theme_name == "light"
    assert app3.collect_options() == ConversionOptions()
    close_app(app3)


def test_dual_install_banner_actions_and_dismiss_persistence(root, calls, tmp_path):
    from pdf2musescore.install_detect import (
        MSIX_STORE_URI,
        WINDOWS_INSTALLED_APPS_URI,
        DualInstallNotice,
        InstalledEdition,
    )

    prefs_file = tmp_path / "dual_prefs.json"
    notice_with_store = DualInstallNotice(
        current_kind="standalone",
        current_version="0.4.6",
        other=InstalledEdition(kind="store", version="0.4.3"),
        comparison="current_newer",
        message="Running Standalone .exe v0.4.6, while older Microsoft Store v0.4.3 is also installed.",
        show_store_update=True,
        dismiss_key="standalone:0.4.6|store:0.4.3",
    )

    app1 = App(
        root,
        AppConfig(log_dir=tmp_path / "logs1", work_dir=tmp_path / "work1", check_updates=False),
        musescore_available=lambda: True,
        open_url=lambda u: calls["url"].append(u) or True,
        prefs_path=prefs_file,
        store_mode=False,
        prefetch_models=False,
        dual_install_checker=lambda: notice_with_store,
    )
    try:
        assert app1.edition_badge_lbl is None
        assert app1.dual_install_banner.winfo_manager()
        assert app1.dual_install_store_btn.winfo_manager()
        assert "0.4.3" in app1.dual_install_lbl.cget("text")

        app1.on_open_store_page()
        app1.on_open_installed_apps()
        assert calls["url"] == [MSIX_STORE_URI, WINDOWS_INSTALLED_APPS_URI]

        app1.toggle_theme()
        assert app1.theme_name == "light"

        app1.on_dismiss_dual_install()
        assert not app1.dual_install_banner.winfo_manager()
        assert app1._dismissed_dual_install == "standalone:0.4.6|store:0.4.3"
    finally:
        close_app(app1)

    # Same dismiss_key stays suppressed across restarts
    app2 = App(
        root,
        AppConfig(log_dir=tmp_path / "logs2", work_dir=tmp_path / "work2", check_updates=False),
        musescore_available=lambda: True,
        prefs_path=prefs_file,
        store_mode=False,
        prefetch_models=False,
        dual_install_checker=lambda: notice_with_store,
    )
    try:
        assert not app2.dual_install_banner.winfo_manager()
    finally:
        close_app(app2)

    # Changed versions or show_store_update=False shows banner again without Store button
    notice_same = DualInstallNotice(
        current_kind="standalone",
        current_version="0.4.6",
        other=InstalledEdition(kind="store", version="0.4.6"),
        comparison="same",
        message="Both Standalone .exe (v0.4.6) and Microsoft Store (v0.4.6) are installed.",
        show_store_update=False,
        dismiss_key="standalone:0.4.6|store:0.4.6",
    )
    app3 = App(
        root,
        AppConfig(log_dir=tmp_path / "logs3", work_dir=tmp_path / "work3", check_updates=False),
        musescore_available=lambda: True,
        prefs_path=prefs_file,
        store_mode=False,
        prefetch_models=False,
        dual_install_checker=lambda: notice_same,
    )
    try:
        assert app3.dual_install_banner.winfo_manager()
        assert not app3.dual_install_store_btn.winfo_manager()
    finally:
        close_app(app3)

    # Failing dual_install_checker never crashes startup
    app4 = App(
        root,
        AppConfig(log_dir=tmp_path / "logs4", work_dir=tmp_path / "work4", check_updates=False),
        musescore_available=lambda: True,
        prefs_path=tmp_path / "p4.json",
        prefetch_models=False,
        dual_install_checker=lambda: (_ for _ in ()).throw(RuntimeError("reg fail")),
    )
    try:
        assert not app4.dual_install_banner.winfo_manager()
    finally:
        close_app(app4)
