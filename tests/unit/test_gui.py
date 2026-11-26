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
    staged_xml_path = None

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
            raise ConversionError("friendly message", detail="OmrError: bad scan", stage="Reading music")
        elif self.behaviour == "export_error":
            raise ConversionError(
                "MuseScore could not save the file.",
                detail="ExportError: MuseScore export failed (code 1320): err: [2009] Incomplete measure",
                stage="Saving MuseScore file...",
                staged_musicxml=FakePipeline.staged_xml_path,
                musescore_error="err: [2009] Incomplete measure",
            )
        elif self.behaviour == "crash":
            raise RuntimeError("boom")
        return out


def close_app(app):
    if app.worker is not None and app.worker.is_alive():
        app.worker.join(timeout=2)
    if getattr(app, "_report_thread", None) is not None and app._report_thread.is_alive():
        app._report_thread.join(timeout=2)
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
        "flash": [],
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
    bug_report_url="",
    report_submitter=None,
):
    logging.getLogger("pdf2musescore").setLevel("DEBUG")
    kwargs = {}
    if installer_downloader is not None:
        kwargs["installer_downloader"] = installer_downloader
    if report_submitter is not None:
        kwargs["report_submitter"] = report_submitter
    return App(
        root,
        AppConfig(
            log_dir=tmp_path / "logs",
            work_dir=tmp_path / "work",
            check_updates=check_updates,
            bug_report_url=bug_report_url,
        ),
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
        flash_taskbar=lambda w: calls["flash"].append(w),
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
    assert app.log_section.is_expanded is False
    assert app.part_options_section.is_expanded is False
    assert not app._pages_row.winfo_manager()
    assert not app.abort_btn.winfo_manager()
    assert not app.progress.winfo_manager()
    assert str(app.load_btn.cget("style")) == "CompactPrimary.TButton"
    assert app.load_btn.cget("text") == "Load File..."
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
        ("error", True, ["friendly message", "OmrError: bad scan", "Fingerprint:"]),
        ("crash", True, ["Unexpected error.", "RuntimeError: boom", "Fingerprint:"]),
        ("confirm", False, ["declined", "Fingerprint:"]),
    ],
)
def test_errors_auto_expand_log(root, calls, tmp_path, behaviour, yes, expected_parts):
    FakePipeline.behaviour = behaviour
    app = make_app(root, calls, tmp_path, yes=yes)
    app.on_load()
    app.log_section.set_expanded(False)
    app.on_convert()
    pump(app, root, lambda: app._error_popup is not None)
    popup = app._error_popup
    assert popup is not None and popup.winfo_exists()
    preview = popup.preview_text.get("1.0", "end")
    for part in expected_parts:
        assert part in preview
    assert popup.save_xml_btn is None
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

    # Session 1: starts in dark mode, log closed, advanced closed -> change everything & close
    app1 = make_app(root, calls, tmp_path, open_path=str(sample_pdf), prefs_path=prefs_file)
    assert app1.theme_name == "dark"
    assert app1.advanced_section.is_expanded is False
    assert app1.log_section.is_expanded is False

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
    assert app2.log_section.is_expanded is True
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


def test_gui_full_score_mode_toggle_presets_busy_validation_and_persistence(root, calls, tmp_path):
    FakePipeline.behaviour = "ok"
    prefs_file = tmp_path / "fs_prefs.json"
    app = make_app(root, calls, tmp_path, prefs_path=prefs_file)
    try:
        # Default: Single Part mode
        assert app.full_score_var.get() is False
        assert app.dpi_var.get() == "300"
        assert str(app.instrument_combo["state"]) == "normal"
        assert str(app.full_score_width_entry["state"]) == "disabled"
        assert app.part_options_section.winfo_manager()
        assert not app._full_score_width_frame.winfo_manager()
        assert not app.score_mode_hint_lbl.winfo_manager()

        # Switch to Full Score Mode -> bumps DPI to 500, hides single-part section, shows width
        app.set_score_mode(True)
        assert app.full_score_var.get() is True
        assert app.dpi_var.get() == "500"
        assert str(app.instrument_combo["state"]) == "disabled"
        assert str(app.clef_combo["state"]) == "disabled"
        assert str(app.transpose_entry["state"]) == "disabled"
        assert str(app.full_score_width_entry["state"]) == "normal"
        assert not app.part_options_section.winfo_manager()
        assert app._full_score_width_frame.winfo_manager()
        assert app.score_mode_hint_lbl.winfo_manager()
        assert app.single_part_btn.cget("style") == "Preset.TButton"
        assert app.full_score_btn.cget("style") == "PresetActive.TButton"

        # Step transpose is ignored in Full Score Mode
        app.step_transpose(2)
        assert app.transpose_var.get() == "0"

        # Applying Faded Scan preset preserves 500 DPI in Full Score Mode
        app.apply_preset("faded")
        assert app.dpi_var.get() == "500"
        assert app.preprocess_var.get() == "faded"
        assert app.deskew_var.get() is True
        assert app.preset_badge_var.get() == ""

        # Busy -> idle preserves Full Score disabled state on single-part widgets
        app._set_busy(True)
        assert str(app.full_score_btn["state"]) == "disabled"
        app._set_busy(False)
        assert str(app.full_score_btn["state"]) == "normal"
        assert str(app.instrument_combo["state"]) == "disabled"
        assert str(app.full_score_width_entry["state"]) == "normal"

        # Customize width and verify collect_options + convert
        app.full_score_width_var.set("bad")
        app.on_load()
        app.on_convert()
        assert calls["error"] and "Full Score Width" in calls["error"][-1]
        calls["error"].clear()

        app.full_score_width_var.set("4096")
        opts = app.collect_options()
        assert opts.full_score is True and opts.full_score_target_width == 4096 and opts.render_dpi == 500
        app.on_convert()
        pump(app, root, lambda: bool(app.result_banner.winfo_manager()))
        assert FakePipeline.last_options.full_score is True
        assert FakePipeline.last_options.full_score_target_width == 4096

        # Switch back to Single Part restores 300 DPI and re-enables single-part widgets
        app.set_score_mode(False)
        assert app.dpi_var.get() == "300"
        assert str(app.instrument_combo["state"]) == "normal"
        assert str(app.clef_combo["state"]) == "readonly"
        assert str(app.full_score_width_entry["state"]) == "disabled"

        # Switch back ON and verify Reset restores everything to Single Part defaults
        app.set_score_mode(True)
        app.on_reset_options()
        assert app.full_score_var.get() is False
        assert app.dpi_var.get() == "300"
        assert app.full_score_width_var.get() == "3840"

        # Set Full Score again so persistence check sees it on next App startup
        app.set_score_mode(True)
        app.full_score_width_var.set("4096")
        app.save_preferences()
    finally:
        close_app(app)

    app2 = make_app(root, calls, tmp_path, prefs_path=prefs_file)
    try:
        assert app2.full_score_var.get() is True
        assert app2.full_score_width_var.get() == "4096"
        assert str(app2.instrument_combo["state"]) == "disabled"
        assert str(app2.full_score_width_entry["state"]) == "normal"
    finally:
        close_app(app2)


def test_gui_activity_log_sticky_bottom_autoscroll(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path)
    try:
        app.log_section.set_expanded(True)
        root.update_idletasks()
        root.update()

        # Populate enough lines to exceed the 4-line text widget viewport -> stays at bottom
        for i in range(30):
            app._append_log((logging.INFO, f"line {i}"))
        root.update_idletasks()
        assert app._is_log_at_bottom() is True
        assert app.log_text.yview()[1] >= 0.999

        # User scrolls up to top -> new log lines must NOT jump back to bottom
        app.log_text.yview_moveto(0.0)
        root.update_idletasks()
        assert app._is_log_at_bottom() is False
        app._append_log((logging.INFO, "new line while reading history"))
        root.update_idletasks()
        assert app.log_text.yview()[0] == 0.0
        assert app._is_log_at_bottom() is False

        # User scrolls back to bottom -> auto-scroll resumes for subsequent lines
        app.log_text.yview_moveto(1.0)
        root.update_idletasks()
        assert app._is_log_at_bottom() is True
        app._append_log((logging.INFO, "tail resumed"))
        root.update_idletasks()
        assert app.log_text.yview()[1] >= 0.999

        # Prior warnings/errors badge is cleared when starting a new conversion
        app._append_log((logging.WARNING, "old warning"))
        assert app.log_section.badge_lbl.winfo_manager()
        assert "1 warning" in app.log_section.badge_lbl.cget("text")
        FakePipeline.behaviour = "ok"
        app.start_conversion(tmp_path / "in.pdf", tmp_path / "out.mscz")
        assert app._log_warning_count == 0 and app._log_error_count == 0
        assert not app.log_section.badge_lbl.winfo_manager()
        pump(app, root, lambda: bool(app.result_banner.winfo_manager()))
    finally:
        close_app(app)


def test_gui_completion_popup_and_taskbar_flash(root, calls, tmp_path):
    app = make_app(root, calls, tmp_path)
    try:
        FakePipeline.behaviour = "ok"
        app.on_load()
        app.on_convert()
        pump(app, root, lambda: bool(app.result_banner.winfo_manager()))

        assert len(calls["flash"]) == 1
        assert calls["flash"][0] is root
        popup = app._completion_popup
        assert popup is not None and popup.winfo_exists()
        assert "song.mscz" in " ".join(
            str(w.cget("text"))
            for f in popup.winfo_children()
            for sub in f.winfo_children()
            for w in ([sub] + list(sub.winfo_children()))
            if hasattr(w, "cget") and "text" in w.keys()
        )

        # Theme toggle updates open completion popup cleanly
        app.toggle_theme()
        assert popup.cget("bg") == app.palette.card

        # Open button triggers open_file and closes popup
        popup.open_btn.invoke()
        assert calls["open_file"] == [str(Path("C:/x/song.mscz"))]
        assert app._completion_popup is None

        # Run again -> Reveal button triggers reveal_file and closes popup
        app.on_convert()
        pump(app, root, lambda: app._completion_popup is not None)
        app._completion_popup.reveal_btn.invoke()
        assert calls["reveal"] == [Path("C:/x/song.mscz")]
        assert app._completion_popup is None

        # Run again -> Close button closes popup; background error flashes taskbar
        app.on_convert()
        pump(app, root, lambda: app._completion_popup is not None)
        app._completion_popup.close_btn.invoke()
        assert app._completion_popup is None

        # Starting a new conversion closes an existing open popup automatically
        app.on_convert()
        pump(app, root, lambda: app._completion_popup is not None)
        FakePipeline.behaviour = "error"
        app.on_convert()
        assert app._completion_popup is None
        pump(app, root, lambda: app._error_popup is not None)
        assert len(calls["flash"]) == 5
    finally:
        close_app(app)


def test_gui_error_dialog_actions_save_xml_zip_copy_and_submit(root, calls, tmp_path):
    import zipfile
    from pdf2musescore.bug_report import BugReportError, BugReportResult

    staged_xml = tmp_path / "work" / "score.musicxml"
    staged_xml.parent.mkdir(parents=True, exist_ok=True)
    staged_xml.write_text("<score-partwise><part id='P1'/></score-partwise>", encoding="utf-8")

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "pdf2musescore.log").write_text(
        r"2026-11-25 INFO Converting C:\Users\Alice\Private.pdf" + "\n", encoding="utf-8"
    )

    saved_xml_out = tmp_path / "rescued" / "song.musicxml"
    saved_zip_out = tmp_path / "rescued" / "diag.zip"
    save_answers = [str(tmp_path / "x" / "song.mscz"), str(saved_xml_out), str(saved_zip_out)]

    def fake_ask_save(**k):
        calls["save_kwargs"] = k
        return save_answers.pop(0)

    # 1. Without Worker URL -> Send Bug Report opens pre-filled GitHub Issue URL
    FakePipeline.behaviour = "export_error"
    FakePipeline.staged_xml_path = staged_xml
    app = make_app(
        root,
        calls,
        tmp_path,
        open_path=r"C:\Users\Alice\Private.pdf",
        bug_report_url="",
    )
    app.ask_save = fake_ask_save
    try:
        app.on_load()
        app.on_convert()
        pump(app, root, lambda: app._error_popup is not None)
        popup = app._error_popup
        assert popup is not None and popup.save_xml_btn is not None

        # Preview is sanitized and shows MuseScore 4 error
        preview = popup.preview_text.get("1.0", "end")
        assert "[2009] Incomplete measure" in preview
        assert "Alice" not in preview
        assert "Private.pdf" not in preview

        # User note dynamically updates sanitized preview
        popup.note_text.insert("1.0", "Failed on bar 46")
        popup.refresh_preview()
        assert "Failed on bar 46" in popup.preview_text.get("1.0", "end")

        # Theme toggle updates open error popup cleanly
        app.toggle_theme()
        assert popup.cget("bg") == app.palette.card

        # Copy Report copies sanitized report to clipboard
        popup.copy_btn.invoke()
        assert "Copied" in popup.status_var.get()
        assert "Failed on bar 46" in root.clipboard_get()

        # Save MusicXML anyway copies staged score.musicxml
        popup.save_xml_btn.invoke()
        assert saved_xml_out.is_file()
        assert "<score-partwise>" in saved_xml_out.read_text(encoding="utf-8")

        # Save Diagnostic ZIP (with optional score.musicxml enabled)
        popup.include_xml_var.set(True)
        popup.save_zip_btn.invoke()
        assert saved_zip_out.is_file()
        with zipfile.ZipFile(saved_zip_out) as zf:
            assert {"bug_report.md", "pdf2musescore.log", "score.musicxml"}.issubset(set(zf.namelist()))
            assert "Alice" not in zf.read("pdf2musescore.log").decode("utf-8")

        # Send Bug Report without Worker URL -> opens pre-filled GitHub Issue URL
        popup.send_btn.invoke()
        assert calls["url"] and "github.com/tomermayost/pdf2musescore/issues/new?" in calls["url"][-1]
        assert "Opened pre-filled GitHub Issue" in popup.status_var.get()
        popup.close_btn.invoke()
        assert app._error_popup is None
    finally:
        FakePipeline.staged_xml_path = None
        close_app(app)

    # 2. With Worker URL -> sends in background, handles error fallback, then succeeds and shows clickable link
    calls["url"].clear()
    submit_modes = ["fail", "ok"]
    submitted_payloads = []

    def fake_submit(report, url):
        submitted_payloads.append((report, url))
        if submit_modes.pop(0) == "fail":
            raise BugReportError("HTTP 503: offline")
        return BugReportResult(
            issue_url="https://github.com/tomermayost/pdf2musescore/issues/77",
            issue_number=77,
            duplicate=True,
        )

    FakePipeline.behaviour = "error"
    app2 = make_app(
        root,
        calls,
        tmp_path,
        bug_report_url="https://bugs.example.workers.dev/report",
        report_submitter=fake_submit,
    )
    try:
        app2.on_load()
        app2.on_convert()
        pump(app2, root, lambda: app2._error_popup is not None)
        popup2 = app2._error_popup
        assert popup2 is not None

        # First attempt fails -> shows fallback GitHub button and clicking it opens GitHub URL
        popup2.send_btn.invoke()
        pump(app2, root, lambda: "Could not send automatically" in popup2.status_var.get())
        assert popup2.fallback_gh_btn.winfo_manager()
        popup2.fallback_gh_btn.invoke()
        assert calls["url"] and "github.com/tomermayost/pdf2musescore/issues/new?" in calls["url"][-1]

        # Retry succeeds -> displays clickable Issue link and disables Send button
        popup2.send_btn.invoke()
        pump(app2, root, lambda: "Issue #77" in popup2.status_var.get())
        assert str(popup2.send_btn["state"]) == "disabled"
        assert "issues/77" in popup2.issue_link_btn.cget("text")
        app2.toggle_theme()
        popup2.issue_link_btn.invoke()
        assert calls["url"][-1] == "https://github.com/tomermayost/pdf2musescore/issues/77"
    finally:
        close_app(app2)


def test_gui_previous_crash_banner_and_persistence(root, calls, tmp_path):
    from pdf2musescore.bug_report import BugReportResult

    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    prefs_file = tmp_path / "crash_prefs.json"
    crash_file = log_dir / "crash.log"

    # Clean header-only crash.log -> no banner
    crash_file.write_text("\n=== 2026-11-25T20:00:00 pdf2musescore 0.4.7 pid=12345 ===\n", encoding="utf-8")
    app_clean = make_app(root, calls, tmp_path, prefs_path=prefs_file)
    try:
        assert not app_clean.crash_banner.winfo_manager()
    finally:
        close_app(app_clean)

    # Real prior crash -> shows banner, Report Previous Crash opens dialog, sending dismisses & persists
    crash_file.write_text(
        "\n=== 2026-11-25T20:05:00 pdf2musescore 0.4.7 pid=12345 ===\n"
        "Windows fatal exception: access violation\n",
        encoding="utf-8",
    )
    app_crash = make_app(
        root,
        calls,
        tmp_path,
        prefs_path=prefs_file,
        bug_report_url="https://bugs.example.workers.dev/report",
        report_submitter=lambda _r, _u: BugReportResult(
            issue_url="https://github.com/tomermayost/pdf2musescore/issues/88",
            issue_number=88,
            duplicate=False,
        ),
    )
    try:
        assert app_crash.crash_banner.winfo_manager()
        assert "2026-11-25 20:05:00" in app_crash.crash_banner_lbl.cget("text")
        app_crash.crash_report_btn.invoke()
        assert app_crash._error_popup is not None
        assert "access violation" in app_crash._error_popup.preview_text.get("1.0", "end")

        app_crash._error_popup.send_btn.invoke()
        pump(app_crash, root, lambda: not app_crash.crash_banner.winfo_manager())
        assert app_crash._dismissed_crash
    finally:
        close_app(app_crash)

    # Restart with same crash.log -> banner remains suppressed
    app_after = make_app(root, calls, tmp_path, prefs_path=prefs_file)
    try:
        assert not app_after.crash_banner.winfo_manager()
    finally:
        close_app(app_after)

    # New crash -> banner appears again and dismiss button persists dismissal
    crash_file.write_text(
        "\n=== 2026-11-25T21:00:00 pdf2musescore 0.4.7 pid=54321 ===\n"
        "Fatal Python error: Segmentation fault\n",
        encoding="utf-8",
    )
    app_new = make_app(root, calls, tmp_path, prefs_path=prefs_file)
    try:
        assert app_new.crash_banner.winfo_manager()
        app_new.crash_dismiss_btn.invoke()
        assert not app_new.crash_banner.winfo_manager()
    finally:
        close_app(app_new)

    app_new_after = make_app(root, calls, tmp_path, prefs_path=prefs_file)
    try:
        assert not app_new_after.crash_banner.winfo_manager()
    finally:
        close_app(app_new_after)


def test_gui_progressive_disclosure_cta_badges_and_resize(root, calls, tmp_path, monkeypatch):
    FakePipeline.behaviour = "ok"
    app = make_app(root, calls, tmp_path, page_counter=lambda p: 1 if "single" in p.name else 4)
    try:
        # 1. Empty state: hero Load File button, flush preview column, divider above drawers, hidden controls
        assert str(app.load_btn.cget("style")) == "CompactPrimary.TButton"
        assert app.load_btn.cget("text") == "Load File..."
        sticky_val = str(app.preview_pane.grid_info()["sticky"])
        assert "n" in sticky_val and "s" in sticky_val
        score_slaves = app._drawers_separator.master.pack_slaves()
        assert (
            score_slaves.index(app._drawers_separator)
            < score_slaves.index(app.part_options_section)
            < score_slaves.index(app.advanced_section)
        )
        app.set_score_mode(True)
        assert not app.part_options_section.winfo_manager()
        app.set_score_mode(False)
        score_slaves_restored = app._drawers_separator.master.pack_slaves()
        assert (
            score_slaves_restored.index(app._drawers_separator)
            < score_slaves_restored.index(app.part_options_section)
            < score_slaves_restored.index(app.advanced_section)
        )
        assert not app._pages_row.winfo_manager()
        assert not app.preview_pane._nav_frame.winfo_manager()
        assert not app.preview_pane.pick_btn.winfo_manager()
        assert not app.abort_btn.winfo_manager()
        assert not app.progress.winfo_manager()

        # 2. Load 1-page file -> secondary Change File button, pages & preview nav remain hidden
        single_pdf = tmp_path / "single.pdf"
        single_pdf.write_bytes(b"%PDF-1.4")
        assert app.load_source(single_pdf) is True
        assert str(app.load_btn.cget("style")) == "Secondary.TButton"
        assert app.load_btn.cget("text") == "Change File..."
        assert not app._pages_row.winfo_manager()
        assert not app.preview_pane._nav_frame.winfo_manager()
        assert not app.preview_pane.pick_btn.winfo_manager()

        # 3. Load 4-page PDF -> pages row & preview nav/pick appear
        multi_pdf = tmp_path / "multi.pdf"
        multi_pdf.write_bytes(b"%PDF-1.4")
        assert app.load_source(multi_pdf) is True
        assert app._pages_row.winfo_manager()
        assert app.preview_pane._nav_frame.winfo_manager()
        assert app.preview_pane.pick_btn.winfo_manager()

        # 4. Collapsible Instrument/Clef/Transpose live badge + theme refresh + reset
        assert not app.part_options_section.badge_lbl.winfo_manager()
        app.instrument_var.set("Trombone")
        app.clef_var.set("bass")
        app.step_transpose(2)
        assert app.part_options_section.badge_lbl.winfo_manager()
        assert app.part_options_section.badge_lbl.cget("text") == "Trombone · Bass · +2"
        app.toggle_theme()
        assert app.part_options_section.badge_lbl.cget("background") == app.palette.accent_soft
        app.transpose_var.set("bad")
        assert "invalid" in app.part_options_section.badge_lbl.cget("text").lower()
        app.on_reset_options()
        assert not app.part_options_section.badge_lbl.winfo_manager()

        # 5. Model prefetch progress reveals progress bar without Abort, then hides on ready
        app.q.put(("models_progress", (40_000_000, 100_000_000)))
        app.process_queue()
        assert app.progress.winfo_manager()
        assert not app.abort_btn.winfo_manager()
        app.q.put(("models_ready", None))
        app.process_queue()
        assert not app.progress.winfo_manager()

        # 6. Conversion shows Abort + progress during run, hides Abort on completion, hides both on abort
        app.start_conversion(multi_pdf, tmp_path / "out.mscz")
        assert app.abort_btn.winfo_manager()
        assert str(app.abort_btn["state"]) == "normal"
        assert app.progress.winfo_manager()
        pump(app, root, lambda: bool(app.result_banner.winfo_manager()))
        assert not app.abort_btn.winfo_manager()
        assert app.progress.winfo_manager()

        FakePipeline.behaviour = "abort"
        app.start_conversion(multi_pdf, tmp_path / "out2.mscz")
        pump(app, root, lambda: "aborted by user" in app.status_var.get())
        assert not app.abort_btn.winfo_manager()
        assert not app.progress.winfo_manager()

        # 7. Log drawer expand packing toggles between non-expanding collapsed and expanding open
        assert int(app.log_section.pack_info()["expand"]) == 0
        app.log_section.toggle()
        assert int(app.log_section.pack_info()["expand"]) == 1
        app.log_section.toggle()
        assert int(app.log_section.pack_info()["expand"]) == 0

        # 8. _fit_window_to_content grows on expand, shrinks on collapse, skips zoomed/withdrawn
        geo_calls: list[str] = []
        minsize_calls: list[tuple[int, int]] = []
        state_holder = {"state": "withdrawn", "req_h": 640, "cur_h": 500}

        monkeypatch.setattr(root, "wm_state", lambda: state_holder["state"])
        monkeypatch.setattr(root, "winfo_reqwidth", lambda: 860)
        monkeypatch.setattr(root, "winfo_reqheight", lambda: state_holder["req_h"])
        monkeypatch.setattr(root, "winfo_width", lambda: 860)
        monkeypatch.setattr(root, "winfo_height", lambda: state_holder["cur_h"])
        monkeypatch.setattr(root, "geometry", lambda g: geo_calls.append(g))
        monkeypatch.setattr(root, "minsize", lambda w, h: minsize_calls.append((w, h)))

        app._fit_window_to_content(shrink=False)
        assert geo_calls == []
        state_holder["state"] = "zoomed"
        app._fit_window_to_content(shrink=True)
        assert geo_calls == []

        state_holder["state"] = "normal"
        app._fit_window_to_content(shrink=False)
        assert geo_calls[-1] == "860x648"

        state_holder["cur_h"] = 648
        state_holder["req_h"] = 430
        app._fit_window_to_content(shrink=True)
        assert geo_calls[-1] == "860x438"
        assert minsize_calls[-1] == (780, 400)
    finally:
        close_app(app)
