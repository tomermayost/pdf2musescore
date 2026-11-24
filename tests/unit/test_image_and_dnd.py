"""Unit tests for direct image input, native Windows drag-and-drop, and footer links."""

from __future__ import annotations

import ctypes
import gc
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from pdf2musescore import win_dnd
from pdf2musescore.config import AppConfig
from pdf2musescore.engines.homr_engine import (
    CancelledError,
    _default_render_pdf,
    get_pdf_page_count,
    is_image_file,
    is_supported_input,
    load_pil_image_rgb,
)
from pdf2musescore.gui import GITHUB_FEEDBACK_URL, GITHUB_REPO_URL, App
from pdf2musescore.pipeline import ConversionError, Pipeline
from pdf2musescore.ui_widgets import render_pdf_page_image
from tests.unit.test_musicxml import score, write_mxl


@pytest.mark.parametrize(
    "name,expected_img,expected_sup",
    [
        ("score.pdf", False, True),
        ("SCORE.PDF", False, True),
        ("photo.jpg", True, True),
        ("photo.JPEG", True, True),
        ("scan.png", True, True),
        ("scan.webp", True, True),
        ("scan.bmp", True, True),
        ("scan.tif", True, True),
        ("scan.TIFF", True, True),
        ("notes.txt", False, False),
        ("README", False, False),
    ],
)
def test_input_extension_helpers(name: str, expected_img: bool, expected_sup: bool) -> None:
    p = Path(name)
    assert is_image_file(p) is expected_img
    assert is_supported_input(p) is expected_sup


def test_load_pil_image_rgb_transparency_exif_and_multipage_tiff(tmp_path: Path) -> None:
    # Transparent RGBA PNG -> transparent areas composite onto white (255, 255, 255)
    rgba_path = tmp_path / "transparent.png"
    rgba = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    rgba.putpixel((10, 10), (15, 25, 35, 255))
    rgba.save(rgba_path, format="PNG")

    loaded_rgba = load_pil_image_rgb(rgba_path)
    assert loaded_rgba.mode == "RGB"
    assert loaded_rgba.getpixel((0, 0)) == (255, 255, 255)
    assert loaded_rgba.getpixel((10, 10)) == (15, 25, 35)

    # Palette PNG with transparency key
    pal_path = tmp_path / "palette.png"
    pal_src = Image.new("RGBA", (12, 12), (255, 255, 255, 0))
    pal_src.putpixel((6, 6), (0, 0, 0, 255))
    pal_src.convert("P", palette=Image.Palette.ADAPTIVE).save(pal_path, format="PNG", transparency=0)
    loaded_pal = load_pil_image_rgb(pal_path)
    assert loaded_pal.mode == "RGB"

    # EXIF orientation=6 (90 deg CW) JPEG transposes width <-> height
    jpg_path = tmp_path / "phone_photo.jpg"
    raw_jpg = Image.new("RGB", (40, 80), (240, 240, 240))
    exif = Image.Exif()
    exif[0x0112] = 6
    raw_jpg.save(jpg_path, format="JPEG", exif=exif.tobytes())

    assert get_pdf_page_count(jpg_path) == 1
    loaded_jpg = load_pil_image_rgb(jpg_path)
    assert loaded_jpg.size == (80, 40)

    # Multi-frame TIFF support
    tif_path = tmp_path / "multi.tiff"
    f1 = Image.new("RGB", (30, 30), (200, 10, 10))
    f2 = Image.new("RGB", (30, 30), (10, 10, 200))
    f1.save(tif_path, format="TIFF", save_all=True, append_images=[f2])

    assert get_pdf_page_count(tif_path) == 2
    p2 = load_pil_image_rgb(tif_path, page_number=2)
    assert p2.getpixel((5, 5))[2] > 180

    with pytest.raises(ValueError, match="out of range"):
        load_pil_image_rgb(tif_path, page_number=3)


def test_default_render_pdf_and_preview_image_inputs(tmp_path: Path) -> None:
    tif_path = tmp_path / "scan.tif"
    f1 = Image.new("RGB", (120, 160), (255, 255, 255))
    f2 = Image.new("RGB", (120, 160), (245, 245, 245))
    f1.save(tif_path, format="TIFF", save_all=True, append_images=[f2])

    out_dir = tmp_path / "omr"
    out_dir.mkdir()
    rendered = _default_render_pdf(tif_path, out_dir, dpi=300, pages_spec="2", preprocess_mode="faded", deskew=False)
    assert [p.name for p in rendered] == ["page_002.png"]
    assert rendered[0].is_file()

    with pytest.raises(CancelledError):
        _default_render_pdf(tif_path, out_dir, dpi=300, is_cancelled=lambda: True)

    thumb = render_pdf_page_image(tif_path, 2, max_size=(80, 100))
    assert thumb.mode == "RGB"
    assert thumb.width <= 80 and thumb.height <= 100

    with pytest.raises(ValueError, match="out of range"):
        render_pdf_page_image(tif_path, 9)


def test_pipeline_accepts_unicode_image_and_rejects_unsupported(tmp_path: Path) -> None:
    heb_dir = tmp_path / "תווים"
    heb_dir.mkdir()
    img_in = heb_dir / "צילום דף.jpg"
    Image.new("RGB", (60, 80), (255, 255, 255)).save(img_in, format="JPEG")

    staged_names: list[str] = []

    class InspectEngine:
        def recognize(self, src: Path, out_dir: Path) -> list[Path]:
            staged_names.append(src.name)
            out_dir.mkdir(parents=True, exist_ok=True)
            return [write_mxl(score(1), out_dir / "page_001.mxl")]

    pipe = Pipeline(
        AppConfig(work_dir=tmp_path / "work"),
        ensure=lambda spec, cfg, confirm: Path("ms.exe"),
        engine_factory=lambda *a, **kw: InspectEngine(),
    )
    out = pipe.convert(img_in, tmp_path / "result.musicxml")
    assert out.is_file()
    assert staged_names == ["input.jpg"]

    bad_in = tmp_path / "notes.txt"
    bad_in.write_text("not music")
    with pytest.raises(ConversionError, match="Unsupported input file"):
        pipe.convert(bad_in, tmp_path / "result.musicxml")


def test_extract_dropped_files_and_drop_handle_uninstall(tmp_path: Path) -> None:
    samples = ["C:/תווים/page.png", "C:/scores/bach.pdf"]
    finished: list[int] = []

    class FakeShell32:
        def DragQueryFileW(self, hdrop, idx, buf, size):
            if idx == 0xFFFFFFFF:
                return len(samples)
            text = samples[idx]
            if buf is None:
                return len(text)
            buf.value = text
            return len(text)

        def DragFinish(self, hdrop):
            finished.append(hdrop)

    paths = win_dnd.extract_dropped_files(42, FakeShell32())
    assert paths == [Path(s) for s in samples]
    assert finished == [42]

    # Empty length entry skipped, DragFinish still guaranteed on exception
    class FailingShell32:
        def DragQueryFileW(self, hdrop, idx, buf, size):
            if idx == 0xFFFFFFFF:
                return 1
            raise OSError("bad hdrop")

        def DragFinish(self, hdrop):
            finished.append(hdrop)

    with pytest.raises(OSError):
        win_dnd.extract_dropped_files(99, FailingShell32())
    assert finished[-1] == 99

    # DropHandle.uninstall restores prev_wndproc once (idempotent) & ignores OS errors
    restored: list[tuple] = []
    accept_calls: list[bool] = []
    fake_u32 = SimpleNamespace(SetWindowLongPtrW=lambda hwnd, idx, val: restored.append((hwnd, idx, val)) or 1)
    fake_s32 = SimpleNamespace(DragAcceptFiles=lambda hwnd, flag: accept_calls.append(flag))
    handle = win_dnd.DropHandle(100, 777, lambda *a: 0, fake_u32, fake_s32)
    handle.uninstall()
    handle.uninstall()
    assert restored == [(100, win_dnd.GWLP_WNDPROC, 777)]
    assert accept_calls == [False]

    bad_u32 = SimpleNamespace(SetWindowLongPtrW=lambda *a: (_ for _ in ()).throw(OSError("gone")))
    bad_handle = win_dnd.DropHandle(100, 777, lambda *a: 0, bad_u32, fake_s32)
    bad_handle.uninstall()


def test_install_file_drop_handler_non_win32_and_mocked_dispatch(monkeypatch) -> None:
    dropped: list[list[Path]] = []
    fake_win = SimpleNamespace(
        update_idletasks=lambda: None,
        winfo_id=lambda: 123,
        after=lambda _ms, fn: fn(),
    )

    monkeypatch.setattr(win_dnd, "WNDPROC", None, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert win_dnd.install_file_drop_handler(fake_win, dropped.append) is None

    monkeypatch.setattr(sys, "platform", "win32")
    assert win_dnd.install_file_drop_handler(fake_win, dropped.append, windll=SimpleNamespace()) is None

    # Mock full Win32 user32 + shell32 subclass flow cleanly
    class _Fn:
        def __init__(self, impl):
            self._impl = impl
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self._impl(*args)

    captured_cb = {}
    passthru_calls = []

    class FakeUser32:
        def __init__(self):
            self.GetParent = _Fn(lambda wid: 500)
            self.SetWindowLongPtrW = _Fn(self._set_ptr)
            self.CallWindowProcW = _Fn(lambda prev, h, m, w, l: passthru_calls.append((prev, h, m, w, l)) or 99)
            self.ChangeWindowMessageFilterEx = _Fn(lambda *a: 1)

        def _set_ptr(self, hwnd, idx, val):
            captured_cb["ptr"] = val
            return 888

    class FakeShell32:
        def __init__(self):
            self.DragAcceptFiles = _Fn(lambda *a: None)
            self.DragQueryFileW = _Fn(self._query)
            self.DragFinish = _Fn(lambda *a: None)

        def _query(self, hdrop, idx, buf, size):
            if idx == 0xFFFFFFFF:
                return 1
            text = "C:/sheet/note.png"
            if buf is None:
                return len(text)
            buf.value = text
            return len(text)

    fake_wd = SimpleNamespace(user32=FakeUser32(), shell32=FakeShell32())
    monkeypatch.setattr(win_dnd, "_WNDPROC", lambda fn: fn)
    monkeypatch.setattr(ctypes, "cast", lambda obj, _tp: SimpleNamespace(value=obj))

    # Avoid int(cb_ptr) failing on Python function object during mock
    orig_set_ptr = fake_wd.user32.SetWindowLongPtrW
    registered = []

    def _capture_set(hwnd, idx, val):
        if callable(val):
            registered.append(val)
            return 888
        return 1

    fake_wd.user32.SetWindowLongPtrW = _Fn(_capture_set)
    handle = win_dnd.install_file_drop_handler(fake_win, dropped.append, windll=fake_wd)
    assert handle is not None and len(registered) == 1

    cb = registered[0]
    assert cb(500, win_dnd.WM_DROPFILES, 12, 0) == 0
    assert dropped == [[Path("C:/sheet/note.png")]]
    assert cb(500, 0x000F, 1, 2) == 99
    assert passthru_calls == [(888, 500, 0x000F, 1, 2)]
    handle.uninstall()

    # Zero hwnd or SetWindowLongPtrW returning 0 fails gracefully without crashing
    fake_win_zero = SimpleNamespace(update_idletasks=lambda: None, winfo_id=lambda: 0, after=lambda _m, f: None)
    fake_wd_zero = SimpleNamespace(
        user32=SimpleNamespace(GetParent=_Fn(lambda _w: 0)),
        shell32=FakeShell32(),
    )
    assert win_dnd.install_file_drop_handler(fake_win_zero, dropped.append, windll=fake_wd_zero) is None

    fake_wd_fail = SimpleNamespace(
        user32=SimpleNamespace(
            GetParent=_Fn(lambda _w: 500),
            SetWindowLongPtrW=_Fn(lambda *a: 0),
            CallWindowProcW=_Fn(lambda *a: 0),
        ),
        shell32=FakeShell32(),
    )
    assert win_dnd.install_file_drop_handler(fake_win, dropped.append, windll=fake_wd_fail) is None


def _close_test_app(app: App) -> None:
    app.destroy()
    for name in list(app.__dict__):
        if name.endswith("_var"):
            delattr(app, name)
    gc.collect()


def test_gui_drag_drop_image_preserves_preset_and_footer_links(root, tmp_path: Path) -> None:
    events: dict[str, list] = {"error": [], "url": []}
    app = App(
        root,
        AppConfig(log_dir=tmp_path / "logs", work_dir=tmp_path / "work", check_updates=False),
        musescore_available=lambda: True,
        show_error=lambda _t, msg: events["error"].append(msg),
        open_url=lambda url: events["url"].append(url) or True,
        prefs_path=tmp_path / "prefs.json",
        enable_dnd=False,
    )
    try:
        # Pick non-default preset and stale multi-page range first
        app.apply_preset("faded")
        app.pages_var.set("2-3")

        jpg_path = tmp_path / "sheet_photo.jpg"
        Image.new("RGB", (100, 140), (255, 255, 255)).save(jpg_path, format="JPEG")

        # Drop unsupported file first -> error popup
        bad_file = tmp_path / "readme.txt"
        bad_file.write_text("x")
        app.on_drop_files([bad_file])
        assert events["error"]
        events["error"].clear()

        # Drop real JPEG (alongside unsupported path) -> loads, clears stale page range, keeps "faded" preset
        app.on_drop_files([bad_file, jpg_path])
        assert app.pdf == jpg_path
        assert app.total_pages == 1
        assert app.pages_var.get() == ""
        assert str(app.preset_buttons["faded"].cget("style")) == "PresetActive.TButton"
        assert app.file_var.get() == "sheet_photo.jpg (1 page)"

        # Busy drop is ignored safely without overwriting loaded file
        app._busy = True
        other_png = tmp_path / "other.png"
        Image.new("RGB", (80, 100), (255, 255, 255)).save(other_png, format="PNG")
        app.on_drop_files([other_png])
        assert app.pdf == jpg_path
        assert "Wait for the current conversion" in app.status_var.get()
        app._busy = False

        # load_source rejects unsupported path
        assert app.load_source(bad_file) is False
        assert events["error"]

        # Footer GitHub & Feedback buttons open exact URLs
        app.github_btn.invoke()
        app.feedback_btn.invoke()
        assert events["url"] == [GITHUB_REPO_URL, GITHUB_FEEDBACK_URL]
    finally:
        _close_test_app(app)


def test_gui_real_dnd_install_and_teardown_cycle(root, tmp_path: Path) -> None:
    for _ in range(2):
        app = App(
            root,
            AppConfig(log_dir=tmp_path / "logs", work_dir=tmp_path / "work", check_updates=False),
            musescore_available=lambda: True,
            prefs_path=tmp_path / "prefs.json",
            enable_dnd=True,
        )
        root.update_idletasks()
        _close_test_app(app)
