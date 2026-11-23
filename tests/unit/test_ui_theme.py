import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace

import pytest

from pdf2musescore import ui_theme
from pdf2musescore.ui_theme import (
    THEMES,
    apply_windows_titlebar_theme,
    configure_ttk_style,
    enable_high_dpi,
    format_elapsed,
    format_page_ranges,
    format_transpose_label,
    get_assets_dir,
    get_theme,
    pick_font_family,
    toggle_page_in_spec,
)


def test_themes_and_get_theme():
    assert set(THEMES.keys()) == {"light", "dark"}
    light = get_theme("light")
    dark = get_theme("DARK")
    assert not light.is_dark
    assert dark.is_dark
    assert get_theme("unknown") == light
    assert get_theme(None) == light  # type: ignore[arg-type]


def test_get_assets_dir_dev_and_frozen(monkeypatch, tmp_path):
    dev_dir = get_assets_dir()
    assert (dev_dir / "icon.ico").is_file()
    assert (dev_dir / "icon.png").is_file()

    frozen_assets = tmp_path / "pdf2musescore" / "assets"
    frozen_assets.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert get_assets_dir() == frozen_assets

    # If _MEIPASS doesn't have the folder, fallback cleanly to package directory
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "missing"), raising=False)
    assert get_assets_dir() == dev_dir


def test_pick_font_family():
    avail = ("Arial", "Segoe UI", "Consolas")
    assert pick_font_family(avail, ("Segoe UI Variable Text", "segoe ui"), "TkDefaultFont") == "Segoe UI"
    assert pick_font_family(avail, ("Cascadia Mono", "JetBrains Mono"), "Consolas") == "Consolas"


@pytest.mark.parametrize(
    "semitones,expected_substr",
    [
        (0, "Original pitch"),
        (2, "+2 (Up M2"),
        (-2, "-2 (Down M2)"),
        (7, "+7 (Up P5"),
        (9, "+9 (Up M6"),
        (12, "+12 (Up 1 octave)"),
        (-12, "-12 (Down 1 octave)"),
        (14, "+14 (Up 1 oct + M2)"),
        (-24, "-24 (Down 2 octaves)"),
        (30, "Out of range"),
    ],
)
def test_format_transpose_label(semitones, expected_substr):
    assert expected_substr in format_transpose_label(semitones)


def test_format_page_ranges_and_toggle():
    assert format_page_ranges([]) == ""
    assert format_page_ranges([3, 1, 2, 5, 7, 8, 5]) == "1-3, 5, 7-8"

    # When blank ("all pages"), picking page 2 selects just page 2 (unless 1-page PDF)
    assert toggle_page_in_spec("", 2, 5) == "2"
    assert toggle_page_in_spec("all", 3, 5) == "3"
    assert toggle_page_in_spec("", 1, 1) == ""
    # Adding / removing pages
    spec = toggle_page_in_spec("2", 3, 5)
    assert spec == "2-3"
    spec = toggle_page_in_spec(spec, 5, 5)
    assert spec == "2-3, 5"
    # Removing the last remaining page reverts to all ("")
    assert toggle_page_in_spec("2", 2, 5) == ""
    # Completing all pages also normalizes to ""
    assert toggle_page_in_spec("1-2", 3, 3) == ""
    # Out-of-range page leaves spec untouched
    assert toggle_page_in_spec("2-3", 9, 5) == "2-3"


def test_format_elapsed():
    assert format_elapsed(-5) == "0:00"
    assert format_elapsed(0) == "0:00"
    assert format_elapsed(9) == "0:09"
    assert format_elapsed(74.8) == "1:14"


def test_configure_ttk_style_on_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No display")
    root.withdraw()
    try:
        style = ttk.Style(root)
        for name in ("light", "dark"):
            configure_ttk_style(style, get_theme(name), "Segoe UI")
        assert style.lookup("Primary.TButton", "background")
    finally:
        root.destroy()


def test_windows_dpi_and_titlebar_helpers(monkeypatch):
    # Test non-win32 fast-exit path
    monkeypatch.setattr(ui_theme.sys, "platform", "linux")
    assert enable_high_dpi() is False
    assert apply_windows_titlebar_theme(SimpleNamespace(), True) is False

    # Test mocked win32 paths (primary, fallback, exception)
    monkeypatch.setattr(ui_theme.sys, "platform", "win32")

    calls = []
    fake_shcore = SimpleNamespace(SetProcessDpiAwareness=lambda mode: calls.append(("shcore", mode)))
    fake_user32_dpi = SimpleNamespace(SetProcessDPIAware=lambda: calls.append(("user32",)))
    monkeypatch.setattr(
        ui_theme.ctypes,
        "windll",
        SimpleNamespace(shcore=fake_shcore, user32=fake_user32_dpi),
        raising=False,
    )
    assert enable_high_dpi() is True
    assert calls == [("shcore", 1)]

    # Fallback when shcore raises
    def raise_err(_mode):
        raise OSError("no shcore")

    calls.clear()
    monkeypatch.setattr(
        ui_theme.ctypes,
        "windll",
        SimpleNamespace(shcore=SimpleNamespace(SetProcessDpiAwareness=raise_err), user32=fake_user32_dpi),
        raising=False,
    )
    assert enable_high_dpi() is True
    assert calls == [("user32",)]

    # DWM titlebar fallback & success
    dwm_attrs = []

    def fake_dwm(_hwnd, attr, _val, _size):
        dwm_attrs.append(attr)
        return 1 if attr == 20 else 0

    fake_win = SimpleNamespace(update_idletasks=lambda: None, winfo_id=lambda: 1234)
    monkeypatch.setattr(
        ui_theme.ctypes,
        "windll",
        SimpleNamespace(
            user32=SimpleNamespace(GetParent=lambda hwnd: hwnd),
            dwmapi=SimpleNamespace(DwmSetWindowAttribute=fake_dwm),
        ),
        raising=False,
    )
    assert apply_windows_titlebar_theme(fake_win, True) is True
    assert dwm_attrs == [20, 19]
