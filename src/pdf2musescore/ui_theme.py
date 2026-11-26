"""Studio visual theme palettes, Windows DPI/titlebar polish, and pure UI helpers."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Sequence

from .engines.homr_engine import parse_page_spec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThemePalette:
    name: str
    bg: str
    card: str
    card_alt: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_subtle: str
    accent: str
    accent_hover: str
    accent_soft: str
    accent_text: str
    success_bg: str
    success_border: str
    success_text: str
    danger: str
    danger_soft: str
    warn_bg: str
    warn_text: str
    warn_border: str
    input_bg: str
    input_border: str
    pill_bg: str
    pill_text: str
    preview_bg: str
    log_bg: str
    log_fg: str
    log_debug: str
    log_warn: str
    log_err: str
    tooltip_bg: str
    tooltip_fg: str
    tooltip_muted: str
    is_dark: bool


THEMES: dict[str, ThemePalette] = {
    "light": ThemePalette(
        name="light",
        bg="#F4F6FA",
        card="#FFFFFF",
        card_alt="#F8FAFC",
        border="#E2E8F0",
        border_strong="#CBD5E1",
        text="#0F172A",
        text_muted="#475569",
        text_subtle="#64748B",
        accent="#2563EB",
        accent_hover="#1D4ED8",
        accent_soft="#EFF6FF",
        accent_text="#FFFFFF",
        success_bg="#ECFDF5",
        success_border="#A7F3D0",
        success_text="#065F46",
        danger="#DC2626",
        danger_soft="#FEF2F2",
        warn_bg="#FFFBEB",
        warn_text="#B45309",
        warn_border="#FDE68A",
        input_bg="#FFFFFF",
        input_border="#CBD5E1",
        pill_bg="#E2E8F0",
        pill_text="#334155",
        preview_bg="#EDF2F7",
        log_bg="#0F172A",
        log_fg="#E2E8F0",
        log_debug="#94A3B8",
        log_warn="#FBBF24",
        log_err="#F87171",
        tooltip_bg="#1E293B",
        tooltip_fg="#F8FAFC",
        tooltip_muted="#CBD5E1",
        is_dark=False,
    ),
    "dark": ThemePalette(
        name="dark",
        bg="#162032",
        card="#1F2B3E",
        card_alt="#192335",
        border="#364760",
        border_strong="#4E6380",
        text="#F8FAFC",
        text_muted="#CBD5E1",
        text_subtle="#94A3B8",
        accent="#3B82F6",
        accent_hover="#2563EB",
        accent_soft="#1E3A5F",
        accent_text="#FFFFFF",
        success_bg="#064E3B",
        success_border="#059669",
        success_text="#D1FAE5",
        danger="#F87171",
        danger_soft="#451A1A",
        warn_bg="#451A03",
        warn_text="#FCD34D",
        warn_border="#92400E",
        input_bg="#111A2B",
        input_border="#4E6380",
        pill_bg="#334155",
        pill_text="#E2E8F0",
        preview_bg="#111A2B",
        log_bg="#060A12",
        log_fg="#E2E8F0",
        log_debug="#94A3B8",
        log_warn="#FBBF24",
        log_err="#F87171",
        tooltip_bg="#020617",
        tooltip_fg="#F8FAFC",
        tooltip_muted="#94A3B8",
        is_dark=True,
    ),
}

_INTERVAL_NAMES: dict[int, str] = {
    1: "m2",
    2: "M2",
    3: "m3",
    4: "M3",
    5: "P4",
    6: "TT",
    7: "P5",
    8: "m6",
    9: "M6",
    10: "m7",
    11: "M7",
    12: "1 octave",
    24: "2 octaves",
}

_COMMON_TRANSPOSE_INSTRUMENTS: dict[int, str] = {
    2: "B♭",
    7: "F",
    9: "E♭",
}


def get_theme(name: str) -> ThemePalette:
    """Return a known studio theme palette, defaulting to ``'light'``."""
    return THEMES.get((name or "").strip().lower(), THEMES["light"])


def get_assets_dir() -> Path:
    """Return the package assets directory (works both in editable installs and PyInstaller bundles)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass_assets = Path(sys._MEIPASS) / "pdf2musescore" / "assets"
        if meipass_assets.is_dir():
            return meipass_assets
    return Path(__file__).resolve().parent / "assets"


def pick_font_family(available: Sequence[str], preferred: Sequence[str], fallback: str) -> str:
    """Pick the first available font family from ``preferred``, or ``fallback``."""
    available_set = {f.lower(): f for f in available}
    for candidate in preferred:
        matched = available_set.get(candidate.lower())
        if matched:
            return matched
    return fallback


def enable_high_dpi() -> bool:
    """Enable crisp per-monitor/system DPI scaling on Windows. Never raises."""
    if sys.platform != "win32":
        return False
    try:
        shcore = getattr(ctypes, "windll", None)
        if shcore is not None and hasattr(shcore, "shcore"):
            shcore.shcore.SetProcessDpiAwareness(1)
            return True
    except Exception as exc:
        log.debug("SetProcessDpiAwareness(1) unavailable: %s", exc)
    try:
        windll = getattr(ctypes, "windll", None)
        if windll is not None and hasattr(windll, "user32"):
            windll.user32.SetProcessDPIAware()
            return True
    except Exception as exc:
        log.debug("SetProcessDPIAware unavailable: %s", exc)
    return False


def apply_windows_titlebar_theme(window: tk.Misc, dark: bool) -> bool:
    """Hint Windows 10/11 DWM to tint the window title bar dark or light. Never raises."""
    if sys.platform != "win32":
        return False
    try:
        windll = getattr(ctypes, "windll", None)
        if windll is None or not hasattr(windll, "dwmapi") or not hasattr(windll, "user32"):
            return False
        window.update_idletasks()
        hwnd = windll.user32.GetParent(window.winfo_id()) or window.winfo_id()
        value = ctypes.c_int(1 if dark else 0)
        # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Windows 11 / Win10 2004+), fallback 19
        res = windll.dwmapi.DwmSetWindowAttribute(
            wintypes.HWND(hwnd),
            20,
            ctypes.byref(value),
            ctypes.sizeof(value),
        )
        if res != 0:
            res = windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                19,
                ctypes.byref(value),
                ctypes.sizeof(value),
            )
        return res == 0
    except Exception as exc:
        log.debug("DWM title bar theme hint skipped: %s", exc)
        return False


class _FLASHWINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("hwnd", wintypes.HWND),
        ("dwFlags", wintypes.DWORD),
        ("uCount", wintypes.UINT),
        ("dwTimeout", wintypes.DWORD),
    ]


def flash_windows_taskbar(window: tk.Misc, *, play_sound: bool = True) -> bool:
    """Flash the Windows taskbar button and optionally play a subtle notification chime."""
    if sys.platform != "win32":
        return False
    flashed = False
    try:
        windll = getattr(ctypes, "windll", None)
        if windll is not None and hasattr(windll, "user32") and hasattr(windll.user32, "FlashWindowEx"):
            window.update_idletasks()
            hwnd = windll.user32.GetParent(window.winfo_id()) or window.winfo_id()
            info = _FLASHWINFO(
                cbSize=ctypes.sizeof(_FLASHWINFO),
                hwnd=wintypes.HWND(hwnd),
                dwFlags=0x00000003 | 0x0000000C,  # FLASHW_ALL | FLASHW_TIMERNOFG
                uCount=0,
                dwTimeout=0,
            )
            windll.user32.FlashWindowEx(ctypes.byref(info))
            flashed = True
    except Exception as exc:
        log.debug("Windows taskbar flash skipped: %s", exc)

    if play_sound:
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception as exc:
            log.debug("Notification beep skipped: %s", exc)
    return flashed


def configure_ttk_style(style: ttk.Style, palette: ThemePalette, ui_font: str) -> None:
    """Configure crisp ``clam``-based ttk styles matching ``palette``."""
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=palette.bg, foreground=palette.text, font=(ui_font, 9))
    style.configure("App.TFrame", background=palette.bg)
    style.configure("Card.TFrame", background=palette.card)
    style.configure("CardAlt.TFrame", background=palette.card_alt)
    style.configure("Banner.TFrame", background=palette.success_bg)
    style.configure("WarnBanner.TFrame", background=palette.warn_bg)

    style.configure("App.TLabel", background=palette.bg, foreground=palette.text, font=(ui_font, 9))
    style.configure("HeaderTitle.TLabel", background=palette.bg, foreground=palette.text, font=(ui_font, 15, "bold"))
    style.configure("HeaderSub.TLabel", background=palette.bg, foreground=palette.text_muted, font=(ui_font, 9))
    style.configure(
        "VersionPill.TLabel",
        background=palette.pill_bg,
        foreground=palette.pill_text,
        font=(ui_font, 8, "bold"),
        padding=(8, 3),
    )

    style.configure("Card.TLabel", background=palette.card, foreground=palette.text, font=(ui_font, 9))
    style.configure("CardTitle.TLabel", background=palette.card, foreground=palette.text, font=(ui_font, 10, "bold"))
    style.configure(
        "SectionHeader.TLabel",
        background=palette.card,
        foreground=palette.text_subtle,
        font=(ui_font, 8, "bold"),
    )
    style.configure("Muted.TLabel", background=palette.card, foreground=palette.text_muted, font=(ui_font, 9))
    style.configure("AppMuted.TLabel", background=palette.bg, foreground=palette.text_muted, font=(ui_font, 9))
    style.configure("Hint.TLabel", background=palette.card, foreground=palette.text_subtle, font=(ui_font, 8))
    style.configure("ErrorHint.TLabel", background=palette.card, foreground=palette.danger, font=(ui_font, 8, "bold"))
    style.configure("AccentHint.TLabel", background=palette.card, foreground=palette.accent, font=(ui_font, 8, "bold"))
    style.configure(
        "BannerText.TLabel",
        background=palette.success_bg,
        foreground=palette.success_text,
        font=(ui_font, 9, "bold"),
    )
    style.configure(
        "WarnBannerText.TLabel",
        background=palette.warn_bg,
        foreground=palette.warn_text,
        font=(ui_font, 8, "bold"),
    )
    style.configure("Footer.TLabel", background=palette.bg, foreground=palette.text_subtle, font=(ui_font, 8))
    style.configure(
        "FooterBadgeOk.TLabel",
        background=palette.bg,
        foreground=palette.success_text if not palette.is_dark else "#34D399",
        font=(ui_font, 8, "bold"),
    )
    style.configure("FooterBadgeMuted.TLabel", background=palette.bg, foreground=palette.text_subtle, font=(ui_font, 8))

    # Buttons
    style.configure(
        "Primary.TButton",
        background=palette.accent,
        foreground=palette.accent_text,
        borderwidth=0,
        focusthickness=0,
        padding=(14, 8),
        font=(ui_font, 10, "bold"),
    )
    style.map(
        "Primary.TButton",
        background=[("disabled", palette.pill_bg), ("pressed", palette.accent_hover), ("active", palette.accent_hover)],
        foreground=[("disabled", palette.text_subtle), ("active", palette.accent_text)],
    )

    style.configure(
        "CompactPrimary.TButton",
        background=palette.accent,
        foreground=palette.accent_text,
        borderwidth=0,
        focusthickness=0,
        padding=(12, 5),
        font=(ui_font, 9, "bold"),
    )
    style.map(
        "CompactPrimary.TButton",
        background=[("disabled", palette.pill_bg), ("pressed", palette.accent_hover), ("active", palette.accent_hover)],
        foreground=[("disabled", palette.text_subtle), ("active", palette.accent_text)],
    )

    style.configure(
        "Secondary.TButton",
        background=palette.card_alt,
        foreground=palette.text,
        bordercolor=palette.border_strong,
        lightcolor=palette.card_alt,
        darkcolor=palette.card_alt,
        borderwidth=1,
        focusthickness=0,
        padding=(10, 4),
        font=(ui_font, 9),
    )
    style.map(
        "Secondary.TButton",
        background=[("disabled", palette.card_alt), ("active", palette.pill_bg)],
        foreground=[("disabled", palette.text_subtle)],
    )

    style.configure(
        "Stepper.TButton",
        background=palette.card_alt,
        foreground=palette.text,
        bordercolor=palette.border_strong,
        borderwidth=1,
        focusthickness=0,
        padding=(6, 2),
        font=(ui_font, 9, "bold"),
    )
    style.map(
        "Stepper.TButton",
        background=[("disabled", palette.card_alt), ("active", palette.pill_bg)],
        foreground=[("disabled", palette.text_subtle)],
    )

    style.configure(
        "Danger.TButton",
        background=palette.danger_soft,
        foreground=palette.danger,
        bordercolor=palette.border_strong,
        borderwidth=1,
        focusthickness=0,
        padding=(12, 8),
        font=(ui_font, 9, "bold"),
    )
    style.map(
        "Danger.TButton",
        background=[("disabled", palette.card_alt), ("active", palette.danger)],
        foreground=[("disabled", palette.text_subtle), ("active", "#FFFFFF")],
    )

    style.configure(
        "Ghost.TButton",
        background=palette.bg,
        foreground=palette.text_muted,
        borderwidth=0,
        focusthickness=0,
        padding=(8, 4),
        font=(ui_font, 8),
    )
    style.map(
        "Ghost.TButton",
        background=[("active", palette.pill_bg)],
        foreground=[("active", palette.text)],
    )

    style.configure(
        "CardGhost.TButton",
        background=palette.card,
        foreground=palette.text_muted,
        borderwidth=0,
        focusthickness=0,
        padding=(6, 4),
        font=(ui_font, 9, "bold"),
    )
    style.map(
        "CardGhost.TButton",
        background=[("active", palette.card_alt)],
        foreground=[("active", palette.accent)],
    )

    disabled_preset_fg = "#64748B" if palette.is_dark else palette.text_subtle
    style.configure(
        "Preset.TButton",
        background=palette.card_alt,
        foreground=palette.text_muted,
        bordercolor=palette.border,
        lightcolor=palette.card_alt,
        darkcolor=palette.card_alt,
        borderwidth=1,
        focusthickness=0,
        padding=(10, 6),
        font=(ui_font, 9),
    )
    style.map(
        "Preset.TButton",
        background=[("disabled", palette.card_alt), ("active", palette.accent_soft)],
        foreground=[("disabled", disabled_preset_fg), ("active", palette.accent)],
        bordercolor=[("disabled", palette.border), ("active", palette.accent)],
        lightcolor=[("disabled", palette.card_alt), ("active", palette.accent_soft)],
        darkcolor=[("disabled", palette.card_alt), ("active", palette.accent_soft)],
    )

    active_bg = "#2563EB"
    active_hover_bg = "#3B82F6" if palette.is_dark else "#1D4ED8"
    active_border = "#60A5FA" if palette.is_dark else "#1D4ED8"
    active_disabled_bg = "#1E3A8A" if palette.is_dark else "#DBEAFE"
    active_disabled_fg = "#DBEAFE" if palette.is_dark else "#1E40AF"
    active_disabled_border = "#60A5FA" if palette.is_dark else "#3B82F6"
    style.configure(
        "PresetActive.TButton",
        background=active_bg,
        foreground="#FFFFFF",
        bordercolor=active_border,
        lightcolor=active_bg,
        darkcolor=active_bg,
        borderwidth=1,
        focusthickness=0,
        padding=(10, 6),
        font=(ui_font, 9, "bold"),
    )
    style.map(
        "PresetActive.TButton",
        background=[("disabled", active_disabled_bg), ("active", active_hover_bg)],
        foreground=[("disabled", active_disabled_fg), ("active", "#FFFFFF")],
        bordercolor=[("disabled", active_disabled_border), ("active", active_border)],
        lightcolor=[("disabled", active_disabled_bg), ("active", active_hover_bg)],
        darkcolor=[("disabled", active_disabled_bg), ("active", active_hover_bg)],
    )

    style.configure(
        "Update.TButton",
        background=palette.warn_bg,
        foreground=palette.warn_text,
        bordercolor=palette.warn_border,
        borderwidth=1,
        focusthickness=0,
        padding=(10, 4),
        font=(ui_font, 8, "bold"),
    )
    style.map(
        "Update.TButton",
        background=[("active", palette.warn_border)],
    )

    style.configure(
        "Success.TButton",
        background=palette.success_text if not palette.is_dark else "#059669",
        foreground="#FFFFFF",
        borderwidth=0,
        focusthickness=0,
        padding=(10, 5),
        font=(ui_font, 8, "bold"),
    )
    style.map(
        "Success.TButton",
        background=[("active", "#047857")],
    )

    # Inputs & Progressbar
    style.configure(
        "Studio.TEntry",
        fieldbackground=palette.input_bg,
        foreground=palette.text,
        bordercolor=palette.input_border,
        insertcolor=palette.text,
        padding=(6, 4),
    )
    style.map("Studio.TEntry", bordercolor=[("focus", palette.accent)])

    style.configure(
        "Studio.TCombobox",
        fieldbackground=palette.input_bg,
        background=palette.card_alt,
        foreground=palette.text,
        bordercolor=palette.input_border,
        arrowcolor=palette.text_muted,
        padding=(5, 4),
    )
    style.map(
        "Studio.TCombobox",
        fieldbackground=[("readonly", palette.input_bg), ("disabled", palette.card_alt)],
        foreground=[("readonly", palette.text), ("disabled", palette.text_subtle)],
        bordercolor=[("focus", palette.accent)],
    )

    style.configure(
        "Studio.TCheckbutton",
        background=palette.card,
        foreground=palette.text,
        indicatorbackground=palette.input_bg,
        indicatorforeground=palette.accent,
        font=(ui_font, 9),
    )
    style.map("Studio.TCheckbutton", background=[("active", palette.card)])

    style.configure(
        "Studio.Horizontal.TProgressbar",
        troughcolor=palette.pill_bg,
        background=palette.accent,
        borderwidth=0,
        thickness=8,
    )
    style.configure("Studio.TSeparator", background=palette.border)

    style.configure(
        "Studio.Vertical.TScrollbar",
        background=palette.border_strong,
        troughcolor=palette.log_bg,
        bordercolor=palette.log_bg,
        lightcolor=palette.border_strong,
        darkcolor=palette.border_strong,
        arrowcolor=palette.text_muted,
        gripcount=0,
    )
    style.map("Studio.Vertical.TScrollbar", background=[("active", palette.text_subtle)])


def format_transpose_label(semitones: int) -> str:
    """Format a musical interval hint for a transpose semitone offset (-24..+24)."""
    if not (-24 <= semitones <= 24):
        return "Out of range (-24..+24)"
    if semitones == 0:
        return "Original pitch"
    direction = "Up" if semitones > 0 else "Down"
    mag = abs(semitones)
    if mag in _INTERVAL_NAMES:
        interval = _INTERVAL_NAMES[mag]
    else:
        octaves, rem = divmod(mag, 12)
        interval = f"{octaves} oct + {_INTERVAL_NAMES[rem]}"
    inst = _COMMON_TRANSPOSE_INSTRUMENTS.get(semitones)
    sign = "+" if semitones > 0 else ""
    base = f"{sign}{semitones} ({direction} {interval}"
    if inst:
        return f"{base} · {inst})"
    return f"{base})"


def format_page_ranges(pages: Sequence[int]) -> str:
    """Format an arbitrary sequence of 1-based page numbers into compact ranges like ``'1-3, 5'``."""
    unique = sorted({int(p) for p in pages if int(p) >= 1})
    if not unique:
        return ""
    ranges: list[str] = []
    start = prev = unique[0]
    for p in unique[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = p
    ranges.append(f"{start}-{prev}" if prev > start else str(start))
    return ", ".join(ranges)


def toggle_page_in_spec(current_spec: str, page: int, total_pages: int) -> str:
    """Toggle ``page`` in a page-range string against ``total_pages``.

    When ``current_spec`` means "all pages" (blank or ``'all'``), picking a specific page
    narrows the selection to just that page (unless the PDF only has 1 page).
    """
    if total_pages <= 0 or not (1 <= page <= total_pages):
        return (current_spec or "").strip()
    text = (current_spec or "").strip()
    if not text or text.lower() == "all":
        return "" if total_pages == 1 else str(page)

    current_pages = set(parse_page_spec(text, total_pages))
    if page in current_pages:
        current_pages.remove(page)
    else:
        current_pages.add(page)

    if not current_pages or len(current_pages) == total_pages:
        return ""
    return format_page_ranges(sorted(current_pages))


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as ``M:SS`` (clamped to >= 0)."""
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    return f"{mins}:{secs:02d}"
