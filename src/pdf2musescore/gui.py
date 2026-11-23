"""Studio desktop window: PDF Preview -> Source & Presets -> Score Setup -> Convert / Open."""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Callable, Optional
import webbrowser

from PIL import Image, ImageTk

from . import APP_NAME, AUTHOR_CREDIT, __version__
from .config import AppConfig
from .deps import MUSESCORE, find_tool
from .engines.homr_engine import CancelledError, get_engine_version, get_pdf_page_count, parse_page_spec
from .options import (
    CLEF_CHOICES,
    INSTRUMENT_PRESETS,
    PREPROCESS_MODES,
    SCAN_PRESETS,
    ConversionOptions,
    OptionsError,
    detect_scan_preset,
    get_parameter_tooltip,
    get_scan_preset,
)
from .pipeline import ConversionError, Pipeline
from .preferences import (
    UserPreferences,
    default_preferences_path,
    load_preferences,
    save_preferences as write_preferences,
)
from .ui_theme import (
    THEMES,
    ThemePalette,
    apply_windows_titlebar_theme,
    configure_ttk_style,
    enable_high_dpi,
    format_elapsed,
    format_transpose_label,
    get_assets_dir,
    get_theme,
    pick_font_family,
    toggle_page_in_spec,
)
from .ui_widgets import CollapsibleSection, PdfPreviewPane, ToolTip, render_pdf_page_image
from .updater import UpdateError, UpdateInfo, check_for_update, download_installer

log = logging.getLogger(__name__)

MSCZ_TYPE = ("MuseScore file", "*.mscz")
XML_TYPE = ("MusicXML file", "*.musicxml")

PAGES_TOOLTIP = (
    "Which Pages to Convert  (Default: all pages)\n"
    "Examples: 2-3 (pages 2 to 3), 1, 4 (only pages 1 and 4)\n\n"
    "Leave empty to convert the entire PDF, type a range, or click "
    "'Pick page' under the preview on the left."
)

PRESET_TOOLTIPS: dict[str, str] = {
    "clean": (
        "Clean PDF  (Recommended for digital sheet music)\n"
        "Settings: 300 DPI · Auto cleanup · No deskew · Normal reading\n\n"
        "Best for crisp, digital PDFs exported directly from notation software."
    ),
    "faded": (
        "Faded Scan  (Recommended for old or light photocopies)\n"
        "Settings: 350 DPI · Faded ink contrast boost · Auto straighten · Careful reading\n\n"
        "Sharpens light staff lines and straightens slightly crooked flatbed scans."
    ),
    "photo": (
        "Phone Photo  (Recommended for camera pictures of sheet music)\n"
        "Settings: 300 DPI · Shadow-removing B&W · Auto straighten · Careful reading\n\n"
        "Removes uneven lighting/shadows from smartphone photos and ignores paper grain."
    ),
}

_TEXT_INPUT_WIDGET_CLASSES = {"TEntry", "Entry", "TCombobox", "Text", "TSpinbox", "Spinbox"}


def _default_open_path(target: str) -> None:  # pragma: no cover - invokes OS shell
    opener = getattr(os, "startfile", None)
    if opener is not None:
        opener(target)


def _default_reveal_file(path: Path, open_folder: Callable[[str], None]) -> None:  # pragma: no cover
    if sys.platform == "win32":
        try:
            subprocess.Popen(["explorer", f"/select,{path.resolve()}"])
            return
        except Exception:
            pass
    open_folder(str(path.parent))


class QueueLogHandler(logging.Handler):
    """Forwards log records (levelno + formatted line) to the GUI thread via a queue."""

    def __init__(self, q: "queue.Queue") -> None:
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put(("log", (record.levelno, self.format(record))))


class App(ttk.Frame):
    POLL_MS = 100

    def __init__(
        self,
        master: tk.Misc,
        config: AppConfig,
        pipeline_factory: Callable[..., Pipeline] = Pipeline,
        musescore_available: Optional[Callable[[], bool]] = None,
        page_counter: Callable[[Path], int] = get_pdf_page_count,
        page_renderer: Optional[Callable[[Path, int, tuple[int, int]], Image.Image]] = None,
        ask_open: Callable[..., str] = filedialog.askopenfilename,
        ask_save: Callable[..., str] = filedialog.asksaveasfilename,
        ask_yes_no: Callable[[str, str], bool] = messagebox.askyesno,
        show_error: Callable[[str, str], None] = messagebox.showerror,
        show_info: Callable[[str, str], None] = messagebox.showinfo,
        open_folder: Callable[[str], None] = _default_open_path,
        open_file: Optional[Callable[[str], None]] = None,
        reveal_file: Optional[Callable[[Path], None]] = None,
        engine_version: Optional[str] = None,
        update_checker: Optional[Callable[[str], Optional[UpdateInfo]]] = None,
        installer_downloader: Callable[..., Path] = download_installer,
        launch_installer: Callable[[str], None] = _default_open_path,
        open_url: Callable[[str], bool] = webbrowser.open,
        on_exit: Optional[Callable[[], None]] = None,
        theme: Optional[str] = None,
        prefs_path: Optional[Path] = None,
    ) -> None:
        self.config_ = config
        self._prefs_path: Path = Path(prefs_path) if prefs_path is not None else default_preferences_path()
        self._prefs: UserPreferences = load_preferences(self._prefs_path, default_dpi=self.config_.render_dpi)
        self._last_dir: str = self._prefs.last_dir
        self.pipeline_factory = pipeline_factory
        self.musescore_available = musescore_available or (
            lambda: find_tool(MUSESCORE, config.musescore_path) is not None
        )
        self.page_counter = page_counter
        self.page_renderer = render_pdf_page_image if page_renderer is None else page_renderer
        self.ask_open, self.ask_save = ask_open, ask_save
        self.ask_yes_no, self.show_error, self.show_info = ask_yes_no, show_error, show_info
        self.open_folder = open_folder
        self.open_file = open_file if open_file is not None else _default_open_path
        self.reveal_file = reveal_file if reveal_file is not None else (
            lambda p: _default_reveal_file(p, self.open_folder)
        )
        self.engine_version = engine_version if engine_version is not None else get_engine_version()
        self.update_checker = check_for_update if update_checker is None else update_checker
        self.installer_downloader = installer_downloader
        self.launch_installer = launch_installer
        self.open_url = open_url
        self._on_exit = on_exit or self._default_exit

        chosen_theme = theme if theme is not None else self._prefs.theme
        self.theme_name = "dark" if (chosen_theme or "").strip().lower() == "dark" else "light"
        self.palette: ThemePalette = get_theme(self.theme_name)
        self._style = ttk.Style(master)
        try:
            avail_fonts = tkfont.families(master)
        except Exception:
            avail_fonts = ()
        self._ui_font = pick_font_family(avail_fonts, ("Segoe UI Variable Text", "Segoe UI"), "TkDefaultFont")
        self._mono_font = pick_font_family(avail_fonts, ("Cascadia Mono", "Consolas"), "TkFixedFont")
        configure_ttk_style(self._style, self.palette, self._ui_font)

        super().__init__(master, style="App.TFrame", padding=(14, 10, 14, 10))

        self.q: "queue.Queue" = queue.Queue()
        self.pdf: Optional[Path] = None
        self.total_pages: int = 0
        self.worker: Optional[threading.Thread] = None
        self.update_thread: Optional[threading.Thread] = None
        self._available_update: Optional[UpdateInfo] = None
        self._last_output: Optional[Path] = None
        self._busy = False
        self._converting = False
        self._started_at = 0.0
        self._cancel_event = threading.Event()
        self._confirm_answer = False
        self._confirm_event = threading.Event()
        self.tooltips: dict[str, ToolTip] = {}
        self.preset_buttons: dict[str, ttk.Button] = {}
        self._cards: list[tk.Frame] = []
        self._trace_guard = False
        self._log_warning_count = 0
        self._log_error_count = 0
        self._elapsed_after_id: Optional[str] = None
        self._shortcut_bindings: list[tuple[str, str]] = []

        self.log_handler = QueueLogHandler(self.q)
        logging.getLogger(APP_NAME).addHandler(self.log_handler)

        self._build()
        self.apply_theme(self.theme_name)
        self._bind_shortcuts()
        self.pack(fill="both", expand=True)

        self._after_id: Optional[str] = self.after(self.POLL_MS, self._poll)
        if self.config_.check_updates:
            self._start_update_check()

    def _default_exit(self) -> None:  # pragma: no cover - terminates interactive Tk root
        try:
            self.master.quit()
            self.master.destroy()
        except Exception:
            pass

    def _attach_tip(self, key: str, *widgets: tk.Widget, custom_text: Optional[str] = None) -> None:
        text = custom_text if custom_text is not None else get_parameter_tooltip(key)
        for idx, w in enumerate(widgets):
            tip = ToolTip(w, text, palette=self.palette)
            if idx == 0:
                self.tooltips[key] = tip

    def _make_card(self, parent: tk.Misc, padx: int = 14, pady: int = 12) -> tuple[tk.Frame, ttk.Frame]:
        outer = tk.Frame(
            parent,
            background=self.palette.card,
            highlightbackground=self.palette.border,
            highlightthickness=1,
            padx=padx,
            pady=pady,
        )
        inner = ttk.Frame(outer, style="Card.TFrame")
        inner.pack(fill="both", expand=True)
        self._cards.append(outer)
        return outer, inner

    # ---------- UI Construction ----------
    def _build(self) -> None:
        self.master.title(f"PDF to MuseScore {__version__} (homr {self.engine_version}) — {AUTHOR_CREDIT}")
        self.master.minsize(780, 520)

        self._has_musescore = bool(self.musescore_available())
        self._build_variables()
        self._build_header()

        # Two-column workspace body (left preview, right cards)
        workspace = ttk.Frame(self, style="App.TFrame")
        workspace.pack(fill="x", pady=(6, 4))
        workspace.columnconfigure(1, weight=1)

        self.preview_pane = PdfPreviewPane(
            workspace,
            on_browse=self.on_load,
            on_toggle_page=self._on_toggle_preview_page,
            page_renderer=self.page_renderer,
            palette=self.palette,
        )
        self.preview_pane.grid(row=0, column=0, sticky="nw", padx=(0, 12))

        right_col = ttk.Frame(workspace, style="App.TFrame")
        right_col.grid(row=0, column=1, sticky="nsew")

        self._build_source_card(right_col)
        self._build_score_card(right_col)
        self._build_action_card(right_col)

        self._build_log_drawer()
        self._build_footer()

        self._attach_traces()
        self._sync_preset_indicators()
        self._sync_transpose_hint()
        self._sync_pages_feedback()

    def _build_variables(self) -> None:
        self.file_var = tk.StringVar(self, value="No PDF selected")
        self.file_meta_var = tk.StringVar(self, value="Choose a sheet music PDF to begin (Ctrl+O)")
        self.pages_var = tk.StringVar(self, value="")
        self.pages_feedback_var = tk.StringVar(self, value="Leave empty to convert all pages")
        self.preset_badge_var = tk.StringVar(self, value="")

        tr_val = self._prefs.transpose_semitones
        self.dpi_var = tk.StringVar(self, value=str(self._prefs.render_dpi))
        self.preprocess_var = tk.StringVar(self, value=self._prefs.preprocess_mode)
        self.deskew_var = tk.BooleanVar(self, value=self._prefs.deskew)
        self.step_size_var = tk.StringVar(self, value=str(self._prefs.segnet_step_size))
        self.min_notehead_var = tk.StringVar(self, value=str(self._prefs.min_notehead_size))

        self.transpose_var = tk.StringVar(self, value=f"+{tr_val}" if tr_val > 0 else str(tr_val))
        self.transpose_hint_var = tk.StringVar(self, value="Original pitch")
        self.clef_var = tk.StringVar(self, value=self._prefs.clef_override)
        self.instrument_var = tk.StringVar(self, value=self._prefs.instrument_name)
        self.keep_breaks_var = tk.BooleanVar(self, value=self._prefs.keep_line_breaks)

        self.status_var = tk.StringVar(self, value="Load a PDF with sheet music to start.")
        self.pct_var = tk.StringVar(self, value="0%")
        self.elapsed_var = tk.StringVar(self, value="")
        self.result_msg_var = tk.StringVar(self, value="")

    def _build_header(self) -> None:
        header = ttk.Frame(self, style="App.TFrame")
        header.pack(fill="x", pady=(0, 2))

        ttk.Label(
            header,
            text="Convert sheet music PDFs into editable MuseScore & MusicXML scores",
            style="HeaderSub.TLabel",
        ).pack(side="left", anchor="w")

        right = ttk.Frame(header, style="App.TFrame")
        right.pack(side="right", anchor="e")

        self.theme_btn = ttk.Button(
            right,
            text="🌙 Dark",
            style="Secondary.TButton",
            command=self.toggle_theme,
        )
        self.theme_btn.pack(side="right")

        # Packed dynamically only when a newer GitHub release is found
        self.update_btn = ttk.Button(
            right,
            text="",
            style="Update.TButton",
            command=self.on_update_clicked,
        )

    def _build_source_card(self, parent: ttk.Frame) -> None:
        card_outer, card = self._make_card(parent, padx=14, pady=9)
        card_outer.pack(fill="x", pady=(0, 8))

        top_bar = ttk.Frame(card, style="Card.TFrame")
        top_bar.pack(fill="x")
        ttk.Label(top_bar, text="1. SOURCE & SCAN PRESET", style="SectionHeader.TLabel").pack(side="left")
        self.preset_pill_lbl = ttk.Label(top_bar, textvariable=self.preset_badge_var, style="AccentHint.TLabel")
        self.preset_pill_lbl.pack(side="right")

        file_row = ttk.Frame(card, style="Card.TFrame")
        file_row.pack(fill="x", pady=(6, 6))
        # Pack button first on the right so long paths never squish the Load PDF button
        self.load_btn = ttk.Button(file_row, text="Load PDF...", style="Secondary.TButton", command=self.on_load)
        self.load_btn.pack(side="right", padx=(8, 0))
        file_info = ttk.Frame(file_row, style="Card.TFrame")
        file_info.pack(side="left", fill="x", expand=True)
        ttk.Label(file_info, textvariable=self.file_var, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(file_info, textvariable=self.file_meta_var, style="Hint.TLabel").pack(anchor="w", pady=(1, 0))

        # Pages selector + live feedback
        pages_row = ttk.Frame(card, style="Card.TFrame")
        pages_row.pack(fill="x", pady=(2, 8))
        pages_lbl = ttk.Label(pages_row, text="Pages:", style="Card.TLabel")
        pages_lbl.pack(side="left")
        self.pages_entry = ttk.Entry(pages_row, textvariable=self.pages_var, width=18, style="Studio.TEntry")
        self.pages_entry.pack(side="left", padx=(8, 10))
        self.pages_feedback_lbl = ttk.Label(pages_row, textvariable=self.pages_feedback_var, style="Hint.TLabel")
        self.pages_feedback_lbl.pack(side="left")
        self._attach_tip("pages", self.pages_entry, pages_lbl, custom_text=PAGES_TOOLTIP)

        # Scan type one-click presets
        preset_row = ttk.Frame(card, style="Card.TFrame")
        preset_row.pack(fill="x")
        ttk.Label(preset_row, text="Scan Type:", style="Card.TLabel").pack(side="left", padx=(0, 8))
        for preset in SCAN_PRESETS:
            btn = ttk.Button(
                preset_row,
                text=preset.label,
                style="Preset.TButton",
                command=lambda k=preset.key: self.apply_preset(k),
            )
            btn.pack(side="left", padx=(0, 6))
            self.preset_buttons[preset.key] = btn
            self._attach_tip(f"preset_{preset.key}", btn, custom_text=PRESET_TOOLTIPS[preset.key])

    def _build_score_card(self, parent: ttk.Frame) -> None:
        card_outer, card = self._make_card(parent, padx=14, pady=9)
        card_outer.pack(fill="x", pady=(0, 8))

        ttk.Label(card, text="2. SCORE SETUP", style="SectionHeader.TLabel").pack(anchor="w", pady=(0, 8))

        grid = ttk.Frame(card, style="Card.TFrame")
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)

        # Col 0: Instrument
        c0 = ttk.Frame(grid, style="Card.TFrame")
        c0.grid(row=0, column=0, sticky="new", padx=(0, 10))
        inst_lbl = ttk.Label(c0, text="Instrument", style="Card.TLabel")
        inst_lbl.pack(anchor="w", pady=(0, 3))
        self.instrument_combo = ttk.Combobox(
            c0,
            textvariable=self.instrument_var,
            values=list(INSTRUMENT_PRESETS),
            width=14,
            style="Studio.TCombobox",
        )
        self.instrument_combo.pack(fill="x")
        self._attach_tip("instrument_name", self.instrument_combo, inst_lbl)

        # Col 1: Clef
        c1 = ttk.Frame(grid, style="Card.TFrame")
        c1.grid(row=0, column=1, sticky="new", padx=(0, 10))
        clef_lbl = ttk.Label(c1, text="Clef", style="Card.TLabel")
        clef_lbl.pack(anchor="w", pady=(0, 3))
        self.clef_combo = ttk.Combobox(
            c1,
            textvariable=self.clef_var,
            values=list(CLEF_CHOICES),
            width=10,
            state="readonly",
            style="Studio.TCombobox",
        )
        self.clef_combo.pack(fill="x")
        self._attach_tip("clef_override", self.clef_combo, clef_lbl)

        # Col 2: Transpose stepper
        c2 = ttk.Frame(grid, style="Card.TFrame")
        c2.grid(row=0, column=2, sticky="new")
        tr_lbl = ttk.Label(c2, text="Transpose (Half-Steps)", style="Card.TLabel")
        tr_lbl.pack(anchor="w", pady=(0, 3))
        tr_ctrl = ttk.Frame(c2, style="Card.TFrame")
        tr_ctrl.pack(fill="x")
        self.tr_minus_btn = ttk.Button(
            tr_ctrl, text="−", width=2, style="Stepper.TButton", command=lambda: self.step_transpose(-1)
        )
        self.tr_minus_btn.pack(side="left")
        self.transpose_entry = ttk.Entry(
            tr_ctrl, textvariable=self.transpose_var, width=5, justify="center", style="Studio.TEntry"
        )
        self.transpose_entry.pack(side="left", padx=4)
        self.tr_plus_btn = ttk.Button(
            tr_ctrl, text="+", width=2, style="Stepper.TButton", command=lambda: self.step_transpose(1)
        )
        self.tr_plus_btn.pack(side="left")
        ttk.Label(c2, textvariable=self.transpose_hint_var, style="Hint.TLabel").pack(anchor="w", pady=(2, 0))
        self._attach_tip("transpose_semitones", self.transpose_entry, tr_lbl)

        # Collapsible Advanced AI & Layout Drawer (compact 2-column grid to keep card narrow)
        ttk.Separator(card, orient="horizontal", style="Studio.TSeparator").pack(fill="x", pady=(8, 6))
        self.advanced_section = CollapsibleSection(
            card,
            title="Advanced AI & Layout Settings",
            expanded=self._prefs.advanced_expanded,
            palette=self.palette,
            card_surface=True,
            on_toggle=self._on_drawer_toggled,
        )
        self.advanced_section.pack(fill="x")
        adv = self.advanced_section.content

        adv_grid = ttk.Frame(adv, style="Card.TFrame")
        adv_grid.pack(fill="x")
        adv_grid.columnconfigure(0, weight=1)
        adv_grid.columnconfigure(1, weight=1)

        # Row 0: Scan Quality (DPI) | Page Cleanup
        f_dpi = ttk.Frame(adv_grid, style="Card.TFrame")
        f_dpi.grid(row=0, column=0, sticky="ew", padx=(0, 10), pady=(0, 5))
        dpi_lbl = ttk.Label(f_dpi, text="Scan Quality (DPI):", style="Card.TLabel")
        dpi_lbl.pack(side="left")
        self.dpi_entry = ttk.Entry(f_dpi, textvariable=self.dpi_var, width=6, style="Studio.TEntry")
        self.dpi_entry.pack(side="right")
        self._attach_tip("render_dpi", self.dpi_entry, dpi_lbl)

        f_prep = ttk.Frame(adv_grid, style="Card.TFrame")
        f_prep.grid(row=0, column=1, sticky="ew", pady=(0, 5))
        prep_lbl = ttk.Label(f_prep, text="Page Cleanup:", style="Card.TLabel")
        prep_lbl.pack(side="left")
        self.preprocess_combo = ttk.Combobox(
            f_prep,
            textvariable=self.preprocess_var,
            values=list(PREPROCESS_MODES),
            width=10,
            state="readonly",
            style="Studio.TCombobox",
        )
        self.preprocess_combo.pack(side="right")
        self._attach_tip("preprocess_mode", self.preprocess_combo, prep_lbl)

        # Row 1: Reading Thoroughness | Smallest Note Size
        f_step = ttk.Frame(adv_grid, style="Card.TFrame")
        f_step.grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(0, 5))
        step_lbl = ttk.Label(f_step, text="Reading Thoroughness (px):", style="Card.TLabel")
        step_lbl.pack(side="left")
        self.step_entry = ttk.Entry(f_step, textvariable=self.step_size_var, width=6, style="Studio.TEntry")
        self.step_entry.pack(side="right")
        self._attach_tip("segnet_step_size", self.step_entry, step_lbl)

        f_nh = ttk.Frame(adv_grid, style="Card.TFrame")
        f_nh.grid(row=1, column=1, sticky="ew", pady=(0, 5))
        nh_lbl = ttk.Label(f_nh, text="Smallest Note Size (px):", style="Card.TLabel")
        nh_lbl.pack(side="left")
        self.notehead_entry = ttk.Entry(f_nh, textvariable=self.min_notehead_var, width=6, style="Studio.TEntry")
        self.notehead_entry.pack(side="right")
        self._attach_tip("min_notehead_size", self.notehead_entry, nh_lbl)

        # Row 2: Deskew checkbox | Keep Line Breaks + Reset Defaults
        self.deskew_check = ttk.Checkbutton(
            adv_grid, text="Straighten Tilted Page", variable=self.deskew_var, style="Studio.TCheckbutton"
        )
        self.deskew_check.grid(row=2, column=0, sticky="w", padx=(0, 10))
        self._attach_tip("deskew", self.deskew_check)

        f_brk = ttk.Frame(adv_grid, style="Card.TFrame")
        f_brk.grid(row=2, column=1, sticky="ew")
        self.keep_breaks_check = ttk.Checkbutton(
            f_brk,
            text="Keep Line Breaks",
            variable=self.keep_breaks_var,
            style="Studio.TCheckbutton",
        )
        self.keep_breaks_check.pack(side="left", padx=(0, 12))
        self._attach_tip("keep_line_breaks", self.keep_breaks_check)

        self.reset_opts_btn = ttk.Button(
            f_brk,
            text="Reset",
            style="Secondary.TButton",
            command=self.on_reset_options,
        )
        self.reset_opts_btn.pack(side="right", padx=(8, 0))

        self._option_widgets = (
            self.dpi_entry,
            self.preprocess_combo,
            self.deskew_check,
            self.step_entry,
            self.notehead_entry,
            self.keep_breaks_check,
            self.transpose_entry,
            self.tr_minus_btn,
            self.tr_plus_btn,
            self.clef_combo,
            self.instrument_combo,
            self.reset_opts_btn,
        )

    def _build_action_card(self, parent: ttk.Frame) -> None:
        card_outer, card = self._make_card(parent, padx=14, pady=12)
        card_outer.pack(fill="x")

        btn_row = ttk.Frame(card, style="Card.TFrame")
        btn_row.pack(fill="x")

        cta_label = self._primary_cta_label()
        self.convert_btn = ttk.Button(
            btn_row,
            text=cta_label,
            style="Primary.TButton",
            command=self.on_convert,
            state="disabled",
        )
        self.convert_btn.pack(side="left", fill="x", expand=True)

        self.abort_btn = ttk.Button(
            btn_row,
            text="Abort",
            style="Danger.TButton",
            command=self.on_abort,
            state="disabled",
        )
        self.abort_btn.pack(side="left", padx=(8, 0))

        self.progress = ttk.Progressbar(card, maximum=100, style="Studio.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(10, 4))

        status_row = ttk.Frame(card, style="Card.TFrame")
        status_row.pack(fill="x")
        ttk.Label(status_row, textvariable=self.status_var, style="Muted.TLabel").pack(side="left")
        ttk.Label(status_row, textvariable=self.pct_var, style="Card.TLabel").pack(side="right")
        ttk.Label(status_row, textvariable=self.elapsed_var, style="Hint.TLabel").pack(side="right", padx=(0, 8))

        # Inline success banner (hidden until a conversion completes)
        self.result_banner = tk.Frame(
            card,
            background=self.palette.success_bg,
            highlightbackground=self.palette.success_border,
            highlightthickness=1,
            padx=10,
            pady=6,
        )
        self.reveal_result_btn = ttk.Button(
            self.result_banner,
            text="Show Folder",
            style="Secondary.TButton",
            command=self.on_reveal_result,
        )
        self.reveal_result_btn.pack(side="right", padx=(6, 0))

        self.open_result_btn = ttk.Button(
            self.result_banner,
            text="Open in MuseScore",
            style="Success.TButton",
            command=self.on_open_result,
        )
        self.open_result_btn.pack(side="right", padx=(8, 0))

        self.result_text_lbl = ttk.Label(
            self.result_banner,
            textvariable=self.result_msg_var,
            style="BannerText.TLabel",
        )
        self.result_text_lbl.pack(side="left", fill="x", expand=True)

    def _build_log_drawer(self) -> None:
        self.log_section = CollapsibleSection(
            self,
            title="Activity Log",
            expanded=self._prefs.log_expanded,
            palette=self.palette,
            card_surface=False,
            on_toggle=self._on_drawer_toggled,
        )
        self.log_section.pack(fill="both", expand=True, pady=(4, 0))

        self.log_frame = tk.Frame(
            self.log_section.content,
            background=self.palette.log_bg,
            highlightbackground=self.palette.border_strong,
            highlightthickness=1,
        )
        self.log_frame.pack(fill="both", expand=True)

        self.log_scroll = ttk.Scrollbar(
            self.log_frame,
            orient="vertical",
            style="Studio.Vertical.TScrollbar",
        )
        self.log_scroll.pack(side="right", fill="y")

        self.log_text = tk.Text(
            self.log_frame,
            height=4,
            state="disabled",
            font=(self._mono_font, 8),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=8,
            pady=6,
            yscrollcommand=self.log_scroll.set,
        )
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_scroll.configure(command=self.log_text.yview)

    def _build_footer(self) -> None:
        footer = ttk.Frame(self, style="App.TFrame")
        footer.pack(fill="x", pady=(8, 0))

        self.credit_label = ttk.Label(footer, text=AUTHOR_CREDIT, style="Footer.TLabel")
        self.credit_label.pack(side="left")

        has_ms = self._has_musescore
        ms_text = "● MuseScore detected" if has_ms else "○ MusicXML export mode"
        ms_style = "FooterBadgeOk.TLabel" if has_ms else "FooterBadgeMuted.TLabel"
        self.ms_badge_lbl = ttk.Label(footer, text=ms_text, style=ms_style)
        self.ms_badge_lbl.pack(side="left", padx=(12, 0))

        self.logs_btn = ttk.Button(footer, text="Open log folder", style="Ghost.TButton", command=self.on_open_logs)
        self.logs_btn.pack(side="right")

    def _primary_cta_label(self) -> str:
        if self._has_musescore:
            return "♪  Convert & Save as MuseScore (.mscz)..."
        return "♪  Convert & Save as MusicXML (.musicxml)..."

    # ---------- Traces & Reactive UI State ----------
    def _attach_traces(self) -> None:
        for var in (
            self.dpi_var,
            self.preprocess_var,
            self.deskew_var,
            self.step_size_var,
            self.min_notehead_var,
            self.keep_breaks_var,
        ):
            var.trace_add("write", lambda *_a: self._sync_preset_indicators())

        self.transpose_var.trace_add("write", lambda *_a: self._sync_transpose_hint())
        self.pages_var.trace_add("write", lambda *_a: self._sync_pages_feedback())

    def _sync_preset_indicators(self) -> None:
        if self._trace_guard:
            return
        matched: Optional[str] = None
        adv_customized = False
        try:
            opts = self.collect_options()
            matched = detect_scan_preset(opts)
            defaults = ConversionOptions()
            adv_customized = any(
                (
                    opts.render_dpi != defaults.render_dpi,
                    opts.preprocess_mode != defaults.preprocess_mode,
                    opts.deskew != defaults.deskew,
                    opts.segnet_step_size != defaults.segnet_step_size,
                    opts.min_notehead_size != defaults.min_notehead_size,
                    opts.keep_line_breaks != defaults.keep_line_breaks,
                )
            )
        except OptionsError:
            adv_customized = True

        for key, btn in self.preset_buttons.items():
            btn.configure(style="PresetActive.TButton" if key == matched else "Preset.TButton")

        self.preset_badge_var.set("" if matched is not None else "Custom")

        self.advanced_section.set_badge("Customized" if adv_customized else "", tone="accent")

    def _sync_transpose_hint(self) -> None:
        raw = self.transpose_var.get().strip()
        try:
            semi = int(raw)
        except ValueError:
            self.transpose_hint_var.set("Enter integer (-24..+24)")
            return
        self.transpose_hint_var.set(format_transpose_label(semi))

    def _sync_pages_feedback(self) -> None:
        spec = self.pages_var.get().strip()
        if not self.pdf or self.total_pages <= 0:
            self.pages_feedback_var.set(f"Range: {spec}" if spec else "Leave empty to convert all pages")
            self.pages_feedback_lbl.configure(style="Hint.TLabel")
            self.preview_pane.update_selection_state(None)
            return

        if not spec or spec.lower() == "all":
            self.pages_feedback_var.set(
                f"Converting all {self.total_pages} page{'s' if self.total_pages != 1 else ''}"
            )
            self.pages_feedback_lbl.configure(style="Hint.TLabel")
            self.preview_pane.update_selection_state(None)
            return

        try:
            selected = parse_page_spec(spec, self.total_pages)
        except Exception as exc:
            self.pages_feedback_var.set(str(exc))
            self.pages_feedback_lbl.configure(style="ErrorHint.TLabel")
            self.preview_pane.update_selection_state(None)
            return

        self.pages_feedback_var.set(f"Converting {len(selected)} of {self.total_pages} page(s)")
        self.pages_feedback_lbl.configure(style="AccentHint.TLabel")
        self.preview_pane.update_selection_state(set(selected))

    def _on_toggle_preview_page(self, page: int) -> None:
        if not self.pdf or self.total_pages <= 0 or self._busy:
            return
        try:
            new_spec = toggle_page_in_spec(self.pages_var.get(), page, self.total_pages)
        except Exception:
            new_spec = str(page)
        self.pages_var.set(new_spec)

    # ---------- Theme & Shortcuts ----------
    def apply_theme(self, name: str) -> None:
        self.theme_name = "dark" if (name or "").strip().lower() == "dark" else "light"
        p = get_theme(self.theme_name)
        self.palette = p

        configure_ttk_style(self._style, p, self._ui_font)
        try:
            self.master.configure(background=p.bg)
            self.master.option_add("*TCombobox*Listbox.background", p.input_bg)
            self.master.option_add("*TCombobox*Listbox.foreground", p.text)
            self.master.option_add("*TCombobox*Listbox.selectBackground", p.accent)
            self.master.option_add("*TCombobox*Listbox.selectForeground", p.accent_text)
        except Exception:
            pass
        apply_windows_titlebar_theme(self.master, p.is_dark)

        for card in self._cards:
            card.configure(background=p.card, highlightbackground=p.border)

        self.preview_pane.apply_theme(p)
        self.advanced_section.apply_theme(p)
        self.log_section.apply_theme(p)
        self.result_banner.configure(background=p.success_bg, highlightbackground=p.success_border)
        self.log_frame.configure(background=p.log_bg, highlightbackground=p.border_strong)

        self.log_text.configure(
            background=p.log_bg,
            foreground=p.log_fg,
            insertbackground=p.log_fg,
        )
        self.log_text.tag_configure("DEBUG", foreground=p.log_debug)
        self.log_text.tag_configure("INFO", foreground=p.log_fg)
        self.log_text.tag_configure("WARNING", foreground=p.log_warn)
        self.log_text.tag_configure("ERROR", foreground=p.log_err)
        self.log_text.tag_configure("CRITICAL", foreground=p.log_err)

        for tip in self.tooltips.values():
            tip.set_palette(p)

        self.theme_btn.configure(text="☀ Light" if p.is_dark else "🌙 Dark")
        self._sync_preset_indicators()

    def toggle_theme(self) -> None:
        self.apply_theme("light" if self.theme_name == "dark" else "dark")
        self.save_preferences()

    def _on_drawer_toggled(self, _expanded: bool) -> None:
        self._fit_window_to_content()
        self.save_preferences()

    @staticmethod
    def _safe_int(raw: str, lo: int, hi: int, fallback: int) -> int:
        try:
            val = int(raw.strip())
        except ValueError:
            return fallback
        return val if lo <= val <= hi else fallback

    def snapshot_preferences(self) -> UserPreferences:
        """Capture current UI settings cleanly, keeping previous valid values if an entry is mid-edit."""
        prep = (self.preprocess_var.get() or "auto").strip().lower()
        if prep not in PREPROCESS_MODES:
            prep = self._prefs.preprocess_mode
        clef = (self.clef_var.get() or "auto").strip().lower()
        if clef not in CLEF_CHOICES:
            clef = self._prefs.clef_override
        inst = (self.instrument_var.get() or "").strip() or "auto"
        return UserPreferences(
            theme=self.theme_name,
            advanced_expanded=bool(self.advanced_section.is_expanded),
            log_expanded=bool(self.log_section.is_expanded),
            render_dpi=self._safe_int(self.dpi_var.get(), 100, 600, self._prefs.render_dpi),
            preprocess_mode=prep,
            deskew=bool(self.deskew_var.get()),
            segnet_step_size=self._safe_int(self.step_size_var.get(), 80, 320, self._prefs.segnet_step_size),
            min_notehead_size=self._safe_int(self.min_notehead_var.get(), 2, 12, self._prefs.min_notehead_size),
            transpose_semitones=self._safe_int(self.transpose_var.get(), -24, 24, self._prefs.transpose_semitones),
            clef_override=clef,
            instrument_name=inst,
            keep_line_breaks=bool(self.keep_breaks_var.get()),
            last_dir=self._last_dir,
        )

    def save_preferences(self) -> bool:
        """Persist current user preferences cleanly to disk."""
        prefs = self.snapshot_preferences()
        self._prefs = prefs
        return write_preferences(prefs, self._prefs_path)

    def _bind_shortcuts(self) -> None:
        def bind(seq: str, fn: Callable[[tk.Event], Optional[str]]) -> None:
            fid = self.master.bind(seq, fn, add="+")
            if fid:
                self._shortcut_bindings.append((seq, fid))

        bind("<Control-o>", self._on_shortcut_load)
        bind("<Control-O>", self._on_shortcut_load)
        bind("<Control-Return>", self._on_shortcut_convert)
        bind("<Escape>", self._on_shortcut_abort)
        bind("<Left>", lambda e: self._on_shortcut_page(e, -1))
        bind("<Right>", lambda e: self._on_shortcut_page(e, +1))

    def _on_shortcut_load(self, _event: tk.Event) -> Optional[str]:
        if not self._busy:
            self.on_load()
            return "break"
        return None

    def _on_shortcut_convert(self, _event: tk.Event) -> Optional[str]:
        if not self._busy and self.pdf is not None:
            self.on_convert()
            return "break"
        return None

    def _on_shortcut_abort(self, _event: tk.Event) -> Optional[str]:
        if self._busy:
            self.on_abort()
            return "break"
        return None

    def _on_shortcut_page(self, event: tk.Event, delta: int) -> Optional[str]:
        widget = getattr(event, "widget", None)
        if widget is not None:
            try:
                if widget.winfo_class() in _TEXT_INPUT_WIDGET_CLASSES:
                    return None
            except Exception:
                pass
        if delta < 0:
            self.preview_pane.go_prev()
        else:
            self.preview_pane.go_next()
        return "break"

    # ---------- Options, Presets & Steppers ----------
    def apply_preset(self, preset_key: str) -> None:
        preset = get_scan_preset(preset_key)
        self._trace_guard = True
        try:
            self.dpi_var.set(str(preset.render_dpi))
            self.preprocess_var.set(preset.preprocess_mode)
            self.deskew_var.set(preset.deskew)
            self.step_size_var.set(str(preset.segnet_step_size))
            self.min_notehead_var.set(str(preset.min_notehead_size))
        finally:
            self._trace_guard = False
        self._sync_preset_indicators()
        self.save_preferences()

    def step_transpose(self, delta: int) -> None:
        if self._busy:
            return
        try:
            current = int(self.transpose_var.get().strip())
        except ValueError:
            current = 0
        clamped = max(-24, min(24, current + delta))
        self.transpose_var.set(f"+{clamped}" if clamped > 0 else str(clamped))
        self.save_preferences()

    def on_reset_options(self) -> None:
        self._trace_guard = True
        try:
            self.dpi_var.set(str(self.config_.render_dpi))
            self.preprocess_var.set("auto")
            self.deskew_var.set(False)
            self.step_size_var.set("320")
            self.min_notehead_var.set("4")
            self.keep_breaks_var.set(False)
            self.transpose_var.set("0")
            self.clef_var.set("auto")
            self.instrument_var.set("auto")
        finally:
            self._trace_guard = False
        self._sync_preset_indicators()
        self._sync_transpose_hint()
        self.save_preferences()

    def collect_options(self) -> ConversionOptions:
        try:
            dpi = int(self.dpi_var.get().strip())
        except ValueError as exc:
            raise OptionsError(f"Render DPI must be an integer (e.g. 300), got {self.dpi_var.get()!r}") from exc
        try:
            step = int(self.step_size_var.get().strip())
        except ValueError as exc:
            raise OptionsError(
                f"SegNet Tile Step must be an integer (e.g. 320), got {self.step_size_var.get()!r}"
            ) from exc
        try:
            min_nh = int(self.min_notehead_var.get().strip())
        except ValueError as exc:
            raise OptionsError(
                f"Min Notehead Size must be an integer (e.g. 4), got {self.min_notehead_var.get()!r}"
            ) from exc
        try:
            transpose = int(self.transpose_var.get().strip())
        except ValueError as exc:
            raise OptionsError(
                f"Transpose semitones must be an integer (e.g. 0 or +2), got {self.transpose_var.get()!r}"
            ) from exc

        return ConversionOptions(
            render_dpi=dpi,
            preprocess_mode=self.preprocess_var.get(),
            deskew=bool(self.deskew_var.get()),
            segnet_step_size=step,
            min_notehead_size=min_nh,
            transpose_semitones=transpose,
            clef_override=self.clef_var.get(),
            instrument_name=self.instrument_var.get(),
            keep_line_breaks=bool(self.keep_breaks_var.get()),
        )

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.load_btn.configure(state=state)
        self.pages_entry.configure(state=state)
        for btn in self.preset_buttons.values():
            btn.configure(state=state)
        for w in self._option_widgets:
            if w in (self.preprocess_combo, self.clef_combo):
                w.configure(state="disabled" if busy else "readonly")
            else:
                w.configure(state=state)
        self.convert_btn.configure(state="disabled" if busy or not self.pdf else "normal")
        self.abort_btn.configure(state="normal" if busy else "disabled")
        self.preview_pane.set_busy(busy)
        if self._available_update is not None:
            self.update_btn.configure(state=state)
        if not busy:
            self._converting = False
            if self._elapsed_after_id:
                try:
                    self.after_cancel(self._elapsed_after_id)
                except Exception:
                    pass
                self._elapsed_after_id = None

    def _tick_elapsed(self) -> None:
        if not self._busy or not self._converting:
            self._elapsed_after_id = None
            return
        self.elapsed_var.set(format_elapsed(time.monotonic() - self._started_at))
        self._elapsed_after_id = self.after(1000, self._tick_elapsed)

    def _fit_window_to_content(self) -> None:
        """Gently grow top-level window height if expanded drawers exceed current height."""
        try:
            self.master.update_idletasks()
            state = self.master.wm_state()
            if state == "withdrawn":
                return
            req_w = self.master.winfo_reqwidth()
            req_h = self.master.winfo_reqheight()
            cur_w = self.master.winfo_width()
            cur_h = self.master.winfo_height()
            max_w = max(820, self.master.winfo_screenwidth() - 60)
            max_h = max(560, self.master.winfo_screenheight() - 80)
            target_w = min(max_w, max(cur_w, req_w))
            target_h = min(max_h, max(cur_h, req_h))
            if target_w != cur_w or target_h != cur_h:
                self.master.geometry(f"{target_w}x{target_h}")
        except Exception:
            pass

    def _hide_result_banner(self) -> None:
        if self.result_banner.winfo_manager():
            self.result_banner.pack_forget()

    def _show_result_banner(self, out_path: Path) -> None:
        self._last_output = out_path
        short_name = out_path.name if len(out_path.name) <= 28 else out_path.name[:25] + "..."
        self.result_msg_var.set(f"✓  Saved {short_name}")
        is_mscz = out_path.suffix.lower() == ".mscz"
        self.open_result_btn.configure(
            text="Open in MuseScore" if (is_mscz or self.musescore_available()) else "Open Score"
        )
        if not self.result_banner.winfo_manager():
            self.result_banner.pack(fill="x", pady=(8, 0))
            self._fit_window_to_content()

    # ---------- Actions & Updates ----------
    def _start_update_check(self) -> None:
        def check() -> None:
            try:
                info = self.update_checker(__version__)
            except Exception as exc:
                log.warning("Update check failed (continuing offline): %s", exc)
                return
            if info is not None:
                self.q.put(("update_available", info))

        self.update_thread = threading.Thread(target=check, daemon=True, name="update-check")
        self.update_thread.start()

    def on_update_clicked(self) -> None:
        info = self._available_update
        if info is None or self._busy:
            return
        yes = bool(
            self.ask_yes_no(
                "Update available",
                f"A newer version of PDF to MuseScore (v{info.version}) is available "
                f"(you have v{__version__}).\n\nWould you like to download and install it now?",
            )
        )
        if not yes:
            log.info("Update to v%s deferred by user.", info.version)
            return
        if not info.download_url.lower().endswith(".exe"):
            log.info("Opening release page: %s", info.release_url)
            self.open_url(info.release_url)
            return

        self._set_busy(True)
        self.abort_btn.configure(state="disabled")
        self.progress["value"] = 0
        self.pct_var.set("0%")
        self.elapsed_var.set("")
        self.status_var.set(f"Downloading update v{info.version}...")
        dest_dir = Path(self.config_.work_dir).resolve().parent / "updates"

        def download_work() -> None:
            try:
                installer = self.installer_downloader(
                    info,
                    dest_dir,
                    progress=lambda p, m: self.q.put(("progress", (p, m))),
                )
                self.q.put(("update_ready", installer))
            except UpdateError as exc:
                log.error("Update download failed: %s", exc)
                self.q.put(("update_error", str(exc)))
            except Exception as exc:
                log.exception("Unexpected error during update download")
                self.q.put(("update_error", f"Unexpected error: {exc}"))

        self.worker = threading.Thread(target=download_work, daemon=True, name="update-download")
        self.worker.start()

    def on_load(self) -> None:
        open_kwargs: dict[str, object] = {
            "title": "Choose a PDF",
            "filetypes": [("PDF files", "*.pdf")],
        }
        if self._last_dir and Path(self._last_dir).is_dir():
            open_kwargs["initialdir"] = self._last_dir
        path = self.ask_open(**open_kwargs)
        if not path:
            log.debug("Load cancelled")
            return
        self.pdf = Path(path)
        self._last_dir = str(self.pdf.parent)
        self._hide_result_banner()
        parent_str = str(self.pdf.parent)
        if len(parent_str) > 46:
            parent_str = "..." + parent_str[-43:]
        try:
            n_pages = int(self.page_counter(self.pdf))
            self.total_pages = max(1, n_pages)
            self.file_var.set(f"{self.pdf.name} ({n_pages} page{'s' if n_pages != 1 else ''})")
            self.file_meta_var.set(parent_str)
        except Exception:
            self.total_pages = 1
            self.file_var.set(self.pdf.name)
            self.file_meta_var.set(parent_str)

        self.preview_pane.set_pdf(self.pdf, total_pages=self.total_pages)
        self._sync_pages_feedback()
        self.progress["value"] = 0
        self.pct_var.set("0%")
        self.elapsed_var.set("")
        self.status_var.set("Ready. Pick a preset or click Convert & Save.")
        log.info("Selected PDF: %s", self.pdf)
        self.save_preferences()
        self._set_busy(False)

    def on_convert(self) -> None:
        if not self.pdf or self._busy:
            return
        try:
            options = self.collect_options()
        except OptionsError as exc:
            self.show_error("Invalid parameter", str(exc))
            return
        self.save_preferences()
        types = [MSCZ_TYPE, XML_TYPE] if self.musescore_available() else [XML_TYPE, MSCZ_TYPE]
        ext = types[0][1][1:]
        out = self.ask_save(
            title="Save as",
            initialfile=self.pdf.stem + ext,
            defaultextension=ext,
            filetypes=types,
        )
        if not out:
            log.debug("Save cancelled")
            return
        self.start_conversion(self.pdf, Path(out), self.pages_var.get().strip() or None, options=options)

    def on_abort(self) -> None:
        log.info("Abort requested by user...")
        self._cancel_event.set()
        self.abort_btn.configure(state="disabled")
        self.status_var.set("Aborting conversion (stopping after current step)...")

    def start_conversion(
        self,
        pdf: Path,
        out: Path,
        pages_spec: Optional[str] = None,
        options: Optional[ConversionOptions] = None,
    ) -> None:
        self._cancel_event.clear()
        self._hide_result_banner()
        self._converting = True
        self._started_at = time.monotonic()
        self._set_busy(True)
        self.progress["value"] = 0
        self.pct_var.set("0%")
        self.elapsed_var.set("0:00")
        self._tick_elapsed()

        pipeline = self.pipeline_factory(
            config=self.config_,
            confirm=self._confirm_from_worker,
            progress=lambda p, m: self.q.put(("progress", (p, m))),
            is_cancelled=self._cancel_event.is_set,
        )

        def work() -> None:
            try:
                self.q.put(("done", pipeline.convert(pdf, out, pages_spec=pages_spec, options=options)))
            except CancelledError:
                self.q.put(("aborted", None))
            except ConversionError as exc:
                log.error("Conversion failed: %s", exc)
                self.q.put(("error", exc.user_message))
            except Exception as exc:  # unexpected - log full traceback
                log.exception("Unexpected error")
                self.q.put(("error", f"Unexpected error: {exc}"))

        self.worker = threading.Thread(target=work, daemon=True, name="conversion")
        self.worker.start()

    def _confirm_from_worker(self, message: str) -> bool:
        self._confirm_event.clear()
        self.q.put(("confirm", message))
        self._confirm_event.wait()
        return self._confirm_answer

    def on_open_logs(self) -> None:
        path = Path(self.config_.log_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        self.open_folder(str(path))

    def on_open_result(self) -> None:
        if self._last_output is None:
            return
        self.open_file(str(self._last_output))

    def on_reveal_result(self) -> None:
        if self._last_output is None:
            return
        self.reveal_file(self._last_output)

    # ---------- Queue Processing (GUI Thread) ----------
    def _append_log(self, payload: object) -> None:
        if isinstance(payload, tuple) and len(payload) == 2:
            levelno, text = int(payload[0]), str(payload[1])
        else:
            levelno, text = logging.INFO, str(payload)

        if levelno >= logging.ERROR:
            tag = "ERROR"
            self._log_error_count += 1
        elif levelno >= logging.WARNING:
            tag = "WARNING"
            self._log_warning_count += 1
        elif levelno <= logging.DEBUG:
            tag = "DEBUG"
        else:
            tag = "INFO"

        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n", (tag,))
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

        if self._log_error_count > 0:
            self.log_section.set_badge(f"{self._log_error_count} error(s)", tone="danger")
        elif self._log_warning_count > 0:
            self.log_section.set_badge(f"{self._log_warning_count} warning(s)", tone="warn")

    def process_queue(self) -> None:
        while True:
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                return
            if kind == "log":
                self._append_log(payload)
            elif kind == "progress":
                pct, msg = payload
                self.progress["value"] = pct
                self.pct_var.set(f"{int(pct)}%")
                self.status_var.set(msg)
            elif kind == "confirm":
                self._confirm_answer = bool(
                    self.ask_yes_no(
                        "Install required program",
                        payload + "\n\nWindows may ask for permission to make changes. Internet is required.",
                    )
                )
                self._confirm_event.set()
            elif kind == "done":
                out_path = Path(payload)
                self._set_busy(False)
                self.progress["value"] = 100
                self.pct_var.set("100%")
                self.status_var.set(f"Done! Saved {out_path.name}")
                self._show_result_banner(out_path)
            elif kind == "aborted":
                self._set_busy(False)
                self.progress["value"] = 0
                self.pct_var.set("0%")
                self.status_var.set("Conversion aborted by user.")
            elif kind == "error":
                self._set_busy(False)
                self.progress["value"] = 0
                self.pct_var.set("0%")
                self.status_var.set("Conversion failed. See Activity Log for details.")
                self.log_section.set_expanded(True)
                self.show_error("Conversion failed", payload)
            elif kind == "update_available":
                self._available_update = payload
                self.update_btn.configure(
                    text=f"↑ Update v{payload.version}",
                    state="disabled" if self._busy else "normal",
                )
                if not self.update_btn.winfo_manager():
                    self.update_btn.pack(side="right", padx=(0, 8))
                if not self._busy:
                    self.on_update_clicked()
            elif kind == "update_ready":
                ver = self._available_update.version if self._available_update else ""
                self.progress["value"] = 100
                self.pct_var.set("100%")
                self.status_var.set(f"Launching installer for v{ver}...")
                log.info("Launching downloaded installer: %s", payload)
                self.launch_installer(str(payload))
                self._on_exit()
            elif kind == "update_error":
                self._set_busy(False)
                self.progress["value"] = 0
                self.pct_var.set("0%")
                self.status_var.set("Update failed. See Activity Log for details.")
                self.log_section.set_expanded(True)
                self.show_error("Update failed", payload)

    def _poll(self) -> None:
        self.process_queue()
        self._after_id = self.after(self.POLL_MS, self._poll)

    def destroy(self) -> None:
        try:
            self.save_preferences()
        except Exception:
            pass
        for tip in self.tooltips.values():
            tip.hide()
        for aid_name in ("_after_id", "_elapsed_after_id"):
            aid = getattr(self, aid_name, None)
            if aid:
                try:
                    self.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, aid_name, None)
        for seq, fid in self._shortcut_bindings:
            try:
                self.master.unbind(seq, fid)
            except Exception:
                pass
        self._shortcut_bindings.clear()
        logging.getLogger(APP_NAME).removeHandler(self.log_handler)
        super().destroy()


def run_gui(config: AppConfig) -> None:  # pragma: no cover - interactive main loop
    enable_high_dpi()
    root = tk.Tk()
    assets = get_assets_dir()
    ico_path = assets / "icon.ico"
    png_path = assets / "icon.png"
    try:
        if ico_path.is_file():
            root.iconbitmap(str(ico_path))
        if png_path.is_file():
            icon_photo = ImageTk.PhotoImage(Image.open(png_path), master=root)
            root.iconphoto(True, icon_photo)
            root._app_icon_ref = icon_photo  # keep alive on root
    except Exception as exc:  # pragma: no cover
        log.debug("Window icon setup skipped: %s", exc)

    app = App(root, config)

    def on_window_close() -> None:
        app.save_preferences()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_window_close)
    root.update_idletasks()
    scr_w = root.winfo_screenwidth()
    scr_h = root.winfo_screenheight()
    width = min(max(920, root.winfo_reqwidth() + 8), max(820, scr_w - 60))
    height = min(max(620, root.winfo_reqheight() + 8), max(540, scr_h - 80))
    root.minsize(min(width, root.winfo_reqwidth()), min(height, root.winfo_reqheight()))
    x = max(0, (scr_w - width) // 2)
    y = max(0, (scr_h - height) // 2 - 20)
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()
