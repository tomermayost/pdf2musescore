"""Tkinter desktop window: Load PDF -> Pages & Engine/Score Options -> Convert & Save as / Abort."""

from __future__ import annotations

import logging
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Callable, Optional

from . import APP_NAME, AUTHOR_CREDIT, __version__
from .config import AppConfig
from .deps import MUSESCORE, find_tool
from .engines.homr_engine import CancelledError, get_pdf_page_count
from .options import (
    CLEF_CHOICES,
    INSTRUMENT_PRESETS,
    PREPROCESS_MODES,
    ConversionOptions,
    OptionsError,
    format_parameter_guide,
)
from .pipeline import ConversionError, Pipeline

log = logging.getLogger(__name__)

MSCZ_TYPE = ("MuseScore file", "*.mscz")
XML_TYPE = ("MusicXML file", "*.musicxml")


class QueueLogHandler(logging.Handler):
    """Forwards log records to the GUI thread via a queue."""

    def __init__(self, q: "queue.Queue") -> None:
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put(("log", self.format(record)))


class App(ttk.Frame):
    POLL_MS = 100

    def __init__(self, master: tk.Misc, config: AppConfig,
                 pipeline_factory: Callable[..., Pipeline] = Pipeline,
                 musescore_available: Optional[Callable[[], bool]] = None,
                 page_counter: Callable[[Path], int] = get_pdf_page_count,
                 ask_open: Callable[..., str] = filedialog.askopenfilename,
                 ask_save: Callable[..., str] = filedialog.asksaveasfilename,
                 ask_yes_no: Callable[[str, str], bool] = messagebox.askyesno,
                 show_error: Callable[[str, str], None] = messagebox.showerror,
                 show_info: Callable[[str, str], None] = messagebox.showinfo,
                 open_folder: Callable[[str], None] = getattr(os, "startfile", lambda p: None)) -> None:
        super().__init__(master, padding=10)
        self.config_ = config
        self.pipeline_factory = pipeline_factory
        self.musescore_available = musescore_available or (
            lambda: find_tool(MUSESCORE, config.musescore_path) is not None)
        self.page_counter = page_counter
        self.ask_open, self.ask_save = ask_open, ask_save
        self.ask_yes_no, self.show_error, self.show_info = ask_yes_no, show_error, show_info
        self.open_folder = open_folder

        self.q: "queue.Queue" = queue.Queue()
        self.pdf: Optional[Path] = None
        self.worker: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._confirm_answer = False
        self._confirm_event = threading.Event()

        self.log_handler = QueueLogHandler(self.q)
        logging.getLogger(APP_NAME).addHandler(self.log_handler)

        self._build()
        self.pack(fill="both", expand=True)
        self._after_id: Optional[str] = self.after(self.POLL_MS, self._poll)

    # ---------- UI ----------
    def _build(self) -> None:
        self.master.title(f"PDF to MuseScore {__version__} — {AUTHOR_CREDIT}")
        self.master.minsize(680, 560)
        row = ttk.Frame(self)
        row.pack(fill="x")
        self.load_btn = ttk.Button(row, text="Load PDF...", command=self.on_load)
        self.load_btn.pack(side="left")
        self.file_var = tk.StringVar(self, value="No file selected")
        ttk.Label(row, textvariable=self.file_var).pack(side="left", padx=8)

        pages_row = ttk.Frame(self)
        pages_row.pack(fill="x", pady=(8, 0))
        ttk.Label(pages_row, text="Pages (optional, e.g. 2-3 or 1, 4):").pack(side="left")
        self.pages_var = tk.StringVar(self, value="")
        self.pages_entry = ttk.Entry(pages_row, textvariable=self.pages_var, width=16)
        self.pages_entry.pack(side="left", padx=6)
        ttk.Label(pages_row, text="(leave blank for all pages)").pack(side="left")

        # Section 2 & 3 Parameters Frame
        opts_frame = ttk.LabelFrame(self, text="Engine & Score Parameters", padding=8)
        opts_frame.pack(fill="x", pady=(8, 4))

        # Section 2: PDF Rendering & AI Engine
        self.dpi_var = tk.StringVar(self, value=str(self.config_.render_dpi))
        self.preprocess_var = tk.StringVar(self, value="auto")
        self.deskew_var = tk.BooleanVar(self, value=False)
        self.step_size_var = tk.StringVar(self, value="320")
        self.min_notehead_var = tk.StringVar(self, value="4")

        # Section 3: Score Post-Processing
        self.transpose_var = tk.StringVar(self, value="0")
        self.clef_var = tk.StringVar(self, value="auto")
        self.instrument_var = tk.StringVar(self, value="auto")
        self.title_var = tk.StringVar(self, value="")

        r0 = ttk.Frame(opts_frame)
        r0.pack(fill="x", pady=2)
        ttk.Label(r0, text="Render DPI (100-600, e.g. 300/400):").pack(side="left")
        self.dpi_entry = ttk.Entry(r0, textvariable=self.dpi_var, width=6)
        self.dpi_entry.pack(side="left", padx=(4, 12))

        ttk.Label(r0, text="Preprocess:").pack(side="left")
        self.preprocess_combo = ttk.Combobox(
            r0, textvariable=self.preprocess_var, values=list(PREPROCESS_MODES), width=10, state="readonly"
        )
        self.preprocess_combo.pack(side="left", padx=(4, 12))

        self.deskew_check = ttk.Checkbutton(r0, text="Auto-Deskew tilted scan", variable=self.deskew_var)
        self.deskew_check.pack(side="left")

        r1 = ttk.Frame(opts_frame)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="SegNet Tile Step px (80-320, e.g. 240/320):").pack(side="left")
        self.step_entry = ttk.Entry(r1, textvariable=self.step_size_var, width=6)
        self.step_entry.pack(side="left", padx=(4, 12))

        ttk.Label(r1, text="Min Notehead px (2-12, default 4):").pack(side="left")
        self.notehead_entry = ttk.Entry(r1, textvariable=self.min_notehead_var, width=5)
        self.notehead_entry.pack(side="left", padx=(4, 12))

        r2 = ttk.Frame(opts_frame)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Transpose semitones (-24..+24, e.g. +2):").pack(side="left")
        self.transpose_entry = ttk.Entry(r2, textvariable=self.transpose_var, width=5)
        self.transpose_entry.pack(side="left", padx=(4, 12))

        ttk.Label(r2, text="Clef:").pack(side="left")
        self.clef_combo = ttk.Combobox(
            r2, textvariable=self.clef_var, values=list(CLEF_CHOICES), width=8, state="readonly"
        )
        self.clef_combo.pack(side="left", padx=(4, 12))

        ttk.Label(r2, text="Instrument:").pack(side="left")
        self.instrument_combo = ttk.Combobox(
            r2, textvariable=self.instrument_var, values=list(INSTRUMENT_PRESETS), width=11
        )
        self.instrument_combo.pack(side="left", padx=(4, 0))

        r3 = ttk.Frame(opts_frame)
        r3.pack(fill="x", pady=(4, 0))
        ttk.Label(r3, text="Title Override (blank = auto):").pack(side="left")
        self.title_entry = ttk.Entry(r3, textvariable=self.title_var, width=28)
        self.title_entry.pack(side="left", padx=(4, 8), fill="x", expand=True)

        self.reset_opts_btn = ttk.Button(r3, text="Reset Defaults", command=self.on_reset_options)
        self.reset_opts_btn.pack(side="right", padx=(4, 0))
        self.help_opts_btn = ttk.Button(r3, text="? Parameter Guide & Examples", command=self.on_show_parameter_guide)
        self.help_opts_btn.pack(side="right")

        self._option_widgets = (
            self.dpi_entry,
            self.preprocess_combo,
            self.deskew_check,
            self.step_entry,
            self.notehead_entry,
            self.transpose_entry,
            self.clef_combo,
            self.instrument_combo,
            self.title_entry,
            self.reset_opts_btn,
        )

        action_row = ttk.Frame(self)
        action_row.pack(fill="x", pady=(8, 4))
        self.convert_btn = ttk.Button(action_row, text="Convert & Save as...", command=self.on_convert,
                                      state="disabled")
        self.convert_btn.pack(side="left", fill="x", expand=True)
        self.abort_btn = ttk.Button(action_row, text="Abort", command=self.on_abort, state="disabled")
        self.abort_btn.pack(side="left", padx=(6, 0))

        self.progress = ttk.Progressbar(self, maximum=100)
        self.progress.pack(fill="x")
        self.status_var = tk.StringVar(self, value="Load a PDF with sheet music to start.")
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w", pady=4)

        self.log_text = scrolledtext.ScrolledText(self, height=10, state="disabled", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)
        footer = ttk.Frame(self)
        footer.pack(fill="x", pady=(6, 0))
        self.credit_label = ttk.Label(footer, text=AUTHOR_CREDIT)
        self.credit_label.pack(side="left")
        ttk.Button(footer, text="Open log folder", command=self.on_open_logs).pack(side="right")

    def on_reset_options(self) -> None:
        self.dpi_var.set(str(self.config_.render_dpi))
        self.preprocess_var.set("auto")
        self.deskew_var.set(False)
        self.step_size_var.set("320")
        self.min_notehead_var.set("4")
        self.transpose_var.set("0")
        self.clef_var.set("auto")
        self.instrument_var.set("auto")
        self.title_var.set("")

    def on_show_parameter_guide(self) -> None:
        self.show_info("Parameter Guide & Examples", format_parameter_guide())

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
            title_override=self.title_var.get(),
        )

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.load_btn.configure(state=state)
        self.pages_entry.configure(state=state)
        for w in self._option_widgets:
            if w in (self.preprocess_combo, self.clef_combo):
                w.configure(state="disabled" if busy else "readonly")
            else:
                w.configure(state=state)
        self.convert_btn.configure(state="disabled" if busy or not self.pdf else "normal")
        self.abort_btn.configure(state="normal" if busy else "disabled")

    # ---------- actions ----------
    def on_load(self) -> None:
        path = self.ask_open(title="Choose a PDF", filetypes=[("PDF files", "*.pdf")])
        if not path:
            log.debug("Load cancelled")
            return
        self.pdf = Path(path)
        try:
            n_pages = self.page_counter(self.pdf)
            self.file_var.set(f"{self.pdf.name} ({n_pages} page{'s' if n_pages != 1 else ''})")
        except Exception:
            self.file_var.set(self.pdf.name)
        self.status_var.set("Ready. Click 'Convert & Save as...'.")
        log.info("Selected PDF: %s", self.pdf)
        self._set_busy(False)

    def on_convert(self) -> None:
        if not self.pdf:
            return
        try:
            options = self.collect_options()
        except OptionsError as exc:
            self.show_error("Invalid parameter", str(exc))
            return
        types = [MSCZ_TYPE, XML_TYPE] if self.musescore_available() else [XML_TYPE, MSCZ_TYPE]
        ext = types[0][1][1:]
        out = self.ask_save(title="Save as", initialfile=self.pdf.stem + ext,
                            defaultextension=ext, filetypes=types)
        if not out:
            log.debug("Save cancelled")
            return
        self.start_conversion(self.pdf, Path(out), self.pages_var.get().strip() or None, options=options)

    def on_abort(self) -> None:
        log.info("Abort requested by user...")
        self._cancel_event.set()
        self.abort_btn.configure(state="disabled")
        self.status_var.set("Aborting conversion (stopping after current step)...")

    def start_conversion(self, pdf: Path, out: Path, pages_spec: Optional[str] = None,
                         options: Optional[ConversionOptions] = None) -> None:
        self._cancel_event.clear()
        self._set_busy(True)
        self.progress["value"] = 0
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

    # ---------- queue processing (GUI thread) ----------
    def process_queue(self) -> None:
        while True:
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                return
            if kind == "log":
                self.log_text.configure(state="normal")
                self.log_text.insert("end", payload + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
            elif kind == "progress":
                pct, msg = payload
                self.progress["value"] = pct
                self.status_var.set(msg)
            elif kind == "confirm":
                self._confirm_answer = bool(self.ask_yes_no(
                    "Install required program",
                    payload + "\n\nWindows may ask for permission to make changes. Internet is required."))
                self._confirm_event.set()
            elif kind == "done":
                self._set_busy(False)
                self.show_info("Done", f"Saved:\n{payload}")
            elif kind == "aborted":
                self._set_busy(False)
                self.progress["value"] = 0
                self.status_var.set("Conversion aborted by user.")
            elif kind == "error":
                self._set_busy(False)
                self.progress["value"] = 0
                self.status_var.set("Conversion failed. See the log below for details.")
                self.show_error("Conversion failed", payload)

    def _poll(self) -> None:
        self.process_queue()
        self._after_id = self.after(self.POLL_MS, self._poll)

    def destroy(self) -> None:
        if getattr(self, "_after_id", None):
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        logging.getLogger(APP_NAME).removeHandler(self.log_handler)
        super().destroy()


def run_gui(config: AppConfig) -> None:  # pragma: no cover - interactive main loop
    root = tk.Tk()
    App(root, config)
    root.mainloop()
