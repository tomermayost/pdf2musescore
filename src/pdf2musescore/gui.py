"""Tkinter desktop window: Load PDF -> (optional Pages) -> Convert & Save as / Abort."""

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
        self.after(self.POLL_MS, self._poll)

    # ---------- UI ----------
    def _build(self) -> None:
        self.master.title(f"PDF to MuseScore {__version__} — {AUTHOR_CREDIT}")
        self.master.minsize(580, 400)
        row = ttk.Frame(self)
        row.pack(fill="x")
        self.load_btn = ttk.Button(row, text="Load PDF...", command=self.on_load)
        self.load_btn.pack(side="left")
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(row, textvariable=self.file_var).pack(side="left", padx=8)

        pages_row = ttk.Frame(self)
        pages_row.pack(fill="x", pady=(8, 0))
        ttk.Label(pages_row, text="Pages (optional, e.g. 2-3 or 1, 4):").pack(side="left")
        self.pages_var = tk.StringVar(value="")
        self.pages_entry = ttk.Entry(pages_row, textvariable=self.pages_var, width=18)
        self.pages_entry.pack(side="left", padx=6)
        ttk.Label(pages_row, text="(leave blank for all pages)").pack(side="left")

        action_row = ttk.Frame(self)
        action_row.pack(fill="x", pady=(10, 4))
        self.convert_btn = ttk.Button(action_row, text="Convert & Save as...", command=self.on_convert,
                                      state="disabled")
        self.convert_btn.pack(side="left", fill="x", expand=True)
        self.abort_btn = ttk.Button(action_row, text="Abort", command=self.on_abort, state="disabled")
        self.abort_btn.pack(side="left", padx=(6, 0))

        self.progress = ttk.Progressbar(self, maximum=100)
        self.progress.pack(fill="x")
        self.status_var = tk.StringVar(value="Load a PDF with sheet music to start.")
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w", pady=4)

        self.log_text = scrolledtext.ScrolledText(self, height=12, state="disabled", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)
        footer = ttk.Frame(self)
        footer.pack(fill="x", pady=(6, 0))
        self.credit_label = ttk.Label(footer, text=AUTHOR_CREDIT)
        self.credit_label.pack(side="left")
        ttk.Button(footer, text="Open log folder", command=self.on_open_logs).pack(side="right")

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.load_btn.configure(state=state)
        self.pages_entry.configure(state=state)
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
        types = [MSCZ_TYPE, XML_TYPE] if self.musescore_available() else [XML_TYPE, MSCZ_TYPE]
        ext = types[0][1][1:]
        out = self.ask_save(title="Save as", initialfile=self.pdf.stem + ext,
                            defaultextension=ext, filetypes=types)
        if not out:
            log.debug("Save cancelled")
            return
        self.start_conversion(self.pdf, Path(out), self.pages_var.get().strip() or None)

    def on_abort(self) -> None:
        log.info("Abort requested by user...")
        self._cancel_event.set()
        self.abort_btn.configure(state="disabled")
        self.status_var.set("Aborting conversion (stopping after current step)...")

    def start_conversion(self, pdf: Path, out: Path, pages_spec: Optional[str] = None) -> None:
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
                self.q.put(("done", pipeline.convert(pdf, out, pages_spec=pages_spec)))
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
        self.after(self.POLL_MS, self._poll)

    def destroy(self) -> None:
        logging.getLogger(APP_NAME).removeHandler(self.log_handler)
        super().destroy()


def run_gui(config: AppConfig) -> None:  # pragma: no cover - interactive main loop
    root = tk.Tk()
    App(root, config)
    root.mainloop()
