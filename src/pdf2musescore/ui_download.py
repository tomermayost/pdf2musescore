"""Lightweight installer/CLI progress dialog for one-time AI model downloads."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

from .engines.homr_engine import CancelledError
from .model_store import ModelManager

log = logging.getLogger(__name__)


class ModelDownloadWindow(ttk.Frame):
    """Determinate progress window shown during installer-triggered AI model downloads."""

    POLL_MS = 80

    def __init__(
        self,
        master: tk.Misc,
        manager: ModelManager,
        on_done: Optional[Callable[[int], None]] = None,
    ) -> None:
        super().__init__(master, padding=(18, 16, 18, 16))
        self.manager = manager
        self.on_done = on_done
        self.result_code: Optional[int] = None
        self._finished = False
        self._cancel_event = threading.Event()
        self._q: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._after_id: Optional[str] = None

        if hasattr(master, "title"):
            try:
                master.title("PDF to MuseScore — Downloading AI models")
            except Exception:
                pass
        if hasattr(master, "resizable"):
            try:
                master.resizable(False, False)
            except Exception:
                pass

        ttk.Label(
            self,
            text="Downloading AI music-recognition models",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            self,
            text="One-time download of the AI music-recognition models (~180 MB).",
            wraplength=390,
        ).pack(anchor="w", pady=(4, 12))

        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100, length=390)
        self.progress.pack(fill="x")

        self.status_var = tk.StringVar(self, value="Connecting to model server...")
        self.status_lbl = ttk.Label(self, textvariable=self.status_var, wraplength=390)
        self.status_lbl.pack(anchor="w", pady=(8, 12))

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x")
        self.action_btn = ttk.Button(btn_row, text="Cancel", command=self._on_action)
        self.action_btn.pack(side="right")

        self.pack(fill="both", expand=True)

        self._worker = threading.Thread(target=self._work, daemon=True, name="installer-model-download")
        self._worker.start()
        self._after_id = self.after(self.POLL_MS, self._poll)

    def _work(self) -> None:
        try:
            self.manager.ensure(
                progress=lambda d, t: self._q.put(("progress", (d, t))),
                is_cancelled=self._cancel_event.is_set,
            )
            self._q.put(("done", 0))
        except CancelledError:
            self._q.put(("cancelled", 2))
        except Exception as exc:
            log.error("Model download failed: %s", exc, exc_info=True)
            self._q.put(("error", f"{type(exc).__name__}: {exc}"))

    def _on_action(self) -> None:
        if self.result_code == 1:
            self._finish(1)
            return
        self._cancel_event.set()
        self.action_btn.configure(state="disabled")
        self.status_var.set("Cancelling download...")

    def _finish(self, code: int) -> None:
        if self._finished:
            return
        self._finished = True
        self.result_code = int(code)
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self.on_done is not None:
            self.on_done(self.result_code)

    def process_queue(self) -> None:
        while True:
            try:
                kind, payload = self._q.get_nowait()
            except queue.Empty:
                return
            if kind == "progress":
                done_b, total_b = payload
                denom = max(int(total_b), 1)
                pct = min(99, max(0, int(100 * int(done_b) / denom)))
                self.progress["value"] = pct
                self.status_var.set(f"Downloaded {done_b / 1e6:.0f} of ~{denom / 1e6:.0f} MB ({pct}%)")
            elif kind == "done":
                self.progress["value"] = 100
                self.status_var.set("AI models downloaded and verified.")
                self._finish(0)
                return
            elif kind == "cancelled":
                self.status_var.set("Download cancelled.")
                self._finish(2)
                return
            elif kind == "error":
                self.result_code = 1
                self.status_var.set(
                    f"Download failed: {payload}\n\nThe app will retry automatically when it starts."
                )
                self.action_btn.configure(text="Close", state="normal")
                return

    def _poll(self) -> None:
        self.process_queue()
        if not self._finished:
            self._after_id = self.after(self.POLL_MS, self._poll)

    def destroy(self) -> None:
        self._cancel_event.set()
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        super().destroy()


def run_download_window(manager: ModelManager) -> int:  # pragma: no cover - interactive Tk loop
    root = tk.Tk()
    result = [1]

    def _done(code: int) -> None:
        result[0] = code
        try:
            root.quit()
            root.destroy()
        except Exception:
            pass

    win = ModelDownloadWindow(root, manager, on_done=_done)

    def _on_close() -> None:
        if win.result_code == 1:
            _done(1)
        else:
            win._on_action()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.update_idletasks()
    w = max(430, root.winfo_reqwidth())
    h = max(170, root.winfo_reqheight())
    x = max(0, (root.winfo_screenwidth() - w) // 2)
    y = max(0, (root.winfo_screenheight() - h) // 2 - 20)
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.mainloop()
    return result[0]
