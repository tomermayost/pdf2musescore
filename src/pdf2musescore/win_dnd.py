"""Pure-stdlib native Windows file drag-and-drop support via Win32 WM_DROPFILES."""

from __future__ import annotations

import ctypes
import logging
import queue
import sys
import tkinter as tk
from ctypes import wintypes
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

WM_DROPFILES = 0x0233
WM_COPYGLOBALDATA = 0x0049
GWLP_WNDPROC = -4
MSGFLT_ALLOW = 1
DROP_POLL_MS = 50

_LONG_PTR = ctypes.c_ssize_t
_WNDPROC = ctypes.WINFUNCTYPE(_LONG_PTR, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


def extract_dropped_files(hdrop: int, shell32: Any) -> list[Path]:
    """Read Unicode file paths from a Win32 ``HDROP`` handle and always release ``hdrop``."""
    paths: list[Path] = []
    try:
        count = int(shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0))
        for idx in range(count):
            length = int(shell32.DragQueryFileW(hdrop, idx, None, 0))
            if length <= 0:
                continue
            buf = ctypes.create_unicode_buffer(length + 1)
            shell32.DragQueryFileW(hdrop, idx, buf, length + 1)
            if buf.value:
                paths.append(Path(buf.value))
    finally:
        shell32.DragFinish(hdrop)
    return paths


class DropHandle:
    """Active Win32 subclass handle; drains dropped-file queue on the Tk thread."""

    def __init__(
        self,
        hwnd: int,
        prev_wndproc: int,
        wndproc_cb: Any,
        user32: Any,
        shell32: Any,
        *,
        window: Optional[tk.Misc] = None,
        on_files: Optional[Callable[[list[Path]], None]] = None,
        poll_ms: int = DROP_POLL_MS,
    ) -> None:
        self.hwnd = int(hwnd)
        self.prev_wndproc = int(prev_wndproc)
        # Keep strong reference so Python never collects the ctypes WNDPROC callback while active.
        self._wndproc_cb = wndproc_cb
        self.user32 = user32
        self.shell32 = shell32
        self._window = window
        self._on_files = on_files
        self._poll_ms = max(10, int(poll_ms))
        self._pending: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._pump_after_id: Optional[str] = None
        self._uninstalled = False

    def enqueue(self, item: Any) -> None:
        """Queue dropped paths (or an extraction exception) safely from the Win32 WNDPROC."""
        self._pending.put(item)

    def start_pump(self) -> None:
        """Schedule the Tk-thread queue poller if a window is attached."""
        if not self._uninstalled and self._window is not None:
            self._pump_after_id = self._window.after(self._poll_ms, self.pump)

    def pump(self) -> None:
        """Drain queued drops on the Tk thread and reschedule while active."""
        self._pump_after_id = None
        if self._uninstalled:
            return
        while True:
            try:
                item = self._pending.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, BaseException):
                log.error("Failed to read dropped files", exc_info=item)
                continue
            paths = list(item)
            if not paths:
                continue
            log.info("Received %d dropped file(s): %s", len(paths), [str(p) for p in paths])
            if self._on_files is not None:
                try:
                    self._on_files(paths)
                except Exception:
                    log.exception("Failed to handle dropped file(s)")
        if not self._uninstalled and self._window is not None:
            try:
                self._pump_after_id = self._window.after(self._poll_ms, self.pump)
            except Exception as exc:
                log.debug("Drag-and-drop pump reschedule skipped: %s", exc)

    def uninstall(self) -> None:
        """Restore the original Win32 window procedure and disable drop acceptance (idempotent)."""
        if self._uninstalled:
            return
        self._uninstalled = True
        if self._pump_after_id is not None and self._window is not None:
            try:
                self._window.after_cancel(self._pump_after_id)
            except Exception as exc:
                log.debug("Drag-and-drop pump cancel skipped: %s", exc)
            finally:
                self._pump_after_id = None
        try:
            setter = getattr(self.user32, "SetWindowLongPtrW", None) or self.user32.SetWindowLongW
            if self.prev_wndproc:
                setter(self.hwnd, GWLP_WNDPROC, self.prev_wndproc)
            self.shell32.DragAcceptFiles(self.hwnd, False)
            log.debug("Uninstalled native Windows file drop handler (hwnd=%s)", self.hwnd)
        except Exception as exc:
            log.debug("Drag-and-drop uninstall skipped: %s", exc)
        finally:
            self._wndproc_cb = None
            self._on_files = None
            self._window = None


def _configure_win32_apis(user32: Any, shell32: Any) -> Any:
    """Declare explicit 64-bit-safe ctypes signatures and return SetWindowLongPtr callable."""
    user32.GetParent.argtypes = [wintypes.HWND]
    user32.GetParent.restype = wintypes.HWND

    setter = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
    setter.argtypes = [wintypes.HWND, ctypes.c_int, _LONG_PTR]
    setter.restype = _LONG_PTR

    user32.CallWindowProcW.argtypes = [_LONG_PTR, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.CallWindowProcW.restype = _LONG_PTR

    shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]
    shell32.DragAcceptFiles.restype = None

    shell32.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
    shell32.DragQueryFileW.restype = wintypes.UINT

    shell32.DragFinish.argtypes = [wintypes.HANDLE]
    shell32.DragFinish.restype = None
    return setter


def install_file_drop_handler(
    window: tk.Misc,
    on_files: Callable[[list[Path]], None],
    *,
    windll: Any = None,
) -> Optional[DropHandle]:
    """Enable native Windows Explorer drag-and-drop onto ``window``; returns ``None`` if unavailable."""
    if sys.platform != "win32":
        return None
    try:
        wd = windll if windll is not None else getattr(ctypes, "windll", None)
        if wd is None or not hasattr(wd, "user32") or not hasattr(wd, "shell32"):
            return None
        user32 = wd.user32
        shell32 = wd.shell32

        window.update_idletasks()
        wid = int(window.winfo_id())
        setter = _configure_win32_apis(user32, shell32)
        parent_id = int(user32.GetParent(wid) or 0)
        hwnd = parent_id or wid
        if not hwnd:
            return None

        shell32.DragAcceptFiles(hwnd, True)

        # Allow WM_DROPFILES even when dragging between elevation boundaries if supported.
        change_filter = getattr(user32, "ChangeWindowMessageFilterEx", None)
        if change_filter is not None:
            try:
                change_filter(hwnd, WM_DROPFILES, MSGFLT_ALLOW, None)
                change_filter(hwnd, WM_COPYGLOBALDATA, MSGFLT_ALLOW, None)
            except Exception as exc:  # pragma: no cover - defensive UIPI fallback
                log.debug("ChangeWindowMessageFilterEx skipped: %s", exc)

        prev_holder = [0]
        handle_holder: list[Optional[DropHandle]] = [None]

        # NEVER call tkinter/Tcl methods or Python logging from inside this raw Win32 WNDPROC!
        # Windows dispatches WM_DROPFILES inside Tcl_DoOneEvent when _tkinter is not in a
        # Tcl->Python callback and its saved Python thread state (tcl_tstate) is NULL.
        # Any nested tkinter call corrupts that thread state and triggers a fatal
        # PyEval_RestoreThread(NULL) abort. Enqueue purely in Python and drain in DropHandle.pump().
        @_WNDPROC
        def _wndproc(cb_hwnd: int, msg: int, wparam: int, lparam: int) -> int:
            try:
                if msg == WM_DROPFILES:
                    target = handle_holder[0]
                    try:
                        paths = extract_dropped_files(wparam, shell32)
                    except Exception as exc:
                        if target is not None:
                            target.enqueue(exc)
                    else:
                        if target is not None:
                            target.enqueue(paths)
                    return 0
                prev_proc = prev_holder[0]
                if prev_proc:
                    return int(user32.CallWindowProcW(prev_proc, cb_hwnd, msg, wparam, lparam))
                return 0
            except BaseException:
                return 0

        cb_ptr = ctypes.cast(_wndproc, ctypes.c_void_p).value
        prev = int(setter(hwnd, GWLP_WNDPROC, cb_ptr))
        if not prev:
            shell32.DragAcceptFiles(hwnd, False)
            return None
        prev_holder[0] = prev
        handle = DropHandle(
            hwnd=hwnd,
            prev_wndproc=prev,
            wndproc_cb=_wndproc,
            user32=user32,
            shell32=shell32,
            window=window,
            on_files=on_files,
        )
        handle_holder[0] = handle
        handle.start_pump()
        log.debug("Installed native Windows file drop handler (hwnd=%s, poll_ms=%s)", hwnd, handle._poll_ms)
        return handle
    except Exception as exc:
        log.debug("Could not install native Windows drag-and-drop: %s", exc)
        return None
