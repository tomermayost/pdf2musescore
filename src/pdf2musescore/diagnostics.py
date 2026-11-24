"""Verbose environment diagnostics, stdio capture for third-party libs, and crash hooks."""

from __future__ import annotations

import ctypes
import faulthandler
import importlib.metadata
import io
import logging
import os
import platform
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

from . import APP_NAME, __version__
from .config import is_msix_package


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def get_memory_status() -> Optional[tuple[int, int, int]]:
    """Return ``(total_bytes, avail_bytes, memory_load_pct)`` on Windows, or ``None`` if unavailable."""
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        return int(stat.ullTotalPhys), int(stat.ullAvailPhys), int(stat.dwMemoryLoad)
    except Exception:
        return None


def memory_snapshot() -> str:
    """Return a concise single-line summary of physical memory availability."""
    status = get_memory_status()
    if status is None:
        return "RAM status unavailable"
    total_b, avail_b, load_pct = status
    gb = float(1024**3)
    return f"RAM available {avail_b / gb:.1f} GB of {total_b / gb:.1f} GB ({load_pct}% in use)"


def _pkg_version(*names: str, module_attr: Optional[Callable[[], str]] = None) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    if module_attr is not None:
        try:
            return str(module_attr())
        except Exception:
            pass
    return "unknown"


def _cv2_version() -> str:
    import cv2

    return str(getattr(cv2, "__version__", "unknown"))


def _ort_providers() -> str:
    import onnxruntime as ort

    providers = ort.get_available_providers()
    return ", ".join(str(p) for p in providers) if providers else "none"


def collect_environment_info() -> dict[str, str]:
    """Collect versions, hardware, runtime mode, and RAM status for diagnostic logs."""
    try:
        ort_prov = _ort_providers()
    except Exception as exc:
        ort_prov = f"unavailable ({type(exc).__name__}: {exc})"

    return {
        "app": __version__,
        "homr": _pkg_version("homr"),
        "rapidocr": _pkg_version("rapidocr"),
        "onnxruntime": _pkg_version("onnxruntime"),
        "onnxruntime_providers": ort_prov,
        "numpy": _pkg_version("numpy"),
        "opencv": _pkg_version("opencv-python-headless", "opencv-python", module_attr=_cv2_version),
        "pypdfium2": _pkg_version("pypdfium2"),
        "pillow": _pkg_version("Pillow"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine() or "unknown",
        "cpu_count": str(os.cpu_count() or "unknown"),
        "frozen": str(bool(getattr(sys, "frozen", False))),
        "msix": str(is_msix_package()),
        "memory": memory_snapshot(),
    }


def log_environment(logger: logging.Logger, level: int = logging.INFO) -> dict[str, str]:
    """Log every key from :func:`collect_environment_info` to ``logger``."""
    info = collect_environment_info()
    logger.log(level, "Environment diagnostic info:")
    for key, val in info.items():
        logger.log(level, "  env[%s] = %s", key, val)
    return info


class LoggerWriter(io.TextIOBase):
    """Line-buffered writable text stream that forwards lines to a Python logger."""

    def __init__(
        self,
        logger: logging.Logger,
        level: int = logging.INFO,
        passthrough: Optional[TextIO] = None,
    ) -> None:
        super().__init__()
        self.logger = logger
        self.level = int(level)
        self.passthrough = passthrough
        self._buffer = ""
        self._local = threading.local()

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise io.UnsupportedOperation("LoggerWriter does not have an OS file descriptor")

    def write(self, s: str) -> int:
        if not s:
            return 0
        text = str(s)
        if self.passthrough is not None:
            try:
                self.passthrough.write(text)
            except Exception:
                pass
        if getattr(self._local, "active", False):
            return len(text)
        self._local.active = True
        try:
            self._buffer += text.replace("\r\n", "\n").replace("\r", "\n")
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                cleaned = line.rstrip()
                if cleaned:
                    self.logger.log(self.level, "%s", cleaned)
        finally:
            self._local.active = False
        return len(text)

    def flush(self) -> None:
        if not getattr(self._local, "active", False) and self._buffer:
            self._local.active = True
            try:
                cleaned = self._buffer.rstrip()
                self._buffer = ""
                if cleaned:
                    self.logger.log(self.level, "%s", cleaned)
            finally:
                self._local.active = False
        if self.passthrough is not None:
            try:
                self.passthrough.flush()
            except Exception:
                pass


def install_stdio_capture() -> Callable[[], None]:
    """Redirect ``sys.stdout`` and ``sys.stderr`` through :class:`LoggerWriter` so third-party prints are logged."""
    orig_out = sys.stdout
    orig_err = sys.stderr
    base_out = getattr(orig_out, "passthrough", orig_out)
    base_err = getattr(orig_err, "passthrough", orig_err)

    out_writer = LoggerWriter(logging.getLogger(f"{APP_NAME}.thirdparty.stdout"), logging.INFO, passthrough=base_out)
    err_writer = LoggerWriter(logging.getLogger(f"{APP_NAME}.thirdparty.stderr"), logging.INFO, passthrough=base_err)
    sys.stdout = out_writer
    sys.stderr = err_writer
    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        try:
            out_writer.flush()
            err_writer.flush()
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err

    return restore


class CrashHandlers:
    """Handle for installed ``faulthandler`` and Python exception hooks; call :meth:`uninstall` to restore."""

    def __init__(
        self,
        log_path: Path,
        crash_file: TextIO,
        orig_excepthook: Any,
        orig_threading_hook: Any,
        orig_unraisable_hook: Any,
    ) -> None:
        self.log_path = Path(log_path)
        self._file = crash_file
        self._orig_excepthook = orig_excepthook
        self._orig_threading_hook = orig_threading_hook
        self._orig_unraisable_hook = orig_unraisable_hook
        self._uninstalled = False

    def uninstall(self) -> None:
        """Restore previous exception hooks, disable ``faulthandler``, and close ``crash.log`` (idempotent)."""
        if self._uninstalled:
            return
        self._uninstalled = True
        try:
            faulthandler.disable()
        except Exception:
            pass
        sys.excepthook = self._orig_excepthook
        threading.excepthook = self._orig_threading_hook
        sys.unraisablehook = self._orig_unraisable_hook
        try:
            self._file.close()
        except Exception:
            pass


def install_crash_handlers(log_dir: Path) -> CrashHandlers:
    """Enable native ``faulthandler`` to ``crash.log`` and log unhandled sys/thread/unraisable exceptions."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    crash_path = log_dir / "crash.log"
    fh = crash_path.open("a", encoding="utf-8", buffering=1)
    fh.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} pdf2musescore {__version__} pid={os.getpid()} ===\n")
    fh.flush()
    faulthandler.enable(file=fh, all_threads=True)

    crash_log = logging.getLogger(f"{APP_NAME}.crash")
    orig_excepthook = sys.excepthook
    orig_threading_hook = threading.excepthook
    orig_unraisable_hook = sys.unraisablehook

    def _excepthook(exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> None:
        if not issubclass(exc_type, KeyboardInterrupt):
            crash_log.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
        try:
            orig_excepthook(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    def _threading_hook(args: threading.ExceptHookArgs) -> None:
        tname = getattr(args.thread, "name", "unknown")
        crash_log.critical(
            "Unhandled exception in thread %s",
            tname,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        try:
            orig_threading_hook(args)
        except Exception:
            pass

    def _unraisable_hook(unraisable: sys.UnraisableHookArgs) -> None:
        try:
            obj_repr = repr(unraisable.object)
        except Exception:
            obj_repr = "<unprintable>"
        err_msg = unraisable.err_msg or "Exception ignored in"
        crash_log.error(
            "%s %s",
            err_msg,
            obj_repr,
            exc_info=(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback),
        )
        try:
            orig_unraisable_hook(unraisable)
        except Exception:
            pass

    sys.excepthook = _excepthook
    threading.excepthook = _threading_hook
    sys.unraisablehook = _unraisable_hook
    logging.getLogger(APP_NAME).debug("Crash handlers installed (crash_log=%s)", crash_path)
    return CrashHandlers(crash_path, fh, orig_excepthook, orig_threading_hook, orig_unraisable_hook)
