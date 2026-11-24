"""Unit tests for verbose diagnostics, stdio capture, memory snapshot, and crash hooks."""

from __future__ import annotations

import io
import logging
import sys
import threading
from types import SimpleNamespace

import pytest

from pdf2musescore import diagnostics


def test_memory_status_and_snapshot(monkeypatch) -> None:
    snap = diagnostics.memory_snapshot()
    if sys.platform == "win32":
        status = diagnostics.get_memory_status()
        assert status is not None
        total_b, avail_b, load_pct = status
        assert total_b > 0 and avail_b >= 0 and 0 <= load_pct <= 100
        assert "RAM available" in snap and "GB" in snap

    monkeypatch.setattr(sys, "platform", "linux")
    assert diagnostics.get_memory_status() is None
    assert diagnostics.memory_snapshot() == "RAM status unavailable"

    # Simulate ctypes failure on Windows
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        diagnostics.ctypes,
        "windll",
        SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=lambda _ptr: 0)),
        raising=False,
    )
    assert diagnostics.get_memory_status() is None


def test_collect_and_log_environment_info(monkeypatch, caplog) -> None:
    info = diagnostics.collect_environment_info()
    for key in (
        "app",
        "homr",
        "rapidocr",
        "onnxruntime",
        "onnxruntime_providers",
        "numpy",
        "opencv",
        "pypdfium2",
        "pillow",
        "python",
        "os",
        "machine",
        "cpu_count",
        "frozen",
        "msix",
        "memory",
    ):
        assert key in info and info[key]

    with caplog.at_level("INFO", logger="pdf2musescore.diagtest"):
        logger = logging.getLogger("pdf2musescore.diagtest")
        logged = diagnostics.log_environment(logger)
    assert logged["app"] == info["app"]
    assert "Environment diagnostic info:" in caplog.text
    assert "env[homr] =" in caplog.text

    # Metadata/providers failure fallbacks return 'unknown'/'unavailable'
    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)),
    )
    monkeypatch.setattr(diagnostics, "_ort_providers", lambda: (_ for _ in ()).throw(RuntimeError("no ort")))
    monkeypatch.setattr(diagnostics, "_cv2_version", lambda: (_ for _ in ()).throw(ImportError("no cv2")))
    fallback_info = diagnostics.collect_environment_info()
    assert fallback_info["homr"] == "unknown"
    assert fallback_info["opencv"] == "unknown"
    assert "unavailable" in fallback_info["onnxruntime_providers"]


def test_logger_writer_buffering_passthrough_and_reentrancy(caplog) -> None:
    buf_stream = io.StringIO()
    logger = logging.getLogger("pdf2musescore.writer_test")
    writer = diagnostics.LoggerWriter(logger, logging.INFO, passthrough=buf_stream)

    assert writer.writable() is True
    assert writer.isatty() is False
    assert writer.encoding == "utf-8"
    with pytest.raises(io.UnsupportedOperation):
        writer.fileno()

    with caplog.at_level("INFO", logger="pdf2musescore.writer_test"):
        assert writer.write("") == 0
        writer.write("hello ")
        writer.write("world\r\nstep 1\rstep 2\ntrailing")
        assert [r.message for r in caplog.records] == ["hello world", "step 1", "step 2"]
        writer.flush()
        assert [r.message for r in caplog.records] == ["hello world", "step 1", "step 2", "trailing"]

    assert "trailing" in buf_stream.getvalue()

    # Reentrancy guard prevents infinite recursion if a log handler writes back to writer
    class ReentrantHandler(logging.Handler):
        def emit(self, record):
            writer.write("reentrant line\n")

    h = ReentrantHandler()
    logger.addHandler(h)
    try:
        before = len(caplog.records)
        with caplog.at_level("INFO", logger="pdf2musescore.writer_test"):
            writer.write("outer line\n")
        assert len(caplog.records) == before + 1
        assert caplog.records[-1].message == "outer line"
    finally:
        logger.removeHandler(h)


def test_install_stdio_capture_and_restore(caplog) -> None:
    orig_out, orig_err = sys.stdout, sys.stderr
    restore = diagnostics.install_stdio_capture()
    try:
        with caplog.at_level("INFO", logger="pdf2musescore.thirdparty"):
            print("homr stdout detail")
            print("No staffs found", file=sys.stderr)
            # Second install unwraps passthrough cleanly without double-wrapping
            restore2 = diagnostics.install_stdio_capture()
            try:
                assert isinstance(sys.stderr, diagnostics.LoggerWriter)
                assert sys.stderr.passthrough is getattr(orig_err, "passthrough", orig_err)
            finally:
                restore2()
        assert "homr stdout detail" in caplog.text
        assert "No staffs found" in caplog.text
    finally:
        restore()
        # Ensure original pytest capture streams are fully restored even if double-wrapped
        sys.stdout = orig_out
        sys.stderr = orig_err


def test_install_crash_handlers_writes_crash_log_and_hooks(monkeypatch, tmp_path, caplog) -> None:
    orig_calls: list[str] = []

    def fake_sys_hook(*_a):
        orig_calls.append("sys")

    def fake_thr_hook(*_a):
        orig_calls.append("thr")

    def fake_unr_hook(*_a):
        orig_calls.append("unr")

    monkeypatch.setattr(sys, "excepthook", fake_sys_hook)
    monkeypatch.setattr(threading, "excepthook", fake_thr_hook)
    monkeypatch.setattr(sys, "unraisablehook", fake_unr_hook)

    handlers = diagnostics.install_crash_handlers(tmp_path / "logs")
    try:
        assert handlers.log_path == tmp_path / "logs" / "crash.log"
        crash_text = handlers.log_path.read_text(encoding="utf-8")
        assert "pdf2musescore" in crash_text and "pid=" in crash_text

        with caplog.at_level("ERROR", logger="pdf2musescore.crash"):
            try:
                raise ValueError("top-level crash")
            except ValueError:
                sys.excepthook(*sys.exc_info())

            try:
                raise RuntimeError("worker crash")
            except RuntimeError as exc:
                args = SimpleNamespace(
                    exc_type=RuntimeError,
                    exc_value=exc,
                    exc_traceback=exc.__traceback__,
                    thread=SimpleNamespace(name="worker-1"),
                )
                threading.excepthook(args)

            try:
                raise OSError("gc finalize fail")
            except OSError as exc:
                unr = SimpleNamespace(
                    exc_type=OSError,
                    exc_value=exc,
                    exc_traceback=exc.__traceback__,
                    err_msg="Exception ignored in",
                    object=object(),
                )
                sys.unraisablehook(unr)

        assert orig_calls == ["sys", "thr", "unr"]
        assert "Unhandled exception" in caplog.text
        assert "top-level crash" in caplog.text
        assert "worker-1" in caplog.text and "worker crash" in caplog.text
        assert "gc finalize fail" in caplog.text
    finally:
        handlers.uninstall()
        handlers.uninstall()  # idempotent
        assert sys.excepthook is fake_sys_hook
        assert threading.excepthook is fake_thr_hook
        assert sys.unraisablehook is fake_unr_hook
