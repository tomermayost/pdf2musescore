"""Unit tests for the installer-triggered ModelDownloadWindow dialog."""

from __future__ import annotations

import time
from pathlib import Path

from pdf2musescore.engines.homr_engine import CancelledError
from pdf2musescore.model_store import ModelDownloadError, ModelStatus
from pdf2musescore.ui_download import ModelDownloadWindow


def _pump_until(win: ModelDownloadWindow, root, pred, timeout: float = 4.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        win.process_queue()
        root.update()
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError("timeout waiting for ModelDownloadWindow")


def test_model_download_window_success_error_and_cancel(root, tmp_path: Path) -> None:
    class _FakeMgr:
        def __init__(self, mode: str) -> None:
            self.mode = mode

        def ensure(self, progress=None, is_cancelled=lambda: False):
            if progress:
                progress(60_000_000, 180_000_000)
            if self.mode == "error":
                raise ModelDownloadError("dns lookup failed")
            if self.mode == "wait_cancel":
                end = time.time() + 2.0
                while time.time() < end:
                    if is_cancelled():
                        raise CancelledError("cancelled")
                    time.sleep(0.01)
                raise AssertionError("cancel was never triggered")
            return ModelStatus(ready=True, missing=(), homr_dir=tmp_path, ocr_dir=tmp_path)

    # 1. Success path finishes with code 0
    done_codes: list[int] = []
    win_ok = ModelDownloadWindow(root, _FakeMgr("ok"), on_done=done_codes.append)
    try:
        _pump_until(win_ok, root, lambda: bool(done_codes))
        assert done_codes == [0]
        assert win_ok.progress["value"] == 100
    finally:
        win_ok.destroy()

    # 2. Failure path switches Cancel -> Close and delivers code 1 when clicked
    err_codes: list[int] = []
    win_err = ModelDownloadWindow(root, _FakeMgr("error"), on_done=err_codes.append)
    try:
        _pump_until(win_err, root, lambda: win_err.result_code == 1)
        assert "Download failed" in win_err.status_var.get()
        assert win_err.action_btn.cget("text") == "Close"
        win_err.action_btn.invoke()
        assert err_codes == [1]
    finally:
        win_err.destroy()

    # 3. Cancel button triggers cancellation -> code 2
    cancel_codes: list[int] = []
    win_cancel = ModelDownloadWindow(root, _FakeMgr("wait_cancel"), on_done=cancel_codes.append)
    try:
        win_cancel.action_btn.invoke()
        _pump_until(win_cancel, root, lambda: bool(cancel_codes))
        assert cancel_codes == [2]
    finally:
        win_cancel.destroy()
