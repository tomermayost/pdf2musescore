"""Unit tests for ModelManager verification, staged downloads, adoption, locking, and installer config."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from pdf2musescore.engines.homr_engine import CancelledError
from pdf2musescore.model_store import MANIFEST_NAME, ModelDownloadError, ModelManager

ROOT = Path(__file__).resolve().parents[2]


def _make_fake_manager(tmp_path: Path, *, homr_dl=None, ocr_dl=None, poll_interval: float = 0.02):
    req = lambda: {
        "segnet.onnx": "segmentation",
        "encoder.onnx": "transformer",
        "decoder.onnx": "transformer",
    }

    def default_homr_dl(seg_dir: Path, tr_dir: Path) -> None:
        (seg_dir / "segnet.onnx").write_bytes(b"S" * 100)
        (tr_dir / "encoder.onnx").write_bytes(b"E" * 200)
        (tr_dir / "decoder.onnx").write_bytes(b"D" * 300)

    def default_ocr_dl(ocr_dir: Path) -> None:
        (ocr_dir / "det.onnx").write_bytes(b"O" * 150)

    return ModelManager(
        tmp_path / "models",
        homr_version="0.7.1",
        ocr_version="3.10.0",
        required_homr_files=req,
        download_homr=homr_dl or default_homr_dl,
        init_ocr=ocr_dl or default_ocr_dl,
        poll_interval=poll_interval,
    )


def test_model_manager_status_download_verify_and_repair(tmp_path: Path) -> None:
    mgr = _make_fake_manager(tmp_path)

    st_before = mgr.status()
    assert st_before.ready is False
    assert set(st_before.missing) == {"homr", "rapidocr"}

    # Ensure downloads via staging -> final, creates manifests, cleans staging
    st_after = mgr.ensure()
    assert st_after.ready is True
    assert not (mgr.homr_dir / ".staging").exists()
    assert (mgr.homr_dir / MANIFEST_NAME).is_file()
    assert (mgr.ocr_dir / MANIFEST_NAME).is_file()
    assert (mgr.homr_dir / "segmentation" / "segnet.onnx").read_bytes() == b"S" * 100

    # Calling ensure again when ready is a fast no-op
    called = []
    mgr._download_homr = lambda *_a: called.append("homr")
    mgr._init_ocr = lambda *_a: called.append("ocr")
    assert mgr.ensure().ready is True
    assert called == []

    # Truncating/corrupting a cached file invalidates status and triggers repair
    (mgr.homr_dir / "segmentation" / "segnet.onnx").write_bytes(b"X")
    st_corrupt = mgr.status()
    assert st_corrupt.ready is False
    assert any("segnet.onnx" in item for item in st_corrupt.missing)

    # Delete a file -> also detected as missing
    (mgr.ocr_dir / "det.onnx").unlink()
    st_missing = mgr.status()
    assert st_missing.ready is False
    assert any("det.onnx" in item for item in st_missing.missing)

    # Invalid/corrupt manifest JSON also marks not ready
    (mgr.ocr_dir / MANIFEST_NAME).write_text("{bad json")
    assert mgr.status().ready is False

    # Restore download hooks -> repairs cleanly (truncated file IS replaced, not adopted!)
    mgr2 = _make_fake_manager(tmp_path)
    assert mgr2.ensure().ready is True
    assert (mgr2.homr_dir / "segmentation" / "segnet.onnx").read_bytes() == b"S" * 100

    # When only OCR fails verification, homr is NOT re-downloaded
    (mgr2.ocr_dir / MANIFEST_NAME).unlink()
    homr_re_calls = []
    mgr3 = _make_fake_manager(tmp_path, homr_dl=lambda *_a: homr_re_calls.append("homr"))
    assert mgr3.ensure().ready is True
    assert homr_re_calls == []

    # Corrupt homr manifest JSON also forces full homr verification/download
    (mgr3.homr_dir / MANIFEST_NAME).write_text("[]")
    (mgr3.homr_dir / "segmentation" / "segnet.onnx").write_bytes(b"stale")
    mgr4 = _make_fake_manager(tmp_path)
    assert mgr4.ensure().ready is True
    assert (mgr4.homr_dir / "segmentation" / "segnet.onnx").read_bytes() == b"S" * 100


def test_model_manager_adopts_legacy_cache_without_redownload(tmp_path: Path) -> None:
    mgr = _make_fake_manager(tmp_path)
    (mgr.homr_dir / "segmentation").mkdir(parents=True)
    (mgr.homr_dir / "transformer").mkdir(parents=True)
    (mgr.homr_dir / "segmentation" / "segnet.onnx").write_bytes(b"existing-seg")
    (mgr.homr_dir / "transformer" / "encoder.onnx").write_bytes(b"existing-enc")
    (mgr.homr_dir / "transformer" / "decoder.onnx").write_bytes(b"existing-dec")

    downloads: list[str] = []
    mgr._download_homr = lambda *_a: downloads.append("homr")
    st = mgr.ensure()
    assert st.ready is True
    assert downloads == []
    assert (mgr.homr_dir / "segmentation" / "segnet.onnx").read_bytes() == b"existing-seg"


def test_model_manager_progress_cancel_errors_and_locking(tmp_path: Path) -> None:
    progress_events: list[tuple[int, int]] = []

    def slow_homr(seg_dir: Path, tr_dir: Path) -> None:
        (seg_dir / "segnet.onnx").write_bytes(b"S" * 400)
        time.sleep(0.06)
        (tr_dir / "encoder.onnx").write_bytes(b"E" * 500)
        (tr_dir / "decoder.onnx").write_bytes(b"D" * 600)

    def slow_ocr(ocr_dir: Path) -> None:
        time.sleep(0.06)
        (ocr_dir / "det.onnx").write_bytes(b"O" * 300)

    # Concurrent ensure calls are serialized by lock so download runs only once
    homr_runs = 0
    started = threading.Event()
    saw_locked: list[bool] = []

    def tracked_homr(seg_dir: Path, tr_dir: Path) -> None:
        nonlocal homr_runs
        homr_runs += 1
        saw_locked.append(ModelManager.is_downloading())
        started.set()
        slow_homr(seg_dir, tr_dir)

    mgr1 = _make_fake_manager(tmp_path, homr_dl=tracked_homr, ocr_dl=slow_ocr, poll_interval=0.01)
    mgr2 = _make_fake_manager(tmp_path, homr_dl=tracked_homr, ocr_dl=slow_ocr, poll_interval=0.01)

    t1 = threading.Thread(target=lambda: mgr1.ensure(progress=lambda d, t: progress_events.append((d, t))))
    t2 = threading.Thread(target=lambda: mgr2.ensure())
    t1.start()
    assert started.wait(timeout=2)
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert homr_runs == 1
    assert saw_locked == [True]
    assert ModelManager.is_downloading() is False
    assert mgr1.status().ready is True
    assert len(progress_events) >= 2
    assert progress_events[-1][0] == 1800

    # Cancel before start and mid-download
    mgr_c = _make_fake_manager(tmp_path / "c1")
    with pytest.raises(CancelledError):
        mgr_c.ensure(is_cancelled=lambda: True)
    assert not (mgr_c.homr_dir / MANIFEST_NAME).exists()

    calls = 0

    def cancel_after_homr() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(CancelledError):
        mgr_c.ensure(is_cancelled=cancel_after_homr)
    assert not (mgr_c.ocr_dir / MANIFEST_NAME).exists()

    # Missing file after homr download -> ModelDownloadError
    mgr_bad_h = _make_fake_manager(tmp_path / "e1", homr_dl=lambda seg, tr: None)
    with pytest.raises(ModelDownloadError, match="missing after download"):
        mgr_bad_h.ensure()

    # RapidOCR produces no files -> ModelDownloadError
    mgr_bad_o = _make_fake_manager(tmp_path / "e2", ocr_dl=lambda ocr: None)
    with pytest.raises(ModelDownloadError, match="produced no model files"):
        mgr_bad_o.ensure()

    # Unexpected exception in downloader -> wrapped in ModelDownloadError
    mgr_exc = _make_fake_manager(
        tmp_path / "e3",
        homr_dl=lambda _s, _t: (_ for _ in ()).throw(OSError("connection reset")),
    )
    with pytest.raises(ModelDownloadError, match="OSError: connection reset"):
        mgr_exc.ensure()


def test_installer_iss_has_model_download_and_uninstall_cleanup() -> None:
    iss = (ROOT / "packaging" / "installer.iss").read_text(encoding="utf-8")
    idx_ui = iss.find("--download-models")
    idx_silent = iss.find("--download-models --no-ui")
    idx_launch = iss.find("Flags: nowait postinstall")
    assert 0 <= idx_ui < idx_silent < idx_launch
    assert "[UninstallDelete]" in iss
    assert r"{localappdata}\pdf2musescore\models" in iss
    assert r"{commonappdata}\pdf2musescore\models" in iss


def test_patched_homr_download_timeout_behavior_and_restore() -> None:
    from types import SimpleNamespace
    from pdf2musescore.model_store import patched_homr_download_timeout

    seen: list[object] = []

    def orig_get(url, **kwargs):
        seen.append((url, kwargs.get("timeout")))
        return "ok"

    fake_requests = SimpleNamespace(get=orig_get)
    fake_dl = SimpleNamespace(requests=fake_requests)

    # Raises short/missing timeouts to 60.0 and preserves larger explicit timeouts
    with patched_homr_download_timeout(60.0, download_utils_mod=fake_dl):
        assert fake_requests.get("https://example.com/a", timeout=5) == "ok"
        assert fake_requests.get("https://example.com/b") == "ok"
        assert fake_requests.get("https://example.com/c", timeout=120) == "ok"
    assert seen == [
        ("https://example.com/a", 60.0),
        ("https://example.com/b", 60.0),
        ("https://example.com/c", 120),
    ]
    assert fake_requests.get is orig_get

    # Exception inside context still restores original requests.get
    with pytest.raises(RuntimeError, match="boom"):
        with patched_homr_download_timeout(60.0, download_utils_mod=fake_dl):
            raise RuntimeError("boom")
    assert fake_requests.get is orig_get

    # Missing requests attribute degrades cleanly
    with patched_homr_download_timeout(60.0, download_utils_mod=SimpleNamespace()):
        pass
