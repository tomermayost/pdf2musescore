"""Verified, staged, thread-safe AI model provisioning and integrity tracking."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

from .engines.homr_engine import (
    CancelledError,
    _apply_homr_model_paths,
    configure_model_cache,
    get_engine_version,
    get_rapidocr_version,
    prune_stale_version_dirs,
    sanitize_version,
)

log = logging.getLogger(__name__)

EXPECTED_DOWNLOAD_BYTES = 190_000_000
MANIFEST_NAME = ".pdf2musescore-models.json"
_MODEL_DOWNLOAD_LOCK = threading.Lock()

ModelProgressFn = Callable[[int, int], None]
CancelFn = Callable[[], bool]


class ModelDownloadError(RuntimeError):
    """Raised when downloading or verifying AI model weights fails."""


@dataclass(frozen=True)
class ModelStatus:
    """Read-only availability report for the local AI model cache."""

    ready: bool
    missing: tuple[str, ...]
    homr_dir: Path
    ocr_dir: Path


def _default_required_homr_files() -> dict[str, str]:  # pragma: no cover - imports homr configs
    import homr.segmentation.config as homr_seg_cfg
    import homr.transformer.configs as homr_tr_cfg

    fp = homr_tr_cfg.default_config.filepaths
    return {
        Path(homr_seg_cfg.segnet_path_onnx).name: "segmentation",
        Path(fp.encoder_path).name: "transformer",
        Path(fp.decoder_path).name: "transformer",
    }


def _default_download_homr(seg_staging: Path, tr_staging: Path, final_models_dir: Path) -> None:  # pragma: no cover
    import homr.main as homr_main

    _apply_homr_model_paths(seg_staging, tr_staging, homr_main=homr_main)
    try:
        homr_main.download_weights(segnet_use_gpu=False, transformer_use_gpu=False, coreml_encoder=False)
    finally:
        configure_model_cache(final_models_dir)


def _default_init_ocr(ocr_dir: Path) -> None:  # pragma: no cover
    from rapidocr import RapidOCR

    RapidOCR(params={"Global.model_root_dir": str(ocr_dir)})


def _read_manifest(ver_dir: Path) -> Optional[dict[str, int]]:
    mpath = ver_dir / MANIFEST_NAME
    if not mpath.is_file():
        return None
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
        files = data.get("files") if isinstance(data, dict) else None
        if not isinstance(files, dict) or not files:
            return None
        return {str(k): int(v) for k, v in files.items()}
    except Exception:
        return None


def _verify_manifest(ver_dir: Path, label: str) -> list[str]:
    manifest = _read_manifest(ver_dir)
    if manifest is None:
        return [label]
    problems: list[str] = []
    for rel, expected_size in manifest.items():
        target = ver_dir / rel
        if not target.is_file():
            problems.append(f"{label}/{rel}")
            continue
        try:
            actual_size = target.stat().st_size
        except OSError:
            problems.append(f"{label}/{rel}")
            continue
        if expected_size <= 0 or actual_size != expected_size:
            problems.append(f"{label}/{rel}")
    return problems


def _write_manifest(ver_dir: Path, version: str, files: dict[str, int]) -> None:
    ver_dir.mkdir(parents=True, exist_ok=True)
    mpath = ver_dir / MANIFEST_NAME
    tmp_path = ver_dir / f"{MANIFEST_NAME}.tmp"
    payload = {
        "version": version,
        "created": datetime.now().isoformat(timespec="seconds"),
        "files": files,
    }
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, mpath)


def _dir_bytes(paths: Sequence[Path]) -> int:
    total = 0
    for root in paths:
        if not root.is_dir():
            continue
        for entry in root.rglob("*"):
            if entry.is_file() and entry.name != MANIFEST_NAME and not entry.name.endswith(".tmp"):
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    return total


def _run_with_progress(
    work: Callable[[], None],
    watch_dirs: Sequence[Path],
    base_bytes: int,
    progress: Optional[ModelProgressFn],
    poll_interval: float,
) -> None:
    if progress is None:
        work()
        return

    stop = threading.Event()
    interval = max(0.005, float(poll_interval))

    def monitor() -> None:
        while not stop.wait(interval):
            cur = base_bytes + _dir_bytes(watch_dirs)
            try:
                progress(cur, max(EXPECTED_DOWNLOAD_BYTES, cur))
            except Exception:
                pass

    t = threading.Thread(target=monitor, daemon=True, name="model-progress")
    t.start()
    try:
        work()
    finally:
        stop.set()
        t.join(timeout=2.0)

    cur = base_bytes + _dir_bytes(watch_dirs)
    try:
        progress(cur, max(EXPECTED_DOWNLOAD_BYTES, cur))
    except Exception:
        pass


class ModelManager:
    """Manages version-scoped downloading, staging, and verification of homr & RapidOCR weights."""

    def __init__(
        self,
        models_dir: Path,
        *,
        homr_version: Optional[str] = None,
        ocr_version: Optional[str] = None,
        required_homr_files: Optional[Callable[[], dict[str, str]]] = None,
        download_homr: Optional[Callable[[Path, Path], None]] = None,
        init_ocr: Optional[Callable[[Path], None]] = None,
        poll_interval: float = 0.5,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.homr_version = sanitize_version(homr_version or get_engine_version())
        self.ocr_version = sanitize_version(ocr_version or get_rapidocr_version())
        self.homr_dir = self.models_dir / "homr" / self.homr_version
        self.ocr_dir = self.models_dir / "rapidocr" / self.ocr_version
        self._required_homr_files = required_homr_files or _default_required_homr_files
        self._download_homr = download_homr or (
            lambda seg, tr: _default_download_homr(seg, tr, self.models_dir)
        )
        self._init_ocr = init_ocr or _default_init_ocr
        self._poll_interval = float(poll_interval)

    @staticmethod
    def is_downloading() -> bool:
        """Return True if another thread currently holds the model download lock."""
        return _MODEL_DOWNLOAD_LOCK.locked()

    def status(self) -> ModelStatus:
        """Fast local manifest check (no network, no heavy AI library imports)."""
        missing = _verify_manifest(self.homr_dir, "homr") + _verify_manifest(self.ocr_dir, "rapidocr")
        return ModelStatus(
            ready=len(missing) == 0,
            missing=tuple(missing),
            homr_dir=self.homr_dir,
            ocr_dir=self.ocr_dir,
        )

    def _ensure_homr(
        self,
        progress: Optional[ModelProgressFn],
        is_cancelled: CancelFn,
        *,
        allow_adopt: bool = True,
        bad_files: Sequence[str] = (),
    ) -> dict[str, int]:
        req = self._required_homr_files()
        bad_set = set(bad_files)
        force_all = "homr" in bad_set
        existing: dict[str, int] = {}
        missing_files: list[str] = []
        for fname, subdir in req.items():
            rel = f"{subdir}/{fname}"
            final_file = self.homr_dir / subdir / fname
            is_bad = not allow_adopt and (force_all or f"homr/{rel}" in bad_set)
            if not is_bad and final_file.is_file() and final_file.stat().st_size > 0:
                existing[rel] = final_file.stat().st_size
            else:
                missing_files.append(rel)

        if allow_adopt and not missing_files:
            log.info(
                "Adopting existing homr %s model files (%d file(s)) in %s",
                self.homr_version,
                len(existing),
                self.homr_dir,
            )
            _write_manifest(self.homr_dir, self.homr_version, existing)
            return existing

        log.info("Missing or invalid homr %s model file(s): %s", self.homr_version, missing_files)
        staging_root = self.homr_dir / ".staging"
        shutil.rmtree(staging_root, ignore_errors=True)
        seg_staging = staging_root / "segmentation"
        tr_staging = staging_root / "transformer"
        seg_staging.mkdir(parents=True, exist_ok=True)
        tr_staging.mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
        try:
            _run_with_progress(
                lambda: self._download_homr(seg_staging, tr_staging),
                [staging_root],
                0,
                progress,
                self._poll_interval,
            )
            if is_cancelled():
                raise CancelledError("Model download aborted by user.")

            manifest_files: dict[str, int] = {}
            for fname, subdir in req.items():
                rel = f"{subdir}/{fname}"
                staged = staging_root / subdir / fname
                final = self.homr_dir / subdir / fname
                must_replace = not allow_adopt and (force_all or f"homr/{rel}" in bad_set)
                if staged.is_file() and staged.stat().st_size > 0:
                    final.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staged, final)
                elif must_replace or not (final.is_file() and final.stat().st_size > 0):
                    raise ModelDownloadError(f"Expected homr model file missing after download: {rel}")
                manifest_files[rel] = final.stat().st_size

            _write_manifest(self.homr_dir, self.homr_version, manifest_files)
            prune_stale_version_dirs(self.models_dir / "homr", self.homr_version)
            log.info(
                "Provisioned homr %s models in %.2fs (%d bytes -> %s)",
                self.homr_version,
                time.monotonic() - started,
                sum(manifest_files.values()),
                self.homr_dir,
            )
            return manifest_files
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    def _ensure_ocr(
        self,
        base_bytes: int,
        progress: Optional[ModelProgressFn],
        is_cancelled: CancelFn,
    ) -> dict[str, int]:
        self.ocr_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        _run_with_progress(
            lambda: self._init_ocr(self.ocr_dir),
            [self.ocr_dir],
            base_bytes,
            progress,
            self._poll_interval,
        )
        if is_cancelled():
            raise CancelledError("Model download aborted by user.")

        ocr_files: dict[str, int] = {}
        for entry in sorted(self.ocr_dir.rglob("*")):
            if entry.is_file() and entry.name != MANIFEST_NAME and not entry.name.endswith(".tmp"):
                rel = entry.relative_to(self.ocr_dir).as_posix()
                sz = entry.stat().st_size
                if sz > 0:
                    ocr_files[rel] = sz

        if not ocr_files:
            raise ModelDownloadError(f"RapidOCR produced no model files in {self.ocr_dir}")

        _write_manifest(self.ocr_dir, self.ocr_version, ocr_files)
        prune_stale_version_dirs(self.models_dir / "rapidocr", self.ocr_version)
        log.info(
            "Provisioned rapidocr %s models in %.2fs (%d bytes -> %s)",
            self.ocr_version,
            time.monotonic() - started,
            sum(ocr_files.values()),
            self.ocr_dir,
        )
        return ocr_files

    def ensure(
        self,
        progress: Optional[ModelProgressFn] = None,
        is_cancelled: CancelFn = lambda: False,
    ) -> ModelStatus:
        """Download and verify any missing or incomplete AI models under a thread lock."""
        with _MODEL_DOWNLOAD_LOCK:
            st = self.status()
            if st.ready:
                log.debug("AI models already verified (%s, %s)", st.homr_dir, st.ocr_dir)
                return st

            if is_cancelled():
                raise CancelledError("Model download aborted by user.")

            log.info("Ensuring AI models (missing=%s, dir=%s)", list(st.missing), self.models_dir)
            try:
                homr_problem = _verify_manifest(self.homr_dir, "homr")
                if homr_problem:
                    had_homr_manifest = (self.homr_dir / MANIFEST_NAME).is_file()
                    # Clear invalid manifest before re-ensuring so an interrupted run stays marked incomplete
                    (self.homr_dir / MANIFEST_NAME).unlink(missing_ok=True)
                    homr_files = self._ensure_homr(
                        progress,
                        is_cancelled,
                        allow_adopt=not had_homr_manifest,
                        bad_files=homr_problem,
                    )
                else:
                    homr_files = _read_manifest(self.homr_dir) or {}
                homr_bytes = sum(homr_files.values())

                if is_cancelled():
                    raise CancelledError("Model download aborted by user.")

                ocr_problem = _verify_manifest(self.ocr_dir, "rapidocr")
                if ocr_problem:
                    (self.ocr_dir / MANIFEST_NAME).unlink(missing_ok=True)
                    ocr_files = self._ensure_ocr(homr_bytes, progress, is_cancelled)
                else:
                    ocr_files = _read_manifest(self.ocr_dir) or {}
                total_bytes = homr_bytes + sum(ocr_files.values())

                if progress is not None:
                    progress(total_bytes, max(total_bytes, EXPECTED_DOWNLOAD_BYTES))
                return self.status()
            except (CancelledError, ModelDownloadError):
                raise
            except Exception as exc:
                log.exception("AI model download failed")
                raise ModelDownloadError(f"{type(exc).__name__}: {exc}") from exc
