"""Conversion pipeline: PDF -> homr AI -> combined MusicXML -> (optional) MuseScore .mscz."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import AppConfig, ascii_safe_base_dir
from .deps import MUSESCORE, InstallError, ToolNotFoundError, ensure_tool
from .diagnostics import check_low_memory_warning, memory_snapshot
from .engines.base import OmrEngine, OmrError
from .engines.homr_engine import CancelledError, HomrEngine, is_supported_input
from .exporters.musescore import ExportError, MuseScoreExporter
from .fullscore.align import merge_fullscore_files
from .model_store import ModelDownloadError, ModelManager
from .musicxml import MergeError, merge_files
from .options import ConversionOptions, OptionsError

log = logging.getLogger(__name__)

SUPPORTED_OUTPUTS = (".mscz", ".musicxml")
ProgressFn = Callable[[int, str], None]
ConfirmFn = Callable[[str], bool]
CancelFn = Callable[[], bool]


class ConversionError(RuntimeError):
    """Error with a plain-language message suitable for end users."""

    def __init__(self, user_message: str, detail: str = "", stage: str = "") -> None:
        super().__init__(f"{user_message} {detail}".strip())
        self.user_message = user_message
        self.detail = detail
        self.stage = stage


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _prune_old_jobs(safe_root: Path, keep: int = 3) -> list[Path]:
    """Delete oldest ``job-*`` folders under ``safe_root`` beyond the newest ``keep``."""
    if not safe_root.is_dir() or keep < 0:
        return []
    jobs: list[tuple[float, str, Path]] = []
    for entry in safe_root.iterdir():
        if entry.is_dir() and entry.name.startswith("job-"):
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                mtime = 0.0
            jobs.append((mtime, entry.name, entry))
    if len(jobs) <= keep:
        return []
    jobs.sort(key=lambda item: (item[0], item[1]))
    to_remove = [item[2] for item in jobs[:-keep]] if keep > 0 else [item[2] for item in jobs]
    removed: list[Path] = []
    for old_dir in to_remove:
        try:
            shutil.rmtree(old_dir)
            removed.append(old_dir)
            log.info("Removed old work folder: %s", old_dir)
        except OSError as exc:
            log.warning("Could not remove old work folder %s: %s", old_dir, exc)
    return removed


def _map_conversion_error(exc: Exception, detail: str, stage: str) -> ConversionError:
    if isinstance(exc, OptionsError):
        return ConversionError(str(exc), detail, stage=stage)
    if isinstance(exc, ToolNotFoundError):
        return ConversionError(
            "MuseScore is not installed. Save as MusicXML (.musicxml) or install MuseScore 4 from musescore.org.",
            detail,
            stage=stage,
        )
    if isinstance(exc, InstallError):
        return ConversionError(
            "Installing MuseScore failed. Check your internet connection and try again.",
            detail,
            stage=stage,
        )
    if isinstance(exc, ModelDownloadError):
        return ConversionError(
            "The AI models could not be downloaded. Check your internet connection and try again.",
            detail,
            stage=stage,
        )
    if isinstance(exc, OmrError):
        msg = str(exc)
        if "page" in msg.lower() and ("range" in msg.lower() or "no pages" in msg.lower()):
            return ConversionError(msg, detail, stage=stage)
        return ConversionError(
            "The music could not be read from this PDF. Clear, printed scores work best.",
            detail,
            stage=stage,
        )
    if isinstance(exc, MergeError):
        return ConversionError(
            "The recognized music could not be combined into one score.",
            detail,
            stage=stage,
        )
    if isinstance(exc, ExportError):
        return ConversionError(
            "MuseScore could not save the file. Try saving as MusicXML instead.",
            detail,
            stage=stage,
        )
    return ConversionError(
        "Something unexpected went wrong while converting.",
        detail,
        stage=stage,
    )


def _default_engine_factory(
    dpi: int,
    cb: Optional[Callable[[int, int], None]],
    pages_spec: Optional[str] = None,
    is_cancelled: CancelFn = lambda: False,
    options: Optional[ConversionOptions] = None,
    staff_cb: Optional[Callable[[int, int, int, int], None]] = None,
) -> OmrEngine:
    opts = options or ConversionOptions(render_dpi=dpi)
    return HomrEngine(
        dpi=opts.render_dpi,
        pages_spec=pages_spec,
        is_cancelled=is_cancelled,
        preprocess_mode=opts.preprocess_mode,
        deskew=opts.deskew,
        segnet_step_size=opts.segnet_step_size,
        min_notehead_size=opts.min_notehead_size,
        page_callback=cb,
        full_score=opts.full_score,
        full_score_target_width=opts.full_score_target_width,
        staff_callback=staff_cb,
    )


@dataclass
class Pipeline:
    config: AppConfig
    confirm: ConfirmFn = lambda msg: False
    progress: ProgressFn = lambda pct, msg: None
    is_cancelled: CancelFn = lambda: False
    ensure: Callable[..., Path] = ensure_tool
    engine_factory: Callable[..., OmrEngine] = field(default=_default_engine_factory)
    exporter_factory: Callable[[Path], MuseScoreExporter] = MuseScoreExporter
    model_manager: Optional[ModelManager] = None
    _current_stage: str = field(default="Starting", init=False, repr=False)

    def _step(self, pct: int, msg: str) -> None:
        if self.is_cancelled():
            raise CancelledError("Conversion aborted by user.")
        self._current_stage = msg
        log.info("[%3d%%] %s", pct, msg)
        self.progress(pct, msg)

    def convert(self, pdf: Path, output: Path, pages_spec: Optional[str] = None,
                options: Optional[ConversionOptions] = None) -> Path:
        self._current_stage = "Validating input"
        pdf, output = Path(pdf), Path(output)
        suffix = output.suffix.lower()
        if suffix not in SUPPORTED_OUTPUTS:
            raise ConversionError(
                f"Unsupported output type '{suffix}'. Choose .mscz or .musicxml.",
                stage=self._current_stage,
            )
        if not pdf.is_file():
            raise ConversionError("The selected file could not be found.", str(pdf), stage=self._current_stage)
        if not is_supported_input(pdf):
            raise ConversionError(
                f"Unsupported input file '{pdf.suffix}'. Choose a PDF or sheet-music image (.png, .jpg, .webp, .bmp, .tif).",
                stage=self._current_stage,
            )

        opts = options or ConversionOptions(render_dpi=self.config.render_dpi)
        safe_root = ascii_safe_base_dir(Path(self.config.work_dir).resolve())
        _prune_old_jobs(safe_root, keep=3)
        work = safe_root / f"job-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        try:
            file_size = pdf.stat().st_size
            file_hash = _sha256_file(pdf)
        except OSError:
            file_size = -1
            file_hash = "unavailable"
        log.info(
            "Converting %s (size=%d bytes, sha256=%s, pages=%r, options=%s, %s) -> %s (work dir %s)",
            pdf, file_size, file_hash, pages_spec or "all", opts, memory_snapshot(), output, work,
        )
        check_low_memory_warning(log, render_dpi=opts.render_dpi, full_score=opts.full_score)
        failed = False
        try:
            work.mkdir(parents=True, exist_ok=True)
            output.parent.mkdir(parents=True, exist_ok=True)

            self._step(5, "Checking required programs...")
            musescore = (self.ensure(MUSESCORE, self.config.musescore_path, self.confirm)
                         if suffix == ".mscz" else None)

            staged_input = work / f"input{pdf.suffix.lower()}"
            shutil.copy2(pdf, staged_input)

            manager = self.model_manager
            if manager is None and self.engine_factory is _default_engine_factory:
                manager = ModelManager(Path(self.config.models_dir))
            if manager is not None and not manager.status().ready:
                self._step(10, "Downloading AI models (one time, ~180 MB)...")
                if manager.is_downloading():
                    self.progress(10, "Waiting for AI model download to finish...")
                manager.ensure(
                    progress=lambda d, t: self.progress(
                        10 + int(5 * min(d, max(t, 1)) / max(t, 1)),
                        f"Downloading AI models (one time): {d / 1e6:.0f} of ~{max(t, 1) / 1e6:.0f} MB...",
                    ),
                    is_cancelled=self.is_cancelled,
                )

            self._step(15, "Reading the music with AI...")
            page_cb = lambda idx, total: self._step(
                15 + int(55 * (idx - 1) / max(total, 1)),
                f"Reading page {idx} of {total} with AI...",
            )
            staff_cb = lambda p_idx, p_tot, s_idx, s_tot: self._step(
                15 + int(55 * ((p_idx - 1) + (s_idx - 1) / max(s_tot, 1)) / max(p_tot, 1)),
                f"Reading page {p_idx} of {p_tot} (staff {s_idx}/{s_tot})...",
            )
            if self.engine_factory is _default_engine_factory:
                if opts.full_score:
                    engine = self.engine_factory(
                        opts.render_dpi, page_cb, pages_spec, self.is_cancelled, opts, staff_cb=staff_cb
                    )
                else:
                    engine = self.engine_factory(
                        opts.render_dpi, page_cb, pages_spec, self.is_cancelled, opts
                    )
            else:
                engine = self.engine_factory(
                    opts.render_dpi, page_cb, pages_spec, self.is_cancelled
                )
            if getattr(engine, "models_dir", None) is None and hasattr(engine, "models_dir"):
                engine.models_dir = Path(self.config.models_dir)
            pages_xml = engine.recognize(staged_input, work / "omr")

            staged_xml = work / "score.musicxml"
            if opts.full_score:
                self._step(75, f"Aligning & combining {len(pages_xml)} full-score page(s)...")
                xml = merge_fullscore_files(pages_xml, staged_xml, title=output.stem, options=opts)
            else:
                self._step(75, f"Combining {len(pages_xml)} page(s) into one score...")
                xml = merge_files(pages_xml, staged_xml, title=output.stem, options=opts)

            if musescore is not None:
                self._step(85, "Saving MuseScore file...")
                staged_mscz = work / "score.mscz"
                self.exporter_factory(musescore).export(xml, staged_mscz)
                shutil.copy2(staged_mscz, output)
            else:
                shutil.copy2(xml, output)

            self._step(100, f"Done! Saved {output.name}")
            return output
        except CancelledError:
            log.info("Conversion aborted by user at stage %r", self._current_stage)
            raise
        except Exception as exc:
            failed = True
            detail = f"{type(exc).__name__}: {exc}"
            log.error("Conversion failed during stage %r: %s", self._current_stage, detail, exc_info=True)
            raise _map_conversion_error(exc, detail, self._current_stage) from exc
        finally:
            if failed:
                log.warning("Kept work folder for diagnosis: %s", work)
            elif self.config.log_level == "DEBUG":
                log.debug("Keeping work dir for debugging: %s", work)
            else:
                shutil.rmtree(work, ignore_errors=True)
