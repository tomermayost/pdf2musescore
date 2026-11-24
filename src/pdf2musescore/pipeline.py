"""Conversion pipeline: PDF -> homr AI -> combined MusicXML -> (optional) MuseScore .mscz."""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import AppConfig, ascii_safe_base_dir
from .deps import MUSESCORE, InstallError, ToolNotFoundError, ensure_tool
from .engines.base import OmrEngine, OmrError
from .engines.homr_engine import CancelledError, HomrEngine, is_supported_input
from .exporters.musescore import ExportError, MuseScoreExporter
from .musicxml import MergeError, merge_files
from .options import ConversionOptions, OptionsError

log = logging.getLogger(__name__)

SUPPORTED_OUTPUTS = (".mscz", ".musicxml")
ProgressFn = Callable[[int, str], None]
ConfirmFn = Callable[[str], bool]
CancelFn = Callable[[], bool]


class ConversionError(RuntimeError):
    """Error with a plain-language message suitable for end users."""

    def __init__(self, user_message: str, detail: str = "") -> None:
        super().__init__(f"{user_message} {detail}".strip())
        self.user_message = user_message
        self.detail = detail


def _default_engine_factory(
    dpi: int,
    cb: Optional[Callable[[int, int], None]],
    pages_spec: Optional[str] = None,
    is_cancelled: CancelFn = lambda: False,
    options: Optional[ConversionOptions] = None,
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

    def _step(self, pct: int, msg: str) -> None:
        if self.is_cancelled():
            raise CancelledError("Conversion aborted by user.")
        log.info("[%3d%%] %s", pct, msg)
        self.progress(pct, msg)

    def convert(self, pdf: Path, output: Path, pages_spec: Optional[str] = None,
                options: Optional[ConversionOptions] = None) -> Path:
        pdf, output = Path(pdf), Path(output)
        suffix = output.suffix.lower()
        if suffix not in SUPPORTED_OUTPUTS:
            raise ConversionError(f"Unsupported output type '{suffix}'. Choose .mscz or .musicxml.")
        if not pdf.is_file():
            raise ConversionError("The selected file could not be found.", str(pdf))
        if not is_supported_input(pdf):
            raise ConversionError(
                f"Unsupported input file '{pdf.suffix}'. Choose a PDF or sheet-music image (.png, .jpg, .webp, .bmp, .tif)."
            )

        opts = options or ConversionOptions(render_dpi=self.config.render_dpi)
        safe_root = ascii_safe_base_dir(Path(self.config.work_dir).resolve())
        work = safe_root / f"job-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        log.info(
            "Converting %s (pages=%r, options=%s) -> %s (work dir %s)",
            pdf, pages_spec or "all", opts, output, work,
        )
        try:
            work.mkdir(parents=True, exist_ok=True)
            output.parent.mkdir(parents=True, exist_ok=True)

            self._step(5, "Checking required programs...")
            musescore = (self.ensure(MUSESCORE, self.config.musescore_path, self.confirm)
                         if suffix == ".mscz" else None)

            staged_input = work / f"input{pdf.suffix.lower()}"
            shutil.copy2(pdf, staged_input)

            self._step(15, "Reading the music with AI (first run downloads models once)...")
            page_cb = lambda idx, total: self._step(
                15 + int(55 * (idx - 1) / max(total, 1)),
                f"Reading page {idx} of {total} with AI...",
            )
            if self.engine_factory is _default_engine_factory:
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

            self._step(75, f"Combining {len(pages_xml)} page(s) into one score...")
            staged_xml = work / "score.musicxml"
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
            log.info("Conversion aborted by user")
            raise
        except OptionsError as exc:
            raise ConversionError(str(exc)) from exc
        except ToolNotFoundError as exc:
            raise ConversionError(
                "MuseScore is not installed. Save as MusicXML (.musicxml) or install MuseScore 4 from musescore.org.",
                str(exc),
            ) from exc
        except InstallError as exc:
            raise ConversionError("Installing MuseScore failed. Check your internet connection "
                                  "and try again.", str(exc)) from exc
        except OmrError as exc:
            msg = str(exc)
            if "page" in msg.lower() and ("range" in msg.lower() or "no pages" in msg.lower()):
                raise ConversionError(msg) from exc
            raise ConversionError("The music could not be read from this PDF. Clear, printed scores "
                                  "work best.", msg) from exc
        except MergeError as exc:
            raise ConversionError("The recognized music could not be combined into one score.",
                                  str(exc)) from exc
        except ExportError as exc:
            raise ConversionError("MuseScore could not save the file. Try saving as MusicXML instead.",
                                  str(exc)) from exc
        finally:
            if self.config.log_level == "DEBUG":
                log.debug("Keeping work dir for debugging: %s", work)
            else:
                shutil.rmtree(work, ignore_errors=True)
