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
from .engines.homr_engine import HomrEngine
from .exporters.musescore import ExportError, MuseScoreExporter
from .musicxml import MergeError, merge_files

log = logging.getLogger(__name__)

SUPPORTED_OUTPUTS = (".mscz", ".musicxml")
ProgressFn = Callable[[int, str], None]
ConfirmFn = Callable[[str], bool]


class ConversionError(RuntimeError):
    """Error with a plain-language message suitable for end users."""

    def __init__(self, user_message: str, detail: str = "") -> None:
        super().__init__(f"{user_message} {detail}".strip())
        self.user_message = user_message
        self.detail = detail


@dataclass
class Pipeline:
    config: AppConfig
    confirm: ConfirmFn = lambda msg: False
    progress: ProgressFn = lambda pct, msg: None
    ensure: Callable[..., Path] = ensure_tool
    engine_factory: Callable[[int, Optional[Callable[[int, int], None]]], OmrEngine] = field(
        default=lambda dpi, cb: HomrEngine(dpi=dpi, page_callback=cb))
    exporter_factory: Callable[[Path], MuseScoreExporter] = MuseScoreExporter

    def _step(self, pct: int, msg: str) -> None:
        log.info("[%3d%%] %s", pct, msg)
        self.progress(pct, msg)

    def convert(self, pdf: Path, output: Path) -> Path:
        pdf, output = Path(pdf), Path(output)
        suffix = output.suffix.lower()
        if suffix not in SUPPORTED_OUTPUTS:
            raise ConversionError(f"Unsupported output type '{suffix}'. Choose .mscz or .musicxml.")
        if not pdf.is_file():
            raise ConversionError("The selected PDF file could not be found.", str(pdf))

        safe_root = ascii_safe_base_dir(Path(self.config.work_dir).resolve())
        work = safe_root / f"job-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        log.info("Converting %s -> %s (work dir %s)", pdf, output, work)
        try:
            work.mkdir(parents=True, exist_ok=True)
            output.parent.mkdir(parents=True, exist_ok=True)

            self._step(5, "Checking required programs...")
            musescore = (self.ensure(MUSESCORE, self.config.musescore_path, self.confirm)
                         if suffix == ".mscz" else None)

            staged_pdf = work / "input.pdf"
            shutil.copy2(pdf, staged_pdf)

            self._step(15, "Reading the music with AI (first run downloads models once)...")
            engine = self.engine_factory(
                self.config.render_dpi,
                lambda idx, total: self._step(
                    15 + int(55 * (idx - 1) / max(total, 1)),
                    f"Reading page {idx} of {total} with AI...",
                ),
            )
            pages_xml = engine.recognize(staged_pdf, work / "omr")

            self._step(75, f"Combining {len(pages_xml)} page(s) into one score...")
            staged_xml = work / "score.musicxml"
            xml = merge_files(pages_xml, staged_xml, title=pdf.stem)

            if musescore is not None:
                self._step(85, "Saving MuseScore file...")
                staged_mscz = work / "score.mscz"
                self.exporter_factory(musescore).export(xml, staged_mscz)
                shutil.copy2(staged_mscz, output)
            else:
                shutil.copy2(xml, output)

            self._step(100, f"Done! Saved {output.name}")
            return output
        except ToolNotFoundError as exc:
            raise ConversionError("MuseScore is not installed, so the conversion was cancelled.",
                                  str(exc)) from exc
        except InstallError as exc:
            raise ConversionError("Installing MuseScore failed. Check your internet connection "
                                  "and try again.", str(exc)) from exc
        except OmrError as exc:
            raise ConversionError("The music could not be read from this PDF. Clear, printed scores "
                                  "work best.", str(exc)) from exc
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
