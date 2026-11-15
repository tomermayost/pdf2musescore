"""Conversion pipeline: PDF -> Audiveris -> combined MusicXML -> (optional) MuseScore .mscz."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import AppConfig
from .deps import AUDIVERIS, MUSESCORE, InstallError, ToolNotFoundError, ensure_tool
from .engines.audiveris import AudiverisEngine
from .engines.base import OmrEngine, OmrError
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
    engine_factory: Callable[[Path, int], OmrEngine] = field(
        default=lambda exe, timeout: AudiverisEngine(exe, timeout))
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

        work = Path(self.config.work_dir) / f"job-{time.strftime('%Y%m%d-%H%M%S')}-{pdf.stem}"
        log.info("Converting %s -> %s (work dir %s)", pdf, output, work)
        try:
            self._step(5, "Checking required programs...")
            audiveris = self.ensure(AUDIVERIS, self.config.audiveris_path, self.confirm)
            musescore = (self.ensure(MUSESCORE, self.config.musescore_path, self.confirm)
                         if suffix == ".mscz" else None)

            self._step(15, "Reading the music from the PDF (this can take a few minutes)...")
            engine = self.engine_factory(audiveris, self.config.audiveris_timeout_sec)
            movements = engine.recognize(pdf, work / "omr")

            self._step(75, f"Combining {len(movements)} movement(s) into one score...")
            xml_target = output if suffix == ".musicxml" else work / f"{pdf.stem}.musicxml"
            xml = merge_files(movements, xml_target, title=pdf.stem)

            if musescore is not None:
                self._step(85, "Saving MuseScore file...")
                self.exporter_factory(musescore).export(xml, output)

            self._step(100, f"Done! Saved {output.name}")
            return output
        except ToolNotFoundError as exc:
            raise ConversionError("A required program is not installed, so the conversion was cancelled.",
                                  str(exc)) from exc
        except InstallError as exc:
            raise ConversionError("Installing a required program failed. Check your internet connection "
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
