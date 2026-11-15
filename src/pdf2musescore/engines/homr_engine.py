"""homr deep-learning OMR engine (renders PDF pages via pypdfium2 and runs homr)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

from .base import OmrEngine, OmrError

log = logging.getLogger(__name__)


class CancelledError(RuntimeError):
    """Raised when the user aborts an in-progress conversion."""


_RANGE_TOKEN_RE = re.compile(r"^(\d+)(?:\s*-\s*(\d+))?$")


def get_pdf_page_count(pdf: Path) -> int:
    """Return the total number of pages in ``pdf``."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    return len(doc)


def parse_page_spec(spec: str | None, total_pages: int) -> list[int]:
    """Parse a 1-based page specification like ``'2-3, 5'`` or ``''``/``'all'`` into sorted 1-based page numbers."""
    if total_pages <= 0:
        raise OmrError("PDF has no pages")
    text = (spec or "").strip().lower()
    if not text or text == "all":
        return list(range(1, total_pages + 1))

    selected: set[int] = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        m = _RANGE_TOKEN_RE.match(token)
        if not m:
            raise OmrError(f"Invalid page range {token!r}. Use e.g. '2-3' or '1, 3-4'.")
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) is not None else start
        if start < 1 or end < start:
            raise OmrError(f"Invalid page range {token!r}. Page numbers must start at 1 and be in order.")
        if end > total_pages:
            raise OmrError(f"Page {end} is out of range (PDF has {total_pages} page(s)).")
        selected.update(range(start, end + 1))

    if not selected:
        raise OmrError("No pages selected.")
    return sorted(selected)


def _default_render_pdf(pdf: Path, out_dir: Path, dpi: int,
                        pages_spec: Optional[str] = None,
                        is_cancelled: Callable[[], bool] = lambda: False) -> list[Path]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    n_pages = len(doc)
    if n_pages == 0:
        raise OmrError(f"PDF has no pages: {pdf}")
    page_numbers = parse_page_spec(pages_spec, n_pages)
    scale = dpi / 72.0
    pages: list[Path] = []
    for pno in page_numbers:
        if is_cancelled():
            raise CancelledError("Conversion aborted by user.")
        pil = doc[pno - 1].render(scale=scale).to_pil()
        img_path = out_dir / f"page_{pno:03d}.png"
        pil.save(img_path)
        pages.append(img_path)
    return pages


def _default_run_homr(image_paths: Sequence[Path],
                      page_cb: Optional[Callable[[int, int], None]] = None,
                      is_cancelled: Callable[[], bool] = lambda: False) -> list[Path]:  # pragma: no cover - integration
    import onnxruntime as ort
    from homr.main import ProcessingConfig, XmlGeneratorArguments, download_weights, process_image
    from homr.title_detection import download_ocr_weights

    if is_cancelled():
        raise CancelledError("Conversion aborted by user.")

    ort.set_default_logger_severity(3)
    download_weights(False)
    download_ocr_weights()
    cfg = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        use_gpu_inference=False,
    )
    xargs = XmlGeneratorArguments(large_page=False, metronome=None, tempo=None)

    outputs: list[Path] = []
    total = len(image_paths)
    for idx, img in enumerate(image_paths, start=1):
        if is_cancelled():
            raise CancelledError("Conversion aborted by user.")
        if page_cb:
            page_cb(idx, total)
        try:
            process_image(str(img), cfg, xargs)
            xml_path = img.with_suffix(".musicxml")
            if xml_path.is_file():
                outputs.append(xml_path)
            else:
                log.warning("homr produced no MusicXML for page %d (%s)", idx, img.name)
        except Exception as exc:
            log.warning("homr failed on page %d (%s): %s", idx, img.name, exc)
    return outputs


class HomrEngine(OmrEngine):
    name = "homr"

    def __init__(self, dpi: int = 300,
                 pages_spec: Optional[str] = None,
                 is_cancelled: Callable[[], bool] = lambda: False,
                 renderer: Callable[..., list[Path]] = _default_render_pdf,
                 runner: Callable[..., list[Path]] = _default_run_homr,
                 page_callback: Optional[Callable[[int, int], None]] = None) -> None:
        self.dpi = dpi
        self.pages_spec = pages_spec
        self._is_cancelled = is_cancelled
        self._render = renderer
        self._run = runner
        self._page_cb = page_callback

    def recognize(self, pdf: Path, out_dir: Path) -> list[Path]:
        pdf, out_dir = Path(pdf), Path(out_dir)
        if not pdf.is_file():
            raise OmrError(f"Input PDF not found: {pdf}")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            pages = self._render(pdf, out_dir, self.dpi, self.pages_spec, self._is_cancelled)
        except (OmrError, CancelledError):
            raise
        except Exception as exc:
            raise OmrError(f"Failed to render PDF pages: {exc}") from exc

        log.info("Rendered %d page(s) at %d DPI; running homr AI...", len(pages), self.dpi)
        try:
            outputs = sorted(self._run(pages, self._page_cb, self._is_cancelled))
        except CancelledError:
            raise
        except Exception as exc:
            raise OmrError(f"homr recognition failed: {exc}") from exc

        if not outputs:
            raise OmrError("homr produced no MusicXML output (score may be unreadable)")
        log.info("homr produced %d file(s): %s", len(outputs), [p.name for p in outputs])
        return outputs
