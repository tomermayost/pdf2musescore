"""homr deep-learning OMR engine (renders PDF pages via pypdfium2 and runs homr)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from .base import OmrEngine, OmrError

log = logging.getLogger(__name__)


def _default_render_pdf(pdf: Path, out_dir: Path, dpi: int) -> list[Path]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    n_pages = len(doc)
    if n_pages == 0:
        raise OmrError(f"PDF has no pages: {pdf}")
    scale = dpi / 72.0
    pages: list[Path] = []
    for idx in range(n_pages):
        pil = doc[idx].render(scale=scale).to_pil()
        img_path = out_dir / f"page_{idx + 1:03d}.png"
        pil.save(img_path)
        pages.append(img_path)
    return pages


def _default_run_homr(image_paths: list[Path],
                      page_cb: Optional[Callable[[int, int], None]] = None) -> list[Path]:  # pragma: no cover - integration
    import onnxruntime as ort
    from homr.main import ProcessingConfig, XmlGeneratorArguments, download_weights, process_image
    from homr.title_detection import download_ocr_weights

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
                 renderer: Callable[[Path, Path, int], list[Path]] = _default_render_pdf,
                 runner: Callable[[list[Path], Optional[Callable[[int, int], None]]], list[Path]] = _default_run_homr,
                 page_callback: Optional[Callable[[int, int], None]] = None) -> None:
        self.dpi = dpi
        self._render = renderer
        self._run = runner
        self._page_cb = page_callback

    def recognize(self, pdf: Path, out_dir: Path) -> list[Path]:
        pdf, out_dir = Path(pdf), Path(out_dir)
        if not pdf.is_file():
            raise OmrError(f"Input PDF not found: {pdf}")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            pages = self._render(pdf, out_dir, self.dpi)
        except OmrError:
            raise
        except Exception as exc:
            raise OmrError(f"Failed to render PDF pages: {exc}") from exc

        log.info("Rendered %d page(s) at %d DPI; running homr AI...", len(pages), self.dpi)
        try:
            outputs = sorted(self._run(pages, self._page_cb))
        except Exception as exc:
            raise OmrError(f"homr recognition failed: {exc}") from exc

        if not outputs:
            raise OmrError("homr produced no MusicXML output (score may be unreadable)")
        log.info("homr produced %d file(s): %s", len(outputs), [p.name for p in outputs])
        return outputs
