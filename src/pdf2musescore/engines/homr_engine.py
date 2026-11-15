"""homr deep-learning OMR engine (renders PDF pages via pypdfium2, preprocesses, and runs homr)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

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


def deskew_image(bgr: np.ndarray) -> np.ndarray:
    """Estimate page skew from near-horizontal staff lines (up to +/-10 deg) and rotate to level."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=100, minLineLength=max(40, gray.shape[1] // 8), maxLineGap=15
    )
    if lines is None or len(lines) == 0:
        return bgr
    angles: list[float] = []
    for line in np.reshape(lines, (-1, 4)):
        x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
        if x2 == x1:
            continue
        deg = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if -10.0 <= deg <= 10.0:
            angles.append(deg)
    if not angles:
        return bgr
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.05:
        return bgr
    h, w = bgr.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), median_angle, 1.0)
    return cv2.warpAffine(bgr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def preprocess_page_image(bgr: np.ndarray, mode: str = "auto", deskew: bool = False) -> np.ndarray:
    """Apply optional deskew and image enhancement ('auto', 'faded', 'binarize', 'none')."""
    out = deskew_image(bgr) if deskew else bgr
    mode = (mode or "auto").strip().lower()
    if mode in ("auto", "none"):
        return out
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
    if mode == "faded":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        boosted = cv2.convertScaleAbs(clahe, alpha=1.35, beta=-35)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(boosted, -1, kernel)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    if mode == "binarize":
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return out


def _default_render_pdf(pdf: Path, out_dir: Path, dpi: int,
                        pages_spec: Optional[str] = None,
                        is_cancelled: Callable[[], bool] = lambda: False,
                        preprocess_mode: str = "auto",
                        deskew: bool = False) -> list[Path]:
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
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        bgr = preprocess_page_image(bgr, mode=preprocess_mode, deskew=deskew)
        img_path = out_dir / f"page_{pno:03d}.png"
        cv2.imwrite(str(img_path), bgr)
        pages.append(img_path)
    return pages


def _default_run_homr(image_paths: Sequence[Path],
                      page_cb: Optional[Callable[[int, int], None]] = None,
                      is_cancelled: Callable[[], bool] = lambda: False,
                      segnet_step_size: int = 320,
                      min_notehead_size: int = 4,
                      preprocess_mode: str = "auto") -> list[Path]:  # pragma: no cover - integration
    import homr.main as hmain
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

    orig_extract = hmain.extract
    orig_ellipses = hmain.create_bounding_ellipses
    orig_clahe = hmain.color_adjust.apply_clahe

    def _custom_extract(preprocessed, img_path, step_size=320, use_cache=False, use_gpu_inference=False):
        return orig_extract(
            preprocessed,
            img_path,
            step_size=int(segnet_step_size),
            use_cache=use_cache,
            use_gpu_inference=use_gpu_inference,
        )

    def _custom_ellipses(notehead_map, min_size=(4, 4), **kwargs):
        ms = (int(min_notehead_size), int(min_notehead_size))
        return orig_ellipses(notehead_map, min_size=ms, **kwargs)

    hmain.extract = _custom_extract
    hmain.create_bounding_ellipses = _custom_ellipses
    if (preprocess_mode or "auto").strip().lower() == "none":
        hmain.color_adjust.apply_clahe = lambda img: img

    try:
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
    finally:
        hmain.extract = orig_extract
        hmain.create_bounding_ellipses = orig_ellipses
        hmain.color_adjust.apply_clahe = orig_clahe


class HomrEngine(OmrEngine):
    name = "homr"

    def __init__(self, dpi: int = 300,
                 pages_spec: Optional[str] = None,
                 is_cancelled: Callable[[], bool] = lambda: False,
                 preprocess_mode: str = "auto",
                 deskew: bool = False,
                 segnet_step_size: int = 320,
                 min_notehead_size: int = 4,
                 renderer: Callable[..., list[Path]] = _default_render_pdf,
                 runner: Callable[..., list[Path]] = _default_run_homr,
                 page_callback: Optional[Callable[[int, int], None]] = None) -> None:
        self.dpi = dpi
        self.pages_spec = pages_spec
        self._is_cancelled = is_cancelled
        self.preprocess_mode = preprocess_mode
        self.deskew = deskew
        self.segnet_step_size = segnet_step_size
        self.min_notehead_size = min_notehead_size
        self._render = renderer
        self._run = runner
        self._page_cb = page_callback

    def recognize(self, pdf: Path, out_dir: Path) -> list[Path]:
        pdf, out_dir = Path(pdf), Path(out_dir)
        if not pdf.is_file():
            raise OmrError(f"Input PDF not found: {pdf}")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self._render is _default_render_pdf:
                pages = self._render(
                    pdf, out_dir, self.dpi, self.pages_spec, self._is_cancelled,
                    self.preprocess_mode, self.deskew
                )
            else:
                pages = self._render(pdf, out_dir, self.dpi, self.pages_spec, self._is_cancelled)
        except (OmrError, CancelledError):
            raise
        except Exception as exc:
            raise OmrError(f"Failed to render PDF pages: {exc}") from exc

        log.info(
            "Rendered %d page(s) at %d DPI (preprocess=%s, deskew=%s, step=%d, min_notehead=%d); running homr AI...",
            len(pages), self.dpi, self.preprocess_mode, self.deskew,
            self.segnet_step_size, self.min_notehead_size,
        )
        try:
            if self._run is _default_run_homr:
                outputs = sorted(
                    self._run(
                        pages, self._page_cb, self._is_cancelled,
                        self.segnet_step_size, self.min_notehead_size, self.preprocess_mode,
                    )
                )
            else:
                outputs = sorted(self._run(pages, self._page_cb, self._is_cancelled))
        except CancelledError:
            raise
        except Exception as exc:
            raise OmrError(f"homr recognition failed: {exc}") from exc

        if not outputs:
            raise OmrError("homr produced no MusicXML output (score may be unreadable)")
        log.info("homr produced %d file(s): %s", len(outputs), [p.name for p in outputs])
        return outputs
