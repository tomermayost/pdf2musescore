"""homr deep-learning OMR engine (renders PDF pages via pypdfium2, preprocesses, and runs homr)."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..diagnostics import memory_snapshot
from .base import OmrEngine, OmrError

log = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
SUPPORTED_INPUT_EXTS: tuple[str, ...] = (".pdf",) + SUPPORTED_IMAGE_EXTS


def is_image_file(path: Path) -> bool:
    """Return True if ``path`` has a supported sheet-music image file extension."""
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTS


def is_supported_input(path: Path) -> bool:
    """Return True if ``path`` is a supported PDF or sheet-music image file."""
    return Path(path).suffix.lower() in SUPPORTED_INPUT_EXTS


def load_pil_image_rgb(path: Path, page_number: int = 1) -> Image.Image:
    """Load an image file (or specific frame of a multi-page TIFF) as RGB with EXIF orientation and white alpha composite."""
    path = Path(path)
    with Image.open(path) as im:
        n_frames = max(1, int(getattr(im, "n_frames", 1)))
        if page_number < 1 or page_number > n_frames:
            raise ValueError(f"Page {page_number} out of range (1..{n_frames})")
        if n_frames > 1:
            im.seek(page_number - 1)
        frame = ImageOps.exif_transpose(im)
        if frame.mode in ("RGBA", "LA") or (frame.mode == "P" and "transparency" in frame.info):
            rgba = frame.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            return bg
        return frame.convert("RGB")


def get_engine_version() -> str:
    """Return the installed homr OMR engine version, or ``'unknown'`` if metadata is missing."""
    try:
        return importlib.metadata.version("homr")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def get_rapidocr_version() -> str:
    """Return the installed rapidocr version, or ``'unknown'`` if metadata is missing."""
    try:
        return importlib.metadata.version("rapidocr")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def prune_stale_version_dirs(parent_dir: Path, keep_version: str) -> list[Path]:
    """Delete cached engine-version directories under ``parent_dir`` except ``keep_version``."""
    parent_dir = Path(parent_dir)
    if not parent_dir.is_dir():
        return []
    removed: list[Path] = []
    for child in sorted(parent_dir.iterdir()):
        if child.is_dir() and child.name != keep_version:
            try:
                shutil.rmtree(child)
                removed.append(child)
                log.info("Removed stale AI model cache directory: %s", child)
            except OSError as exc:
                log.warning("Could not remove stale AI model cache %s: %s", child, exc)
    return removed


def sanitize_version(raw: str) -> str:
    """Normalize an engine version string into a filesystem-safe directory component."""
    cleaned = (raw or "").strip() or "unknown"
    return re.sub(r"[^0-9A-Za-z._-]", "_", cleaned)


def _apply_homr_model_paths(
    seg_dir: Path,
    tr_dir: Path,
    *,
    homr_main: Any = None,
    homr_seg_cfg: Any = None,
    homr_seg_inf: Any = None,
    homr_tr_cfg: Any = None,
) -> None:
    """Point homr's segmentation and transformer weight paths at ``seg_dir`` and ``tr_dir``."""
    seg_dir.mkdir(parents=True, exist_ok=True)
    tr_dir.mkdir(parents=True, exist_ok=True)

    if homr_main is None:
        import homr.main as homr_main
    if homr_seg_cfg is None:
        import homr.segmentation.config as homr_seg_cfg
    if homr_seg_inf is None:
        import homr.segmentation.inference_segnet as homr_seg_inf
    if homr_tr_cfg is None:
        import homr.transformer.configs as homr_tr_cfg

    new_seg = str(seg_dir / Path(homr_seg_cfg.segnet_path_onnx).name)
    new_seg_fp16 = str(seg_dir / Path(homr_seg_cfg.segnet_path_onnx_fp16).name)
    for mod in (homr_seg_cfg, homr_main, homr_seg_inf):
        if hasattr(mod, "segnet_path_onnx"):
            mod.segnet_path_onnx = new_seg
        if hasattr(mod, "segnet_path_onnx_fp16"):
            mod.segnet_path_onnx_fp16 = new_seg_fp16

    filepaths = homr_tr_cfg.default_config.filepaths
    for attr in ("encoder_path", "decoder_path", "encoder_path_fp16", "decoder_path_fp16"):
        if hasattr(filepaths, attr):
            setattr(filepaths, attr, str(tr_dir / Path(getattr(filepaths, attr)).name))


def configure_model_cache(
    models_dir: Path,
    homr_version: Optional[str] = None,
    rapidocr_version: Optional[str] = None,
    *,
    homr_main: Any = None,
    homr_seg_cfg: Any = None,
    homr_seg_inf: Any = None,
    homr_tr_cfg: Any = None,
) -> tuple[Path, Path]:
    """Redirect downloadable homr and rapidocr ONNX weights into version-scoped writable directories.

    Upgrading either engine changes its version folder, forcing fresh weight downloads
    automatically and cleaning up obsolete model directories. Bundled tokenizer JSON files
    remain untouched in their package directory.
    """
    safe_homr = sanitize_version(homr_version or get_engine_version())
    safe_ocr = sanitize_version(rapidocr_version or get_rapidocr_version())

    homr_root = Path(models_dir) / "homr"
    ocr_root = Path(models_dir) / "rapidocr"
    homr_ver_dir = homr_root / safe_homr
    seg_dir = homr_ver_dir / "segmentation"
    tr_dir = homr_ver_dir / "transformer"
    ocr_ver_dir = ocr_root / safe_ocr

    ocr_ver_dir.mkdir(parents=True, exist_ok=True)
    _apply_homr_model_paths(
        seg_dir,
        tr_dir,
        homr_main=homr_main,
        homr_seg_cfg=homr_seg_cfg,
        homr_seg_inf=homr_seg_inf,
        homr_tr_cfg=homr_tr_cfg,
    )

    prune_stale_version_dirs(homr_root, safe_homr)
    prune_stale_version_dirs(ocr_root, safe_ocr)

    log.info(
        "Configured versioned AI model cache: homr=%s (%s), rapidocr=%s (%s)",
        safe_homr,
        homr_ver_dir,
        safe_ocr,
        ocr_ver_dir,
    )
    return homr_ver_dir, ocr_ver_dir


class CancelledError(RuntimeError):
    """Raised when the user aborts an in-progress conversion."""


_RANGE_TOKEN_RE = re.compile(r"^(\d+)(?:\s*-\s*(\d+))?$")


def get_pdf_page_count(pdf: Path) -> int:
    """Return the total number of pages in ``pdf`` (or frames in a sheet-music image)."""
    path = Path(pdf)
    if is_image_file(path):
        with Image.open(path) as im:
            return max(1, int(getattr(im, "n_frames", 1)))

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        return len(doc)
    finally:
        if hasattr(doc, "close"):
            doc.close()


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
    pdf = Path(pdf)
    if is_image_file(pdf):
        n_pages = get_pdf_page_count(pdf)
        page_numbers = parse_page_spec(pages_spec, n_pages)
        log.info(
            "Rendering image %s (total_frames=%d, selected=%s, preprocess=%s, deskew=%s)",
            pdf.name, n_pages, page_numbers, preprocess_mode, deskew,
        )
        pages: list[Path] = []
        for pno in page_numbers:
            if is_cancelled():
                raise CancelledError("Conversion aborted by user.")
            try:
                pil = load_pil_image_rgb(pdf, page_number=pno)
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                bgr = preprocess_page_image(bgr, mode=preprocess_mode, deskew=deskew)
                h_px, w_px = bgr.shape[:2]
                img_path = out_dir / f"page_{pno:03d}.png"
                if not cv2.imwrite(str(img_path), bgr):
                    raise OmrError(f"Could not save rendered page {pno} ({w_px}x{h_px} px) to {img_path}")
                file_bytes = img_path.stat().st_size if img_path.is_file() else 0
                log.info("Image page %d saved: %s (%dx%d px, %d bytes)", pno, img_path.name, w_px, h_px, file_bytes)
                pages.append(img_path)
            except (OmrError, CancelledError):
                raise
            except MemoryError as exc:
                raise OmrError(
                    f"Out of memory loading image page {pno} — try a smaller image ({memory_snapshot()})"
                ) from exc
            except Exception as exc:
                raise OmrError(f"Failed to prepare image page {pno}: {type(exc).__name__}: {exc}") from exc
        return pages

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    try:
        n_pages = len(doc)
        if n_pages == 0:
            raise OmrError(f"PDF has no pages: {pdf}")
        try:
            pdf_ver = doc.get_version()
        except Exception:
            pdf_ver = "unknown"
        page_numbers = parse_page_spec(pages_spec, n_pages)
        log.info(
            "Rendering PDF %s (version=%s, total_pages=%d, selected=%s, dpi=%d, preprocess=%s, deskew=%s)",
            pdf.name, pdf_ver, n_pages, page_numbers, dpi, preprocess_mode, deskew,
        )
        scale = dpi / 72.0
        pages = []
        for pno in page_numbers:
            if is_cancelled():
                raise CancelledError("Conversion aborted by user.")
            try:
                page = doc[pno - 1]
                w_pt, h_pt = page.get_size()
                pil = page.render(scale=scale).to_pil()
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                bgr = preprocess_page_image(bgr, mode=preprocess_mode, deskew=deskew)
                h_px, w_px = bgr.shape[:2]
                log.info("Page %d: %.0fx%.0f pt -> %dx%d px at %d DPI", pno, w_pt, h_pt, w_px, h_px, dpi)
                img_path = out_dir / f"page_{pno:03d}.png"
                if not cv2.imwrite(str(img_path), bgr):
                    raise OmrError(f"Could not save rendered page {pno} ({w_px}x{h_px} px) to {img_path}")
                file_bytes = img_path.stat().st_size if img_path.is_file() else 0
                log.info("Page %d saved: %s (%d bytes)", pno, img_path.name, file_bytes)
                pages.append(img_path)
            except (OmrError, CancelledError):
                raise
            except MemoryError as exc:
                raise OmrError(
                    f"Out of memory rendering page {pno} at {dpi} DPI — try a lower Render DPI ({memory_snapshot()})"
                ) from exc
            except Exception as exc:
                raise OmrError(f"Failed to render page {pno}: {type(exc).__name__}: {exc}") from exc
        return pages
    finally:
        if hasattr(doc, "close"):
            doc.close()


@dataclass(frozen=True)
class PageFailure:
    """Details of a single page OMR failure for diagnostics and user reporting."""

    page: int
    image: str
    error_type: str
    message: str

    def format_summary(self) -> str:
        return f"page {self.page} ({self.image}): {self.error_type}: {self.message}"


def _image_dimensions(path: Path) -> str:
    try:
        with Image.open(path) as im:
            return f"{im.width}x{im.height} px"
    except Exception:
        return "unknown size"


def run_pages(
    image_paths: Sequence[Path],
    process_one: Callable[[Path], Optional[Path]],
    page_cb: Optional[Callable[[int, int], None]] = None,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> list[Path]:
    """Run ``process_one`` on each rendered page with verbose logging, timing, and error capture."""
    outputs: list[Path] = []
    failures: list[PageFailure] = []
    total = len(image_paths)
    for idx, img in enumerate(image_paths, start=1):
        if is_cancelled():
            raise CancelledError("Conversion aborted by user.")
        if page_cb:
            page_cb(idx, total)
        dims = _image_dimensions(img)
        log.info("homr: page %d/%d start (%s, %s, %s)", idx, total, img.name, dims, memory_snapshot())
        started = time.monotonic()
        try:
            xml_path = process_one(img)
            elapsed = time.monotonic() - started
            if xml_path is not None and Path(xml_path).is_file():
                outputs.append(Path(xml_path))
                log.info("homr: page %d/%d done in %.2fs -> %s", idx, total, elapsed, Path(xml_path).name)
            else:
                log.warning("homr produced no MusicXML for page %d (%s) after %.2fs", idx, img.name, elapsed)
                failures.append(PageFailure(idx, img.name, "NoOutput", "homr produced no MusicXML"))
        except CancelledError:
            raise
        except Exception as exc:
            elapsed = time.monotonic() - started
            log.error(
                "homr failed on page %d (%s) after %.2fs: %s: %s",
                idx,
                img.name,
                elapsed,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            failures.append(PageFailure(idx, img.name, type(exc).__name__, str(exc) or type(exc).__name__))

    if failures and outputs:
        summary = "; ".join(f.format_summary() for f in failures)
        log.warning("%d of %d page(s) could not be read: %s", len(failures), total, summary)
    elif failures and not outputs:
        summary = "; ".join(f.format_summary() for f in failures)
        raise OmrError(f"homr could not read any of the {total} page(s): {summary}")
    return outputs


def _default_run_homr(image_paths: Sequence[Path],
                      page_cb: Optional[Callable[[int, int], None]] = None,
                      is_cancelled: Callable[[], bool] = lambda: False,
                      segnet_step_size: int = 320,
                      min_notehead_size: int = 4,
                      preprocess_mode: str = "auto",
                      models_dir: Optional[Path] = None) -> list[Path]:  # pragma: no cover - integration
    import homr.main as hmain
    import homr.title_detection as htitle
    import onnxruntime as ort
    from homr.main import ProcessingConfig, XmlGeneratorArguments, download_weights, process_image

    if is_cancelled():
        raise CancelledError("Conversion aborted by user.")

    ort.set_default_logger_severity(3)
    ocr_ver_dir: Optional[Path] = None
    if models_dir is not None:
        _, ocr_ver_dir = configure_model_cache(models_dir)
    download_weights(segnet_use_gpu=False, transformer_use_gpu=False, coreml_encoder=False)
    if htitle._reader is None:
        if ocr_ver_dir is not None:
            from rapidocr import RapidOCR
            htitle._reader = RapidOCR(params={"Global.model_root_dir": str(ocr_ver_dir)})
        else:
            htitle.download_ocr_weights()
    cfg = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        transformer_use_gpu=False,
        segnet_use_gpu=False,
        coreml_encoder=False,
        title_detection=True,
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
        def _process_one(img: Path) -> Optional[Path]:
            process_image(str(img), cfg, xargs)
            xml_path = img.with_suffix(".musicxml")
            return xml_path if xml_path.is_file() else None

        return run_pages(image_paths, _process_one, page_cb=page_cb, is_cancelled=is_cancelled)
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
                 page_callback: Optional[Callable[[int, int], None]] = None,
                 models_dir: Optional[Path] = None) -> None:
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
        self.models_dir = models_dir

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
            log.exception("Failed to render PDF pages")
            raise OmrError(f"Failed to render PDF pages: {type(exc).__name__}: {exc}") from exc

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
                        models_dir=self.models_dir,
                    )
                )
            else:
                outputs = sorted(self._run(pages, self._page_cb, self._is_cancelled))
        except (OmrError, CancelledError):
            raise
        except Exception as exc:
            log.exception("homr recognition failed")
            raise OmrError(f"homr recognition failed: {type(exc).__name__}: {exc}") from exc

        if not outputs:
            raise OmrError("homr produced no MusicXML output (score may be unreadable)")
        log.info("homr produced %d file(s): %s", len(outputs), [p.name for p in outputs])
        return outputs
