"""Page-level OMR runner for Full Score Mode."""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2

from ..diagnostics import memory_snapshot
from ..engines.base import OmrError
from ..engines.homr_engine import CancelledError, PageFailure, configure_model_cache
from ..rehearsal import PositionedRehearsal, extract_rehearsal_marks_in_band
from ..tempo import PositionedTempo, TempoMarking, extract_tempo_markings_in_band
from .align import PageSidecar, write_page_sidecar
from .chunking import detect_system_barlines_with_skew, staff_barlines_from_system
from .homr_patch import DEFAULT_TARGET_WIDTH, fullscore_homr_context
from .instruments import extract_instrument_labels
from .staves import flatten_filter_and_extend_staves

log = logging.getLogger(__name__)

_PAGE_NUM_RE = re.compile(r"page_(\d+)", re.IGNORECASE)


def _parse_page_number(img_path: Path, fallback_idx: int) -> int:
    m = _PAGE_NUM_RE.search(img_path.stem)
    if m:
        return int(m.group(1))
    return fallback_idx


def _is_non_music_page_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in (
            "no staffs found",
            "no noteheads or clefs found",
            "no staves found",
        )
    )


def _default_load_homr_modules() -> dict[str, Any]:  # pragma: no cover - integration imports
    import homr.main as hmain
    import homr.music_xml_generator as hxml
    import homr.resize as hresize
    import homr.staff_parsing as hsp
    from homr.model import MultiStaff
    from homr.transformer.configs import Config

    return {
        "main": hmain,
        "xml": hxml,
        "resize": hresize,
        "staff_parsing": hsp,
        "MultiStaff": MultiStaff,
        "Config": Config,
    }


def extract_page_header_markings(
    img_path: Path,
    *,
    top_staff_y: float,
    median_gap: float,
    sys_min_x: float = 0.0,
    sys_max_x: float = 0.0,
    ocr_reader: Any = None,
    ocr_dir: Optional[Path] = None,
) -> tuple[Optional[TempoMarking], list[PositionedTempo], list[PositionedRehearsal]]:
    """Extract rehearsal marks and positioned tempos/expressions above ``top_staff_y`` in one OCR pass."""
    if top_staff_y < 25:
        return None, [], []
    page_img = cv2.imread(str(img_path))
    if page_img is None:
        log.warning("Could not read %s for header OCR", img_path)
        return None, [], []

    from .instruments import _run_rapidocr_on_crop

    h_px = page_img.shape[0]
    y_bot = max(1, min(int(top_staff_y) + 8, h_px))
    header = cv2.copyMakeBorder(
        page_img[:y_bot, :], 0, 32, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    boxes_raw: Sequence[Any] = []
    txts_raw: Sequence[str] = []
    try:
        boxes_raw, txts_raw = _run_rapidocr_on_crop(
            header, ocr_reader=ocr_reader, ocr_dir=ocr_dir
        )
    except Exception as exc:
        log.warning("Header OCR failed on %s: %s", Path(img_path).name, exc)
        return None, [], []

    min_reh_y = max(0, int(top_staff_y) - int(max(170.0, median_gap * 2.2)))
    max_reh_y = int(top_staff_y - max(15.0, 0.15 * median_gap))
    rehearsal_marks: list[PositionedRehearsal] = []
    try:
        rehearsal_marks = extract_rehearsal_marks_in_band(
            page_img,
            0,
            y_bot,
            sys_min_x=float(sys_min_x),
            sys_max_x=float(sys_max_x),
            unboxed_min_y=min_reh_y,
            unboxed_max_y=max_reh_y,
            ocr_reader=ocr_reader,
            ocr_dir=ocr_dir,
            precomputed_band_ocr=(boxes_raw, txts_raw),
            source_name=Path(img_path).name,
        )
    except Exception as exc:
        log.warning("Rehearsal mark OCR failed on %s: %s", Path(img_path).name, exc)

    exclude_boxes = [
        (r.x_left, r.y_top, r.x_right, r.y_bottom)
        for r in rehearsal_marks
    ]
    positioned_tempos: list[PositionedTempo] = []
    try:
        positioned_tempos = extract_tempo_markings_in_band(
            page_img,
            0,
            y_bot,
            sys_min_x=float(sys_min_x),
            sys_max_x=float(sys_max_x),
            ocr_reader=ocr_reader,
            ocr_dir=ocr_dir,
            expr_min_y=min_reh_y,
            precomputed_ocr=(boxes_raw, txts_raw),
            exclude_boxes=exclude_boxes,
            source_name=Path(img_path).name,
        )
    except Exception as exc:
        log.warning("Tempo OCR failed on %s: %s", Path(img_path).name, exc)

    primary_tempo: Optional[TempoMarking] = None
    for pt in positioned_tempos:
        if not pt.tempo.is_expression:
            primary_tempo = pt.tempo
            break
    return primary_tempo, positioned_tempos, rehearsal_marks


def run_fullscore_homr(
    image_paths: Sequence[Path],
    *,
    page_cb: Optional[Callable[[int, int], None]] = None,
    staff_cb: Optional[Callable[[int, int, int, int], None]] = None,
    is_cancelled: Callable[[], bool] = lambda: False,
    target_width: int = DEFAULT_TARGET_WIDTH,
    segnet_step_size: int = 320,
    min_notehead_size: int = 4,
    preprocess_mode: str = "auto",
    enable_staff_chunking: bool = True,
    models_dir: Optional[Path] = None,
    homr_modules: Optional[dict[str, Any]] = None,
    ocr_reader_factory: Optional[Callable[[Optional[Path]], Any]] = None,
) -> list[Path]:
    """Run Full Score Mode OMR on ``image_paths`` inside a scoped homr patch context."""
    if is_cancelled():
        raise CancelledError("Conversion aborted by user.")

    mods = homr_modules if homr_modules is not None else _default_load_homr_modules()
    hmain = mods["main"]
    hxml = mods["xml"]
    hresize = mods["resize"]
    hsp = mods["staff_parsing"]
    multi_staff_cls = mods["MultiStaff"]
    tr_config_cls = mods["Config"]

    ocr_ver_dir: Optional[Path] = None
    if homr_modules is None:  # pragma: no cover - real model provisioning
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        if models_dir is not None:
            _, ocr_ver_dir = configure_model_cache(models_dir)
        from ..model_store import patched_homr_download_timeout

        with patched_homr_download_timeout():
            hmain.download_weights(segnet_use_gpu=False, transformer_use_gpu=False, coreml_encoder=False)

    ocr_reader: Any = None
    if ocr_reader_factory is not None:
        ocr_reader = ocr_reader_factory(ocr_ver_dir)
    elif homr_modules is None:  # pragma: no cover - real RapidOCR init
        from rapidocr import RapidOCR

        ocr_reader = (
            RapidOCR(params={"Global.model_root_dir": str(ocr_ver_dir)})
            if ocr_ver_dir is not None
            else RapidOCR()
        )

    cfg = hmain.ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        transformer_use_gpu=False,
        segnet_use_gpu=False,
        coreml_encoder=False,
        title_detection=False,
    )
    xargs = hxml.XmlGeneratorArguments(large_page=False, metronome=None, tempo=None)

    outputs: list[Path] = []
    failures: list[PageFailure] = []
    skipped_pages: list[int] = []
    total_pages = len(image_paths)

    active_page: dict[str, Any] = {
        "idx": 1,
        "staff_count": 0,
        "barlines": [],
        "barlines_y_mid": 0.0,
        "barlines_skew": 0.0,
    }

    def _barlines_for_staff(staff: Any) -> list[int]:
        bls = active_page["barlines"]
        if not bls:
            return []
        return staff_barlines_from_system(
            bls,
            staff,
            y_mid=float(active_page["barlines_y_mid"]),
            dx_per_dy=float(active_page["barlines_skew"]),
        )

    def _on_staff_progress(staff_idx: int) -> None:
        if staff_cb is not None and active_page["staff_count"] > 0:
            staff_cb(
                active_page["idx"],
                total_pages,
                min(active_page["staff_count"], staff_idx + 1),
                active_page["staff_count"],
            )

    with fullscore_homr_context(
        target_width=int(target_width),
        segnet_step_size=int(segnet_step_size),
        min_notehead_size=int(min_notehead_size),
        preprocess_mode=preprocess_mode,
        enable_staff_chunking=enable_staff_chunking,
        barlines_provider=_barlines_for_staff,
        is_cancelled=is_cancelled,
        staff_progress_cb=_on_staff_progress,
        homr_resize=hresize,
        homr_staff_parsing=hsp,
        homr_main=hmain,
    ):
        for idx, img in enumerate(image_paths, start=1):
            img_path = Path(img)
            if is_cancelled():
                raise CancelledError("Conversion aborted by user.")
            if page_cb is not None:
                page_cb(idx, total_pages)

            pno = _parse_page_number(img_path, idx)
            log.info("Full score OMR: page %d/%d (%s, %s)", idx, total_pages, img_path.name, memory_snapshot())
            started = time.monotonic()
            debug_obj = None
            try:
                multi_staffs, preprocessed_img, debug_obj, _title_future, raw_staff_count, to_input_image = (
                    hmain.detect_staffs_in_image(str(img_path), cfg)
                )
                log.info(
                    "Page %d detection: raw_staff_count=%d, initial MultiStaff counts=%s",
                    pno,
                    raw_staff_count,
                    [len(ms.staffs) for ms in multi_staffs],
                )

                layout = flatten_filter_and_extend_staves(multi_staffs, to_input_image)
                if not layout.kept_staffs:
                    log.info("Skipping non-music/cover page %d (%s): 0 staves kept", pno, img_path.name)
                    skipped_pages.append(pno)
                    continue

                labels = extract_instrument_labels(
                    img_path,
                    layout.staff_cys,
                    layout.crop_x,
                    layout.median_gap,
                    layout.y_lo,
                    layout.y_hi,
                    ocr_reader=ocr_reader,
                    ocr_dir=ocr_ver_dir,
                )

                in_sys_min_x, _ = to_input_image((float(layout.sys_min_x), 0.0))
                in_sys_max_x, _ = to_input_image((float(layout.sys_max_x), 0.0))
                tempo, positioned_tempos, rehearsal_marks = extract_page_header_markings(
                    img_path,
                    top_staff_y=layout.top_staff_y,
                    median_gap=layout.median_gap,
                    sys_min_x=float(in_sys_min_x),
                    sys_max_x=float(in_sys_max_x),
                    ocr_reader=ocr_reader,
                    ocr_dir=ocr_ver_dir,
                )

                active_page["idx"] = idx
                active_page["staff_count"] = len(layout.kept_staffs)
                if enable_staff_chunking:
                    try:
                        bls, y_mid, dx_per_dy = detect_system_barlines_with_skew(
                            preprocessed_img, layout.kept_staffs
                        )
                        active_page["barlines"] = bls
                        active_page["barlines_y_mid"] = y_mid
                        active_page["barlines_skew"] = dx_per_dy
                    except Exception as exc:
                        log.warning(
                            "System barline detection failed on page %d (%s); using single-pass OMR",
                            pno,
                            exc,
                        )
                        active_page["barlines"] = []
                        active_page["barlines_y_mid"] = 0.0
                        active_page["barlines_skew"] = 0.0
                else:
                    active_page["barlines"] = []
                    active_page["barlines_y_mid"] = 0.0
                    active_page["barlines_skew"] = 0.0

                full_system = [multi_staff_cls(layout.kept_staffs, [])]
                tr_cfg = tr_config_cls()
                tr_cfg.use_gpu_inference = False
                tr_cfg.use_coreml_encoder = False

                result_staffs = hsp.parse_staffs(
                    debug_obj,
                    full_system,
                    preprocessed_img,
                    config=tr_cfg,
                    selected_staff=-1,
                    page_to_input_image=to_input_image,
                )

                xml_el = hxml.generate_xml(xargs, result_staffs, title=f"Page {pno}")
                xml_path = img_path.with_suffix(".musicxml")
                ET.ElementTree(xml_el).write(xml_path, encoding="unicode", xml_declaration=True)

                write_page_sidecar(
                    PageSidecar(
                        page_number=pno,
                        staff_count=len(layout.kept_staffs),
                        staff_cys=layout.staff_cys,
                        staff_y_norms=layout.staff_y_norms,
                        labels=labels,
                        sys_min_x=int(in_sys_min_x),
                        sys_max_x=int(in_sys_max_x),
                        crop_x=layout.crop_x,
                        tempo=tempo,
                        positioned_tempos=tuple(positioned_tempos),
                        rehearsal_marks=tuple(rehearsal_marks),
                    ),
                    xml_path,
                )
                outputs.append(xml_path)
                log.info(
                    "Full score page %d/%d done in %.2fs -> %s (%d staves)",
                    idx,
                    total_pages,
                    time.monotonic() - started,
                    xml_path.name,
                    len(layout.kept_staffs),
                )
            except CancelledError:
                raise
            except Exception as exc:
                elapsed = time.monotonic() - started
                if _is_non_music_page_error(exc):
                    log.info(
                        "Skipping non-music/cover page %d (%s) after %.2fs: %s",
                        pno,
                        img_path.name,
                        elapsed,
                        exc,
                    )
                    skipped_pages.append(pno)
                    continue
                log.error(
                    "Full score OMR failed on page %d (%s) after %.2fs: %s: %s",
                    pno,
                    img_path.name,
                    elapsed,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                failures.append(PageFailure(pno, img_path.name, type(exc).__name__, str(exc) or type(exc).__name__))
            finally:
                if debug_obj is not None and hasattr(debug_obj, "clean_debug_files_from_previous_runs"):
                    try:
                        debug_obj.clean_debug_files_from_previous_runs()
                    except Exception:
                        pass

    if skipped_pages:
        log.info("Skipped %d non-music page(s): %s", len(skipped_pages), skipped_pages)
    if failures and outputs:
        summary = "; ".join(f.format_summary() for f in failures)
        log.warning("%d of %d full-score page(s) failed: %s", len(failures), total_pages, summary)
    elif not outputs:
        if failures:
            summary = "; ".join(f.format_summary() for f in failures)
            raise OmrError(f"homr could not read any of the {total_pages} full-score page(s): {summary}")
        raise OmrError(f"No music staves found across {total_pages} selected full-score page(s).")

    return sorted(outputs)
