"""Thread-safe, strictly scoped context manager for homr Full Score Mode patches."""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np

from ..engines.homr_engine import CancelledError

from .chunking import parse_staff_image_chunked

log = logging.getLogger(__name__)

_HOMR_PATCH_LOCK = threading.Lock()
DEFAULT_TARGET_WIDTH = 3840
MAX_TARGET_HEIGHT = 7680


def calc_fullscore_image_size(
    width: int,
    height: int,
    target_width: int = DEFAULT_TARGET_WIDTH,
    max_height: int = MAX_TARGET_HEIGHT,
) -> tuple[int, int]:
    """Scale image dimensions to ``target_width`` while capping height to avoid OOM."""
    if width <= 0 or height <= 0:
        return max(1, int(width)), max(1, int(height))

    if width == target_width and height <= max_height:
        return width, height

    ratio = target_width / float(width)
    tar_w = int(target_width)
    tar_h = max(1, round(height * ratio))

    if tar_h > max_height:
        scale_h = max_height / float(height)
        tar_h = int(max_height)
        tar_w = max(1, round(width * scale_h))
        log.warning(
            "Clamped full-score target height from %d px to %d px (resized %dx%d -> %dx%d)",
            round(height * ratio),
            max_height,
            width,
            height,
            tar_w,
            tar_h,
        )
    return tar_w, tar_h


@contextlib.contextmanager
def fullscore_homr_context(
    *,
    target_width: int = DEFAULT_TARGET_WIDTH,
    segnet_step_size: int = 320,
    min_notehead_size: int = 4,
    preprocess_mode: str = "auto",
    enable_staff_chunking: bool = True,
    barlines_provider: Optional[Callable[[Any], Sequence[int]]] = None,
    is_cancelled: Callable[[], bool] = lambda: False,
    staff_progress_cb: Optional[Callable[[int], None]] = None,
    homr_resize: Any = None,
    homr_staff_parsing: Any = None,
    homr_main: Any = None,
) -> Iterator[None]:
    """Temporarily patch homr internals for high-res multi-staff full-score OMR, restoring all on exit."""
    if homr_resize is None:  # pragma: no cover - thin default import
        import homr.resize as homr_resize
    if homr_staff_parsing is None:  # pragma: no cover - thin default import
        import homr.staff_parsing as homr_staff_parsing
    if homr_main is None:  # pragma: no cover - thin default import
        import homr.main as homr_main

    with _HOMR_PATCH_LOCK:
        orig_calc_size = homr_resize.calc_target_image_size
        orig_calc_region = homr_staff_parsing._calculate_region
        orig_parse_staff_image = homr_staff_parsing.parse_staff_image
        orig_extract = homr_main.extract
        orig_ellipses = homr_main.create_bounding_ellipses
        color_adjust_mod = getattr(homr_main, "color_adjust", None)
        orig_apply_clahe = getattr(color_adjust_mod, "apply_clahe", None) if color_adjust_mod is not None else None

        def _fullscore_calc_size(width: int, height: int) -> tuple[int, int]:
            return calc_fullscore_image_size(width, height, target_width=int(target_width))

        def _safe_calculate_region(staff: Any, regions: Any) -> np.ndarray:
            reg = orig_calc_region(staff, regions)
            u = max(float(getattr(staff, "average_unit_size", 4.0)), 4.0)
            x_min, y_min, x_max, y_max = int(reg[0]), int(reg[1]), int(reg[2]), int(reg[3])

            if y_max - y_min < 2 * u or y_min >= y_max:
                fb_y_min = int(staff.min_y - 3 * u)
                fb_y_max = int(staff.max_y + 3 * u)
                if fb_y_max <= fb_y_min:
                    fb_y_max = fb_y_min + max(10, int(6 * u))
                log.info(
                    "Degenerate staff region y=[%d,%d] (unit=%.1f); fallback y=[%d,%d]",
                    y_min,
                    y_max,
                    u,
                    fb_y_min,
                    fb_y_max,
                )
                y_min, y_max = fb_y_min, fb_y_max

            if x_max <= x_min:
                fb_x_min = int(staff.min_x - 2 * u)
                fb_x_max = max(int(staff.max_x + 2 * u), fb_x_min + max(20, int(4 * u)))
                log.info(
                    "Degenerate staff region x=[%d,%d] (unit=%.1f); fallback x=[%d,%d]",
                    x_min,
                    x_max,
                    u,
                    fb_x_min,
                    fb_x_max,
                )
                x_min, x_max = fb_x_min, fb_x_max

            return np.array([x_min, y_min, x_max, y_max])

        default_identity = getattr(homr_staff_parsing, "identity", lambda pt: pt)

        def _safe_parse_staff_image(
            debug: Any,
            index: int,
            staff: Any,
            image: Any,
            regions: Any,
            config: Any,
            page_to_input_image: Any = default_identity,
        ) -> list[Any]:
            if is_cancelled():
                raise CancelledError("Conversion aborted by user.")
            if staff_progress_cb is not None:
                staff_progress_cb(int(index))
            try:
                barlines_x = barlines_provider(staff) if (enable_staff_chunking and barlines_provider is not None) else ()
                if len(barlines_x) >= 3:
                    return parse_staff_image_chunked(
                        debug,
                        index,
                        staff,
                        image,
                        regions,
                        config,
                        page_to_input_image=page_to_input_image,
                        barlines_x=barlines_x,
                        orig_parse_staff_image=orig_parse_staff_image,
                        homr_staff_parsing=homr_staff_parsing,
                        is_cancelled=is_cancelled,
                    )
                return orig_parse_staff_image(
                    debug,
                    index,
                    staff,
                    image,
                    regions,
                    config,
                    page_to_input_image=page_to_input_image,
                )
            except CancelledError:
                raise
            except Exception as exc:
                log.error(
                    "parse_staff_image failed on staff %d (%s: %s); skipping staff",
                    index,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                return []

        def _custom_extract(
            preprocessed: Any,
            img_path: Any,
            step_size: int = 320,
            use_cache: bool = False,
            use_gpu_inference: bool = False,
        ) -> Any:
            return orig_extract(
                preprocessed,
                img_path,
                step_size=int(segnet_step_size),
                use_cache=use_cache,
                use_gpu_inference=use_gpu_inference,
            )

        def _custom_ellipses(notehead_map: Any, min_size: tuple[int, int] = (4, 4), **kwargs: Any) -> Any:
            ms = (int(min_notehead_size), int(min_notehead_size))
            return orig_ellipses(notehead_map, min_size=ms, **kwargs)

        homr_resize.calc_target_image_size = _fullscore_calc_size
        homr_staff_parsing._calculate_region = _safe_calculate_region
        homr_staff_parsing.parse_staff_image = _safe_parse_staff_image
        homr_main.extract = _custom_extract
        homr_main.create_bounding_ellipses = _custom_ellipses
        if (preprocess_mode or "auto").strip().lower() == "none" and color_adjust_mod is not None and orig_apply_clahe is not None:
            color_adjust_mod.apply_clahe = lambda img: img

        log.info(
            "Activated Full Score homr context (target_width=%d, step_size=%d, min_notehead=%d, preprocess=%s)",
            int(target_width),
            int(segnet_step_size),
            int(min_notehead_size),
            preprocess_mode,
        )
        try:
            yield
        finally:
            homr_resize.calc_target_image_size = orig_calc_size
            homr_staff_parsing._calculate_region = orig_calc_region
            homr_staff_parsing.parse_staff_image = orig_parse_staff_image
            homr_main.extract = orig_extract
            homr_main.create_bounding_ellipses = orig_ellipses
            if color_adjust_mod is not None and orig_apply_clahe is not None:
                color_adjust_mod.apply_clahe = orig_apply_clahe
            log.debug("Restored original homr functions after Full Score context")
