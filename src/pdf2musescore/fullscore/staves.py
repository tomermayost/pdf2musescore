"""Multi-staff layout normalization for Full Score Mode."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StaffLayoutResult:
    """Normalized orchestral system staves and input-image geometry for OCR."""

    kept_staffs: list[Any]
    sys_min_x: int
    sys_max_x: int
    crop_x: int
    median_gap: float
    staff_cys: list[float]
    staff_y_norms: list[float]
    y_lo: float
    y_hi: float
    top_staff_y: float = 0.0
    staff_left_x: float = 0.0
    dropped_narrow_count: int = 0
    deduped_overlap_count: int = 0
    extended_count: int = 0


def _identity_pt(pt: tuple[float, float]) -> tuple[float, float]:
    return float(pt[0]), float(pt[1])


def flatten_filter_and_extend_staves(
    multi_staffs: Sequence[Any],
    to_input_image: Optional[Callable[[tuple[float, float]], tuple[float, float]]] = None,
    *,
    min_width_ratio: float = 0.45,
    overlap_unit_tol: float = 1.5,
    wide_ratio: float = 0.85,
    extend_margin_px: int = 40,
) -> StaffLayoutResult:
    """Split false grand-staves, drop narrow fragments/duplicates, and extend short staves to system X-span."""
    map_pt = to_input_image if to_input_image is not None else _identity_pt

    flat_raw: list[Any] = []
    for ms in multi_staffs:
        for staff in getattr(ms, "staffs", ()):
            if getattr(staff, "is_grandstaff", False):
                parts = staff.split()
                if parts is not None:
                    flat_raw.extend(parts)
                else:
                    flat_raw.append(staff)
            else:
                flat_raw.append(staff)

    flat_sorted = sorted(flat_raw, key=lambda s: float(s.min_y))
    if not flat_sorted:
        return StaffLayoutResult(
            kept_staffs=[],
            sys_min_x=0,
            sys_max_x=0,
            crop_x=100,
            median_gap=120.0,
            staff_cys=[],
            staff_y_norms=[],
            y_lo=0.0,
            y_hi=0.0,
        )

    widths = [float(s.max_x - s.min_x) for s in flat_sorted]
    median_width = float(np.median(widths))
    min_allowed_width = min_width_ratio * median_width
    log.info(
        "Raw flattened staves=%d, median_width=%.1f px, min_allowed_width=%.1f px",
        len(flat_sorted),
        median_width,
        min_allowed_width,
    )

    wide_enough: list[Any] = []
    dropped_narrow = 0
    for idx, staff in enumerate(flat_sorted):
        w = float(staff.max_x - staff.min_x)
        cy = 0.5 * (float(staff.min_y) + float(staff.max_y))
        status = "OK" if w >= min_allowed_width else "DROP_NARROW"
        log.debug(
            "  raw[%02d]: x=[%.0f..%.0f] w=%.0f y=[%.0f..%.0f] cy=%.0f unit=%.1f -> %s",
            idx,
            staff.min_x,
            staff.max_x,
            w,
            staff.min_y,
            staff.max_y,
            cy,
            float(getattr(staff, "average_unit_size", 0.0)),
            status,
        )
        if status == "OK":
            wide_enough.append(staff)
        else:
            dropped_narrow += 1

    if not wide_enough:
        return StaffLayoutResult(
            kept_staffs=[],
            sys_min_x=0,
            sys_max_x=0,
            crop_x=100,
            median_gap=120.0,
            staff_cys=[],
            staff_y_norms=[],
            y_lo=0.0,
            y_hi=0.0,
            dropped_narrow_count=dropped_narrow,
        )

    kept: list[Any] = []
    deduped_overlap = 0
    for staff in wide_enough:
        cy = 0.5 * (float(staff.min_y) + float(staff.max_y))
        w = float(staff.max_x - staff.min_x)
        if kept:
            prev = kept[-1]
            prev_cy = 0.5 * (float(prev.min_y) + float(prev.max_y))
            tol = overlap_unit_tol * max(
                float(getattr(staff, "average_unit_size", 4.0)),
                float(getattr(prev, "average_unit_size", 4.0)),
            )
            if abs(cy - prev_cy) <= tol:
                prev_w = float(prev.max_x - prev.min_x)
                deduped_overlap += 1
                log.debug(
                    "Overlapping staff at cy=%.0f (prev_cy=%.0f, tol=%.1f): keeping wider (w=%.0f vs %.0f)",
                    cy,
                    prev_cy,
                    tol,
                    max(w, prev_w),
                    min(w, prev_w),
                )
                if w > prev_w:
                    kept[-1] = staff
                continue
        kept.append(staff)

    wide_staffs = [s for s in kept if float(s.max_x - s.min_x) >= wide_ratio * median_width] or kept
    sys_min_x = int(np.median([float(s.min_x) for s in wide_staffs]))
    sys_max_x = int(np.median([float(s.max_x) for s in wide_staffs]))
    log.info("System full X-span from wide staves: [%d..%d]", sys_min_x, sys_max_x)

    extended_kept: list[Any] = []
    extended_count = 0
    for idx, staff in enumerate(kept):
        if staff.min_x > sys_min_x + extend_margin_px or staff.max_x < sys_max_x - extend_margin_px:
            old_x0, old_x1 = int(staff.min_x), int(staff.max_x)
            new_min_x = min(old_x0, sys_min_x)
            new_max_x = max(old_x1, sys_max_x)
            s_ext = staff.extend_to_x_range(new_min_x, new_max_x)
            s_ext.symbols = list(getattr(staff, "symbols", ()))
            s_ext.is_grandstaff = False
            extended_count += 1
            log.info(
                "  Extended staff[%02d] X-range: [%d..%d] -> [%d..%d] (preserved %d symbols)",
                idx,
                old_x0,
                old_x1,
                int(s_ext.min_x),
                int(s_ext.max_x),
                len(s_ext.symbols),
            )
            extended_kept.append(s_ext)
        else:
            staff.is_grandstaff = False
            extended_kept.append(staff)

    staff_left_xs: list[float] = []
    staff_top_ys: list[float] = []
    staff_bot_ys: list[float] = []
    staff_cys: list[float] = []
    for staff in extended_kept:
        lx0, ty0 = map_pt((float(sys_min_x), float(staff.min_y)))
        _, by0 = map_pt((float(sys_min_x), float(staff.max_y)))
        staff_left_xs.append(float(lx0))
        staff_top_ys.append(float(ty0))
        staff_bot_ys.append(float(by0))
        staff_cys.append(0.5 * (float(ty0) + float(by0)))

    gaps = [staff_cys[i + 1] - staff_cys[i] for i in range(len(staff_cys) - 1)]
    median_gap = float(np.median(gaps)) if gaps else 120.0
    if gaps and median_gap > 0 and max(gaps) > 3.5 * median_gap:
        log.warning(
            "Large vertical gap detected between staves (max_gap=%.1f px vs median_gap=%.1f px) — page may have >1 system",
            max(gaps),
            median_gap,
        )

    crop_x = max(100, int(np.median(staff_left_xs)) - 5)
    y_lo = staff_top_ys[0] - 1.5 * median_gap
    y_hi = staff_bot_ys[-1] + 1.0 * median_gap

    if len(staff_cys) == 1:
        staff_y_norms = [0.5]
    else:
        span = max(1.0, staff_cys[-1] - staff_cys[0])
        staff_y_norms = [(cy - staff_cys[0]) / span for cy in staff_cys]

    log.info(
        "Kept %d staves (dropped_narrow=%d, deduped_overlap=%d, extended=%d, crop_x=%d, median_gap=%.1f)",
        len(extended_kept),
        dropped_narrow,
        deduped_overlap,
        extended_count,
        crop_x,
        median_gap,
    )
    return StaffLayoutResult(
        kept_staffs=extended_kept,
        sys_min_x=sys_min_x,
        sys_max_x=sys_max_x,
        crop_x=crop_x,
        median_gap=median_gap,
        staff_cys=staff_cys,
        staff_y_norms=staff_y_norms,
        y_lo=y_lo,
        y_hi=y_hi,
        top_staff_y=float(staff_top_ys[0]),
        staff_left_x=float(np.median(staff_left_xs)),
        dropped_narrow_count=dropped_narrow,
        deduped_overlap_count=deduped_overlap,
        extended_count=extended_count,
    )
