"""Native-scale barline-chunked OMR processing for dense multi-staff full scores."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np

from ..engines.homr_engine import CancelledError

log = logging.getLogger(__name__)

TR_OMR_MAX_WIDTH = 1280
TR_OMR_MAX_HEIGHT = 256
MIN_NATIVE_5LINE_PX = 36.0
TARGET_MIN_5LINE_PX = 40.0
TARGET_MAX_5LINE_PX = 58.0
MAX_CHUNKS_PER_STAFF = 4


@dataclass(frozen=True)
class StaffChunkPlan:
    """Contiguous slice of segments inside a dewarped staff strip."""

    measure_start: int
    measure_end: int
    x_start: int
    x_end: int
    expected_5line_px: float
    crosses_tie_left: bool = False
    keep_x_start: int = 0
    keep_x_end: int = 0


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a 2D uint8/float-compatible grayscale view of ``image``."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _erase_staff_lines(gray: np.ndarray, unit: float) -> np.ndarray:
    """Suppress thin horizontal staff lines via vertical morphological closing."""
    k = max(5, int(round(0.42 * float(unit))) | 1)
    kernel = np.ones((k, 1), dtype=np.uint8)
    src = gray if gray.dtype == np.uint8 else np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)


def _canvas_fit_size(height: int, width: int) -> tuple[int, int]:
    """Return ``(canvas_w, canvas_h)`` matching TrOMR aspect-preserving resize."""
    h = max(1, int(height))
    w = max(1, int(width))
    tr_ratio = float(TR_OMR_MAX_HEIGHT) / float(TR_OMR_MAX_WIDTH)
    if (float(h) / float(w)) > tr_ratio:
        return max(1, int(w / float(h) * TR_OMR_MAX_HEIGHT)), TR_OMR_MAX_HEIGHT
    return TR_OMR_MAX_WIDTH, max(1, int(h / float(w) * TR_OMR_MAX_WIDTH))


def _canvas_y_offset(content_height: int) -> int:
    """Vertical top-padding offset used by TrOMR when centering a staff on canvas."""
    return (TR_OMR_MAX_HEIGHT - int(content_height)) // 2


def _staff_y_profile(staff: Any, width: int, height: int) -> tuple[float, float, np.ndarray]:
    """Return ``(median_cy, unit_size, cy_by_x[0..width-1])`` for ``staff``."""
    grid = getattr(staff, "grid", None)
    xs_grid = np.arange(width, dtype=np.float64)
    if grid:
        pts_x: list[float] = []
        pts_y: list[float] = []
        spans: list[float] = []
        for pt in grid:
            y_vals = getattr(pt, "y", None)
            if y_vals is not None and len(y_vals) >= 2:
                pts_x.append(float(pt.x))
                pts_y.append(0.5 * (float(y_vals[0]) + float(y_vals[-1])))
                spans.append(float(y_vals[-1] - y_vals[0]))
        if pts_x:
            order = np.argsort(pts_x)
            sx = np.asarray(pts_x, dtype=np.float64)[order]
            sy = np.asarray(pts_y, dtype=np.float64)[order]
            cy_by_x = np.interp(xs_grid, sx, sy)
            med_cy = float(np.median(sy))
            five_line_h = float(np.median(spans)) if spans else 0.0
            raw_unit = float(getattr(staff, "average_unit_size", 0.0))
            unit = max(raw_unit, five_line_h / 4.0 if five_line_h > 0 else 0.0, 4.0)
            return med_cy, unit, np.clip(cy_by_x, 0.0, float(max(0, height - 1)))

    min_y = float(getattr(staff, "min_y", 0.0))
    max_y = float(getattr(staff, "max_y", min_y + 16.0))
    med_cy = 0.5 * (min_y + max_y)
    raw_unit = float(getattr(staff, "average_unit_size", 0.0))
    unit = max(raw_unit, (max_y - min_y) / 4.0, 4.0)
    cy_by_x = np.full(width, max(0.0, min(med_cy, float(max(0, height - 1)))), dtype=np.float64)
    return med_cy, unit, cy_by_x


def _score_barline_peaks(acc: np.ndarray, min_sep: int, min_peak: float) -> float:
    """Sum squared heights of well-separated barline consensus peaks for skew estimation."""
    a = acc.copy()
    w = len(a)
    total = 0.0
    for _ in range(16):
        idx = int(np.argmax(a))
        val = float(a[idx])
        if val < min_peak:
            break
        total += val * val
        a[max(0, idx - min_sep) : min(w, idx + min_sep + 1)] = 0.0
    return total


def detect_system_barlines_with_skew(
    image: Optional[np.ndarray],
    staffs: Sequence[Any],
    *,
    min_vote_ratio: float = 0.52,
) -> tuple[list[int], float, float]:
    """Detect system barlines ``(barlines_at_y_mid, y_mid, dx_per_dy)`` using per-staff projections.

    Handles both crisp digital scores and scanned/skewed orchestral pages as well as
    barlines interrupted between instrument choirs.
    """
    if image is None or not hasattr(image, "ndim") or image.ndim < 2:
        return [], 0.0, 0.0
    h, w = int(image.shape[0]), int(image.shape[1])
    n_staves = len(staffs)
    if h < 30 or w < 200 or n_staves < 2:
        return [], 0.0, 0.0

    gray = _to_grayscale(image).astype(np.float32)
    h_blur = cv2.blur(gray, (25, 1))
    v_diff = np.maximum(0.0, h_blur - gray)
    v_dil = cv2.dilate(v_diff, np.ones((1, 5), dtype=np.float32))

    profiles = [_staff_y_profile(s, w, h) for s in staffs]
    cys = np.asarray([p[0] for p in profiles], dtype=np.float64)
    med_unit = float(np.median([p[1] for p in profiles]))
    med_sh = max(12.0, 4.0 * med_unit)

    min_xs = [float(getattr(s, "min_x", 0.0)) for s in staffs]
    max_xs = [float(getattr(s, "max_x", float(w))) for s in staffs]
    valid_span = any(mx > mn + 40.0 for mn, mx in zip(min_xs, max_xs))
    if valid_span:
        sx_lo = max(8, int(round(min(min_xs) - 2.0 * med_sh)))
        sx_hi = min(w - 8, int(round(max(max_xs) + 3.0 * med_sh)))
    else:
        sx_lo = max(8, int(round(0.015 * w)))
        sx_hi = min(w - 8, int(round(0.985 * w)))

    xs_grid = np.arange(w, dtype=np.int32)
    hits = np.zeros((n_staves, w), dtype=np.float32)
    for idx, (staff, (_cy, unit, cy_by_x)) in enumerate(zip(staffs, profiles)):
        half = max(4, int(round(0.42 * 4.0 * unit)))
        yc = np.clip(np.round(cy_by_x).astype(np.int32), half, max(half, h - half - 1))
        if yc[0] + half >= h or yc[0] - half < 0:
            continue
        offsets = np.arange(-half, half + 1, dtype=np.int32)[:, None]
        ys = np.clip(yc[None, :] + offsets, 0, h - 1)
        band = v_dil[ys, xs_grid[None, :]]
        span = (band > 8.0).mean(axis=0)
        strength = band.mean(axis=0)
        row = ((span >= 0.84) & (strength > 10.0)).astype(np.float32)

        # Fuse SegNet-detected BarLines on this staff as weak corroborating evidence
        get_bls = getattr(staff, "get_bar_lines", None)
        if callable(get_bls):
            try:
                for bl in get_bls():
                    center = getattr(bl, "center", None)
                    if center is not None:
                        cx = int(round(float(center[0])))
                        if sx_lo <= cx < sx_hi:
                            row[max(0, cx - 3) : min(w, cx + 4)] = 1.0
            except Exception:
                pass
        hits[idx] = row

    hits[:, :sx_lo] = 0.0
    hits[:, sx_hi:] = 0.0
    hits_wide = cv2.dilate(hits, np.ones((1, 11), dtype=np.float32))
    hits_narrow = cv2.dilate(hits, np.ones((1, 5), dtype=np.float32))

    y_mid = 0.5 * float(np.min(cys) + np.max(cys))
    y_span = max(1.0, float(np.max(cys) - np.min(cys)))
    y_norm = (cys - y_mid) / y_span

    def _accumulate(src_hits: np.ndarray, dx_total: int) -> np.ndarray:
        acc = np.zeros(w, dtype=np.float32)
        for i in range(n_staves):
            sh_x = int(round(dx_total * y_norm[i]))
            if sh_x == 0:
                acc += src_hits[i]
            elif sh_x > 0 and sh_x < w:
                acc[:-sh_x] += src_hits[i, sh_x:]
            elif sh_x < 0 and (-sh_x) < w:
                acc[-sh_x:] += src_hits[i, : w + sh_x]
        return acc

    min_sep = max(20, int(round(3.0 * med_sh)))
    min_peak = max(1.5, 0.40 * n_staves)
    max_skew_dx = min(160, max(48, int(round(0.045 * y_span))))

    candidates_coarse = sorted(set(list(range(-max_skew_dx, max_skew_dx + 1, 4)) + [0]))
    coarse_dx = 0
    best_coarse_sc = -1.0
    for dx in candidates_coarse:
        acc = _accumulate(hits_wide, dx)
        sc = _score_barline_peaks(acc, min_sep, min_peak) - 1e-5 * abs(dx)
        if sc > best_coarse_sc:
            best_coarse_sc = sc
            coarse_dx = dx

    fine_lo = max(-max_skew_dx, coarse_dx - 16)
    fine_hi = min(max_skew_dx, coarse_dx + 16)
    fine_scores: list[tuple[int, float]] = []
    max_fine_sc = -1.0
    for dx in range(fine_lo, fine_hi + 1):
        acc = _accumulate(hits, dx)
        sc = _score_barline_peaks(acc, min_sep, min_peak)
        fine_scores.append((dx, sc))
        if sc > max_fine_sc:
            max_fine_sc = sc

    top_dxs = [dx for dx, sc in fine_scores if max_fine_sc > 0 and sc >= 0.99 * max_fine_sc]
    if 0 in top_dxs:
        best_dx = 0
    elif top_dxs:
        best_dx = top_dxs[len(top_dxs) // 2]
    else:
        best_dx = coarse_dx

    # Re-evaluate at best_dx with un-broadened strokes so adjacent time-sigs / courtesy stubs stay separate
    final_hits = hits if abs(best_dx) <= 4 else hits_narrow
    best_acc = _accumulate(final_hits, best_dx)
    min_votes = max(2.0, float(min_vote_ratio) * float(n_staves))
    cols = np.where(best_acc >= min_votes)[0]
    clusters: list[list[int]] = []
    for col in cols:
        c = int(col)
        if not clusters or (c - clusters[-1][-1]) > 10:
            clusters.append([c])
        else:
            clusters[-1].append(c)

    max_cluster_w = max(28, int(round(0.65 * med_sh)))
    raw: list[int] = [
        int(round(float(np.mean(g))))
        for g in clusters
        if (g[-1] - g[0]) <= max_cluster_w and sx_lo <= np.mean(g) <= sx_hi
    ]

    # Collapse tight double-barline / repeat strokes (<= 0.35 * staff_h) without merging courtesy stubs
    double_tol = max(10.0, 0.35 * med_sh)
    merged: list[list[int]] = []
    for x in raw:
        if merged and (x - merged[-1][-1]) <= double_tol:
            merged[-1].append(x)
        else:
            merged.append([x])
    pts = [int(round(float(np.mean(g)))) for g in merged]

    # Filter opening time-signature false hits and trailing courtesy stubs
    if len(pts) >= 3:
        left, right = pts[0], pts[-1]
        interior = [
            x
            for x in pts[1:-1]
            if (x - left) >= 4.5 * med_sh and (right - x) >= 3.5 * med_sh
        ]
        pts = [left] + interior + [right]

    dx_per_dy = float(best_dx) / y_span if y_span > 1.0 else 0.0
    if len(pts) < 3:
        log.debug(
            "System barlines: fewer than 3 boundaries detected (%s, staves=%d, max_votes=%.1f) — chunking disabled",
            pts,
            n_staves,
            float(best_acc.max()) if best_acc.size else 0.0,
        )
        return [], y_mid, 0.0

    log.info(
        "Detected %d system barline boundaries (%d measures) across %d staves (skew_dx=%+d px): %s",
        len(pts),
        len(pts) - 1,
        n_staves,
        best_dx,
        pts,
    )
    return pts, y_mid, dx_per_dy


def detect_system_barlines(
    image: Optional[np.ndarray],
    staffs: Sequence[Any],
    *,
    min_vote_ratio: float = 0.52,
) -> list[int]:
    """Detect system barline X-coordinates using per-staff vertical-contrast consensus."""
    pts, _y_mid, _dx_per_dy = detect_system_barlines_with_skew(
        image, staffs, min_vote_ratio=min_vote_ratio
    )
    return pts


def staff_barlines_from_system(
    barlines_x: Sequence[int],
    staff: Any,
    y_mid: float = 0.0,
    dx_per_dy: float = 0.0,
) -> list[int]:
    """Shift system median barline X-coordinates to match a specific staff's vertical position."""
    if not barlines_x or abs(dx_per_dy) < 1e-6:
        return [int(x) for x in barlines_x]
    cy = 0.5 * (float(getattr(staff, "min_y", 0.0)) + float(getattr(staff, "max_y", 0.0)))
    shift = int(round(dx_per_dy * (cy - y_mid)))
    return [int(x + shift) for x in barlines_x]


def should_chunk_staff(
    staff: Any,
    image: Optional[np.ndarray],
    region: Sequence[int] | np.ndarray,
    barlines_x: Sequence[int],
    *,
    min_native_5line_px: float = MIN_NATIVE_5LINE_PX,
) -> tuple[bool, str]:
    """Return ``(should_chunk, reason)`` deciding if a staff needs native-scale chunked OMR."""
    if len(barlines_x) < 3:
        return False, "insufficient_barlines"
    if image is None or not hasattr(image, "ndim") or image.ndim < 2:
        return False, "missing_image"
    grid = getattr(staff, "grid", None)
    if not grid or len(grid) < 2:
        return False, "missing_staff_grid"
    if len(region) < 4:
        return False, "invalid_region"

    reg_w = max(1.0, float(region[2]) - float(region[0]))
    reg_h = max(1.0, float(region[3]) - float(region[1]))
    unit = max(float(getattr(staff, "average_unit_size", 4.0)), 1.0)
    five_line_h = 4.0 * unit
    canvas_w, _canvas_h = _canvas_fit_size(int(round(reg_h)), int(round(reg_w)))
    full_5line_px = five_line_h * (float(canvas_w) / reg_w)
    if full_5line_px >= min_native_5line_px:
        return False, f"native_scale_ok:{full_5line_px:.1f}px"

    n_notes = 0
    get_num_notes = getattr(staff, "get_number_of_notes", None)
    if callable(get_num_notes):
        try:
            n_notes = int(get_num_notes())
        except Exception:
            n_notes = 0

    if n_notes >= 3:
        return True, f"compressed:{full_5line_px:.1f}px,notes={n_notes}"

    # Check non-staff ink inside measure interiors so rest-only staves bypass fast
    gray = _to_grayscale(image)
    h, w = int(gray.shape[0]), int(gray.shape[1])
    _cy, _u, cy_by_x = _staff_y_profile(staff, w, h)
    half_band = max(4, int(round(0.65 * five_line_h)))
    cy_mid = int(round(float(np.median(cy_by_x))))
    y0 = max(0, cy_mid - half_band)
    y1 = min(h, cy_mid + half_band + 1)
    if y1 <= y0 + 4:
        return True, f"compressed:{full_5line_px:.1f}px,narrow_band"

    band = _erase_staff_lines(np.asarray(gray[y0:y1, :], dtype=np.uint8), unit)
    row_bg = np.percentile(band, 90, axis=1, keepdims=True)
    ink = np.maximum(0.0, row_bg - band.astype(np.float32)) > 35.0
    col_ink_rows = ink.sum(axis=0)
    active_cols = col_ink_rows >= 2

    pad = max(6, int(round(0.30 * five_line_h)))
    n_measures = len(barlines_x) - 1
    max_bar_cols = 0
    total_cols = 0
    for mi in range(n_measures):
        x_a = max(0, int(barlines_x[mi]) + pad)
        if mi == 0:
            x_a = max(x_a, min(int(barlines_x[1]) - pad, int(barlines_x[0]) + int(round(4.8 * five_line_h))))
        x_b = min(w, int(barlines_x[mi + 1]) - pad)
        if x_b <= x_a:
            continue
        cnt = int(active_cols[x_a:x_b].sum())
        max_bar_cols = max(max_bar_cols, cnt)
        total_cols += cnt

    max_thresh = max(24, int(round(0.48 * five_line_h)))
    tot_thresh = max(48, int(round(0.55 * five_line_h * n_measures)))
    if max_bar_cols < max_thresh and total_cols < tot_thresh:
        return False, f"empty_rest_only:max_cols={max_bar_cols},total={total_cols}"

    return True, f"compressed:{full_5line_px:.1f}px,ink_cols={total_cols}"


def _has_cross_barline_arc(
    strip: np.ndarray,
    bar_x: int,
    staff_y0: int,
    staff_y1: int,
    unit: float,
) -> bool:
    """Detect whether a tie/slur arc bridges horizontally across ``bar_x`` in ``strip``."""
    h, w = int(strip.shape[0]), int(strip.shape[1])
    if bar_x <= 4 or bar_x >= w - 4:
        return False
    gray = _to_grayscale(strip).astype(np.float32)
    paper_bg = float(np.percentile(gray, 90))
    ink = (paper_bg - gray) > 25.0

    # Mask local horizontal staff lines around bar_x so wobbly dewarped lines never mimic ties
    sy_lo = max(0, int(round(staff_y0)) - 3)
    sy_hi = min(h, int(round(staff_y1)) + 4)
    wx0 = max(0, int(round(bar_x - 4.0 * unit)))
    wx1 = min(w, int(round(bar_x + 4.0 * unit)) + 1)
    if sy_hi > sy_lo and wx1 > wx0:
        win = gray[sy_lo:sy_hi, wx0:wx1]
        is_line_row = np.percentile(win, 75, axis=1) < (paper_bg - 25.0)
        if np.any(is_line_row):
            dilated = np.convolve(is_line_row.astype(np.int32), np.ones(5, dtype=np.int32), mode="same") > 0
            ink[sy_lo:sy_hi][dilated, :] = False

    y0 = max(0, int(round(staff_y0 - 2.2 * unit)))
    y1 = min(h, int(round(staff_y1 + 2.2 * unit)) + 1)
    if y1 <= y0 + 2:
        return False

    gap = max(2, int(round(0.25 * unit)))
    reach = max(gap + 3, int(round(1.6 * unit)))
    l0, l1 = max(0, bar_x - reach), max(0, bar_x - gap)
    r0, r1 = min(w, bar_x + gap + 1), min(w, bar_x + reach + 1)
    if l1 <= l0 or r1 <= r0:
        return False

    left_patch = ink[y0:y1, l0:l1]
    right_patch = ink[y0:y1, r0:r1]

    # Measure vertical thickness of ink runs to distinguish thin horizontal tie arcs from chords/barlines
    left_row_hits = left_patch.sum(axis=1) >= max(2, min(left_patch.shape[1], int(round(0.45 * unit))))
    right_row_hits = right_patch.sum(axis=1) >= max(2, min(right_patch.shape[1], int(round(0.45 * unit))))
    tol_rows = max(1, int(round(0.5 * unit)))

    def _thin_run_rows(mask: np.ndarray) -> set[int]:
        out: set[int] = set()
        run_start: Optional[int] = None
        for r, val in enumerate(mask):
            if val and run_start is None:
                run_start = r
            elif not val and run_start is not None:
                if (r - run_start) <= max(6, int(round(0.6 * unit))):
                    out.update(range(run_start, r))
                run_start = None
        if run_start is not None and (len(mask) - run_start) <= max(6, int(round(0.6 * unit))):
            out.update(range(run_start, len(mask)))
        return out

    l_rows = _thin_run_rows(left_row_hits)
    r_rows = _thin_run_rows(right_row_hits)
    for lr in l_rows:
        if any(abs(lr - rr) <= tol_rows for rr in r_rows):
            return True
    return False


def plan_staff_chunks(
    strip_barlines_x: Sequence[int],
    strip_height: int,
    five_line_h: float,
    *,
    strip_width: Optional[int] = None,
    unit: Optional[float] = None,
    tie_barline_flags: Optional[Sequence[bool]] = None,
) -> list[StaffChunkPlan]:
    """Partition ``[0 .. strip_w]`` at clean interior barlines with right-context overlap."""
    if not strip_barlines_x:
        return []

    u = float(unit) if unit is not None else max(4.0, float(five_line_h) / 4.0)
    bar_pad_left = max(4, int(round(0.30 * u)))
    right_ctx = max(72, int(round(12.0 * u)))
    max_w = int(strip_width) if strip_width is not None else int(strip_barlines_x[-1]) + max(6, int(round(0.45 * u))) + 8

    # Build monotonic well-separated interior split points over [0 .. max_w]
    edge_margin = max(60, int(round(3.5 * five_line_h)))
    min_split_gap = max(50, int(round(2.5 * five_line_h)))
    raw_ties = list(tie_barline_flags) if tie_barline_flags is not None else [False] * len(strip_barlines_x)
    candidates = sorted(
        ((int(bx), bool(raw_ties[i]) if i < len(raw_ties) else False) for i, bx in enumerate(strip_barlines_x)),
        key=lambda item: item[0],
    )
    interior: list[int] = []
    interior_ties: list[bool] = []
    for bx, tflag in candidates:
        if bx < edge_margin or bx > (max_w - edge_margin):
            continue
        if interior and (bx - interior[-1]) < min_split_gap:
            continue
        interior.append(bx)
        interior_ties.append(tflag)

    bounds = [0] + interior + [max_w]
    bound_ties = [False] + interior_ties + [False]
    n_segs = len(bounds) - 1
    if len(bounds) < 3 or n_segs < 2:
        return []

    header_est_w = max(80, int(round(12.0 * u)))
    target_mid = 0.5 * (TARGET_MIN_5LINE_PX + TARGET_MAX_5LINE_PX)

    def _chunk_bounds_and_scale(m_start: int, m_end: int) -> tuple[int, int, int, int, float]:
        keep_xs = int(bounds[m_start])
        keep_xe = int(bounds[m_end])
        xs = 0 if m_start == 0 else max(0, keep_xs + bar_pad_left)
        xe = max_w if m_end == n_segs else min(max_w, keep_xe + right_ctx)
        chunk_w = max(1, (xe - xs) + (0 if m_start == 0 else header_est_w))
        cw, _ch = _canvas_fit_size(max(1, int(strip_height)), chunk_w)
        five_px = float(five_line_h) * (float(cw) / float(chunk_w))
        return keep_xs, keep_xe, xs, xe, five_px

    # DP over segment boundaries 0..n_segs: minimize (num_chunks, tie_splits, max_scale_penalty)
    inf_cost = (10**9, 10**9, 1e9)
    dp: list[tuple[int, int, float]] = [inf_cost] * (n_segs + 1)
    parent: list[Optional[tuple[int, int, int, int, int, float]]] = [None] * (n_segs + 1)
    dp[0] = (0, 0, 0.0)

    for end_m in range(1, n_segs + 1):
        for start_m in range(end_m):
            if dp[start_m] >= inf_cost:
                continue
            keep_xs, keep_xe, xs, xe, five_px = _chunk_bounds_and_scale(start_m, end_m)
            if (end_m - start_m) > 1 and five_px < TARGET_MIN_5LINE_PX:
                continue
            n_ch = dp[start_m][0] + 1
            if n_ch > MAX_CHUNKS_PER_STAFF and (end_m - start_m) > 1:
                continue
            crosses_tie = 1 if (0 < start_m < len(bound_ties) and bound_ties[start_m]) else 0
            t_splits = dp[start_m][1] + crosses_tie
            dev = abs(five_px - target_mid)
            if five_px < TARGET_MIN_5LINE_PX:
                dev += 50.0 * (TARGET_MIN_5LINE_PX - five_px)
            cand_cost = (n_ch, t_splits, max(dp[start_m][2], dev))
            if cand_cost < dp[end_m]:
                dp[end_m] = cand_cost
                parent[end_m] = (start_m, keep_xs, keep_xe, xs, xe, five_px)

    if dp[n_segs] >= inf_cost:
        plans: list[StaffChunkPlan] = []
        for m_idx in range(n_segs):
            keep_xs, keep_xe, xs, xe, five_px = _chunk_bounds_and_scale(m_idx, m_idx + 1)
            t_flag = bool(0 < m_idx < len(bound_ties) and bound_ties[m_idx])
            plans.append(StaffChunkPlan(m_idx, m_idx + 1, xs, xe, five_px, t_flag, keep_xs, keep_xe))
        return plans

    rev_plans: list[StaffChunkPlan] = []
    cur = n_segs
    while cur > 0:
        info = parent[cur]
        assert info is not None
        start_m, keep_xs, keep_xe, xs, xe, five_px = info
        t_flag = bool(0 < start_m < len(bound_ties) and bound_ties[start_m])
        rev_plans.append(StaffChunkPlan(start_m, cur, xs, xe, five_px, t_flag, keep_xs, keep_xe))
        cur = start_m
    rev_plans.reverse()
    return rev_plans


def extract_clef_key_header(
    strip: np.ndarray,
    b0_x: int,
    b1_x: int,
    staff_y0: int,
    staff_y1: int,
    unit: float,
    chunk0_symbols: Sequence[Any],
    chunk0_x_in_strip: Sequence[Optional[float]],
) -> tuple[np.ndarray, int]:
    """Slice ``[Clef + Key Signature]`` from ``strip`` using Chunk 0 token coordinates + spacer."""
    h, w = int(strip.shape[0]), int(strip.shape[1])
    gray = _to_grayscale(strip)
    med_col = np.median(gray, axis=1, keepdims=True)

    # Cut strictly between Chunk 0 clef/key tokens and first meter/dynamic/note/rest token
    key_fifths = 0
    last_header_x: Optional[float] = None
    first_body_x: Optional[float] = None
    for idx, sym in enumerate(chunk0_symbols):
        rhythm = str(getattr(sym, "rhythm", ""))
        sx = chunk0_x_in_strip[idx] if idx < len(chunk0_x_in_strip) else None
        valid_x = sx is not None and not math.isnan(sx)
        if rhythm.startswith("keySignature_") and first_body_x is None:
            try:
                key_fifths = min(7, abs(int(rhythm.split("_", 1)[1])))
            except ValueError:
                key_fifths = 0
            if valid_x:
                last_header_x = float(sx) if last_header_x is None else max(last_header_x, float(sx))
        elif rhythm.startswith("clef_") and first_body_x is None:
            if valid_x:
                last_header_x = float(sx) if last_header_x is None else max(last_header_x, float(sx))
        elif rhythm.startswith(("timeSignature", "dynamic_", "note_", "rest_")):
            if valid_x and float(sx) > float(b0_x) + 1.5 * unit:
                first_body_x = float(sx)
            break
        elif _is_barline_token(rhythm) and idx > 0:
            break

    min_cut = min(w, max(6, int(round(float(b0_x) + 2.4 * unit))))
    max_safe_x = float(b0_x) + (4.8 + 1.35 * key_fifths) * unit
    max_cut = min(w, max(min_cut, int(round(min(max_safe_x, float(b1_x) - 0.8 * unit)))))

    if last_header_x is not None and first_body_x is not None and first_body_x > last_header_x + 0.5 * unit:
        ideal = min(
            0.5 * (last_header_x + first_body_x),
            last_header_x + 2.2 * unit,
            first_body_x - 0.55 * unit,
        )
        header_cut_x = int(round(ideal))
    elif last_header_x is not None:
        header_cut_x = int(round(last_header_x + 2.0 * unit))
    elif first_body_x is not None:
        header_cut_x = int(round(first_body_x - 0.55 * unit))
    else:
        header_cut_x = int(round(float(b0_x) + (3.6 + 1.15 * key_fifths) * unit))

    header_cut_x = max(min_cut, min(max_cut, header_cut_x))

    raw_hdr = strip[:, :header_cut_x].copy()
    # Scrub leaked dynamics / tempo text above/below staff margin so Chunks 1..K don't duplicate them
    keep_y0 = max(0, int(round(staff_y0 - 1.8 * unit)))
    keep_y1 = min(h, int(round(staff_y1 + 1.8 * unit)) + 1)
    bg_rows = np.repeat(np.round(med_col).astype(strip.dtype), raw_hdr.shape[1], axis=1)
    if raw_hdr.ndim == 2:
        if keep_y0 > 0:
            raw_hdr[:keep_y0, :] = bg_rows[:keep_y0, :]
        if keep_y1 < h:
            raw_hdr[keep_y1:, :] = bg_rows[keep_y1:, :]
    else:
        for ch_i in range(raw_hdr.shape[2]):
            if keep_y0 > 0:
                raw_hdr[:keep_y0, :, ch_i] = bg_rows[:keep_y0, :]
            if keep_y1 < h:
                raw_hdr[keep_y1:, :, ch_i] = bg_rows[keep_y1:, :]

    spacer_w = max(6, int(round(1.1 * unit)))
    if strip.ndim == 2:
        spacer = np.repeat(np.round(med_col).astype(strip.dtype), spacer_w, axis=1)
    else:
        spacer = np.repeat(np.round(med_col).astype(strip.dtype)[:, :, None], spacer_w, axis=1)
        spacer = np.repeat(spacer, strip.shape[2], axis=2)

    header_img = np.concatenate([raw_hdr, spacer], axis=1)
    log.debug(
        "Extracted clef/key header: cut_x=%d (b0=%d, b1=%d, fifths=%d, last_hx=%s, first_bx=%s) -> %dx%d",
        header_cut_x,
        b0_x,
        b1_x,
        key_fifths,
        f"{last_header_x:.1f}" if last_header_x is not None else "None",
        f"{first_body_x:.1f}" if first_body_x is not None else "None",
        header_img.shape[1],
        header_img.shape[0],
    )
    return header_img, int(header_img.shape[1])


def _is_barline_token(rhythm: str) -> bool:
    return "barline" in rhythm or "repeat" in rhythm


def _add_slur_mark(sym: Any, mark: str) -> None:
    slur = str(getattr(sym, "slur", "_"))
    if mark in slur:
        return
    if hasattr(sym, "add_slurs"):
        updated = sym.add_slurs([mark])
        sym.slur = updated.slur
        return
    parts = [p for p in slur.split("_") if p and p != "."]
    parts.append(mark)
    sym.slur = "_".join(sorted(parts))


def _strip_chunk_header(
    symbols: Sequence[Any],
    strip_xs: Sequence[Optional[float]],
    keep_start: float,
    unit: float,
) -> tuple[list[Any], Optional[str]]:
    """Strip synthetic ``[Clef + Key]`` header tokens and leading boundary barlines from ``Chunks 1..K-1``."""
    leading_repeat: Optional[str] = None
    first_body_idx: Optional[int] = None
    n_syms = len(symbols)

    for idx, sym in enumerate(symbols):
        rhythm = str(getattr(sym, "rhythm", ""))
        sx = strip_xs[idx] if idx < len(strip_xs) else None
        valid_x = sx is not None and not math.isnan(sx)

        if rhythm.startswith(("clef_", "keySignature_")):
            if not valid_x or float(sx) < keep_start + 0.5 * unit:
                continue
        elif rhythm.startswith(("timeSignature", "dynamic_")):
            if valid_x and float(sx) < keep_start - 0.5 * unit:
                continue
        elif _is_barline_token(rhythm):
            if "repeat" in rhythm:
                leading_repeat = rhythm
            continue

        first_body_idx = idx
        break

    if first_body_idx is None:
        return [], leading_repeat

    body: list[Any] = []
    for idx in range(first_body_idx, n_syms):
        sym = symbols[idx]
        sx = strip_xs[idx] if idx < len(strip_xs) else None
        rhythm = str(getattr(sym, "rhythm", ""))
        if (
            not body
            and sx is not None
            and not math.isnan(sx)
            and float(sx) < keep_start - 0.5 * unit
            and rhythm.startswith(("clef_", "keySignature_"))
        ):
            continue
        body.append(sym)
    return body, leading_repeat


def _extract_musical_slots(symbols: Sequence[Any]) -> list[tuple[str, frozenset[str]]]:
    """Normalize ``note_*``/``rest_*`` symbols (collapsing chords) for overlap matching."""
    slots: list[tuple[str, frozenset[str]]] = []
    in_chord = False
    for sym in symbols:
        rhythm = str(getattr(sym, "rhythm", ""))
        if rhythm == "chord":
            in_chord = True
            continue
        if rhythm.startswith(("note_", "rest_")) or rhythm == "measureRest":
            pitch = str(getattr(sym, "pitch", "_"))
            p_set = frozenset({pitch}) if pitch not in ("_", ".", "") else frozenset()
            if in_chord and slots:
                prev_r, prev_p = slots[-1]
                slots[-1] = (prev_r, prev_p | p_set)
            else:
                slots.append((rhythm, p_set))
            in_chord = False
        else:
            in_chord = False
    return slots


def _slots_match_prefix(
    stub_slots: Sequence[tuple[str, frozenset[str]]],
    next_bar_slots: Sequence[tuple[str, frozenset[str]]],
    *,
    allow_equal_len: bool,
) -> bool:
    """Return True when ``stub_slots`` matches the opening sequence of ``next_bar_slots``."""
    if not stub_slots or not next_bar_slots:
        return False
    if len(stub_slots) > len(next_bar_slots):
        return False
    if not allow_equal_len and len(stub_slots) == len(next_bar_slots):
        return False
    for (r1, p1), (r2, p2) in zip(stub_slots, next_bar_slots):
        if r1 != r2:
            return False
        if p1 and p2 and not (p1 & p2):
            return False
    return True


def _trim_chunk_overlap(
    body: Sequence[Any],
    next_body: Sequence[Any],
    expected_bars: int,
) -> list[Any]:
    """Trim right-context overlap from a non-final chunk via coordinate-free sequence matching."""
    bar_idxs = [i for i, sym in enumerate(body) if _is_barline_token(str(getattr(sym, "rhythm", "")))]
    if not bar_idxs:
        return list(body)

    first_next_bar = next(
        (i for i, sym in enumerate(next_body) if _is_barline_token(str(getattr(sym, "rhythm", "")))),
        len(next_body),
    )
    next_bar_slots = _extract_musical_slots(next_body[:first_next_bar])

    cut_end = bar_idxs[-1] + 1
    while len(bar_idxs) >= 2:
        b_prev = bar_idxs[-2]
        b_last = bar_idxs[-1]
        stub = body[b_prev + 1 : b_last]
        stub_slots = _extract_musical_slots(stub)
        excess_bars = expected_bars > 0 and len(bar_idxs) > expected_bars
        is_empty_stub = len(stub_slots) == 0
        is_prefix_overlap = _slots_match_prefix(
            stub_slots,
            next_bar_slots,
            allow_equal_len=excess_bars,
        )
        if is_empty_stub or is_prefix_overlap or excess_bars:
            log.debug(
                "Trimmed overlap stub at barline %d/%d (expected_bars=%d, empty=%s, prefix=%s, excess=%s, stub=%s)",
                len(bar_idxs) - 1,
                len(bar_idxs),
                expected_bars,
                is_empty_stub,
                is_prefix_overlap,
                excess_bars,
                [(r, sorted(p)) for r, p in stub_slots],
            )
            bar_idxs.pop()
            cut_end = b_prev + 1
        else:
            break
    return list(body[:cut_end])


def _strip_trailing_cautionary_metadata(symbols: list[Any]) -> list[Any]:
    """Drop trailing cautionary-only tokens after the final barline with no music in the bar."""
    last_bar_idx = -1
    for i, sym in enumerate(symbols):
        if _is_barline_token(str(getattr(sym, "rhythm", ""))):
            last_bar_idx = i
    if last_bar_idx < 0 or last_bar_idx == len(symbols) - 1:
        return symbols
    trailing = symbols[last_bar_idx + 1 :]
    if all(
        str(getattr(s, "rhythm", "")).startswith(("timeSignature", "keySignature_", "clef_"))
        for s in trailing
    ):
        return symbols[: last_bar_idx + 1]
    return symbols


def stitch_chunk_symbols(
    chunk_results: Sequence[tuple[StaffChunkPlan, Sequence[Any], Sequence[Optional[float]], int]],
    strip_barlines_x: Sequence[int],
    unit: float,
) -> list[Any]:
    """Concatenate decoded chunks via coordinate-free sequence overlap stitching and tie bridging."""
    if not chunk_results:
        return []

    n_total_chunks = len(chunk_results)

    # Pre-clean Chunks 1..K-1 so each Chunk k can validate its overlap stub against Chunk k+1's clean opening
    prepared_bodies: list[list[Any]] = [list(chunk_results[0][1])]
    leading_repeats: list[Optional[str]] = [None]
    for plan, symbols, strip_xs, _header_w in chunk_results[1:]:
        keep_start = float(plan.keep_x_start if plan.keep_x_start > 0 else plan.x_start)
        body, leading_rep = _strip_chunk_header(symbols, strip_xs, keep_start, unit)
        prepared_bodies.append(body)
        leading_repeats.append(leading_rep)

    stitched: list[Any] = []
    for chunk_idx, (plan, _symbols, _strip_xs, _header_w) in enumerate(chunk_results):
        n_segs = max(1, plan.measure_end - plan.measure_start)
        body = prepared_bodies[chunk_idx]

        if chunk_idx < n_total_chunks - 1:
            kept = _trim_chunk_overlap(body, prepared_bodies[chunk_idx + 1], n_segs)
        else:
            kept = _strip_trailing_cautionary_metadata(list(body))

        if chunk_idx == 0:
            stitched.extend(kept)
            continue

        if not kept and leading_repeats[chunk_idx] is None:
            continue

        if stitched and not _is_barline_token(str(getattr(stitched[-1], "rhythm", ""))):
            stitched.append(type(stitched[-1])("barline"))
            log.debug(
                "Inserted missing boundary barline before chunk at segment %d",
                plan.measure_start,
            )

        rep = leading_repeats[chunk_idx]
        if rep is not None and stitched and _is_barline_token(str(getattr(stitched[-1], "rhythm", ""))):
            stitched[-1].rhythm = rep

        if not kept:
            continue

        if plan.crosses_tie_left and stitched and kept:
            prev_note = next(
                (s for s in reversed(stitched) if str(getattr(s, "rhythm", "")).startswith("note_")),
                None,
            )
            next_note = next(
                (s for s in kept if str(getattr(s, "rhythm", "")).startswith("note_")),
                None,
            )
            if (
                prev_note is not None
                and next_note is not None
                and getattr(prev_note, "pitch", ".") not in (".", "_")
                and getattr(prev_note, "pitch", None) == getattr(next_note, "pitch", None)
                and getattr(prev_note, "lift", None) == getattr(next_note, "lift", None)
            ):
                _add_slur_mark(prev_note, "slurStart")
                _add_slur_mark(next_note, "slurStop")
                log.debug(
                    "Bridged cross-chunk tie at boundary segment %d (pitch=%s%s)",
                    plan.measure_start,
                    getattr(prev_note, "pitch", "?"),
                    getattr(prev_note, "lift", ""),
                )

        stitched.extend(kept)

    return stitched


def _fallback_parse_staff(
    orig_parse_staff_image: Callable[..., list[Any]],
    debug: Any,
    index: int,
    staff: Any,
    image: np.ndarray,
    regions: Any,
    config: Any,
    page_to_input_image: Any,
) -> list[Any]:
    return orig_parse_staff_image(
        debug,
        index,
        staff,
        image,
        regions,
        config,
        page_to_input_image=page_to_input_image,
    )


def parse_staff_image_chunked(
    debug: Any,
    index: int,
    staff: Any,
    image: np.ndarray,
    regions: Any,
    config: Any,
    page_to_input_image: Callable[[tuple[float, float]], tuple[float, float]],
    *,
    barlines_x: Sequence[int],
    orig_parse_staff_image: Callable[..., list[Any]],
    homr_staff_parsing: Any,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> list[Any]:
    """Parse one compressed dense-score staff using native-scale barline chunks."""
    if is_cancelled():
        raise CancelledError("Conversion aborted by user.")

    calc_region = getattr(homr_staff_parsing, "_calculate_region", None)
    if calc_region is None or len(barlines_x) < 3:
        return _fallback_parse_staff(
            orig_parse_staff_image, debug, index, staff, image, regions, config, page_to_input_image
        )

    region = calc_region(staff, regions)
    should_chunk, reason = should_chunk_staff(staff, image, region, barlines_x)
    if not should_chunk:
        log.debug("Staff %02d single-pass (%s)", index, reason)
        return _fallback_parse_staff(
            orig_parse_staff_image, debug, index, staff, image, regions, config, page_to_input_image
        )

    try:
        crop_with_top = homr_staff_parsing.crop_image_and_return_new_top
        dewarp_staff_fn = homr_staff_parsing._dewarp_staff
        dewarp_staff_img_fn = homr_staff_parsing.dewarp_staff_image
        rm_edges_fn = homr_staff_parsing.remove_black_contours_at_edges_of_image
        add_canvas_fn = homr_staff_parsing.add_image_into_tr_omr_canvas
        canvas_size_fn = homr_staff_parsing.get_tr_omr_canvas_size
        parse_tromr_fn = homr_staff_parsing.parse_staff_tromr
        get_center_fn = homr_staff_parsing._get_symbol_center
        undo_crop_fn = homr_staff_parsing.undo_crop
        chain_fn = homr_staff_parsing.chain
        if is_cancelled():
            raise CancelledError("Conversion aborted by user.")

        # 1. Single 1.0x high-resolution dewarp across full staff width
        crop_reg = np.array([int(region[0]), int(region[1]), int(region[2]), int(region[3])])
        reg_step1 = crop_reg + np.array([-10, -50, 10, 50])
        crop1, top_left1 = crop_with_top(image, *reg_step1)
        staff_s1 = dewarp_staff_fn(staff, None, top_left1, 1.0)
        dewarp = dewarp_staff_img_fn(crop1, staff_s1, index, debug)
        dewarped = dewarp.dewarp(crop1)
        reg_step2 = crop_reg - np.array([*top_left1, *top_left1])
        strip, top_left2 = crop_with_top(dewarped, *reg_step2)
        unit = max(float(getattr(staff, "average_unit_size", 4.0)), 4.0)
        strip = rm_edges_fn(strip, unit)
        if strip is None or strip.size == 0 or strip.shape[0] < 12 or strip.shape[1] < 40:
            log.warning("Staff %02d degenerate strip after 1x dewarp — falling back", index)
            return _fallback_parse_staff(
                orig_parse_staff_image, debug, index, staff, image, regions, config, page_to_input_image
            )

        strip_to_page = chain_fn(undo_crop_fn(*top_left2), dewarp.undewarp_point, undo_crop_fn(*top_left1))
        cy_page = 0.5 * (float(getattr(staff, "min_y", 0.0)) + float(getattr(staff, "max_y", 0.0)))

        def _page_to_strip(pt: tuple[float, float]) -> tuple[float, float]:
            dx, dy = dewarp.dewarp_point((pt[0] - float(top_left1[0]), pt[1] - float(top_left1[1])))
            return dx - float(top_left2[0]), dy - float(top_left2[1])

        strip_w = int(strip.shape[1])
        raw_strip_bls = [
            max(0, min(strip_w - 1, int(round(_page_to_strip((float(bx), cy_page))[0]))))
            for bx in barlines_x
        ]
        strip_barlines_x = sorted(set(raw_strip_bls))

        _pts_0 = _page_to_strip((float(barlines_x[0]), float(getattr(staff, "min_y", cy_page - 2.0 * unit))))
        _pts_1 = _page_to_strip((float(barlines_x[0]), float(getattr(staff, "max_y", cy_page + 2.0 * unit))))
        staff_y0 = int(round(_pts_0[1]))
        staff_y1 = int(round(_pts_1[1]))
        if staff_y1 <= staff_y0:
            staff_y0 = int(round(strip.shape[0] * 0.5 - 2.0 * unit))
            staff_y1 = int(round(strip.shape[0] * 0.5 + 2.0 * unit))

        five_line_h = float(max(8, staff_y1 - staff_y0))
        tie_flags = [
            _has_cross_barline_arc(strip, bx, staff_y0, staff_y1, unit)
            for bx in strip_barlines_x
        ]

        plans = plan_staff_chunks(
            strip_barlines_x,
            int(strip.shape[0]),
            five_line_h,
            strip_width=strip_w,
            unit=unit,
            tie_barline_flags=tie_flags,
        )
        if len(plans) <= 1:
            log.debug("Staff %02d planned %d chunk(s) — running single-pass", index, len(plans))
            return _fallback_parse_staff(
                orig_parse_staff_image, debug, index, staff, image, regions, config, page_to_input_image
            )

        def _decode_chunk(chunk_img: np.ndarray, plan: StaffChunkPlan, header_w: int) -> tuple[list[Any], list[Optional[float]]]:
            if is_cancelled():
                raise CancelledError("Conversion aborted by user.")
            canvas = add_canvas_fn(chunk_img)
            sc = canvas_size_fn(chunk_img.shape)
            scale_x = float(sc[0]) / float(max(1, chunk_img.shape[1]))
            scale_y = float(sc[1]) / float(max(1, chunk_img.shape[0]))
            y_off = float(_canvas_y_offset(int(sc[1])))
            syms = parse_tromr_fn(staff_image=canvas, staff=staff, config=config)
            strip_xs: list[Optional[float]] = []
            for sym in syms:
                center = get_center_fn(sym)
                if center is None:
                    strip_xs.append(None)
                    continue
                x_chunk = float(center[0]) / max(1e-6, scale_x)
                y_chunk = (float(center[1]) - y_off) / max(1e-6, scale_y)
                x_strip = x_chunk if header_w == 0 else ((x_chunk - float(header_w)) + float(plan.x_start))
                strip_xs.append(x_strip)
                sym.image_coordinates = page_to_input_image(strip_to_page((x_strip, y_chunk)))
            return syms, strip_xs

        # 2. Decode Chunk 0 and dynamically extract clean [Clef + Key Signature] header
        plan0 = plans[0]
        chunk0_img = strip[:, plan0.x_start : plan0.x_end]
        syms0, xs0 = _decode_chunk(chunk0_img, plan0, 0)
        hdr_b0 = strip_barlines_x[0] if strip_barlines_x[0] < int(round(4.0 * five_line_h)) else 0
        hdr_b1 = int(plan0.keep_x_end if plan0.keep_x_end > 0 else plan0.x_end)
        header_img, header_w = extract_clef_key_header(
            strip,
            hdr_b0,
            hdr_b1,
            staff_y0,
            staff_y1,
            unit,
            syms0,
            xs0,
        )

        chunk_outputs: list[tuple[StaffChunkPlan, list[Any], list[Optional[float]], int]] = [
            (plan0, syms0, xs0, 0)
        ]
        for plan in plans[1:]:
            body = strip[:, plan.x_start : plan.x_end]
            if body.shape[1] < 10:
                log.warning("Staff %02d chunk [%d..%d] too narrow (%d px) — falling back", index, plan.measure_start, plan.measure_end, body.shape[1])
                return _fallback_parse_staff(
                    orig_parse_staff_image, debug, index, staff, image, regions, config, page_to_input_image
                )
            spliced = np.concatenate([header_img, body], axis=1)
            syms_k, xs_k = _decode_chunk(spliced, plan, header_w)
            chunk_outputs.append((plan, syms_k, xs_k, header_w))

        stitched = stitch_chunk_symbols(chunk_outputs, strip_barlines_x, unit)
        if not stitched:
            log.warning("Staff %02d chunk stitching returned 0 symbols — falling back", index)
            return _fallback_parse_staff(
                orig_parse_staff_image, debug, index, staff, image, regions, config, page_to_input_image
            )

        log.debug(
            "Staff %02d chunked OMR complete (%s): %d chunks -> %d symbols (5-line scales=%s)",
            index,
            reason,
            len(plans),
            len(stitched),
            [round(p.expected_5line_px, 1) for p in plans],
        )
        return stitched
    except CancelledError:
        raise
    except Exception as exc:
        log.warning(
            "Chunked OMR failed on staff %02d (%s: %s) — falling back to single-pass",
            index,
            type(exc).__name__,
            exc,
        )
        return _fallback_parse_staff(
            orig_parse_staff_image, debug, index, staff, image, regions, config, page_to_input_image
        )
