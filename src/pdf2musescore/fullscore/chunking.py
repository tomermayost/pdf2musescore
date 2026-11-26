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
TARGET_MIN_5LINE_PX = 42.0
TARGET_MAX_5LINE_PX = 58.0
MAX_CHUNKS_PER_STAFF = 4


@dataclass(frozen=True)
class StaffChunkPlan:
    """Contiguous slice of whole measures inside a dewarped staff strip."""

    measure_start: int
    measure_end: int
    x_start: int
    x_end: int
    expected_5line_px: float
    crosses_tie_left: bool = False


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a 2D uint8/float-compatible grayscale view of ``image``."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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

    if n_notes >= 1:
        return True, f"compressed:{full_5line_px:.1f}px,notes={n_notes}"

    # Check non-staff ink inside measure interiors so rest-only staves bypass fast
    gray = _to_grayscale(image).astype(np.float32)
    h, w = int(gray.shape[0]), int(gray.shape[1])
    _cy, _u, cy_by_x = _staff_y_profile(staff, w, h)
    half_band = max(4, int(round(1.1 * five_line_h)))
    cy_mid = int(round(float(np.median(cy_by_x))))
    y0 = max(0, cy_mid - half_band)
    y1 = min(h, cy_mid + half_band + 1)
    if y1 <= y0 + 4:
        return True, f"compressed:{full_5line_px:.1f}px,narrow_band"

    band = gray[y0:y1, :]
    row_bg = np.median(band, axis=1, keepdims=True)
    ink = np.maximum(0.0, row_bg - band) > 25.0
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

    max_thresh = max(16, int(round(0.32 * five_line_h)))
    tot_thresh = max(36, int(round(0.65 * five_line_h * n_measures)))
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
    med_col = np.median(gray, axis=1, keepdims=True)
    ink = np.maximum(0.0, med_col - gray) > 22.0

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
    """Partition measures ``0..M-1`` into contiguous TrOMR chunks at native ~42-58px scale."""
    n_bounds = len(strip_barlines_x)
    n_meas = n_bounds - 1
    if n_meas < 2:
        return []

    u = float(unit) if unit is not None else max(4.0, float(five_line_h) / 4.0)
    bar_pad_right = max(6, int(round(0.45 * u)))
    bar_pad_left = max(4, int(round(0.30 * u)))
    max_w = int(strip_width) if strip_width is not None else int(strip_barlines_x[-1]) + bar_pad_right + 8
    header_est_w = min(int(strip_barlines_x[1] - strip_barlines_x[0]), max(40, int(round(4.5 * five_line_h))))
    target_mid = 0.5 * (TARGET_MIN_5LINE_PX + TARGET_MAX_5LINE_PX)

    def _chunk_bounds_and_scale(m_start: int, m_end: int) -> tuple[int, int, float]:
        xe = min(max_w, int(strip_barlines_x[m_end]) + bar_pad_right)
        if m_start == 0:
            xs = 0
            chunk_w = max(1, xe - xs)
        else:
            xs = max(0, int(strip_barlines_x[m_start]) + bar_pad_left)
            chunk_w = max(1, (xe - xs) + header_est_w)
        cw, _ch = _canvas_fit_size(max(1, int(strip_height)), chunk_w)
        five_px = float(five_line_h) * (float(cw) / float(chunk_w))
        return xs, xe, five_px

    # Check if full strip already meets target (should have been caught by should_chunk_staff)
    tie_flags = list(tie_barline_flags) if tie_barline_flags is not None else [False] * n_bounds

    # DP over measure boundaries 0..n_meas: minimize (num_chunks, tie_splits, max_scale_penalty)
    inf_cost = (10**9, 10**9, 1e9)
    dp: list[tuple[int, int, float]] = [inf_cost] * (n_meas + 1)
    parent: list[Optional[tuple[int, int, int, float]]] = [None] * (n_meas + 1)
    dp[0] = (0, 0, 0.0)

    for end_m in range(1, n_meas + 1):
        for start_m in range(end_m):
            if dp[start_m] >= inf_cost:
                continue
            xs, xe, five_px = _chunk_bounds_and_scale(start_m, end_m)
            # Reject chunks that remain below MIN target unless single measure already spans >= target width
            if (end_m - start_m) > 1 and five_px < TARGET_MIN_5LINE_PX:
                continue
            n_ch = dp[start_m][0] + 1
            if n_ch > MAX_CHUNKS_PER_STAFF and (end_m - start_m) > 1:
                continue
            crosses_tie = 1 if (start_m > 0 and start_m < len(tie_flags) and tie_flags[start_m]) else 0
            t_splits = dp[start_m][1] + crosses_tie
            dev = abs(five_px - target_mid)
            if five_px < TARGET_MIN_5LINE_PX:
                dev += 50.0 * (TARGET_MIN_5LINE_PX - five_px)
            cand_cost = (n_ch, t_splits, max(dp[start_m][2], dev))
            if cand_cost < dp[end_m]:
                dp[end_m] = cand_cost
                parent[end_m] = (start_m, xs, xe, five_px)

    if dp[n_meas] >= inf_cost:
        # Single measure wider than max_chunk_w edge case: fall back to 1-measure chunks
        plans: list[StaffChunkPlan] = []
        for m_idx in range(n_meas):
            xs, xe, five_px = _chunk_bounds_and_scale(m_idx, m_idx + 1)
            t_flag = bool(m_idx > 0 and m_idx < len(tie_flags) and tie_flags[m_idx])
            plans.append(StaffChunkPlan(m_idx, m_idx + 1, xs, xe, five_px, t_flag))
        return plans

    rev_plans: list[StaffChunkPlan] = []
    cur = n_meas
    while cur > 0:
        info = parent[cur]
        assert info is not None
        start_m, xs, xe, five_px = info
        t_flag = bool(start_m > 0 and start_m < len(tie_flags) and tie_flags[start_m])
        rev_plans.append(StaffChunkPlan(start_m, cur, xs, xe, five_px, t_flag))
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
    """Slice cleanly ``[Clef + Key Signature]`` from ``strip`` and append a blank 5-line spacer."""
    h, w = int(strip.shape[0]), int(strip.shape[1])
    gray = _to_grayscale(strip)
    med_col = np.median(gray, axis=1, keepdims=True)

    # Parse key signature count and first time-sig / dynamic / body-note X from Chunk 0 predictions
    key_fifths = 0
    first_body_x: Optional[float] = None
    for idx, sym in enumerate(chunk0_symbols):
        rhythm = str(getattr(sym, "rhythm", ""))
        sx = chunk0_x_in_strip[idx] if idx < len(chunk0_x_in_strip) else None
        if rhythm.startswith("keySignature_") and first_body_x is None:
            try:
                key_fifths = min(7, abs(int(rhythm.split("_", 1)[1])))
            except ValueError:
                key_fifths = 0
        elif rhythm.startswith(("timeSignature", "dynamic_", "note_", "rest_")):
            if sx is not None and not math.isnan(sx) and sx > float(b0_x) + 1.5 * unit:
                first_body_x = float(sx) if first_body_x is None else min(first_body_x, float(sx))
            break
        elif "barline" in rhythm or "repeat" in rhythm:
            if idx > 0:
                break

    band_y0 = max(0, int(round(staff_y0 - 1.3 * unit)))
    band_y1 = min(h, int(round(staff_y1 + 1.3 * unit)) + 1)
    core = gray[band_y0:band_y1, :].astype(np.float32)
    core_bg = med_col[band_y0:band_y1, :]
    ink = np.maximum(0.0, core_bg - core) > 24.0
    col_active = ink.sum(axis=0) >= 2

    search_lo = max(0, int(b0_x) + max(2, int(round(0.3 * unit))))
    search_hi = min(w - 1, int(b1_x) - max(4, int(round(0.8 * unit))))
    comps: list[list[int]] = []
    merge_gap = max(3, int(round(0.35 * unit)))
    for x in range(search_lo, max(search_lo, search_hi)):
        if col_active[x]:
            if not comps or (x - comps[-1][1]) > merge_gap:
                comps.append([x, x])
            else:
                comps[-1][1] = x

    max_header_x = float(b0_x) + (4.4 + 1.35 * key_fifths) * unit
    if first_body_x is not None:
        capped = first_body_x - 0.55 * unit
        if capped >= float(b0_x) + 2.2 * unit:
            max_header_x = min(max_header_x, capped)

    selected: list[list[int]] = []
    max_comps = 1 + key_fifths + 1
    body_guard = (first_body_x - 0.35 * unit) if first_body_x is not None else None
    for c_lo, c_hi in comps:
        if float(c_lo) > max_header_x:
            break
        if body_guard is not None and float(c_hi) >= body_guard and len(selected) >= 1:
            break
        if first_body_x is not None and float(c_lo) - 2.0 <= first_body_x <= float(c_hi) + 2.0 and len(selected) >= 1:
            break
        if selected:
            gap = c_lo - selected[-1][1]
            if key_fifths == 0 and gap > 1.35 * unit:
                break
            if len(selected) >= max_comps and gap > 0.8 * unit:
                break
        selected.append([c_lo, c_hi])
        if len(selected) >= max_comps:
            break

    if selected:
        header_cut_x = selected[-1][1] + max(4, int(round(0.45 * unit)))
    else:
        header_cut_x = int(round(float(b0_x) + (3.5 + 1.15 * key_fifths) * unit))

    min_cut = min(w, max(search_lo + max(6, int(round(2.0 * unit))), int(b0_x) + int(round(2.4 * unit))))
    max_cut = min(w, max(min_cut, int(b1_x) - max(4, int(round(0.8 * unit)))))
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
        "Extracted clef/key header: cut_x=%d (b0=%d, b1=%d, fifths=%d, first_body_x=%s) -> %dx%d",
        header_cut_x,
        b0_x,
        b1_x,
        key_fifths,
        f"{first_body_x:.1f}" if first_body_x is not None else "None",
        header_img.shape[1],
        header_img.shape[0],
    )
    return header_img, int(header_img.shape[1])


def _is_barline_token(rhythm: str) -> bool:
    return "barline" in rhythm or "repeat" in rhythm


def _is_special_barline(rhythm: str) -> bool:
    return _is_barline_token(rhythm) and rhythm != "barline"


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


def _trim_chunk_to_measure_count(
    symbols: Sequence[Any], expected_measures: int
) -> list[Any]:
    """Keep symbols up through the ``expected_measures``-th closing barline."""
    if expected_measures <= 0:
        return list(symbols)
    out: list[Any] = []
    bars_seen = 0
    for sym in symbols:
        rhythm = str(getattr(sym, "rhythm", ""))
        out.append(sym)
        if _is_barline_token(rhythm):
            bars_seen += 1
            if bars_seen >= expected_measures:
                break
    return out


def stitch_chunk_symbols(
    chunk_results: Sequence[tuple[StaffChunkPlan, Sequence[Any], Sequence[Optional[float]], int]],
    strip_barlines_x: Sequence[int],
    unit: float,
) -> list[Any]:
    """Concatenate decoded chunks deterministically, stripping synthetic headers and bridging ties."""
    if not chunk_results:
        return []

    stitched: list[Any] = []
    n_total_chunks = len(chunk_results)

    for chunk_idx, (plan, symbols, strip_xs, _header_w) in enumerate(chunk_results):
        n_meas = max(1, plan.measure_end - plan.measure_start)
        if chunk_idx == 0:
            kept = _trim_chunk_to_measure_count(symbols, n_meas) if n_total_chunks > 1 else list(symbols)
            stitched.extend(kept)
            continue

        # For Chunks 1..K-1, drop synthetic header tokens and optional duplicate boundary barline
        n_syms = len(symbols)
        first_body_idx: Optional[int] = None
        for idx, sym in enumerate(symbols):
            rhythm = str(getattr(sym, "rhythm", ""))
            sx = strip_xs[idx] if idx < len(strip_xs) else None
            in_header_zone = sx is not None and not math.isnan(sx) and sx <= float(plan.x_start) + 0.35 * unit
            is_header_meta = rhythm.startswith(("clef_", "keySignature_")) or (
                rhythm.startswith(("timeSignature", "dynamic_")) and in_header_zone
            )
            if is_header_meta:
                continue
            if _is_barline_token(rhythm) and first_body_idx is None:
                # Duplicate boundary barline after synthetic header -> merge style into closing barline of prev chunk
                if _is_special_barline(rhythm):
                    for prev in reversed(stitched):
                        if _is_barline_token(str(getattr(prev, "rhythm", ""))):
                            prev.rhythm = rhythm
                            break
                continue
            first_body_idx = idx
            break

        if first_body_idx is None:
            continue

        tail: list[Any] = []
        for idx in range(first_body_idx, n_syms):
            sym = symbols[idx]
            sx = strip_xs[idx] if idx < len(strip_xs) else None
            # Guard against stray header tokens appearing right before real notes without coordinates
            rhythm = str(getattr(sym, "rhythm", ""))
            if (
                not tail
                and sx is not None
                and not math.isnan(sx)
                and sx < float(plan.x_start) - 0.5 * unit
                and rhythm.startswith(("clef_", "keySignature_"))
            ):
                continue
            tail.append(sym)

        kept_tail = _trim_chunk_to_measure_count(tail, n_meas) if chunk_idx < n_total_chunks - 1 else tail

        if stitched and not _is_barline_token(str(getattr(stitched[-1], "rhythm", ""))):
            stitched.append(type(stitched[-1])("barline"))
            log.debug(
                "Inserted missing boundary barline before chunk at measure %d",
                plan.measure_start,
            )

        if plan.crosses_tie_left and stitched and kept_tail:
            prev_note = next(
                (s for s in reversed(stitched) if str(getattr(s, "rhythm", "")).startswith("note_")),
                None,
            )
            next_note = next(
                (s for s in kept_tail if str(getattr(s, "rhythm", "")).startswith("note_")),
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
                    "Bridged cross-chunk tie at boundary measure %d (pitch=%s%s)",
                    plan.measure_start,
                    getattr(prev_note, "pitch", "?"),
                    getattr(prev_note, "lift", ""),
                )

        stitched.extend(kept_tail)

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

        strip_barlines_x = [int(round(_page_to_strip((float(bx), cy_page))[0])) for bx in barlines_x]
        # Clamp first/last gently if slightly beyond edges while enforcing strictly increasing interior
        strip_w = int(strip.shape[1])
        strip_barlines_x[0] = max(0, min(strip_w - 2, strip_barlines_x[0]))
        strip_barlines_x[-1] = max(strip_barlines_x[0] + 2, min(strip_w - 1, strip_barlines_x[-1]))
        if any(
            strip_barlines_x[k + 1] <= strip_barlines_x[k] + max(10, int(round(3.0 * unit)))
            for k in range(len(strip_barlines_x) - 1)
        ):
            log.warning("Staff %02d non-monotonic projected barlines %s — falling back", index, strip_barlines_x)
            return _fallback_parse_staff(
                orig_parse_staff_image, debug, index, staff, image, regions, config, page_to_input_image
            )

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
            if 0 < k < len(strip_barlines_x) - 1
            else False
            for k, bx in enumerate(strip_barlines_x)
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
        header_img, header_w = extract_clef_key_header(
            strip,
            strip_barlines_x[0],
            strip_barlines_x[1],
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
