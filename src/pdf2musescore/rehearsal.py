"""Rehearsal mark (boxed numbers/letters and unboxed bold letters) detection and MusicXML injection."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

_BOXED_REHEARSAL_RE = re.compile(r"^(?:[A-Z]{1,2}|[1-9]\d{0,2}|[A-Z][1-9]\d?)$")
_UNBOXED_REHEARSAL_RE = re.compile(r"^[A-Z]$")
_UNBOXED_NEXT_BLOCKLIST = frozenset(
    {
        "tempo",
        "cappella",
        "la",
        "flat",
        "sharp",
        "major",
        "minor",
        "dur",
        "moll",
        "sol",
        "solo",
        "tutti",
        "capo",
        "segno",
        "fine",
    }
)
_SECTION_BAR_STYLES = frozenset({"light-light", "heavy-heavy", "light-heavy"})


@dataclass(frozen=True)
class RehearsalMark:
    """Detected rehearsal mark label and enclosure style ('square' or 'none')."""

    label: str
    enclosure: str = "square"
    raw_ocr: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "label": self.label,
            "enclosure": self.enclosure,
            "raw_ocr": self.raw_ocr,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["RehearsalMark"]:
        """Deserialize from a dictionary; return ``None`` if invalid."""
        if not isinstance(raw, dict):
            return None
        label = str(raw.get("label", "") or "").strip()
        if not label:
            return None
        enc = str(raw.get("enclosure", "square") or "square").strip().lower()
        if enc not in ("square", "none"):
            enc = "square"
        return cls(
            label=label,
            enclosure=enc,
            raw_ocr=str(raw.get("raw_ocr", "") or ""),
        )


@dataclass(frozen=True)
class PositionedRehearsal:
    """Rehearsal mark together with full-page bounding coordinates and normalized system X."""

    mark: RehearsalMark
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float
    u_norm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "mark": self.mark.to_dict(),
            "x_left": float(self.x_left),
            "x_right": float(self.x_right),
            "y_top": float(self.y_top),
            "y_bottom": float(self.y_bottom),
            "u_norm": float(self.u_norm),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["PositionedRehearsal"]:
        """Deserialize from a dictionary; return ``None`` if invalid."""
        if not isinstance(raw, dict):
            return None
        mark = RehearsalMark.from_dict(raw.get("mark"))
        if mark is None:
            return None
        try:
            x_left = float(raw.get("x_left", 0.0))
            x_right = float(raw.get("x_right", 0.0))
            y_top = float(raw.get("y_top", 0.0))
            y_bottom = float(raw.get("y_bottom", 0.0))
            u_norm = float(raw.get("u_norm", 0.0))
        except (TypeError, ValueError):
            return None
        return cls(
            mark=mark,
            x_left=x_left,
            x_right=x_right,
            y_top=y_top,
            y_bottom=y_bottom,
            u_norm=max(0.0, min(1.0, u_norm)),
        )


def _clean_boxed_label(raw_txt: str) -> Optional[str]:
    """Validate and normalize OCR text extracted from inside a rehearsal box."""
    cleaned = re.sub(r"[\s\[\]\(\)\.\,\:\;\|'`\"_~]+", "", raw_txt or "")
    if not cleaned:
        return None
    # RapidOCR commonly reads a boxed rehearsal letter 'O' as digit '0'; rehearsal numbers never start at 0
    if cleaned == "0":
        cleaned = "O"
    # Reject multi-character lowercase noise inside boxes (e.g. dynamics or articulation artifacts)
    if any("a" <= ch <= "z" for ch in cleaned):
        return None
    return cleaned if _BOXED_REHEARSAL_RE.match(cleaned) else None


def _run_interior_ocr(
    img_crop: np.ndarray,
    *,
    ocr_reader: Any = None,
    ocr_dir: Optional[Path] = None,
    ocr_runner: Optional[Callable[[np.ndarray], tuple[Sequence[Any], Sequence[str]]]] = None,
) -> tuple[Optional[str], str]:
    """Upscale and pad an interior box crop before running OCR."""
    if img_crop.size == 0:
        return None, ""
    crop = img_crop
    ih, iw = crop.shape[:2]
    if min(ih, iw) < 40:
        scale = 48.0 / float(max(1, min(ih, iw)))
        crop = cv2.resize(
            crop,
            (max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
        ih, iw = crop.shape[:2]
    margin = max(24, int(0.65 * max(ih, iw)))
    if crop.ndim == 2:
        canvas: np.ndarray = np.full((ih + 2 * margin, iw + 2 * margin), 255, dtype=np.uint8)
    else:
        canvas = np.full((ih + 2 * margin, iw + 2 * margin, crop.shape[2]), 255, dtype=np.uint8)
    canvas[margin : margin + ih, margin : margin + iw] = crop

    from .fullscore.instruments import _run_rapidocr_on_crop

    _boxes, txts = _run_rapidocr_on_crop(
        canvas, ocr_reader=ocr_reader, ocr_dir=ocr_dir, ocr_runner=ocr_runner
    )
    if not txts:
        return None, ""
    raw_joined = "".join(str(t).strip() for t in txts if str(t).strip())
    return _clean_boxed_label(raw_joined), raw_joined


def detect_boxed_rehearsal_marks_in_band(
    img: np.ndarray,
    y_top: int,
    y_bot: int,
    *,
    sys_min_x: float = 0.0,
    sys_max_x: float = 0.0,
    ocr_reader: Any = None,
    ocr_dir: Optional[Path] = None,
    ocr_runner: Optional[Callable[[np.ndarray], tuple[Sequence[Any], Sequence[str]]]] = None,
) -> list[PositionedRehearsal]:
    """Detect enclosed square/rectangular rehearsal marks within vertical band ``[y_top, y_bot)``."""
    h_px, w_px = img.shape[:2]
    y0 = max(0, int(y_top))
    y1 = min(h_px, int(y_bot))
    if y1 - y0 < 20 or w_px < 40:
        return []

    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray_band = gray_full[y0:y1, :]
    bw = (gray_band < 190).astype(np.uint8)
    contours, hierarchy = cv2.findContours(bw * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_s = max(20, int(0.011 * w_px))
    max_h = max(48, int(0.055 * w_px))
    max_w = max(72, int(0.095 * w_px))
    has_sys_span = sys_max_x > sys_min_x + 20.0
    x_base = float(sys_min_x) if has_sys_span else 0.0
    span = float(sys_max_x - sys_min_x) if has_sys_span else float(max(1, w_px))

    hier_arr = hierarchy[0] if hierarchy is not None and len(contours) > 0 else []
    ext_items = [
        (cv2.boundingRect(contours[i]), contours[i])
        for i in range(len(contours))
        if len(hier_arr) == 0 or hier_arr[i][3] == -1
    ]

    found: list[PositionedRehearsal] = []
    for (x, y, ww, hh), _cnt in sorted(ext_items, key=lambda item: item[0][:2]):
        if not (min_s <= ww <= max_w and min_s <= hh <= max_h):
            continue
        aspect = ww / float(hh)
        if not (0.6 <= aspect <= 2.5):
            continue

        pad = max(3, int(0.11 * min(ww, hh)))
        top_cov = float((bw[y : y + pad, x : x + ww].max(axis=0) > 0).mean())
        bot_cov = float((bw[y + hh - pad : y + hh, x : x + ww].max(axis=0) > 0).mean())
        lef_cov = float((bw[y : y + hh, x : x + pad].max(axis=1) > 0).mean())
        rig_cov = float((bw[y : y + hh, x + ww - pad : x + ww].max(axis=1) > 0).mean())
        if min(top_cov, bot_cov, lef_cov, rig_cov) < 0.85:
            continue

        inset = max(4, int(0.12 * min(ww, hh)))
        if 2 * inset >= min(ww, hh) - 4:
            continue
        inner_bw = bw[y + inset : y + hh - inset, x + inset : x + ww - inset]
        if inner_bw.size == 0:
            continue
        fill = float(inner_bw.mean())
        if not (0.04 <= fill <= 0.60):
            continue

        in_crop = img[y0 + y + inset : y0 + y + hh - inset, x + inset : x + ww - inset]
        label, raw_ocr = _run_interior_ocr(
            in_crop,
            ocr_reader=ocr_reader,
            ocr_dir=ocr_dir,
            ocr_runner=ocr_runner,
        )
        if label is None:
            continue

        cx = float(x) + 0.5 * float(ww)
        u_norm = max(0.0, min(1.0, (cx - x_base) / span))
        found.append(
            PositionedRehearsal(
                mark=RehearsalMark(label=label, enclosure="square", raw_ocr=raw_ocr),
                x_left=float(x),
                x_right=float(x + ww),
                y_top=float(y0 + y),
                y_bottom=float(y0 + y + hh),
                u_norm=u_norm,
            )
        )

    # Secondary pass: recover boxed marks whose outer edge touches adjacent text (e.g. 'P' touching 'Andante')
    # via their rectangular interior hole contour.
    hole_items = [
        (cv2.boundingRect(contours[i]), contours[i])
        for i in range(len(contours))
        if len(hier_arr) > i and hier_arr[i][3] != -1
    ]
    band_h = bw.shape[0]
    for (hx, hy, hw, hh), cnt in sorted(hole_items, key=lambda item: item[0][:2]):
        if not (int(0.75 * min_s) <= hw <= max_w and int(0.75 * min_s) <= hh <= max_h):
            continue
        aspect = hw / float(hh)
        if not (0.6 <= aspect <= 2.5):
            continue
        if cv2.contourArea(cnt) / float(hw * hh) < 0.80:
            continue

        rim_pad = max(2, int(0.08 * min(hw, hh)))
        rx0 = max(0, hx - rim_pad)
        ry0 = max(0, hy - rim_pad)
        rx1 = min(w_px, hx + hw + rim_pad)
        ry1 = min(band_h, hy + hh + rim_pad)
        if hx <= rx0 or hy <= ry0 or hx + hw >= rx1 or hy + hh >= ry1:
            continue

        top_cov = float((bw[ry0:hy, hx : hx + hw].max(axis=0) > 0).mean())
        bot_cov = float((bw[hy + hh : ry1, hx : hx + hw].max(axis=0) > 0).mean())
        lef_cov = float((bw[hy : hy + hh, rx0:hx].max(axis=1) > 0).mean())
        rig_cov = float((bw[hy : hy + hh, hx + hw : rx1].max(axis=1) > 0).mean())
        if min(top_cov, bot_cov, lef_cov, rig_cov) < 0.85:
            continue

        border_px = max(2, int(0.06 * min(hw, hh)))
        bx0 = float(max(0, hx - border_px))
        by0 = float(y0 + max(0, hy - border_px))
        bx1 = float(min(w_px, hx + hw + border_px))
        by1 = float(y0 + min(band_h, hy + hh + border_px))

        dup = any(
            max(0.0, min(bx1, ex.x_right) - max(bx0, ex.x_left))
            * max(0.0, min(by1, ex.y_bottom) - max(by0, ex.y_top))
            > 0.45 * min((bx1 - bx0) * (by1 - by0), (ex.x_right - ex.x_left) * (ex.y_bottom - ex.y_top))
            for ex in found
        )
        if dup:
            continue

        hole_inset = max(2, int(0.05 * min(hw, hh)))
        if 2 * hole_inset >= min(hw, hh) - 4:
            continue
        inner_bw = bw[hy + hole_inset : hy + hh - hole_inset, hx + hole_inset : hx + hw - hole_inset]
        if inner_bw.size == 0:
            continue
        fill = float(inner_bw.mean())
        if not (0.04 <= fill <= 0.60):
            continue

        in_crop = img[y0 + hy + hole_inset : y0 + hy + hh - hole_inset, hx + hole_inset : hx + hw - hole_inset]
        label, raw_ocr = _run_interior_ocr(
            in_crop,
            ocr_reader=ocr_reader,
            ocr_dir=ocr_dir,
            ocr_runner=ocr_runner,
        )
        if label is None:
            continue

        cx = 0.5 * (bx0 + bx1)
        u_norm = max(0.0, min(1.0, (cx - x_base) / span))
        log.debug("Recovered touching boxed rehearsal mark %r via interior hole (%dx%d)", label, hw, hh)
        found.append(
            PositionedRehearsal(
                mark=RehearsalMark(label=label, enclosure="square", raw_ocr=raw_ocr),
                x_left=bx0,
                x_right=bx1,
                y_top=by0,
                y_bottom=by1,
                u_norm=u_norm,
            )
        )
    return sorted(found, key=lambda item: item.x_left)


def detect_unboxed_rehearsal_marks_in_band(
    img: np.ndarray,
    y_top: int,
    y_bot: int,
    *,
    sys_min_x: float = 0.0,
    sys_max_x: float = 0.0,
    min_y: Optional[int] = None,
    max_y: Optional[int] = None,
    ocr_reader: Any = None,
    ocr_dir: Optional[Path] = None,
    ocr_runner: Optional[Callable[[np.ndarray], tuple[Sequence[Any], Sequence[str]]]] = None,
    precomputed_ocr: Optional[tuple[Sequence[Any], Sequence[str]]] = None,
) -> list[PositionedRehearsal]:
    """Detect tall bold single-letter unboxed rehearsal marks (e.g. ``A``, ``B``) above a system."""
    h_px, w_px = img.shape[:2]
    y0 = max(0, int(y_top))
    y1 = min(h_px, int(y_bot))
    if y1 - y0 < 20 or w_px < 40:
        return []

    if precomputed_ocr is not None:
        boxes_raw, txts_raw = precomputed_ocr
    else:
        from .fullscore.instruments import _run_rapidocr_on_crop

        boxes_raw, txts_raw = _run_rapidocr_on_crop(
            img[y0:y1, :], ocr_reader=ocr_reader, ocr_dir=ocr_dir, ocr_runner=ocr_runner
        )
    if not boxes_raw or not txts_raw:
        return []

    items: list[tuple[float, float, float, float, str]] = []
    for box, txt in zip(boxes_raw, txts_raw):
        clean_t = str(txt).strip()
        if not clean_t:
            continue
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        items.append((min(xs), min(ys), max(xs), max(ys), clean_t))

    has_sys_span = sys_max_x > sys_min_x + 20.0
    x_base = float(sys_min_x) if has_sys_span else 0.0
    span = float(sys_max_x - sys_min_x) if has_sys_span else float(max(1, w_px))
    min_h = max(24.0, 0.013 * float(w_px))
    max_h = max(60.0, 0.045 * float(w_px))

    found: list[PositionedRehearsal] = []
    for idx, (left, top, right, bottom, txt) in enumerate(items):
        if not _UNBOXED_REHEARSAL_RE.match(txt):
            continue
        bh = bottom - top
        bw = right - left
        if bh < min_h or bh > max_h or not (0.40 <= bw / max(1.0, bh) <= 1.6):
            continue
        if min_y is not None and (y0 + bottom) < min_y:
            continue
        if max_y is not None and (y0 + bottom) > max_y:
            continue
        if has_sys_span and left < sys_min_x - 0.05 * span:
            continue

        blocked = False
        for o_idx, (ol, ot, or_, ob, otxt) in enumerate(items):
            if o_idx == idx:
                continue
            y_ov = min(bottom, ob) - max(top, ot)
            if y_ov <= 0.4 * min(bh, max(1.0, ob - ot)):
                continue
            # Split word preceding this letter, or trailing word like "A tempo" / "B flat"
            if 0.0 <= left - or_ <= 1.2 * bh:
                blocked = True
                break
            first_next = otxt.split()[0].lower().rstrip(".,;:") if otxt.split() else ""
            if 0.0 <= ol - right <= 1.2 * bh and first_next in _UNBOXED_NEXT_BLOCKLIST:
                blocked = True
                break
        if blocked:
            continue

        cx = 0.5 * (left + right)
        u_norm = max(0.0, min(1.0, (cx - x_base) / span))
        found.append(
            PositionedRehearsal(
                mark=RehearsalMark(label=txt, enclosure="none", raw_ocr=txt),
                x_left=left,
                x_right=right,
                y_top=float(y0 + top),
                y_bottom=float(y0 + bottom),
                u_norm=u_norm,
            )
        )
    return sorted(found, key=lambda item: item.x_left)


def extract_rehearsal_marks_in_band(
    img: np.ndarray,
    y_top: int,
    y_bot: int,
    *,
    sys_min_x: float = 0.0,
    sys_max_x: float = 0.0,
    unboxed_min_y: Optional[int] = None,
    unboxed_max_y: Optional[int] = None,
    ocr_reader: Any = None,
    ocr_dir: Optional[Path] = None,
    box_ocr_runner: Optional[Callable[[np.ndarray], tuple[Sequence[Any], Sequence[str]]]] = None,
    band_ocr_runner: Optional[Callable[[np.ndarray], tuple[Sequence[Any], Sequence[str]]]] = None,
    precomputed_band_ocr: Optional[tuple[Sequence[Any], Sequence[str]]] = None,
    detect_unboxed: bool = True,
    source_name: str = "",
) -> list[PositionedRehearsal]:
    """Detect boxed and optional unboxed rehearsal marks in ``[y_top, y_bot)`` sorted left-to-right."""
    boxed = detect_boxed_rehearsal_marks_in_band(
        img,
        y_top,
        y_bot,
        sys_min_x=sys_min_x,
        sys_max_x=sys_max_x,
        ocr_reader=ocr_reader,
        ocr_dir=ocr_dir,
        ocr_runner=box_ocr_runner,
    )
    combined = list(boxed)
    if detect_unboxed:
        unboxed = detect_unboxed_rehearsal_marks_in_band(
            img,
            y_top,
            y_bot,
            sys_min_x=sys_min_x,
            sys_max_x=sys_max_x,
            min_y=unboxed_min_y,
            max_y=unboxed_max_y,
            ocr_reader=ocr_reader,
            ocr_dir=ocr_dir,
            ocr_runner=band_ocr_runner,
            precomputed_ocr=precomputed_band_ocr,
        )
        for ub in unboxed:
            overlaps_boxed = any(
                min(ub.x_right, bx.x_right) - max(ub.x_left, bx.x_left) > 4.0
                and min(ub.y_bottom, bx.y_bottom) - max(ub.y_top, bx.y_top) > 4.0
                for bx in boxed
            )
            if not overlaps_boxed:
                combined.append(ub)

    result = sorted(combined, key=lambda item: item.x_left)
    for pr in result:
        log.info(
            "Rehearsal mark OCR on %s (y=%d..%d) detected %r (enclosure=%s, u=%.2f)",
            source_name or "image",
            int(y_top),
            int(y_bot),
            pr.mark.label,
            pr.mark.enclosure,
            pr.u_norm,
        )
    return result


def assign_boxed_rehearsal_marks_to_systems(
    img: np.ndarray,
    sys_boxes: Sequence[tuple[float, float, float, float]],
    *,
    avg_h: float,
    ocr_reader: Any = None,
    ocr_dir: Optional[Path] = None,
    ocr_runner: Optional[Callable[[np.ndarray], tuple[Sequence[Any], Sequence[str]]]] = None,
    source_name: str = "",
) -> list[list[PositionedRehearsal]]:
    """Detect boxed rehearsal marks across the full page and assign each to the first system below it."""
    per_sys: list[list[PositionedRehearsal]] = [[] for _ in sys_boxes]
    if not sys_boxes:
        return per_sys
    h_px, w_px = img.shape[:2]
    all_boxed = detect_boxed_rehearsal_marks_in_band(
        img,
        0,
        h_px,
        ocr_reader=ocr_reader,
        ocr_dir=ocr_dir,
        ocr_runner=ocr_runner,
    )
    max_dist = max(180.0, float(avg_h) * 3.5)
    for pr in all_boxed:
        target_idx: Optional[int] = None
        for s_idx, (sx0, sx1, sy0, _sy1) in enumerate(sys_boxes):
            if sy0 >= pr.y_bottom - 15.0:
                if sy0 - pr.y_bottom <= max_dist:
                    target_idx = s_idx
                break
        if target_idx is None:
            continue
        sx0, sx1, _sy0, _sy1 = sys_boxes[target_idx]
        span = max(1.0, float(sx1 - sx0)) if sx1 > sx0 + 20.0 else float(max(1, w_px))
        x_base = float(sx0) if sx1 > sx0 + 20.0 else 0.0
        cx = 0.5 * (pr.x_left + pr.x_right)
        u_norm = max(0.0, min(1.0, (cx - x_base) / span))
        assigned = PositionedRehearsal(
            mark=pr.mark,
            x_left=pr.x_left,
            x_right=pr.x_right,
            y_top=pr.y_top,
            y_bottom=pr.y_bottom,
            u_norm=u_norm,
        )
        per_sys[target_idx].append(assigned)
        log.info(
            "Boxed rehearsal mark on %s sys=%d detected %r (y=%.0f..%.0f, u=%.2f)",
            source_name or "image",
            target_idx,
            assigned.mark.label,
            assigned.y_top,
            assigned.y_bottom,
            assigned.u_norm,
        )
    return per_sys


def apply_rehearsal_mark(
    root: ET.Element,
    mark: Optional[RehearsalMark],
    *,
    measure_index: int = 0,
) -> bool:
    """Inject a MusicXML 4.0 ``<rehearsal>`` direction into the first part at ``measure_index``."""
    if mark is None or not mark.label.strip():
        return False
    first_part = root.find("part")
    if first_part is None:
        return False
    measures = first_part.findall("measure")
    if measure_index < 0 or measure_index >= len(measures):
        return False
    measure = measures[measure_index]

    if measure.find("./direction/direction-type/rehearsal") is not None:
        return False

    direction = ET.Element("direction", {"placement": "above"})
    dt = ET.SubElement(direction, "direction-type")
    reh = ET.SubElement(
        dt,
        "rehearsal",
        {"enclosure": mark.enclosure, "font-weight": "bold"},
    )
    reh.text = mark.label.strip()

    insert_pos = 0
    for idx, child in enumerate(list(measure)):
        if child.tag in ("print", "attributes"):
            insert_pos = idx + 1
        elif child.tag in ("direction", "note"):
            break
    measure.insert(insert_pos, direction)
    log.info(
        "Injected rehearsal mark %r (enclosure=%s) into %s measure %s",
        mark.label,
        mark.enclosure,
        first_part.get("id", "P1"),
        measure.get("number", str(measure_index + 1)),
    )
    return True


def _structural_boundaries(measures: Sequence[ET.Element], indices: Sequence[int]) -> list[int]:
    boundaries: list[int] = []
    for j in range(1, len(indices)):
        g = indices[j]
        cur_m = measures[g]
        prev_m = measures[g - 1]
        has_attr_change = cur_m.find("./attributes/key") is not None or cur_m.find("./attributes/time") is not None
        has_bar = any(
            bl.find("repeat") is not None
            or (bl.findtext("bar-style") or "").strip().lower() in _SECTION_BAR_STYLES
            for bl in prev_m.findall("barline")
        )
        if has_attr_change or has_bar:
            boundaries.append(j)
    return boundaries


def map_system_rehearsal_marks_to_measures(
    root: ET.Element,
    system_marks: Sequence[Sequence[PositionedRehearsal]],
    *,
    system_barlines: Optional[Sequence[Sequence[int]]] = None,
    system_x_bounds: Optional[Sequence[tuple[float, float]]] = None,
) -> int:
    """Map per-system rehearsal marks onto MusicXML measures using barlines or structural snapping."""
    first_part = root.find("part")
    if first_part is None:
        return 0
    measures = first_part.findall("measure")
    if not measures:
        return 0

    systems: list[list[int]] = [[0]]
    for mi in range(1, len(measures)):
        pr = measures[mi].find("print")
        if pr is not None and pr.get("new-system") == "yes":
            systems.append([mi])
        else:
            systems[-1].append(mi)

    injected = 0
    for sys_idx, marks in enumerate(system_marks):
        if sys_idx >= len(systems) or not marks:
            continue
        sys_indices = systems[sys_idx]
        k = len(sys_indices)
        bars = list(system_barlines[sys_idx]) if (system_barlines is not None and sys_idx < len(system_barlines)) else []
        x_bounds = (
            system_x_bounds[sys_idx]
            if (system_x_bounds is not None and sys_idx < len(system_x_bounds))
            else None
        )
        has_exact_bars = bool(bars and len(bars) == k and x_bounds is not None)
        start_edges = [float(x_bounds[0])] + [float(b) for b in bars[:-1]] if has_exact_bars else []
        boundaries = _structural_boundaries(measures, sys_indices)

        for pr in sorted(marks, key=lambda item: (0 if item.mark.enclosure == "square" else 1, item.x_left)):
            if has_exact_bars:
                cx = 0.5 * (pr.x_left + pr.x_right)
                local_idx = min(range(k), key=lambda j: abs(cx - start_edges[j]))
            elif pr.u_norm < 0.08 or k == 1:
                local_idx = 0
            else:
                est_float = pr.u_norm * k
                near = [
                    j
                    for j in boundaries
                    if abs(j - est_float) <= 1.25
                    and measures[sys_indices[j]].find("./direction/direction-type/rehearsal") is None
                ]
                if near:
                    local_idx = min(near, key=lambda j: abs(j - est_float))
                else:
                    local_idx = min(k - 1, max(0, int(round(est_float))))
            if apply_rehearsal_mark(root, pr.mark, measure_index=sys_indices[local_idx]):
                injected += 1
    return injected


def map_range_rehearsal_marks_to_measures(
    root: ET.Element,
    marks: Sequence[PositionedRehearsal],
    *,
    start_index: int,
    measure_count: int,
) -> int:
    """Map full-score page rehearsal marks across contiguous measures ``[start_index .. start_index + measure_count)``."""
    if not marks or measure_count <= 0:
        return 0
    first_part = root.find("part")
    if first_part is None:
        return 0
    measures = first_part.findall("measure")
    if start_index < 0 or start_index >= len(measures):
        return 0
    end_index = min(len(measures), start_index + measure_count)
    indices = list(range(start_index, end_index))
    k = len(indices)
    boundaries = _structural_boundaries(measures, indices)

    injected = 0
    for pr in marks:
        if pr.u_norm < 0.12 or k == 1:
            local_idx = 0
        else:
            est_float = pr.u_norm * k
            near = [
                j
                for j in boundaries
                if abs(j - est_float) <= 1.25
                and measures[indices[j]].find("./direction/direction-type/rehearsal") is None
            ]
            if near:
                local_idx = min(near, key=lambda j: abs(j - est_float))
            else:
                local_idx = min(k - 1, max(0, int(round(est_float))))
        if apply_rehearsal_mark(root, pr.mark, measure_index=indices[local_idx]):
            injected += 1
    return injected
