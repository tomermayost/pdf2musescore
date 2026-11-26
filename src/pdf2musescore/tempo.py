"""OCR tempo marking detection, parsing, and MusicXML <direction> injection."""

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

_DEFAULT_TEMPO_BPM: dict[str, int] = {
    "tempo di menuetto": 112,
    "tempo di valse": 120,
    "alla marcia": 116,
    "prestissimo": 200,
    "larghetto": 60,
    "adagietto": 74,
    "andantino": 92,
    "allegretto": 116,
    "moderately": 104,
    "moderato": 108,
    "maestoso": 88,
    "brightly": 132,
    "quickly": 144,
    "allegro": 132,
    "andante": 84,
    "adagio": 68,
    "vivace": 152,
    "presto": 176,
    "slowly": 66,
    "lively": 136,
    "freely": 84,
    "grave": 40,
    "largo": 48,
    "lento": 52,
    "vivo": 156,
    "fast": 140,
}

_UNIT_FACTORS: dict[str, float] = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
}

_TEMPO_HEADWORDS_PAT = (
    r"tempo\s+di\s+(?:menuetto|minuetto|valse|waltz|marcia|march|\w+)|"
    r"(?:(?:poco|molto|sempre)\s+)?(?:meno|pi[uù])\s+mosso|"
    r"alla\s+marcia|a\s+tempo|"
    r"prestissimo|larghetto|adagietto|andantino|allegretto|moderately|"
    r"moderato|maestoso|brightly|quickly|allegro|andante|adagio|"
    r"vivace|presto|slowly|lively|freely|grave|largo|lento|vivo|fast"
)

_TEMPO_HEADWORD_RE = re.compile(rf"\b(?:{_TEMPO_HEADWORDS_PAT})\b", re.IGNORECASE)
_TEMPO_START_RE = re.compile(rf"^(?:{_TEMPO_HEADWORDS_PAT})\b", re.IGNORECASE)

_BOILERPLATE_RE = re.compile(
    r"\b(?:copyright|full\s+score|commissioned|all\s+rights\s+reserved|printed\s+in|"
    r"edition|page\s+\d+|op\.|bwv|concert\s+band|orchestra)\b",
    re.IGNORECASE,
)

_METRO_EQ_RE = re.compile(
    r"(?:(?:(?<![A-Za-z])(?P<unit>♩|𝅗𝅥|♪|𝅗𝅥|q|h|e|d|j)(?P<dot>\.)?|(?:m\.?m\.?|ca\.?|c\.))\s*)?"
    r"[=≈]\s*[,.'`~]*\s*(?P<bpm>[0-9sSBbOoDlI|Zz]{2,3})(?:\b|[)\]])",
    re.IGNORECASE,
)
_METRO_BPM_RE = re.compile(
    r"(?:\b(?:bpm\s*[:=]?\s*(?P<bpm1>\d{2,3})|(?P<bpm2>\d{2,3})\s*bpm)\b|"
    r"\bm\.?m\.?\s*[:=]?\s*(?P<bpm3>\d{2,3})\b)",
    re.IGNORECASE,
)
_BPM_OCR_DIGIT_MAP = str.maketrans(
    {"s": "8", "S": "8", "B": "8", "b": "6", "O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1", "Z": "2", "z": "2"}
)
_PAREN_IN_ONE_RE = re.compile(r"\(\s*in\s+[1-4tlI|]\s*\)", re.IGNORECASE)
_PAREN_METRONOME_RE = re.compile(r"\([^)]*[=≈][^)]*\)")
_SECTION_BAR_STYLES = frozenset({"light-light", "heavy-heavy", "light-heavy"})

_EXPR_MODIFIER_PAT = r"(?:poco\s+a\s+poco|poco|molto|sempre|pi[uù]|meno)"
_EXPR_TERMS_PAT = (
    r"tranguillo|tranquillo|trang\.{1,3}|tranq\.{1,3}|"
    r"ritardando|ritard\.{1,3}|rit\.{1,3}|"
    r"rallentando|rall\.{1,3}|"
    r"accelerando|accel\.{1,3}|"
    r"a\s+tempo|"
    r"espressivo|espress\.{1,3}|"
    r"cantabile|dolce|stringendo|"
    r"allargando|allarg\.{1,3}|"
    r"calando|morendo|rubato|subito"
)
_EXPRESSION_RE = re.compile(
    rf"(?P<full>(?:(?P<mod>{_EXPR_MODIFIER_PAT})\s+)?(?P<term>{_EXPR_TERMS_PAT}))(?!\w)",
    re.IGNORECASE,
)
_EXPR_CANONICAL_TERMS: dict[str, str] = {
    "trang": "tranq.",
    "trang.": "tranq.",
    "trang..": "tranq.",
    "trang...": "tranq.",
    "tranq": "tranq.",
    "tranq.": "tranq.",
    "tranq..": "tranq.",
    "tranq...": "tranq.",
    "tranguillo": "tranquillo",
    "tranquillo": "tranquillo",
    "rit": "rit.",
    "rit.": "rit.",
    "rit..": "rit.",
    "rit...": "rit.",
    "ritard": "ritard.",
    "ritard.": "ritard.",
    "ritard..": "ritard.",
    "ritard...": "ritard.",
    "rall": "rall.",
    "rall.": "rall.",
    "rall..": "rall.",
    "rall...": "rall.",
    "accel": "accel.",
    "accel.": "accel.",
    "accel..": "accel.",
    "accel...": "accel.",
    "espress": "espress.",
    "espress.": "espress.",
    "espress..": "espress.",
    "espress...": "espress.",
    "allarg": "allarg.",
    "allarg.": "allarg.",
    "allarg..": "allarg.",
    "allarg...": "allarg.",
}
_XML_SPACE_ATTR = "{http://www.w3.org/XML/1998/namespace}space"

_UNIT_MAP: dict[str, str] = {
    "h": "half",
    "𝅗𝅥": "half",
    "𝅗𝅥": "half",
    "e": "eighth",
    "♪": "eighth",
}


def _repair_bpm_token(tok: str) -> Optional[int]:
    """Repair common single-character OCR digit misreads (e.g. ``'s4'`` -> ``84``) and validate range."""
    if not tok or not any(ch.isdigit() for ch in tok):
        return None
    repaired = tok.translate(_BPM_OCR_DIGIT_MAP)
    if not repaired.isdigit():
        return None
    val = int(repaired)
    return val if 24 <= val <= 260 else None


@dataclass(frozen=True)
class TempoMarking:
    """Parsed tempo or above-staff expression indication (text and/or metronome BPM)."""

    text: str = ""
    bpm: Optional[int] = None
    beat_unit: str = "quarter"
    beat_unit_dot: bool = False
    raw_ocr: str = ""
    is_expression: bool = False

    def quarter_bpm(self) -> Optional[int]:
        """Return effective quarter-note BPM for MusicXML ``<sound tempo='...'>``."""
        if self.is_expression and self.bpm is None:
            return None
        if self.bpm is not None:
            factor = _UNIT_FACTORS.get(self.beat_unit, 1.0)
            if self.beat_unit_dot:
                factor *= 1.5
            val = int(round(self.bpm * factor))
            return max(20, min(300, val))
        if self.text:
            low = self.text.lower()
            m_first = _TEMPO_HEADWORD_RE.search(low)
            if m_first is not None:
                kw = re.sub(r"\s+", " ", m_first.group(0)).replace("più", "piu")
                if kw in _DEFAULT_TEMPO_BPM:
                    return _DEFAULT_TEMPO_BPM[kw]
            for kw, default_bpm in _DEFAULT_TEMPO_BPM.items():
                if re.search(rf"\b{re.escape(kw)}\b", low):
                    return default_bpm
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "text": self.text,
            "bpm": self.bpm,
            "beat_unit": self.beat_unit,
            "beat_unit_dot": self.beat_unit_dot,
            "raw_ocr": self.raw_ocr,
            "is_expression": self.is_expression,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["TempoMarking"]:
        """Deserialize from a dictionary; return ``None`` if empty or invalid."""
        if not isinstance(raw, dict):
            return None
        text = str(raw.get("text", "") or "").strip()
        bpm_raw = raw.get("bpm")
        bpm: Optional[int] = None
        if bpm_raw is not None:
            try:
                bpm = int(bpm_raw)
            except (TypeError, ValueError):
                bpm = None
        if not text and bpm is None:
            return None
        unit = str(raw.get("beat_unit", "quarter") or "quarter").strip().lower()
        if unit not in _UNIT_FACTORS:
            unit = "quarter"
        return cls(
            text=text,
            bpm=bpm,
            beat_unit=unit,
            beat_unit_dot=bool(raw.get("beat_unit_dot", False)),
            raw_ocr=str(raw.get("raw_ocr", "") or ""),
            is_expression=bool(raw.get("is_expression", False)),
        )


def _clean_tempo_prefix(prefix: str, *, has_bpm: bool = False) -> str:
    """Strip OCR notehead artifacts (e.g. ``'•'``, ``'₀'``) and glued rehearsal letters from tempo text."""
    txt = (prefix or "").strip(" \t\r\n([,:;-–—•·°◦○●₀♩♪𝅗𝅥𝅗𝅥♦▪■□")
    txt = re.sub(r"^\d{1,2}\s+(?=[A-Za-z])", "", txt).strip()
    txt = re.sub(r"(?:^|\s+)[^A-Za-z0-9()]+(?=\s|$)", " ", txt).strip()
    txt = re.sub(r"\s+", " ", txt).strip(" \t\r\n([,:;-–—•·°◦○●₀♩♪𝅗𝅥𝅗𝅥♦▪■□")
    if not re.search(r"[A-Za-z]{2,}", txt):
        return ""
    m_reh = re.match(r"^([A-Z])\s+(.+)$", txt)
    if m_reh is not None:
        letter, rest = m_reh.group(1), m_reh.group(2).strip()
        if not (letter == "A" and rest.lower().startswith("tempo")):
            if _TEMPO_START_RE.match(rest) or (has_bpm and letter != "A" and rest[0].isupper()):
                txt = rest
    return txt


def parse_tempo_text(raw: str) -> Optional[TempoMarking]:
    """Parse a single OCR line into a ``TempoMarking``, or return ``None`` for non-tempo text."""
    cleaned = re.sub(r"\s+", " ", (raw or "")).strip()
    if not cleaned or len(cleaned) > 60:
        return None
    if _BOILERPLATE_RE.search(cleaned):
        return None
    cleaned = re.sub(r"\(\s*in\s+[tlI|]\s*\)", "(in 1)", cleaned, flags=re.IGNORECASE)

    bpm: Optional[int] = None
    beat_unit = "quarter"
    beat_unit_dot = False
    match_start: Optional[int] = None

    m_eq = _METRO_EQ_RE.search(cleaned)
    if m_eq is not None:
        cand = _repair_bpm_token(m_eq.group("bpm"))
        if cand is not None:
            bpm = cand
            match_start = m_eq.start()
            unit_tok = (m_eq.group("unit") or "").lower()
            beat_unit = _UNIT_MAP.get(unit_tok, "quarter")
            beat_unit_dot = bool(unit_tok and m_eq.group("dot"))
    else:
        m_bpm = _METRO_BPM_RE.search(cleaned)
        if m_bpm is not None:
            cand_txt = m_bpm.group("bpm1") or m_bpm.group("bpm2") or m_bpm.group("bpm3")
            cand = int(cand_txt)
            if 24 <= cand <= 260:
                bpm = cand
                match_start = m_bpm.start()

    if bpm is not None:
        prefix = _clean_tempo_prefix(cleaned[:match_start], has_bpm=True)
        words = prefix.split()
        if prefix and (_TEMPO_HEADWORD_RE.search(prefix) or (len(words) <= 4 and not re.search(r"\d", prefix))):
            text = prefix
        else:
            text = ""
        return TempoMarking(
            text=text,
            bpm=bpm,
            beat_unit=beat_unit,
            beat_unit_dot=beat_unit_dot,
            raw_ocr=cleaned,
        )

    # Word-only tempo marking (e.g. "Allegro ma non troppo", "Allegro (in 1) alla Spagnola")
    stripped_paren = _PAREN_METRONOME_RE.sub("", cleaned)
    text_only = stripped_paren.strip(" \t\r\n.,:;-–—")
    text_only = text_only.lstrip("(").rstrip(")") if text_only.count("(") != text_only.count(")") else text_only
    text_only = _clean_tempo_prefix(text_only, has_bpm=False)
    check_str = _PAREN_IN_ONE_RE.sub("", text_only)
    if not text_only or re.search(r"\d|[=≈]", check_str):
        return None
    if len(text_only.split()) > 6:
        return None
    if not _TEMPO_START_RE.match(text_only):
        return None
    return TempoMarking(text=text_only, bpm=None, raw_ocr=cleaned)


def parse_expression_tokens(raw: str) -> list[tuple[str, int, int]]:
    """Parse an OCR line containing expression/tempo-modifier markings into ``(text, start, end)`` spans.

    Splits combined OCR boxes such as ``'trang. rit.'`` -> ``[('tranq.', 0, 6), ('rit.', 7, 11)]``.
    Only returns matches when every alphanumeric word in the line is accounted for.
    """
    cleaned = re.sub(r"\s+", " ", (raw or "")).strip()
    if not cleaned or len(cleaned) > 60 or _BOILERPLATE_RE.search(cleaned):
        return []

    spans: list[tuple[str, int, int]] = []
    consumed = bytearray(len(cleaned))
    for match in _EXPRESSION_RE.finditer(cleaned):
        start, end = match.span("full")
        if start > 0 and cleaned[start - 1].isalnum():
            continue
        for pos in range(start, end):
            consumed[pos] = 1
        mod = match.group("mod")
        term = match.group("term")
        term_norm = _EXPR_CANONICAL_TERMS.get(term.lower(), term.lower())
        full_norm = f"{re.sub(r'\\s+', ' ', mod.lower()).strip()} {term_norm}" if mod else term_norm
        spans.append((full_norm, start, end))

    if not spans:
        return []
    leftover = "".join(ch for idx, ch in enumerate(cleaned) if not consumed[idx])
    if re.search(r"[A-Za-z0-9]", leftover):
        return []
    return spans


def _merge_horizontal_ocr_boxes(
    boxes: Sequence[Any],
    txts: Sequence[str],
) -> list[tuple[float, float, float, float, str]]:
    """Merge horizontally adjacent OCR boxes on the same line (e.g. ``'Allegro'`` + ``'= 120'``)."""
    items: list[tuple[float, float, float, float, str]] = []
    for box, text in zip(boxes, txts):
        clean_t = str(text).strip()
        if not clean_t:
            continue
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        items.append((min(xs), min(ys), max(xs), max(ys), clean_t))

    merged: list[list[Any]] = []
    for left, top, right, bottom, text in sorted(items, key=lambda it: (it[0], it[1])):
        cur_h = max(1.0, bottom - top)
        is_metro = bool(re.match(r"^(?:[=≈(♩♪𝅗𝅥𝅗₀•·°◦○●]|[0-9]?\s*[dqhe]\s*[=≈]|m\.?m\.?|bpm\b)", text, re.IGNORECASE))
        max_gap_mult = 2.4 if is_metro else 1.2
        min_ov_mult = 0.35 if is_metro else 0.45
        for prev in merged:
            prev_h = max(1.0, prev[3] - prev[1])
            y_overlap = min(prev[3], bottom) - max(prev[1], top)
            x_gap = left - prev[2]
            min_gap = (-0.8 * min(prev_h, cur_h)) if is_metro else (-0.3 * min(prev_h, cur_h))
            if (
                y_overlap > min_ov_mult * min(prev_h, cur_h)
                and min_gap <= x_gap <= max_gap_mult * max(prev_h, cur_h)
                and (not is_metro or left >= prev[0] + 0.3 * (prev[2] - prev[0]))
            ):
                prev[0] = min(prev[0], left)
                prev[1] = min(prev[1], top)
                prev[2] = max(prev[2], right)
                prev[3] = max(prev[3], bottom)
                prev[4] = f"{prev[4]} {text}"
                break
        else:
            merged.append([left, top, right, bottom, text])

    return [(float(m[0]), float(m[1]), float(m[2]), float(m[3]), str(m[4])) for m in merged]


@dataclass(frozen=True)
class PositionedTempo:
    """A tempo marking with its horizontal bounding box and normalized system X coordinate."""

    tempo: TempoMarking
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float
    u_norm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "tempo": self.tempo.to_dict(),
            "x_left": float(self.x_left),
            "x_right": float(self.x_right),
            "y_top": float(self.y_top),
            "y_bottom": float(self.y_bottom),
            "u_norm": float(self.u_norm),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["PositionedTempo"]:
        """Deserialize from a dictionary; return ``None`` if invalid."""
        if not isinstance(raw, dict):
            return None
        tempo = TempoMarking.from_dict(raw.get("tempo"))
        if tempo is None:
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
            tempo=tempo,
            x_left=x_left,
            x_right=x_right,
            y_top=y_top,
            y_bottom=y_bottom,
            u_norm=max(0.0, min(1.0, u_norm)),
        )


def _overlaps_excluded_box(
    box: Sequence[Any],
    y0: int,
    exclude_boxes: Sequence[tuple[float, float, float, float]],
) -> bool:
    if not exclude_boxes:
        return False
    xs = [float(pt[0]) for pt in box]
    ys = [float(pt[1]) + y0 for pt in box]
    bx0, bx1 = min(xs), max(xs)
    by0, by1 = min(ys), max(ys)
    cx = 0.5 * (bx0 + bx1)
    cy = 0.5 * (by0 + by1)
    b_area = max(1.0, (bx1 - bx0) * (by1 - by0))
    for ex0, ey0, ex1, ey1 in exclude_boxes:
        if (ex0 - 4.0) <= cx <= (ex1 + 4.0) and (ey0 - 4.0) <= cy <= (ey1 + 4.0):
            return True
        inter_w = max(0.0, min(bx1, ex1) - max(bx0, ex0))
        inter_h = max(0.0, min(by1, ey1) - max(by0, ey0))
        if (inter_w * inter_h) >= 0.50 * b_area:
            return True
    return False


def extract_tempo_markings_in_band(
    img: np.ndarray,
    y_top: int,
    y_bot: int,
    *,
    sys_min_x: float = 0.0,
    sys_max_x: float = 0.0,
    ocr_reader: Any = None,
    ocr_dir: Optional[Path] = None,
    ocr_runner: Optional[Callable[[np.ndarray], tuple[Sequence[Any], Sequence[str]]]] = None,
    source_name: str = "",
    include_expressions: bool = True,
    expr_min_y: Optional[int] = None,
    precomputed_ocr: Optional[tuple[Sequence[Any], Sequence[str]]] = None,
    exclude_boxes: Sequence[tuple[float, float, float, float]] = (),
) -> list[PositionedTempo]:
    """Extract all tempo and expression markings from band ``[y_top, y_bot)`` sorted left-to-right."""
    h_px, w_px = img.shape[:2]
    y0 = max(0, int(y_top))
    y1 = min(h_px, int(y_bot))
    if y1 - y0 < 15:
        return []

    if precomputed_ocr is not None:
        boxes_raw, txts_raw = precomputed_ocr
    else:
        from .fullscore.instruments import _run_rapidocr_on_crop

        crop = img[y0:y1, :]
        boxes_raw, txts_raw = _run_rapidocr_on_crop(
            crop, ocr_reader=ocr_reader, ocr_dir=ocr_dir, ocr_runner=ocr_runner
        )
    if not boxes_raw or not txts_raw:
        return []

    if exclude_boxes:
        filtered = [
            (b, t)
            for b, t in zip(boxes_raw, txts_raw)
            if not _overlaps_excluded_box(b, y0, exclude_boxes)
        ]
        boxes_raw = [b for b, _ in filtered]
        txts_raw = [t for _, t in filtered]
        if not boxes_raw:
            return []

    merged_boxes = _merge_horizontal_ocr_boxes(boxes_raw, txts_raw)
    has_sys_span = sys_max_x > sys_min_x + 20.0
    x_base = float(sys_min_x) if has_sys_span else 0.0
    span = float(sys_max_x - sys_min_x) if has_sys_span else float(max(1, w_px))

    raw_cands: list[tuple[tuple[int, int, float], PositionedTempo]] = []
    for left, top, right, bottom, txt in merged_boxes:
        tempo = parse_tempo_text(txt)
        if tempo is not None:
            u_norm = max(0.0, min(1.0, (float(left) - x_base) / span))
            pt = PositionedTempo(
                tempo=tempo,
                x_left=float(left),
                x_right=float(right),
                y_top=float(y0 + top),
                y_bottom=float(y0 + bottom),
                u_norm=u_norm,
            )
            score = (1 if tempo.bpm is not None else 0, 1 if tempo.text else 0, float(bottom))
            raw_cands.append((score, pt))
        elif include_expressions and (expr_min_y is None or (y0 + bottom) >= expr_min_y):
            expr_spans = parse_expression_tokens(txt)
            if not expr_spans:
                continue
            cleaned_len = max(1, len(re.sub(r"\s+", " ", txt).strip()))
            box_w = max(1.0, float(right - left))
            for expr_txt, c_start, c_end in expr_spans:
                sub_l = float(left) + box_w * (float(c_start) / cleaned_len)
                sub_r = float(left) + box_w * (float(c_end) / cleaned_len)
                sub_cx = 0.5 * (sub_l + sub_r)
                u_norm = max(0.0, min(1.0, (sub_cx - x_base) / span))
                pt = PositionedTempo(
                    tempo=TempoMarking(text=expr_txt, bpm=None, raw_ocr=txt, is_expression=True),
                    x_left=sub_l,
                    x_right=sub_r,
                    y_top=float(y0 + top),
                    y_bottom=float(y0 + bottom),
                    u_norm=u_norm,
                )
                raw_cands.append(((0, 1, float(bottom)), pt))

    if not raw_cands:
        return []

    # Deduplicate horizontally overlapping detections per category (main tempo vs expression)
    kept: list[tuple[tuple[int, int, float], PositionedTempo]] = []
    for score, pt in sorted(raw_cands, key=lambda item: item[0], reverse=True):
        overlaps = any(
            ex.tempo.is_expression == pt.tempo.is_expression
            and min(pt.x_right, ex.x_right) - max(pt.x_left, ex.x_left) > 10.0
            for _, ex in kept
        )
        if not overlaps:
            kept.append((score, pt))

    result = sorted((pt for _, pt in kept), key=lambda item: item.x_left)
    for pt in result:
        log.info(
            "%s band OCR on %s (y=%d..%d) detected %r (bpm=%s, u=%.2f)",
            "Expression" if pt.tempo.is_expression else "Tempo",
            source_name or "image",
            y0,
            y1,
            pt.tempo.text or pt.tempo.raw_ocr,
            pt.tempo.bpm,
            pt.u_norm,
        )
    return result


def detect_staff_barlines(
    gray: np.ndarray,
    sy0: int,
    sy1: int,
    sx0: int,
    sx1: int,
) -> list[int]:
    """Detect X-coordinates of interior and closing barlines across a staff band ``[sy0..sy1]``."""
    h_img, w_img = gray.shape[:2]
    y0 = max(0, int(sy0))
    y1 = min(h_img - 1, int(sy1))
    sh = y1 - y0 + 1
    if sh < 12 or sx1 - sx0 < 40:
        return []

    sp = max(4, int(round(sh / 4.0)))
    band = (gray[y0 : y1 + 1, :] < 180).astype(np.uint8)
    band_dil = cv2.dilate(band, np.ones((1, 3), np.uint8))
    col_in = band_dil.mean(axis=0)

    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, sh), 1))
    no_staff = cv2.subtract(band, cv2.morphologyEx(band, cv2.MORPH_OPEN, h_kern))

    min_x = max(0, int(sx0) + int(0.06 * (sx1 - sx0)))
    max_x = min(w_img - 1, int(sx1) + 12)
    cand = [int(c) for c in np.where(col_in >= 0.90)[0] if min_x <= c <= max_x]
    if not cand:
        return []

    clusters: list[list[int]] = []
    gap_limit = max(24, int(0.35 * sh))
    for c in cand:
        if not clusters or c - clusters[-1][-1] > gap_limit:
            clusters.append([c])
        else:
            clusters[-1].append(c)

    win_h = max(6, int(0.75 * sp))
    win_w = max(5, int(0.55 * sp))
    end_margin = max(40, int(0.6 * sh))
    bar_xs: list[int] = []
    for cl in clusters:
        cx = int(round(float(np.mean(cl))))
        c0, c1 = cl[0], cl[-1]
        is_end = (sx1 - cx) < end_margin
        l_strip = no_staff[:, max(0, c0 - win_w - 2) : max(0, c0 - 1)]
        r_strip = no_staff[:, min(w_img, c1 + 2) : min(w_img, c1 + win_w + 3)]
        max_blob = 0.0
        for side in (l_strip, r_strip):
            if side.shape[1] >= 3 and side.shape[0] > win_h:
                csum = np.cumsum(side.mean(axis=1))
                win_means = (csum[win_h:] - csum[:-win_h]) / float(win_h)
                if win_means.size:
                    max_blob = max(max_blob, float(win_means.max()))
        if is_end or max_blob < 0.48:
            bar_xs.append(cx)
    return bar_xs


def _interval_index(x_pos: float, edges: Sequence[float], k: int) -> int:
    for j in range(k):
        if x_pos <= edges[j + 1]:
            return j
    return k - 1


def _find_structural_boundary_near(
    measures: Sequence[ET.Element],
    sys_indices: Sequence[int],
    center_idx: int,
    radius: int,
) -> list[int]:
    k = len(sys_indices)
    boundaries: list[int] = []
    for j in range(max(1, center_idx - radius), min(k, center_idx + radius + 1)):
        g = sys_indices[j]
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


def map_system_tempos_to_measures(
    root: ET.Element,
    system_tempos: Sequence[Sequence[PositionedTempo]],
    *,
    system_barlines: Optional[Sequence[Sequence[int]]] = None,
    system_x_bounds: Optional[Sequence[tuple[float, float]]] = None,
) -> int:
    """Map per-system positioned tempos and expressions onto MusicXML measures."""
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
    for sys_idx, tempos in enumerate(system_tempos):
        if sys_idx >= len(systems) or not tempos:
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
        edges = [float(x_bounds[0])] + [float(b) for b in bars] if has_exact_bars else []

        for pt in tempos:
            if pt.tempo.is_expression:
                if has_exact_bars:
                    cx = 0.5 * (pt.x_left + pt.x_right)
                    local_idx = _interval_index(cx, edges, k)
                else:
                    local_idx = min(k - 1, max(0, int(pt.u_norm * k)))
            elif pt.u_norm < 0.12 or k == 1:
                local_idx = 0
            elif has_exact_bars:
                probe_x = pt.x_left + 0.15 * (pt.x_right - pt.x_left)
                est_idx = _interval_index(probe_x, edges, k)
                boundaries = _find_structural_boundary_near(measures, sys_indices, est_idx, radius=1)
                local_idx = min(boundaries, key=lambda j: abs(j - est_idx)) if boundaries else est_idx
            else:
                est_float = pt.u_norm * k
                est_idx = min(k - 1, max(0, int(est_float)))
                boundaries = _find_structural_boundary_near(measures, sys_indices, est_idx, radius=2)
                local_idx = min(boundaries, key=lambda j: abs(j - est_float)) if boundaries else est_idx

            if apply_tempo_marking(root, pt.tempo, measure_index=sys_indices[local_idx]):
                injected += 1
    return injected


def map_range_tempos_to_measures(
    root: ET.Element,
    tempos: Sequence[PositionedTempo],
    *,
    start_index: int,
    measure_count: int,
    last_tempo_key: Optional[tuple[str, Optional[int], str, bool]] = None,
) -> tuple[int, Optional[tuple[str, Optional[int], str, bool]]]:
    """Map full-score page tempos/expressions across ``[start_index .. start_index + measure_count)``."""
    if not tempos or measure_count <= 0:
        return 0, last_tempo_key
    first_part = root.find("part")
    if first_part is None:
        return 0, last_tempo_key
    measures = first_part.findall("measure")
    if start_index < 0 or start_index >= len(measures):
        return 0, last_tempo_key
    end_index = min(len(measures), start_index + measure_count)
    indices = list(range(start_index, end_index))
    k = len(indices)

    injected = 0
    for pt in tempos:
        if pt.tempo.is_expression:
            local_idx = min(k - 1, max(0, int(pt.u_norm * k)))
            if apply_tempo_marking(root, pt.tempo, measure_index=indices[local_idx]):
                injected += 1
            continue

        t_key = (pt.tempo.text.lower(), pt.tempo.bpm, pt.tempo.beat_unit, pt.tempo.beat_unit_dot)
        if t_key == last_tempo_key:
            continue

        if pt.u_norm < 0.12 or k == 1:
            local_idx = 0
        else:
            est_float = pt.u_norm * k
            est_idx = min(k - 1, max(0, int(est_float)))
            boundaries = _find_structural_boundary_near(measures, indices, est_idx, radius=2)
            local_idx = min(boundaries, key=lambda j: abs(j - est_float)) if boundaries else est_idx

        if apply_tempo_marking(root, pt.tempo, measure_index=indices[local_idx]):
            last_tempo_key = t_key
            injected += 1
    return injected, last_tempo_key


def extract_tempo_marking(
    png_path: Path,
    top_staff_y: float,
    *,
    staff_left_x: float = 0.0,
    ocr_reader: Any = None,
    ocr_dir: Optional[Path] = None,
    ocr_runner: Optional[Callable[[np.ndarray], tuple[Sequence[Any], Sequence[str]]]] = None,
) -> Optional[TempoMarking]:
    """Run RapidOCR above ``top_staff_y`` on ``png_path`` to extract a tempo marking."""
    if top_staff_y < 25:
        return None
    img = cv2.imread(str(png_path))
    if img is None:
        log.warning("Could not read %s for tempo OCR", png_path)
        return None

    from .fullscore.instruments import _run_rapidocr_on_crop

    h_px, w_px = img.shape[:2]
    crop_h = max(1, min(int(top_staff_y) + 8, h_px))
    header = img[:crop_h, :]
    boxes_raw, txts_raw = _run_rapidocr_on_crop(
        header, ocr_reader=ocr_reader, ocr_dir=ocr_dir, ocr_runner=ocr_runner
    )
    if not boxes_raw or not txts_raw:
        log.debug("Tempo OCR on %s (y<%.1f): empty OCR output", Path(png_path).name, top_staff_y)
        return None

    merged_boxes = _merge_horizontal_ocr_boxes(boxes_raw, txts_raw)
    candidates: list[tuple[tuple[int, int, int, float], TempoMarking, tuple[float, float, float, float]]] = []
    for left, top, right, bottom, txt in merged_boxes:
        tempo = parse_tempo_text(txt)
        if tempo is None:
            continue
        has_bpm = 1 if tempo.bpm is not None else 0
        has_text = 1 if tempo.text else 0
        cx = 0.5 * (left + right)
        cy = 0.5 * (top + bottom)
        near_left = 1 if (left >= staff_left_x - 30 or (w_px > 0 and cx / float(w_px) < 0.6)) else 0
        score = (has_bpm, has_text, near_left, cy)
        candidates.append((score, tempo, (left, top, right, bottom)))

    if not candidates:
        log.debug("Tempo OCR on %s (y<%.1f): no tempo marking found", Path(png_path).name, top_staff_y)
        return None

    _best_score, best_tempo, bbox = max(candidates, key=lambda item: item[0])
    log.info(
        "Tempo OCR on %s detected %r (bpm=%s, unit=%s%s, box=[%.0f,%.0f..%.0f,%.0f])",
        Path(png_path).name,
        best_tempo.text or best_tempo.raw_ocr,
        best_tempo.bpm,
        best_tempo.beat_unit,
        "." if best_tempo.beat_unit_dot else "",
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
    )
    return best_tempo


def apply_tempo_marking(
    root: ET.Element,
    tempo: Optional[TempoMarking],
    *,
    measure_index: int = 0,
) -> bool:
    """Inject a MusicXML 4.0 tempo ``<direction>`` into the first part at ``measure_index``."""
    if tempo is None or (not tempo.text and tempo.bpm is None):
        return False

    first_part = root.find("part")
    if first_part is None:
        return False
    measures = first_part.findall("measure")
    if measure_index < 0 or measure_index >= len(measures):
        return False
    measure = measures[measure_index]

    # Idempotent per category: expressions deduplicate by matching words text; main tempos by metronome/sound
    if tempo.is_expression:
        target_low = tempo.text.strip().lower()
        for w_el in measure.findall("./direction/direction-type/words"):
            if (w_el.text or "").strip().lower() == target_low:
                return False
        direction = ET.Element("direction", {"placement": "above"})
        dt_words = ET.SubElement(direction, "direction-type")
        words = ET.SubElement(dt_words, "words", {"font-style": "italic"})
        words.text = tempo.text
    else:
        for existing in measure.findall("direction"):
            if existing.find(".//metronome") is not None or existing.find("./sound[@tempo]") is not None:
                return False

        direction = ET.Element("direction", {"placement": "above"})
        if tempo.text:
            dt_words = ET.SubElement(direction, "direction-type")
            word_attrib = {"font-weight": "bold"}
            if tempo.bpm is not None:
                word_attrib[_XML_SPACE_ATTR] = "preserve"
            words = ET.SubElement(dt_words, "words", word_attrib)
            words.text = f"{tempo.text} " if tempo.bpm is not None else tempo.text
        if tempo.bpm is not None:
            dt_metro = ET.SubElement(direction, "direction-type")
            metro = ET.SubElement(dt_metro, "metronome")
            ET.SubElement(metro, "beat-unit").text = tempo.beat_unit
            if tempo.beat_unit_dot:
                ET.SubElement(metro, "beat-unit-dot")
            ET.SubElement(metro, "per-minute").text = str(tempo.bpm)
        q_bpm = tempo.quarter_bpm()
        if q_bpm is not None:
            ET.SubElement(direction, "sound", {"tempo": str(q_bpm)})

    insert_pos = 0
    for idx, child in enumerate(list(measure)):
        if child.tag in ("print", "attributes") or (tempo.is_expression and child.tag == "direction"):
            insert_pos = idx + 1
        elif child.tag == "note":
            break
    measure.insert(insert_pos, direction)
    log.info(
        "Injected %s marking into %s measure %s: text=%r bpm=%s sound_tempo=%s",
        "expression" if tempo.is_expression else "tempo",
        first_part.get("id", "P1"),
        measure.get("number", str(measure_index + 1)),
        tempo.text,
        tempo.bpm,
        tempo.quarter_bpm(),
    )
    return True
