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
    r"alla\s+marcia|meno\s+mosso|pi[uù]\s+mosso|a\s+tempo|"
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
    r"[=≈]\s*(?P<bpm>\d{2,3})(?:\b|[)\]])",
    re.IGNORECASE,
)
_METRO_BPM_RE = re.compile(
    r"(?:\b(?:bpm\s*[:=]?\s*(?P<bpm1>\d{2,3})|(?P<bpm2>\d{2,3})\s*bpm)\b|"
    r"\bm\.?m\.?\s*[:=]?\s*(?P<bpm3>\d{2,3})\b)",
    re.IGNORECASE,
)

_UNIT_MAP: dict[str, str] = {
    "h": "half",
    "𝅗𝅥": "half",
    "𝅗𝅥": "half",
    "e": "eighth",
    "♪": "eighth",
}


@dataclass(frozen=True)
class TempoMarking:
    """Parsed tempo indication (text and/or metronome BPM)."""

    text: str = ""
    bpm: Optional[int] = None
    beat_unit: str = "quarter"
    beat_unit_dot: bool = False
    raw_ocr: str = ""

    def quarter_bpm(self) -> Optional[int]:
        """Return effective quarter-note BPM for MusicXML ``<sound tempo='...'>``."""
        if self.bpm is not None:
            factor = _UNIT_FACTORS.get(self.beat_unit, 1.0)
            if self.beat_unit_dot:
                factor *= 1.5
            val = int(round(self.bpm * factor))
            return max(20, min(300, val))
        if self.text:
            low = self.text.lower()
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
        )


def parse_tempo_text(raw: str) -> Optional[TempoMarking]:
    """Parse a single OCR line into a ``TempoMarking``, or return ``None`` for non-tempo text."""
    cleaned = re.sub(r"\s+", " ", (raw or "")).strip()
    if not cleaned or len(cleaned) > 60:
        return None
    if _BOILERPLATE_RE.search(cleaned):
        return None

    bpm: Optional[int] = None
    beat_unit = "quarter"
    beat_unit_dot = False
    match_start: Optional[int] = None

    m_eq = _METRO_EQ_RE.search(cleaned)
    if m_eq is not None:
        cand = int(m_eq.group("bpm"))
        if 24 <= cand <= 260:
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
        prefix = cleaned[:match_start].strip(" \t\r\n([,:;-–—")
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

    # Word-only tempo marking (e.g. "Allegro ma non troppo")
    text_only = cleaned.strip(" \t\r\n.,:;-–—()")
    if not text_only or re.search(r"\d|[=≈]", text_only):
        return None
    if len(text_only.split()) > 5:
        return None
    if not _TEMPO_START_RE.match(text_only):
        return None
    return TempoMarking(text=text_only, bpm=None, raw_ocr=cleaned)


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
        for prev in merged:
            prev_h = max(1.0, prev[3] - prev[1])
            y_overlap = min(prev[3], bottom) - max(prev[1], top)
            x_gap = left - prev[2]
            if y_overlap > 0.45 * min(prev_h, cur_h) and -0.3 * min(prev_h, cur_h) <= x_gap <= 1.2 * max(prev_h, cur_h):
                prev[0] = min(prev[0], left)
                prev[1] = min(prev[1], top)
                prev[2] = max(prev[2], right)
                prev[3] = max(prev[3], bottom)
                prev[4] = f"{prev[4]} {text}"
                break
        else:
            merged.append([left, top, right, bottom, text])

    return [(float(m[0]), float(m[1]), float(m[2]), float(m[3]), str(m[4])) for m in merged]


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

    # Idempotent: avoid duplicating if tempo direction is already present
    for existing in measure.findall("direction"):
        if existing.find(".//metronome") is not None or existing.find("./sound[@tempo]") is not None:
            return False

    direction = ET.Element("direction", {"placement": "above"})
    if tempo.text:
        dt_words = ET.SubElement(direction, "direction-type")
        words = ET.SubElement(dt_words, "words", {"font-weight": "bold"})
        words.text = tempo.text
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
        if child.tag in ("print", "attributes"):
            insert_pos = idx + 1
        elif child.tag == "note":
            break
    measure.insert(insert_pos, direction)
    log.info(
        "Injected tempo marking into %s measure %s: text=%r bpm=%s sound_tempo=%s",
        first_part.get("id", "P1"),
        measure.get("number", str(measure_index + 1)),
        tempo.text,
        tempo.bpm,
        q_bpm,
    )
    return True
