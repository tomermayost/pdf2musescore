"""Full Score alignment, variable-staff DP reconciliation, merging, metadata, and invariant verification."""

from __future__ import annotations

import copy
import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional, Sequence

from ..musicxml import (
    TYPE_TO_QUARTERS as _TYPE_TO_QUARTERS,
    MergeError,
    add_courtesy_accidentals,
    normalize_unpitched_percussion_part,
    read_score,
    repair_spurious_tuplets,
    set_score_part_instrument,
    strip_layout_breaks,
    write_musicxml,
)
from ..options import ConversionOptions
from ..rehearsal import PositionedRehearsal, map_range_rehearsal_marks_to_measures
from ..tempo import PositionedTempo, TempoMarking, apply_tempo_marking, map_range_tempos_to_measures
from .instruments import (
    InstrumentLabel,
    InstrumentPlaybackProfile,
    consolidate_page_labels,
    infer_instrument_profile,
    is_percussion_instrument,
    looks_low_quality_label,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageSidecar:
    """Per-page full-score geometry and OCR labels persisted beside each page's MusicXML."""

    page_number: int
    staff_count: int
    staff_cys: list[float]
    staff_y_norms: list[float]
    labels: list[InstrumentLabel]
    sys_min_x: int = 0
    sys_max_x: int = 0
    crop_x: int = 0
    tempo: Optional[TempoMarking] = None
    positioned_tempos: tuple[PositionedTempo, ...] = ()
    rehearsal_marks: tuple[PositionedRehearsal, ...] = ()


def sidecar_path_for(xml_path: Path) -> Path:
    """Return the ``.fullscore.json`` sidecar path paired with ``xml_path``."""
    return Path(xml_path).with_suffix(".fullscore.json")


def write_page_sidecar(sidecar: PageSidecar, xml_path: Path) -> Path:
    """Persist ``sidecar`` JSON alongside ``xml_path``."""
    path = sidecar_path_for(xml_path)
    payload: dict[str, object] = {
        "page_number": sidecar.page_number,
        "staff_count": sidecar.staff_count,
        "staff_cys": list(sidecar.staff_cys),
        "staff_y_norms": list(sidecar.staff_y_norms),
        "labels": [asdict(lbl) for lbl in sidecar.labels],
        "sys_min_x": sidecar.sys_min_x,
        "sys_max_x": sidecar.sys_max_x,
        "crop_x": sidecar.crop_x,
    }
    if sidecar.tempo is not None:
        payload["tempo"] = sidecar.tempo.to_dict()
    if sidecar.positioned_tempos:
        payload["positioned_tempos"] = [pt.to_dict() for pt in sidecar.positioned_tempos]
    if sidecar.rehearsal_marks:
        payload["rehearsal_marks"] = [r.to_dict() for r in sidecar.rehearsal_marks]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.debug("Wrote full-score sidecar %s (%d staves)", path.name, sidecar.staff_count)
    return path


def read_page_sidecar(xml_path: Path) -> Optional[PageSidecar]:
    """Read sibling ``.fullscore.json`` sidecar if present and valid; otherwise return None."""
    path = sidecar_path_for(xml_path)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        labels = [
            InstrumentLabel(
                full_name=str(item["full_name"]),
                abbreviation=str(item["abbreviation"]),
                raw_ocr=str(item.get("raw_ocr", "")),
                is_fallback=bool(item.get("is_fallback", False)),
                is_abbreviated_source=bool(item.get("is_abbreviated_source", False)),
            )
            for item in raw.get("labels", [])
        ]
        positioned_tempos = tuple(
            pt
            for item in raw.get("positioned_tempos", [])
            if (pt := PositionedTempo.from_dict(item)) is not None
        )
        rehearsals = tuple(
            pr
            for item in raw.get("rehearsal_marks", [])
            if (pr := PositionedRehearsal.from_dict(item)) is not None
        )
        return PageSidecar(
            page_number=int(raw["page_number"]),
            staff_count=int(raw["staff_count"]),
            staff_cys=[float(v) for v in raw.get("staff_cys", [])],
            staff_y_norms=[float(v) for v in raw.get("staff_y_norms", [])],
            labels=labels,
            sys_min_x=int(raw.get("sys_min_x", 0)),
            sys_max_x=int(raw.get("sys_max_x", 0)),
            crop_x=int(raw.get("crop_x", 0)),
            tempo=TempoMarking.from_dict(raw.get("tempo")),
            positioned_tempos=positioned_tempos,
            rehearsal_marks=rehearsals,
        )
    except Exception as exc:
        log.warning("Could not read full-score sidecar %s (%s: %s) — falling back", path.name, type(exc).__name__, exc)
        return None


def _is_rest_only_measure(measure: ET.Element) -> bool:
    notes = measure.findall("note")
    if not notes:
        return True
    return all(n.find("rest") is not None for n in notes)


def _extract_last_time(measure: ET.Element) -> Optional[tuple[str, str]]:
    last_ts: Optional[tuple[str, str]] = None
    for attrs in measure.findall("attributes"):
        for t in attrs.findall("time"):
            b = (t.findtext("beats") or "").strip()
            bt = (t.findtext("beat-type") or "").strip()
            if b and bt:
                last_ts = (b, bt)
    return last_ts


def _remove_all_time_elements(measure: ET.Element) -> None:
    for attrs in list(measure.findall("attributes")):
        for t in list(attrs.findall("time")):
            attrs.remove(t)
        if len(attrs) == 0:
            measure.remove(attrs)


def _set_single_time(measure: ET.Element, beats: str, beat_type: str) -> None:
    _remove_all_time_elements(measure)
    attrs = measure.find("attributes")
    if attrs is None:
        attrs = ET.Element("attributes")
        measure.insert(0, attrs)
    t_el = ET.SubElement(attrs, "time")
    ET.SubElement(t_el, "beats").text = str(beats)
    ET.SubElement(t_el, "beat-type").text = str(beat_type)


def _replace_measure_notes_with_rest(measure: ET.Element, expected_dur: int) -> None:
    for child in list(measure):
        if child.tag in ("note", "backup", "forward", "direction"):
            measure.remove(child)
    note = ET.Element("note")
    ET.SubElement(note, "rest", measure="yes")
    ET.SubElement(note, "duration").text = str(max(1, expected_dur))
    ET.SubElement(note, "voice").text = "1"
    ET.SubElement(note, "staff").text = "1"
    bar = measure.find("barline")
    if bar is not None:
        idx = list(measure).index(bar)
        measure.insert(idx, note)
    else:
        measure.append(note)


def _time_sig_quarters(ts: tuple[str, str]) -> Optional[Fraction]:
    try:
        b = int(ts[0])
        bt = int(ts[1])
    except (TypeError, ValueError):
        return None
    if b <= 0 or bt <= 0:
        return None
    return Fraction(b * 4, bt)


def _fraction_to_time_sig(q: Fraction) -> Optional[tuple[str, str]]:
    if q.denominator == 1 and 2 <= q.numerator <= 7:
        return (str(q.numerator), "4")
    if q.denominator == 2 and q.numerator in (3, 5, 7, 9, 11):
        return (str(q.numerator), "8")
    return None


_STEP_DOWN_TYPE: dict[str, str] = {
    "breve": "whole",
    "whole": "half",
    "half": "quarter",
    "quarter": "eighth",
    "eighth": "16th",
    "16th": "32nd",
    "32nd": "64th",
}


def _duration_to_type_and_dots(dur: int, div: int) -> Optional[tuple[str, int]]:
    """Map an integer MusicXML duration at ``div`` divisions/quarter to ``(type, dots)`` if standard."""
    if dur <= 0:
        return None
    q = Fraction(dur, max(1, div))
    for note_type, base_q in _TYPE_TO_QUARTERS.items():
        if q == base_q:
            return (note_type, 0)
        if q == base_q * Fraction(3, 2):
            return (note_type, 1)
        if q == base_q * Fraction(7, 4):
            return (note_type, 2)
    return None


def _largest_standard_type_and_dots(max_dur: int, div: int) -> Optional[tuple[int, str, int]]:
    """Return ``(ticks, type, dots)`` for the largest standard notation value with ``0 < ticks <= max_dur``."""
    if max_dur <= 0:
        return None
    d_unit = max(1, div)
    best: Optional[tuple[int, int, str, int]] = None  # (ticks, -dots, type, dots)
    for note_type, base_q in _TYPE_TO_QUARTERS.items():
        for dots, mult in ((0, Fraction(1)), (1, Fraction(3, 2)), (2, Fraction(7, 4))):
            ticks_frac = base_q * mult * d_unit
            if ticks_frac.denominator != 1:
                continue
            ticks = int(ticks_frac)
            if 0 < ticks <= max_dur:
                cand = (ticks, -dots, note_type, dots)
                if best is None or cand[:2] > best[:2]:
                    best = cand
    if best is None:
        return None
    return best[0], best[2], best[3]


def _reconcile_duration_type_mismatches(measure: ET.Element, div: int) -> bool:
    """Sync ``<duration>`` when it disagrees with explicit ``<type>``/``<dot>``/``<time-modification>``."""
    d_unit = max(1, div)
    changed = False
    for note in measure.findall("note"):
        if note.find("grace") is not None:
            continue
        note_type = (note.findtext("type") or "").strip()
        base_q = _TYPE_TO_QUARTERS.get(note_type)
        if base_q is None:
            continue
        dots = len(note.findall("dot"))
        if dots == 0:
            dot_mult = Fraction(1)
        elif dots == 1:
            dot_mult = Fraction(3, 2)
        elif dots == 2:
            dot_mult = Fraction(7, 4)
        else:
            continue
        tm_mult = Fraction(1)
        tm = note.find("time-modification")
        if tm is not None:
            an_txt = (tm.findtext("actual-notes") or "").strip()
            nn_txt = (tm.findtext("normal-notes") or "").strip()
            if not (an_txt.isdigit() and nn_txt.isdigit()):
                continue
            actual, normal = int(an_txt), int(nn_txt)
            if actual <= 0 or normal <= 0:
                continue
            tm_mult = Fraction(normal, actual)
        vis_frac = base_q * dot_mult * tm_mult * d_unit
        if vis_frac.denominator != 1 or vis_frac <= 0:
            continue
        vis_ticks = int(vis_frac)
        dur_el = note.find("duration")
        if dur_el is not None and (dur_el.text or "").strip().isdigit() and int(dur_el.text.strip()) != vis_ticks:
            dur_el.text = str(vis_ticks)
            changed = True
    return changed


def _set_note_duration_and_type(
    note: ET.Element,
    new_dur: int,
    new_type: Optional[str] = None,
    new_dots: Optional[int] = None,
    *,
    clear_tuplet: bool = False,
) -> None:
    dur_el = note.find("duration")
    if dur_el is None:
        dur_el = ET.Element("duration")
        ins_idx = 0
        for idx, ch in enumerate(list(note)):
            if ch.tag in ("pitch", "rest", "unpitched"):
                ins_idx = idx + 1
                break
        note.insert(ins_idx, dur_el)
    dur_el.text = str(max(1, int(new_dur)))

    type_el = note.find("type")
    if new_type is not None:
        if type_el is None:
            type_el = ET.Element("type")
            anchor_idx = list(note).index(dur_el) + 1
            voice_el = note.find("voice")
            if voice_el is not None:
                anchor_idx = list(note).index(voice_el) + 1
            note.insert(anchor_idx, type_el)
        type_el.text = new_type

    if new_dots is not None:
        for old_dot in list(note.findall("dot")):
            note.remove(old_dot)
        if new_dots > 0:
            type_el = note.find("type")
            if type_el is not None:
                dot_idx = list(note).index(type_el) + 1
            else:
                voice_el = note.find("voice")
                anchor = voice_el if voice_el is not None else dur_el
                dot_idx = list(note).index(anchor) + 1
            for offset in range(new_dots):
                note.insert(dot_idx + offset, ET.Element("dot"))

    if clear_tuplet:
        for tm in list(note.findall("time-modification")):
            note.remove(tm)


def _primary_beam_text(notes: Sequence[ET.Element]) -> Optional[str]:
    if not notes:
        return None
    beams = notes[0].findall("beam")
    if not beams:
        return None
    for b in beams:
        if b.get("number") in (None, "1"):
            return (b.text or "").strip().lower()
    return (beams[0].text or "").strip().lower()


def _clear_beam_run(note_groups: Sequence[list[ET.Element]], target_idx: int) -> None:
    """Clear ``<beam>`` elements across the contiguous beamed group containing ``target_idx``."""
    cur_beam = _primary_beam_text(note_groups[target_idx])
    if cur_beam is None:
        return
    lo = target_idx
    if cur_beam != "begin":
        while lo > 0:
            prev_beam = _primary_beam_text(note_groups[lo - 1])
            if prev_beam is None:
                break
            lo -= 1
            if prev_beam == "begin":
                break
    hi = target_idx
    if cur_beam != "end":
        while hi + 1 < len(note_groups):
            next_beam = _primary_beam_text(note_groups[hi + 1])
            if next_beam is None:
                break
            hi += 1
            if next_beam == "end":
                break
    for gi in range(lo, hi + 1):
        for n in note_groups[gi]:
            for b_el in list(n.findall("beam")):
                n.remove(b_el)


def _collect_voice_groups(measure: ET.Element) -> tuple[dict[str, list[list[object]]], set[str]]:
    """Return ``(voice_groups, overlap_voices)``.

    Each group is ``[onset, dur, [primary_note, *chord_notes], leading_graces]``.
    """
    pos = 0
    pending_graces: list[ET.Element] = []
    voice_groups: dict[str, list[list[object]]] = {}
    voice_end: dict[str, int] = {}
    overlap_voices: set[str] = set()

    for child in measure:
        if child.tag == "note":
            if child.find("grace") is not None:
                pending_graces.append(child)
                continue
            voice = (child.findtext("voice") or "1").strip() or "1"
            dur_txt = (child.findtext("duration") or "").strip()
            dur = int(dur_txt) if dur_txt.isdigit() else 0
            if child.find("chord") is not None:
                groups = voice_groups.get(voice)
                if groups:
                    groups[-1][2].append(child)
            else:
                if voice in voice_end and pos < voice_end[voice]:
                    overlap_voices.add(voice)
                voice_groups.setdefault(voice, []).append([pos, dur, [child], pending_graces])
                pending_graces = []
                voice_end[voice] = max(voice_end.get(voice, 0), pos + dur)
                pos += dur
        elif child.tag == "backup":
            pending_graces.clear()
            dur_txt = (child.findtext("duration") or "").strip()
            pos = max(0, pos - (int(dur_txt) if dur_txt.isdigit() else 0))
        elif child.tag == "forward":
            pending_graces.clear()
            dur_txt = (child.findtext("duration") or "").strip()
            pos += int(dur_txt) if dur_txt.isdigit() else 0
    return voice_groups, overlap_voices


def _normalize_group_sequence(
    measure: ET.Element,
    groups: list[list[object]],
    expected_dur: int,
    div: int,
    *,
    remove_dropped_from_measure: bool,
) -> list[list[object]]:
    """Run the 3-tier overfull repair on one contiguous voice's note groups."""
    if not groups:
        return groups
    excess = max(int(g[0]) + int(g[1]) for g in groups) - expected_dur
    if excess <= 0:
        return groups

    def _apply_group(
        gi: int,
        new_dur: int,
        new_type: Optional[str],
        new_dots: Optional[int],
        *,
        clear_beam: bool,
        clear_tuplet: bool = False,
    ) -> None:
        nonlocal excess
        old_dur = int(groups[gi][1])
        delta = old_dur - new_dur
        groups[gi][1] = new_dur
        for later in groups[gi + 1 :]:
            later[0] = int(later[0]) - delta
        for note in groups[gi][2]:
            _set_note_duration_and_type(note, new_dur, new_type, new_dots, clear_tuplet=clear_tuplet)
        if clear_beam:
            _clear_beam_run([g[2] for g in groups], gi)
        excess = max(int(g[0]) + int(g[1]) for g in groups) - expected_dur

    # Tier 1: Recover missed beam on dotted pairs (e.g. eighth + dotted-eighth -> 16th + dotted-eighth)
    for max_factor in (1, 2):
        if excess <= 0:
            break
        for idx in range(len(groups) - 1):
            if excess <= 0:
                break
            a = groups[idx][2][0]
            b = groups[idx + 1][2][0]
            if (
                a.find("rest") is not None
                or b.find("rest") is not None
                or a.find("time-modification") is not None
                or b.find("time-modification") is not None
            ):
                continue
            a_type = (a.findtext("type") or "").strip()
            b_type = (b.findtext("type") or "").strip()
            a_dots = len(a.findall("dot"))
            b_dots = len(b.findall("dot"))
            a_dur = int(groups[idx][1])
            b_dur = int(groups[idx + 1][1])

            target_idx: Optional[int] = None
            base_type = ""
            undotted_dur = 0
            if a_dots == 0 and b_dots == 1 and a_type == b_type and a_type in _STEP_DOWN_TYPE and a_dur % 2 == 0:
                target_idx, base_type, undotted_dur = idx, a_type, a_dur
            elif a_dots == 1 and b_dots == 0 and a_type == b_type and b_type in _STEP_DOWN_TYPE and b_dur % 2 == 0:
                target_idx, base_type, undotted_dur = idx + 1, b_type, b_dur

            if target_idx is not None:
                half = undotted_dur // 2
                if 0 < half <= max_factor * excess:
                    _apply_group(
                        target_idx,
                        half,
                        _STEP_DOWN_TYPE[base_type],
                        0,
                        clear_beam=True,
                    )

    # Tier 2: Remove spurious dot from a single-dotted note matching excess
    for exact_only in (True, False):
        if excess <= 0:
            break
        for idx, g in enumerate(groups):
            if excess <= 0:
                break
            n = g[2][0]
            if n.find("rest") is not None or n.find("time-modification") is not None:
                continue
            if len(n.findall("dot")) != 1:
                continue
            dur = int(g[1])
            if dur % 3 != 0:
                continue
            dot_val = dur // 3
            if (dot_val == excess) if exact_only else (0 < dot_val <= excess):
                cur_type = (n.findtext("type") or "").strip() or None
                _apply_group(idx, dur - dot_val, cur_type, 0, clear_beam=False)

    # Tier 3a: Step down one undotted note whose halved value equals exact excess
    if excess > 0:
        for idx in range(len(groups) - 1, -1, -1):
            n = groups[idx][2][0]
            if n.find("rest") is not None or n.find("time-modification") is not None:
                continue
            if len(n.findall("dot")) != 0:
                continue
            cur_type = (n.findtext("type") or "").strip()
            dur = int(groups[idx][1])
            if cur_type in _STEP_DOWN_TYPE and dur % 2 == 0 and (dur // 2) == excess:
                _apply_group(
                    idx,
                    dur // 2,
                    _STEP_DOWN_TYPE[cur_type],
                    0,
                    clear_beam=True,
                )
                break

    # Tier 3a-pair: Step down an adjacent pair of identical undotted notes (e.g. 2 eighths -> 2 sixteenths)
    # when halving both equals exact excess, preferring slurred or beamed pairs.
    if excess > 0:
        pair_candidates: list[tuple[int, int, str, int]] = []
        for idx in range(len(groups) - 1):
            a = groups[idx][2][0]
            b = groups[idx + 1][2][0]
            if (
                a.find("rest") is not None
                or b.find("rest") is not None
                or a.find("time-modification") is not None
                or b.find("time-modification") is not None
            ):
                continue
            if len(a.findall("dot")) != 0 or len(b.findall("dot")) != 0:
                continue
            a_type = (a.findtext("type") or "").strip()
            b_type = (b.findtext("type") or "").strip()
            a_dur = int(groups[idx][1])
            b_dur = int(groups[idx + 1][1])
            if a_type != b_type or a_type not in _STEP_DOWN_TYPE or a_dur != b_dur or a_dur % 2 != 0:
                continue
            if a_dur == excess:
                has_slur_pair = (
                    a.find("./notations/slur[@type='start']") is not None
                    and b.find("./notations/slur[@type='stop']") is not None
                )
                has_beam = _primary_beam_text(groups[idx][2]) is not None
                priority = 2 if has_slur_pair else (1 if has_beam else 0)
                pair_candidates.append((priority, idx, a_type, a_dur))
        if pair_candidates:
            _prio, best_idx, p_type, p_dur = max(pair_candidates, key=lambda item: (item[0], item[1]))
            half_dur = p_dur // 2
            stepped_type = _STEP_DOWN_TYPE[p_type]
            _apply_group(best_idx, half_dur, stepped_type, 0, clear_beam=True)
            _apply_group(best_idx + 1, half_dur, stepped_type, 0, clear_beam=True)

    # Tier 3b: Guaranteed barline clamp for any remaining trailing overflow
    if excess <= 0:
        return groups

    kept_groups: list[list[object]] = []
    for idx, g in enumerate(groups):
        onset = int(g[0])
        dur = int(g[1])
        notes = list(g[2])
        if onset >= expected_dur:
            if remove_dropped_from_measure:
                for gn in g[3]:
                    if gn in list(measure):
                        measure.remove(gn)
                for note in notes:
                    if note in list(measure):
                        measure.remove(note)
            continue
        if onset + dur > expected_dur:
            rem = max(1, expected_dur - onset)
            td = _duration_to_type_and_dots(rem, div)
            if td is not None:
                clamped_dur, clamped_type, clamped_dots = rem, td[0], td[1]
            else:
                fb = _largest_standard_type_and_dots(rem, div)
                if fb is not None:
                    clamped_dur, clamped_type, clamped_dots = fb
                else:
                    clamped_dur, clamped_type, clamped_dots = rem, None, None
            for note in notes:
                _set_note_duration_and_type(
                    note,
                    clamped_dur,
                    clamped_type,
                    clamped_dots,
                    clear_tuplet=True,
                )
            _clear_beam_run([gr[2] for gr in groups], idx)
            g[1] = clamped_dur
        kept_groups.append(g)
    return kept_groups


def _rebuild_measure_canonical_voices(
    measure: ET.Element,
    kept_by_voice: dict[str, list[list[object]]],
) -> None:
    """Re-emit ``measure`` in canonical voice-by-voice order with exact ``<backup>``/``<forward>`` elements."""
    pre_children: list[ET.Element] = []
    barlines: list[ET.Element] = []
    for ch in list(measure):
        if ch.tag in ("note", "backup", "forward"):
            continue
        if ch.tag == "barline":
            barlines.append(ch)
        else:
            pre_children.append(ch)
    for ch in list(measure):
        measure.remove(ch)

    for ch in pre_children:
        measure.append(ch)

    ordered_voices = sorted(
        (v for v, gs in kept_by_voice.items() if gs),
        key=lambda v: (0, int(v)) if v.isdigit() else (1, v),
    )
    prev_span = 0
    for idx, voice in enumerate(ordered_voices):
        groups = kept_by_voice[voice]
        if idx > 0 and prev_span > 0:
            bk = ET.Element("backup")
            ET.SubElement(bk, "duration").text = str(prev_span)
            measure.append(bk)
        cursor = 0
        for g in groups:
            onset = int(g[0])
            dur = int(g[1])
            if onset > cursor:
                fw = ET.Element("forward")
                ET.SubElement(fw, "duration").text = str(onset - cursor)
                measure.append(fw)
                cursor = onset
            for gn in g[3]:
                measure.append(gn)
            for n in g[2]:
                measure.append(n)
            cursor += dur
        prev_span = cursor

    for bl in barlines:
        measure.append(bl)


def _normalize_overfull_measure(measure: ET.Element, expected_dur: int, div: int) -> bool:
    """Repair OMR over-duration glitches so a measure never overflows ``expected_dur`` ticks."""
    if expected_dur <= 0 or _is_rest_only_measure(measure):
        return False

    reconciled = _reconcile_duration_type_mismatches(measure, div)
    voice_groups, overlap_voices = _collect_voice_groups(measure)
    if not voice_groups:
        return reconciled

    is_multivoice = (
        len(voice_groups) > 1
        or measure.find("backup") is not None
        or measure.find("forward") is not None
    )
    m_num = measure.get("number", "?")

    if not is_multivoice:
        any_modified = reconciled
        for voice, groups in voice_groups.items():
            old_end = max(int(g[0]) + int(g[1]) for g in groups)
            if old_end <= expected_dur:
                continue
            kept = _normalize_group_sequence(
                measure,
                groups,
                expected_dur,
                div,
                remove_dropped_from_measure=True,
            )
            new_end = max((int(g[0]) + int(g[1]) for g in kept), default=0)
            any_modified = True
            log.debug(
                "Normalized overfull measure %s voice %s (%d -> %d ticks, limit=%d, div=%d)",
                m_num,
                voice,
                old_end,
                new_end,
                expected_dur,
                div,
            )
        return any_modified

    needs_repair = bool(overlap_voices)
    if not needs_repair:
        for groups in voice_groups.values():
            raw_end = max(int(g[0]) + int(g[1]) for g in groups)
            seq_sum = sum(int(g[1]) for g in groups)
            if raw_end > expected_dur or seq_sum > expected_dur:
                needs_repair = True
                break
    if not needs_repair:
        return reconciled

    kept_by_voice: dict[str, list[list[object]]] = {}
    for voice, groups in voice_groups.items():
        seq_sum = sum(int(g[1]) for g in groups)
        raw_end = max(int(g[0]) + int(g[1]) for g in groups)
        if voice not in overlap_voices and raw_end <= expected_dur and seq_sum <= expected_dur:
            kept_by_voice[voice] = groups
            continue

        lead = max(0, int(groups[0][0]))
        if lead + seq_sum > expected_dur:
            lead = 0
        cursor = lead
        for g in groups:
            g[0] = cursor
            cursor += int(g[1])

        kept = _normalize_group_sequence(
            measure,
            groups,
            expected_dur,
            div,
            remove_dropped_from_measure=False,
        )
        kept_by_voice[voice] = kept
        new_end = max((int(g[0]) + int(g[1]) for g in kept), default=0)
        log.debug(
            "Rebuilt multi-voice measure %s voice %s (%d -> %d ticks, limit=%d, div=%d)",
            m_num,
            voice,
            raw_end,
            new_end,
            expected_dur,
            div,
        )

    _rebuild_measure_canonical_voices(measure, kept_by_voice)
    return True


def _measure_quarter_duration(measure: ET.Element, div: int) -> Optional[Fraction]:
    if _is_rest_only_measure(measure):
        return None
    voice_totals: dict[str, int] = {}
    for note in measure.findall("note"):
        if note.find("chord") is not None or note.find("grace") is not None:
            continue
        dur_txt = (note.findtext("duration") or "").strip()
        if not dur_txt.isdigit():
            continue
        dur = int(dur_txt)
        voice = (note.findtext("voice") or "1").strip() or "1"
        voice_totals[voice] = voice_totals.get(voice, 0) + dur
    if not voice_totals:
        return None
    chosen = voice_totals["1"] if "1" in voice_totals else max(voice_totals.values())
    if chosen <= 0:
        return None
    return Fraction(chosen, max(1, div))


def align_page_score(
    root: ET.Element,
    page_num: int,
    *,
    inherited_time_sig: Optional[tuple[str, str]] = None,
) -> tuple[ET.Element, tuple[str, str]]:
    """Harmonize measure counts and time signatures across all parts of a single page."""
    parts = root.findall("part")
    fallback_ts = inherited_time_sig or ("4", "4")
    if not parts:
        return root, fallback_ts

    part_measures_list = [p.findall("measure") for p in parts]
    counts = [len(ms) for ms in part_measures_list]
    count_freq = Counter(counts)
    pitched_by_count: Counter[int] = Counter()
    for ms, cnt in zip(part_measures_list, counts):
        if any(not _is_rest_only_measure(m) for m in ms):
            pitched_by_count[cnt] += 1

    target_measures = max(
        count_freq.keys(),
        key=lambda c: (count_freq[c], pitched_by_count[c], c),
    )
    if target_measures >= 2:
        min_shorter_parts = max(2, int(0.25 * len(parts)))
        shorter_count = sum(1 for c in counts if c < target_measures)
        last_idx = target_measures - 1
        last_all_rest = all(
            _is_rest_only_measure(ms[last_idx])
            for ms in part_measures_list
            if len(ms) >= target_measures
        )
        prev_has_notes = any(
            not _is_rest_only_measure(ms[last_idx - 1])
            for ms in part_measures_list
            if len(ms) >= target_measures
        )
        if last_all_rest and shorter_count >= min_shorter_parts and prev_has_notes:
            log.info(
                "Page %d: trimmed ghost all-rest trailing measure %d (%d/%d parts shorter, dist=%s)",
                page_num,
                target_measures,
                shorter_count,
                len(parts),
                dict(sorted(count_freq.items())),
            )
            target_measures -= 1

    log.info("Aligning page %d: part_measures_dist=%s, target_measures=%d", page_num, dict(sorted(count_freq.items())), target_measures)
    if target_measures == 0:
        return root, fallback_ts

    part_divs: list[list[int]] = []
    for ms in part_measures_list:
        divs_for_part: list[int] = []
        cur_div = 4
        for m in ms:
            for a in m.findall("attributes"):
                d_txt = a.findtext("divisions")
                if d_txt and d_txt.strip().isdigit():
                    cur_div = max(1, int(d_txt.strip()))
            divs_for_part.append(cur_div)
        part_divs.append(divs_for_part)

    # Precompute per-measure playing-part quarter durations so Measure 1 can guard against
    # isolated OMR continuation duration-fallback misreads (e.g. 5/4 before 6q bars).
    later_q_votes: Counter[Fraction] = Counter()
    later_explicit_inh = False
    inh_q_val = _time_sig_quarters(inherited_time_sig) if inherited_time_sig is not None else None
    for later_mi in range(1, target_measures):
        for ms, divs in zip(part_measures_list, part_divs):
            if later_mi >= len(ms):
                continue
            lm = ms[later_mi]
            lq = _measure_quarter_duration(lm, divs[later_mi])
            if lq is not None:
                later_q_votes[lq] += 1
            lts = _extract_last_time(lm)
            if lts is not None and inherited_time_sig is not None and lts == inherited_time_sig:
                later_explicit_inh = True

    global_ts: list[tuple[str, str]] = []
    active_global_ts: tuple[str, str] = fallback_ts
    min_playing = 2 if len(parts) <= 4 else 3
    for mi in range(target_measures):
        pitched_votes: Counter[tuple[str, str]] = Counter()
        all_votes: Counter[tuple[str, str]] = Counter()
        pitched_q_votes: Counter[Fraction] = Counter()
        for ms, divs in zip(part_measures_list, part_divs):
            if mi >= len(ms):
                continue
            m = ms[mi]
            q_dur = _measure_quarter_duration(m, divs[mi])
            if q_dur is not None:
                pitched_q_votes[q_dur] += 1
            ts = _extract_last_time(m)
            if ts is None:
                continue
            if mi == 0 or ts != active_global_ts:
                all_votes[ts] += 1
                if not _is_rest_only_measure(m):
                    pitched_votes[ts] += 1

        if mi == 0:
            votes = pitched_votes if pitched_votes else all_votes
            if votes:
                chosen_ts = max(
                    votes.keys(),
                    key=lambda ts: (
                        votes[ts],
                        pitched_q_votes.get(_time_sig_quarters(ts), 0)
                        if _time_sig_quarters(ts) is not None
                        else 0,
                    ),
                )
                if inherited_time_sig is not None and chosen_ts != inherited_time_sig:
                    inh_q = inh_q_val
                    ch_q = _time_sig_quarters(chosen_ts)
                    strong_explicit = all_votes.get(chosen_ts, 0) >= 0.60 * len(parts)
                    is_homr_quarter_fallback = (
                        inherited_time_sig[1] == "8" and chosen_ts[1] == "4"
                    )
                    if inh_q is not None and ch_q is not None and ch_q == inh_q and (is_homr_quarter_fallback or not strong_explicit):
                        log.info(
                            "Page %d measure 1: preserved inherited equivalent meter %s/%s over unprinted OMR guess %s/%s (votes=%d/%d)",
                            page_num,
                            inherited_time_sig[0],
                            inherited_time_sig[1],
                            chosen_ts[0],
                            chosen_ts[1],
                            all_votes.get(chosen_ts, 0),
                            len(parts),
                        )
                        chosen_ts = inherited_time_sig
                    elif (
                        inh_q is not None
                        and ch_q is not None
                        and ch_q != inh_q
                        and chosen_ts[1] == "4"
                        and not strong_explicit
                        and not later_explicit_inh
                        and later_q_votes.get(inh_q, 0) > later_q_votes.get(ch_q, 0)
                        and later_q_votes.get(inh_q, 0) >= min_playing
                        and pitched_q_votes.get(ch_q, 0) < max(3, int(0.70 * max(1, sum(pitched_q_votes.values()))))
                    ):
                        log.info(
                            "Page %d measure 1: preserved inherited meter %s/%s over minority OMR guess %s/%s (m1 votes=%d/%d, later %sq consensus=%d)",
                            page_num,
                            inherited_time_sig[0],
                            inherited_time_sig[1],
                            chosen_ts[0],
                            chosen_ts[1],
                            all_votes.get(chosen_ts, 0),
                            len(parts),
                            inh_q,
                            later_q_votes.get(inh_q, 0),
                        )
                        chosen_ts = inherited_time_sig
                active_global_ts = chosen_ts
            else:
                active_global_ts = fallback_ts
        else:
            explicit_accepted = False
            playing_parts = sum(pitched_q_votes.values())
            filtered_all = {
                ts: cnt
                for ts, cnt in all_votes.items()
                if (
                    (ts[0].isdigit() and ts[1].isdigit() and int(ts[1]) < 16 and int(ts[0]) <= 12)
                    or cnt >= 0.50 * len(parts)
                )
            }
            filtered_pitched = {ts: cnt for ts, cnt in pitched_votes.items() if ts in filtered_all}
            if filtered_all:
                cand_votes = filtered_pitched if filtered_pitched else filtered_all
                top_ts = max(
                    cand_votes.keys(),
                    key=lambda ts: (
                        cand_votes[ts],
                        filtered_all[ts],
                        pitched_q_votes.get(_time_sig_quarters(ts), 0)
                        if _time_sig_quarters(ts) is not None
                        else 0,
                    ),
                )
                top_all = filtered_all[top_ts]
                top_pitched = filtered_pitched.get(top_ts, 0)
                active_q = _time_sig_quarters(active_global_ts)
                top_q = _time_sig_quarters(top_ts)
                same_span = active_q is not None and top_q is not None and active_q == top_q
                is_compound_to_quarter_fallback = same_span and active_global_ts[1] == "8" and top_ts[1] == "4"
                if same_span:
                    quorum = (not is_compound_to_quarter_fallback) and (top_all >= max(3, int(0.50 * len(parts))))
                elif playing_parts >= 2 and top_pitched == 0:
                    quorum = False
                else:
                    quorum = (
                        top_all >= max(2, int(0.25 * len(parts)))
                        and (top_pitched >= 2 or playing_parts < 2)
                    ) or (
                        playing_parts >= 3
                        and top_pitched >= max(3, int(0.40 * playing_parts))
                    )
                if (
                    quorum
                    and active_q is not None
                    and top_q is not None
                    and top_q != active_q
                    and playing_parts >= 3
                    and pitched_q_votes.get(active_q, 0) > pitched_q_votes.get(top_q, 0)
                    and top_all < 0.60 * len(parts)
                ):
                    quorum = False

                if quorum:
                    active_global_ts = top_ts
                    explicit_accepted = True
                    log.info(
                        "Page %d measure %d: accepted global time-sig change -> %s/%s (all=%s, pitched=%s)",
                        page_num,
                        mi + 1,
                        top_ts[0],
                        top_ts[1],
                        dict(all_votes),
                        dict(pitched_votes),
                    )
                else:
                    log.debug(
                        "Page %d measure %d: ignored minority/noisy time-sig votes %s (kept %s/%s)",
                        page_num,
                        mi + 1,
                        dict(all_votes),
                        active_global_ts[0],
                        active_global_ts[1],
                    )
            if not explicit_accepted and pitched_q_votes:
                pitched_total_q = sum(pitched_q_votes.values())
                top_q, top_q_cnt = pitched_q_votes.most_common(1)[0]
                active_q = _time_sig_quarters(active_global_ts)
                cand_ts = _fraction_to_time_sig(top_q)
                if (
                    pitched_total_q >= min_playing
                    and cand_ts is not None
                    and active_q is not None
                    and top_q != active_q
                    and top_q_cnt >= min_playing
                    and top_q_cnt >= 0.65 * pitched_total_q
                    and pitched_q_votes.get(active_q, 0) <= 0.25 * pitched_total_q
                ):
                    # Preserve compound /8 meter (e.g. 12/8) when duration consensus reverts to its quarter length
                    if inherited_time_sig is not None and _time_sig_quarters(inherited_time_sig) == top_q:
                        cand_ts = inherited_time_sig
                    else:
                        for prev_ts in reversed(global_ts):
                            if _time_sig_quarters(prev_ts) == top_q:
                                cand_ts = prev_ts
                                break
                    active_global_ts = cand_ts
                    log.info(
                        "Page %d measure %d: inferred global time-sig change -> %s/%s from playing-part duration consensus (%d/%d parts = %s quarters)",
                        page_num,
                        mi + 1,
                        cand_ts[0],
                        cand_ts[1],
                        top_q_cnt,
                        pitched_total_q,
                        top_q,
                    )
        global_ts.append(active_global_ts)

    log.info("Page %d global_ts timeline: %s", page_num, [f"{b}/{bt}" for b, bt in global_ts])

    updated_rest_measures = 0
    repaired_tuplets = 0
    normalized_overfull = 0
    padded_count = 0
    trimmed_count = 0

    for p in parts:
        ms = p.findall("measure")
        if len(ms) > target_measures:
            extras = ms[target_measures:]
            for em in extras:
                p.remove(em)
            trimmed_count += len(extras)
            ms = p.findall("measure")

        div = 4
        for mi, m in enumerate(ms):
            for a in m.findall("attributes"):
                d_txt = a.findtext("divisions")
                if d_txt and d_txt.strip().isdigit():
                    div = max(1, int(d_txt.strip()))

            b, bt = global_ts[mi]
            need_time = (mi == 0) or (global_ts[mi] != global_ts[mi - 1])
            if need_time:
                _set_single_time(m, b, bt)
            else:
                _remove_all_time_elements(m)

            expected_dur = int(round(div * int(b) * 4 / max(1, int(bt))))
            if _is_rest_only_measure(m):
                _replace_measure_notes_with_rest(m, expected_dur)
                updated_rest_measures += 1
            else:
                if repair_spurious_tuplets(m, expected_dur, div, (b, bt)):
                    repaired_tuplets += 1
                if _normalize_overfull_measure(m, expected_dur, div):
                    normalized_overfull += 1

        while len(p.findall("measure")) < target_measures:
            mi = len(p.findall("measure"))
            b, bt = global_ts[mi]
            expected_dur = int(round(div * int(b) * 4 / max(1, int(bt))))
            new_m = ET.Element("measure", number=str(mi + 1))
            if mi == 0 or global_ts[mi] != global_ts[mi - 1]:
                _set_single_time(new_m, b, bt)
            _replace_measure_notes_with_rest(new_m, expected_dur)
            p.append(new_m)
            padded_count += 1

    log.info(
        "Page %d alignment complete: updated_rest_measures=%d, repaired_tuplets=%d, normalized_overfull=%d, padded=%d, trimmed=%d",
        page_num,
        updated_rest_measures,
        repaired_tuplets,
        normalized_overfull,
        padded_count,
        trimmed_count,
    )
    return root, global_ts[-1]


@dataclass(frozen=True)
class StaffMatchSpec:
    """Signature of one part/staff used for cross-page monotonic alignment."""

    label: InstrumentLabel
    clef: Optional[tuple[str, str]]
    fifths: Optional[str]
    y_norm: float
    pitched_notes: int = 0


def _extract_part_signature(part: ET.Element) -> tuple[Optional[tuple[str, str]], Optional[str], int]:
    clef: Optional[tuple[str, str]] = None
    fifths: Optional[str] = None
    for attr in part.findall("./measure/attributes"):
        if clef is None:
            for c_el in attr.findall("clef"):
                if c_el.get("number") in (None, "1"):
                    sign = (c_el.findtext("sign") or "").strip()
                    line = (c_el.findtext("line") or "").strip()
                    if sign and line:
                        clef = (sign, line)
                        break
        if fifths is None:
            k_el = attr.find("key/fifths")
            if k_el is not None and k_el.text is not None and k_el.text.strip():
                fifths = k_el.text.strip()
        if clef is not None and fifths is not None:
            break
    pitched = sum(1 for n in part.findall(".//note") if n.find("rest") is None)
    return clef, fifths, pitched


def _label_tokens(label: InstrumentLabel) -> set[str]:
    text = f"{label.full_name} {label.abbreviation}".lower()
    return {tok for tok in re.findall(r"[a-z0-9]+", text) if len(tok) >= 2}


def _name_similarity(a: InstrumentLabel, b: InstrumentLabel) -> float:
    if a.is_fallback or b.is_fallback:
        return 0.0
    ta = _label_tokens(a)
    tb = _label_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def _pair_alignment_score(item: StaffMatchSpec, master: StaffMatchSpec) -> float:
    score = 1.0 - abs(item.y_norm - master.y_norm)
    sim = _name_similarity(item.label, master.label)
    if not item.label.is_fallback and not master.label.is_fallback:
        score += 4.0 * sim
    if item.clef is not None and master.clef is not None:
        score += 1.2 if item.clef == master.clef else -0.8
    if item.fifths is not None and master.fifths is not None and item.fifths == master.fifths:
        score += 0.6
    return score


def match_staves_monotonic_dp(
    page_items: Sequence[StaffMatchSpec],
    master_specs: Sequence[StaffMatchSpec],
) -> list[int]:
    """Map ``K <= M`` page staves to strictly increasing master indices ``0..M-1``."""
    k = len(page_items)
    m = len(master_specs)
    if k == 0:
        return []
    if k > m:
        raise ValueError(f"Cannot monotonically map {k} items into {m} master slots")
    if k == m:
        return list(range(m))

    neg_inf = float("-inf")
    dp = [[neg_inf] * m for _ in range(k)]
    parent = [[-1] * m for _ in range(k)]

    for i in range(m - k + 1):
        dp[0][i] = _pair_alignment_score(page_items[0], master_specs[i])

    for j in range(1, k):
        best_prev_val = neg_inf
        best_prev_idx = -1
        for i in range(j, m - (k - 1 - j)):
            if dp[j - 1][i - 1] > best_prev_val:
                best_prev_val = dp[j - 1][i - 1]
                best_prev_idx = i - 1
            if best_prev_idx >= 0:
                dp[j][i] = best_prev_val + _pair_alignment_score(page_items[j], master_specs[i])
                parent[j][i] = best_prev_idx

    last_idx = max(range(k - 1, m), key=lambda i: dp[k - 1][i])
    mapping = [0] * k
    cur = last_idx
    for j in range(k - 1, -1, -1):
        mapping[j] = cur
        cur = parent[j][cur]
    return mapping


def _build_specs_for_page(
    root: ET.Element,
    sidecar: Optional[PageSidecar],
) -> list[StaffMatchSpec]:
    parts = root.findall("part")
    n = len(parts)
    specs: list[StaffMatchSpec] = []
    for idx, part in enumerate(parts):
        clef, fifths, pitched = _extract_part_signature(part)
        if sidecar is not None and idx < len(sidecar.labels):
            lbl = sidecar.labels[idx]
        else:
            lbl = InstrumentLabel(full_name=f"Part {idx + 1}", abbreviation=f"P{idx + 1}", is_fallback=True)
        if sidecar is not None and idx < len(sidecar.staff_y_norms):
            y_norm = sidecar.staff_y_norms[idx]
        else:
            y_norm = 0.5 if n <= 1 else idx / float(n - 1)
        specs.append(StaffMatchSpec(label=lbl, clef=clef, fifths=fifths, y_norm=y_norm, pitched_notes=pitched))
    return specs


def _build_rest_part_for_page(
    part_id: str,
    master_spec: StaffMatchSpec,
    reference_part: ET.Element,
) -> ET.Element:
    """Create a full-measure rest <part> matching the measure count & time signatures of ``reference_part``."""
    part = ET.Element("part", id=part_id)
    ref_measures = reference_part.findall("measure")
    active_ts = ("4", "4")
    for mi, ref_m in enumerate(ref_measures):
        ts = _extract_last_time(ref_m)
        if ts is not None:
            active_ts = ts
        b, bt = active_ts
        m_el = ET.SubElement(part, "measure", number=str(mi + 1))
        if mi == 0 or ts is not None:
            attrs = ET.SubElement(m_el, "attributes")
            if mi == 0:
                ET.SubElement(attrs, "divisions").text = "1"
                if master_spec.fifths is not None:
                    key_el = ET.SubElement(attrs, "key")
                    ET.SubElement(key_el, "fifths").text = master_spec.fifths
                if master_spec.clef is not None:
                    c_el = ET.SubElement(attrs, "clef", number="1")
                    ET.SubElement(c_el, "sign").text = master_spec.clef[0]
                    ET.SubElement(c_el, "line").text = master_spec.clef[1]
            if ts is not None:
                t_el = ET.SubElement(attrs, "time")
                ET.SubElement(t_el, "beats").text = b
                ET.SubElement(t_el, "beat-type").text = bt
        expected_dur = max(1, int(round(1 * int(b) * 4 / max(1, int(bt)))))
        _replace_measure_notes_with_rest(m_el, expected_dur)
    return part


def _rebuilt_score_with_parts(
    template_root: ET.Element,
    master_score_parts: Sequence[ET.Element],
    ordered_parts: Sequence[ET.Element],
) -> ET.Element:
    out = copy.deepcopy(template_root)
    pl = out.find("part-list")
    if pl is None:
        pl = ET.Element("part-list")
        out.insert(0, pl)
    for child in list(pl):
        pl.remove(child)
    for idx, sp in enumerate(master_score_parts, start=1):
        spc = copy.deepcopy(sp)
        spc.set("id", f"P{idx}")
        si = spc.find("score-instrument")
        if si is not None:
            si.set("id", f"P{idx}-I1")
        mi = spc.find("midi-instrument")
        if mi is not None:
            mi.set("id", f"P{idx}-I1")
        pl.append(spc)

    for old_p in list(out.findall("part")):
        out.remove(old_p)
    for idx, p in enumerate(ordered_parts, start=1):
        pc = copy.deepcopy(p)
        pc.set("id", f"P{idx}")
        out.append(pc)
    return out


def reconcile_pages_to_master(
    aligned_pages: Sequence[tuple[int, ET.Element, Optional[PageSidecar]]],
) -> tuple[list[ET.Element], list[InstrumentLabel]]:
    """Reconcile pages with possibly differing staff counts onto a single canonical instrument roster."""
    if not aligned_pages:
        raise MergeError("No full-score pages to reconcile")

    counts = [len(root.findall("part")) for _pno, root, _sc in aligned_pages]
    if any(c == 0 for c in counts):
        raise MergeError("One or more full-score pages have 0 parts")

    # Drop obvious empty stray detections on outlier pages when a smaller valid roster matches cleanly
    count_freq = Counter(counts)
    consensus_count = max(
        count_freq.keys(),
        key=lambda c: (count_freq[c], -c),
    )
    consensus_pages = count_freq[consensus_count]
    cleaned_pages: list[tuple[int, ET.Element, Optional[PageSidecar]]] = list(aligned_pages)

    if len(set(counts)) > 1 and (consensus_pages >= 2 or any(c < max(counts) for c in counts)):
        # Use the most common count (or largest count <= consensus with valid labels) as reference
        ref_count = consensus_count if consensus_pages >= 2 else max(
            c for c in counts if c < max(counts)
        )
        ref_idx = counts.index(ref_count)
        ref_specs = _build_specs_for_page(cleaned_pages[ref_idx][1], cleaned_pages[ref_idx][2])
        for pidx, (pno, root, sc) in enumerate(list(cleaned_pages)):
            if len(root.findall("part")) <= ref_count:
                continue
            specs = _build_specs_for_page(root, sc)
            # Match ref_count reference staves into this larger page's staves
            kept_indices = set(match_staves_monotonic_dp(ref_specs, specs))
            extra_indices = [i for i in range(len(specs)) if i not in kept_indices]
            all_extra_stray = all(
                specs[i].pitched_notes == 0 and (specs[i].label.is_fallback or looks_low_quality_label(specs[i].label.full_name))
                for i in extra_indices
            )
            if all_extra_stray:
                new_root = copy.deepcopy(root)
                all_parts = new_root.findall("part")
                all_sps = new_root.findall("./part-list/score-part")
                pl = new_root.find("part-list")
                for ei in sorted(extra_indices, reverse=True):
                    new_root.remove(all_parts[ei])
                    if pl is not None and ei < len(all_sps):
                        pl.remove(all_sps[ei])
                new_sc = None
                if sc is not None:
                    new_sc = PageSidecar(
                        page_number=sc.page_number,
                        staff_count=ref_count,
                        staff_cys=[sc.staff_cys[i] for i in sorted(kept_indices) if i < len(sc.staff_cys)],
                        staff_y_norms=[sc.staff_y_norms[i] for i in sorted(kept_indices) if i < len(sc.staff_y_norms)],
                        labels=[sc.labels[i] for i in sorted(kept_indices) if i < len(sc.labels)],
                        sys_min_x=sc.sys_min_x,
                        sys_max_x=sc.sys_max_x,
                        crop_x=sc.crop_x,
                        tempo=sc.tempo,
                        positioned_tempos=sc.positioned_tempos,
                        rehearsal_marks=sc.rehearsal_marks,
                    )
                cleaned_pages[pidx] = (pno, new_root, new_sc)
                log.info(
                    "Page %d: dropped %d empty/unlabeled stray staff detection(s) %s to match %d-part score",
                    pno,
                    len(extra_indices),
                    extra_indices,
                    ref_count,
                )

    updated_counts = [len(root.findall("part")) for _pno, root, _sc in cleaned_pages]
    max_count = max(updated_counts)

    # Pick richest master page among those with max_count
    max_indices = [i for i, c in enumerate(updated_counts) if c == max_count]
    master_idx = max(
        max_indices,
        key=lambda i: (
            sum(
                1
                for sp in _build_specs_for_page(cleaned_pages[i][1], cleaned_pages[i][2])
                if not sp.label.is_fallback and not looks_low_quality_label(sp.label.full_name)
            ),
            -i,
        ),
    )

    label_pages = [
        _build_specs_for_page(cleaned_pages[i][1], cleaned_pages[i][2])
        for i in max_indices
    ]
    consolidated_labels = consolidate_page_labels([[s.label for s in lp] for lp in label_pages])

    master_root = cleaned_pages[master_idx][1]
    master_raw_specs = _build_specs_for_page(master_root, cleaned_pages[master_idx][2])
    master_specs = [
        StaffMatchSpec(
            label=consolidated_labels[i] if i < len(consolidated_labels) else master_raw_specs[i].label,
            clef=master_raw_specs[i].clef,
            fifths=master_raw_specs[i].fifths,
            y_norm=master_raw_specs[i].y_norm,
            pitched_notes=master_raw_specs[i].pitched_notes,
        )
        for i in range(max_count)
    ]
    master_score_parts = master_root.findall("./part-list/score-part")
    if len(master_score_parts) < max_count:
        master_score_parts = [ET.Element("score-part", id=f"P{i + 1}") for i in range(max_count)]

    reconciled: list[ET.Element] = []
    for pno, root, sc in cleaned_pages:
        parts = root.findall("part")
        if len(parts) == max_count:
            reconciled.append(_rebuilt_score_with_parts(master_root, master_score_parts, parts))
            continue

        page_specs = _build_specs_for_page(root, sc)
        mapped_slots = match_staves_monotonic_dp(page_specs, master_specs)
        slot_to_part = {slot: parts[j] for j, slot in enumerate(mapped_slots)}
        ref_part = parts[0]

        ordered_parts: list[ET.Element] = []
        missing_names: list[str] = []
        for slot in range(max_count):
            if slot in slot_to_part:
                ordered_parts.append(slot_to_part[slot])
            else:
                missing_names.append(master_specs[slot].label.full_name)
                ordered_parts.append(_build_rest_part_for_page(f"P{slot + 1}", master_specs[slot], ref_part))

        log.info(
            "Page %d reconciled %d -> %d parts via monotonic DP (rest-filled %d missing/hidden: %s)",
            pno,
            len(parts),
            max_count,
            len(missing_names),
            ", ".join(missing_names),
        )
        reconciled.append(_rebuilt_score_with_parts(master_root, master_score_parts, ordered_parts))

    return reconciled, consolidated_labels


def dedup_boundary_attributes(
    measure: ET.Element,
    is_later_page_start: bool,
    state: dict[str, object],
) -> int:
    """Remove redundant divisions/key/time/clef on page boundaries while preserving genuine changes."""
    removed = 0
    for attrs in list(measure.findall("attributes")):
        if is_later_page_start:
            div_el = attrs.find("divisions")
            if div_el is not None and div_el.text and div_el.text.strip() == state["divisions"]:
                attrs.remove(div_el)
                removed += 1

            for key_el in list(attrs.findall("key")):
                fifths = key_el.findtext("fifths")
                if fifths is not None and fifths.strip() == state["fifths"]:
                    attrs.remove(key_el)
                    removed += 1

            for time_el in list(attrs.findall("time")):
                pair = (
                    (time_el.findtext("beats") or "").strip(),
                    (time_el.findtext("beat-type") or "").strip(),
                )
                if pair == state["time"]:
                    attrs.remove(time_el)
                    removed += 1

            for clef_el in list(attrs.findall("clef")):
                if clef_el.get("number") in (None, "1"):
                    cpair = (
                        (clef_el.findtext("sign") or "").strip(),
                        (clef_el.findtext("line") or "").strip(),
                    )
                    if cpair == state["clef"]:
                        attrs.remove(clef_el)
                        removed += 1

        if len(attrs) == 0:
            measure.remove(attrs)
            continue

        div_el = attrs.find("divisions")
        if div_el is not None and div_el.text:
            state["divisions"] = div_el.text.strip()
        key_el = attrs.find("key")
        if key_el is not None and key_el.findtext("fifths") is not None:
            state["fifths"] = key_el.findtext("fifths").strip()
        time_el = attrs.find("time")
        if time_el is not None:
            b = (time_el.findtext("beats") or "").strip()
            bt = (time_el.findtext("beat-type") or "").strip()
            if b and bt:
                state["time"] = (b, bt)
        for clef_el in attrs.findall("clef"):
            if clef_el.get("number") in (None, "1"):
                s = (clef_el.findtext("sign") or "").strip()
                ln = (clef_el.findtext("line") or "").strip()
                if s and ln:
                    state["clef"] = (s, ln)
    return removed


def compute_conductor_scaling(part_count: int) -> tuple[str, str, str]:
    """Return adaptive ``(millimeters, page_height_tenths, page_width_tenths)`` for A4 full scores."""
    if part_count >= 24:
        return "3.4", "3494", "2470"
    if part_count >= 16:
        mm = 4.2
    elif part_count >= 8:
        mm = 5.2
    else:
        mm = 6.5
    return f"{mm:.1f}", str(round(297 * 40 / mm)), str(round(210 * 40 / mm))


_SHARP_ORDER: tuple[str, ...] = ("F", "C", "G", "D", "A", "E", "B")
_FLAT_ORDER: tuple[str, ...] = ("B", "E", "A", "D", "G", "C", "F")


def _wrap_fifths(fifths: int) -> int:
    """Wrap a circle-of-fifths key value into ``[-6, +5]``."""
    return ((fifths + 6) % 12) - 6


def _fifths_equivalent(a: int, b: int) -> bool:
    """Return True if two key signatures represent enharmonic equivalents mod 12."""
    return (a - b) % 12 == 0


def _key_signature_alters(fifths: int) -> dict[str, int]:
    """Return the diatonic step alterations (``-1``, ``0``, ``+1``) implied by ``fifths``."""
    alters = {step: 0 for step in "CDEFGAB"}
    wf = _wrap_fifths(fifths)
    if wf > 0:
        for step in _SHARP_ORDER[: min(7, wf)]:
            alters[step] = 1
    elif wf < 0:
        for step in _FLAT_ORDER[: min(7, -wf)]:
            alters[step] = -1
    return alters


def _extract_initial_fifths(part: ET.Element) -> Optional[int]:
    for attr in part.findall("./measure/attributes"):
        txt = attr.findtext("key/fifths")
        if txt is not None and txt.strip():
            try:
                return int(txt.strip())
            except ValueError:
                return None
    return None


def _extract_initial_clef(part: ET.Element) -> Optional[tuple[str, str]]:
    for attr in part.findall("./measure/attributes"):
        for c_el in attr.findall("clef"):
            if c_el.get("number") in (None, "1"):
                sign = (c_el.findtext("sign") or "").strip()
                line = (c_el.findtext("line") or "").strip()
                if sign and line:
                    return (sign, line)
    return None


def detect_score_transposition_mode(
    roots: Sequence[ET.Element],
    labels: Sequence[InstrumentLabel],
) -> tuple[bool, Optional[int]]:
    """Detect whether a full score is written as a Transposed Score or Concert-Pitch (C) Score."""
    if not roots:
        return False, None

    first_parts = roots[0].findall("part")
    profiles: list[InstrumentPlaybackProfile] = []
    for idx in range(max(len(first_parts), len(labels))):
        lbl = labels[idx] if idx < len(labels) else InstrumentLabel(f"Part {idx + 1}", f"P{idx + 1}")
        clef = _extract_initial_clef(first_parts[idx]) if idx < len(first_parts) else None
        profiles.append(infer_instrument_profile(lbl.full_name, clef=clef))

    concert_votes_all: list[int] = []
    transposed_matches = 0
    concert_matches = 0

    for root in roots:
        c_fifths: list[int] = []
        tr_obs: list[tuple[int, int]] = []
        for idx, part in enumerate(root.findall("part")):
            lbl = labels[idx] if idx < len(labels) else InstrumentLabel(f"Part {idx + 1}", f"P{idx + 1}")
            if is_percussion_instrument(lbl.full_name):
                continue
            obs = _extract_initial_fifths(part)
            if obs is None:
                continue
            prof = (
                profiles[idx]
                if idx < len(profiles)
                else infer_instrument_profile(lbl.full_name, clef=_extract_initial_clef(part))
            )
            if prof.fifths_shift == 0:
                c_fifths.append(obs)
            else:
                tr_obs.append((obs, prof.fifths_shift))

        if c_fifths and tr_obs:
            c_freq = Counter(c_fifths)
            page_concert = max(c_freq.keys(), key=lambda f: (c_freq[f], -abs(f)))
            concert_votes_all.append(page_concert)
            for obs, sh in tr_obs:
                if _fifths_equivalent(obs, page_concert + sh):
                    transposed_matches += 1
                if _fifths_equivalent(obs, page_concert):
                    concert_matches += 1
        elif not c_fifths and len(tr_obs) >= 2 and len({sh for _, sh in tr_obs}) >= 2:
            tr_implied = Counter(_wrap_fifths(obs - sh) for obs, sh in tr_obs)
            c_implied = Counter(_wrap_fifths(obs) for obs, _ in tr_obs)
            top_tr_val, top_tr_cnt = tr_implied.most_common(1)[0]
            _top_c_val, top_c_cnt = c_implied.most_common(1)[0]
            transposed_matches += top_tr_cnt
            concert_matches += top_c_cnt
            if top_tr_cnt > top_c_cnt:
                concert_votes_all.append(top_tr_val)

    is_transposed = transposed_matches > concert_matches and transposed_matches >= 1
    concert_fifths = Counter(concert_votes_all).most_common(1)[0][0] if concert_votes_all else None
    if is_transposed:
        log.info(
            "Detected Transposed Full Score (concert fifths=%s; %d transposing match(es) vs %d concert match(es))",
            concert_fifths,
            transposed_matches,
            concert_matches,
        )
    else:
        log.info(
            "Detected Concert-Pitch (C) Full Score (concert fifths=%s; %d concert match(es) vs %d transposing match(es))",
            concert_fifths,
            concert_matches,
            transposed_matches,
        )
    return is_transposed, concert_fifths


def _reconcile_notes_for_key_change(part: ET.Element, old_fifths: int, new_fifths: int) -> int:
    old_alters = _key_signature_alters(old_fifths)
    new_alters = _key_signature_alters(new_fifths)
    updated = 0
    for mi, measure in enumerate(part.findall("measure")):
        if mi > 0 and measure.find("./attributes/key") is not None:
            break
        for note in measure.findall("note"):
            if note.find("rest") is not None or note.find("unpitched") is not None or note.find("accidental") is not None:
                continue
            pitch = note.find("pitch")
            if pitch is None:
                continue
            step = (pitch.findtext("step") or "").strip().upper()
            if step not in old_alters or old_alters[step] == new_alters[step]:
                continue
            alter_el = pitch.find("alter")
            if alter_el is not None and (alter_el.text or "").strip():
                try:
                    cur_alter = int(alter_el.text.strip())
                except ValueError:
                    continue
            else:
                cur_alter = 0
            if cur_alter != old_alters[step]:
                continue

            new_val = new_alters[step]
            if new_val == 0:
                if alter_el is not None:
                    pitch.remove(alter_el)
            else:
                if alter_el is None:
                    step_el = pitch.find("step")
                    ins_idx = list(pitch).index(step_el) + 1 if step_el is not None else 1
                    alter_el = ET.Element("alter")
                    pitch.insert(ins_idx, alter_el)
                alter_el.text = str(new_val)
            updated += 1
    return updated


def harmonize_page_key_signatures(
    root: ET.Element,
    labels: Sequence[InstrumentLabel],
    *,
    is_transposed_score: bool,
    page_num: int,
) -> int:
    """Repair isolated single-staff OMR key-signature misreads using ensemble consensus."""
    parts = root.findall("part")
    entries: list[tuple[ET.Element, InstrumentLabel, int, int, int, ET.Element]] = []

    for idx, part in enumerate(parts):
        lbl = labels[idx] if idx < len(labels) else InstrumentLabel(f"Part {idx + 1}", f"P{idx + 1}")
        if is_percussion_instrument(lbl.full_name):
            continue
        m1 = part.find("measure")
        if m1 is None:
            continue
        fifths_el = m1.find("./attributes/key/fifths")
        if fifths_el is None or not (fifths_el.text or "").strip():
            continue
        try:
            obs = int(fifths_el.text.strip())
        except ValueError:
            continue
        prof = infer_instrument_profile(lbl.full_name, clef=_extract_initial_clef(part))
        shift = prof.fifths_shift if is_transposed_score else 0
        implied_concert = _wrap_fifths(obs - shift)
        entries.append((part, lbl, obs, shift, implied_concert, fifths_el))

    if len(entries) < 3:
        return 0

    concert_counts = Counter(e[4] for e in entries)
    top_concert, top_count = concert_counts.most_common(1)[0]
    if top_count < 3 or top_count < 0.70 * len(entries):
        return 0

    repaired = 0
    for part, lbl, obs, shift, _implied, fifths_el in entries:
        if _fifths_equivalent(obs, top_concert + shift):
            continue
        target_fifths = _wrap_fifths(top_concert + shift)
        fifths_el.text = str(target_fifths)
        notes_updated = _reconcile_notes_for_key_change(part, obs, target_fifths)
        repaired += 1
        log.info(
            "Page %d %s (%s): harmonized OMR key-signature outlier fifths %d -> %d (updated %d implied note alteration(s))",
            page_num,
            part.get("id", "?"),
            lbl.full_name,
            obs,
            target_fifths,
            notes_updated,
        )
    return repaired


def _insert_transpose_element(attrs: ET.Element, profile: InstrumentPlaybackProfile) -> None:
    """Insert a valid MusicXML 4.0 ``<transpose>`` element inside ``<attributes>``."""
    for old_tr in list(attrs.findall("transpose")):
        attrs.remove(old_tr)

    tr_el = ET.Element("transpose")
    ET.SubElement(tr_el, "diatonic").text = str(profile.diatonic)
    ET.SubElement(tr_el, "chromatic").text = str(profile.chromatic)
    if profile.octave_change != 0:
        ET.SubElement(tr_el, "octave-change").text = str(profile.octave_change)

    insert_idx = len(attrs)
    for idx, child in enumerate(list(attrs)):
        if child.tag in ("directive", "measure-style"):
            insert_idx = idx
            break
    attrs.insert(insert_idx, tr_el)


def _validate_transpose_element(tr: ET.Element, part_id: str) -> None:
    chrom_txt = (tr.findtext("chromatic") or "").strip()
    try:
        chrom_val = int(chrom_txt)
    except ValueError as exc:
        raise MergeError(f"Part {part_id} <transpose> has invalid/missing <chromatic>: {chrom_txt!r}") from exc
    if not (-24 <= chrom_val <= 24):
        raise MergeError(f"Part {part_id} <transpose> chromatic {chrom_val} out of range [-24, 24]")

    diat_el = tr.find("diatonic")
    if diat_el is not None:
        diat_txt = (diat_el.text or "").strip()
        try:
            diat_val = int(diat_txt)
        except ValueError as exc:
            raise MergeError(f"Part {part_id} <transpose> has invalid <diatonic>: {diat_txt!r}") from exc
        if not (-14 <= diat_val <= 14):
            raise MergeError(f"Part {part_id} <transpose> diatonic {diat_val} out of range [-14, 14]")

    oct_el = tr.find("octave-change")
    if oct_el is not None:
        oct_txt = (oct_el.text or "").strip()
        try:
            oct_val = int(oct_txt)
        except ValueError as exc:
            raise MergeError(f"Part {part_id} <transpose> has invalid <octave-change>: {oct_txt!r}") from exc
        if not (-3 <= oct_val <= 3):
            raise MergeError(f"Part {part_id} <transpose> octave-change {oct_val} out of range [-3, 3]")


def apply_conductor_metadata(
    root: ET.Element,
    title: str,
    labels: Sequence[InstrumentLabel],
    *,
    keep_line_breaks: bool = False,
    is_transposed_score: Optional[bool] = None,
) -> ET.Element:
    """Apply conductor score title, adaptive scaling, instrument names/sounds/transpositions, and MIDI channels."""
    for xp in ("movement-title",):
        el = root.find(xp)
        if el is not None:
            root.remove(el)
    work = root.find("work")
    if work is None:
        work = ET.Element("work")
        root.insert(0, work)
    wt = work.find("work-title")
    if wt is None:
        wt = ET.SubElement(work, "work-title")
    wt.text = title

    part_count = len(root.findall("part"))
    mm, page_h, page_w = compute_conductor_scaling(part_count)
    old_def = root.find("defaults")
    if old_def is not None:
        root.remove(old_def)
    defaults = ET.Element("defaults")
    scaling = ET.SubElement(defaults, "scaling")
    ET.SubElement(scaling, "millimeters").text = mm
    ET.SubElement(scaling, "tenths").text = "40"
    pl = ET.SubElement(defaults, "page-layout")
    ET.SubElement(pl, "page-height").text = page_h
    ET.SubElement(pl, "page-width").text = page_w
    pm = ET.SubElement(pl, "page-margins", type="both")
    ET.SubElement(pm, "left-margin").text = "140"
    ET.SubElement(pm, "right-margin").text = "120"
    ET.SubElement(pm, "top-margin").text = "120"
    ET.SubElement(pm, "bottom-margin").text = "120"

    ident = root.find("identification")
    insert_pos = list(root).index(ident) + 1 if ident is not None else 1
    root.insert(insert_pos, defaults)

    if not keep_line_breaks:
        strip_layout_breaks(root)

    # Clear any stray raw <transpose> elements before emitting canonical Measure-1 transpositions
    removed_transpose = 0
    for parent in root.iter():
        for tr in list(parent.findall("transpose")):
            parent.remove(tr)
            removed_transpose += 1
    if removed_transpose:
        log.debug("Cleared %d raw <transpose> element(s) prior to canonical emission", removed_transpose)

    if is_transposed_score is None:
        is_transposed_score, _ = detect_score_transposition_mode([root], labels)

    parts_by_id = {p.get("id", ""): p for p in root.findall("part")}
    all_parts = root.findall("part")
    pitched_i = 0
    transposing_parts = 0

    for idx, sp in enumerate(root.findall("./part-list/score-part")):
        part_id = sp.get("id", f"P{idx + 1}")
        part_el = parts_by_id.get(part_id)
        if part_el is None and idx < len(all_parts):
            part_el = all_parts[idx]
        if idx < len(labels):
            inst_name, inst_abbr = labels[idx].full_name, labels[idx].abbreviation
        else:
            inst_name, inst_abbr = f"Part {idx + 1}", f"P{idx + 1}"

        initial_clef = _extract_initial_clef(part_el) if part_el is not None else None
        profile, is_perc = set_score_part_instrument(
            sp,
            inst_name,
            abbreviation=inst_abbr,
            clef=initial_clef,
            pitched_channel_index=pitched_i,
        )
        if is_perc and part_el is not None:
            normalize_unpitched_percussion_part(sp, part_el, inst_name)
        if not is_perc:
            pitched_i += 1

        should_transpose = (is_transposed_score and profile.is_transposing) or (
            not is_transposed_score and profile.fifths_shift == 0 and profile.octave_change != 0
        )
        if should_transpose and part_el is not None:
            m1 = part_el.find("measure")
            if m1 is not None:
                attrs = m1.find("attributes")
                if attrs is None:
                    attrs = ET.Element("attributes")
                    m1.insert(0, attrs)
                _insert_transpose_element(attrs, profile)
                transposing_parts += 1
                log.debug(
                    "Set playback transpose for %s (%s): diatonic=%d chromatic=%d octave=%d sound=%s",
                    part_id,
                    inst_name,
                    profile.diatonic,
                    profile.chromatic,
                    profile.octave_change,
                    profile.sound_id,
                )

    log.info(
        "Applied conductor metadata (%d parts, scaling=%smm, page=%sx%s, transposed_score=%s, transposing_parts=%d)",
        part_count,
        mm,
        page_w,
        page_h,
        is_transposed_score,
        transposing_parts,
    )
    return root


def verify_fullscore_invariants(root: ET.Element) -> dict[str, int]:
    """Verify all 6 MusicXML full-score invariants; raise MergeError on any violation."""
    parts = root.findall("part")
    score_parts = root.findall("./part-list/score-part")
    if not parts:
        raise MergeError("Full score has no <part> elements")
    if len(score_parts) != len(parts):
        raise MergeError(f"part-list count {len(score_parts)} != part count {len(parts)}")

    mcounts = {len(p.findall("measure")) for p in parts}
    if len(mcounts) != 1:
        raise MergeError(f"Inconsistent part measure counts: {sorted(mcounts)}")
    total_m = next(iter(mcounts))
    if total_m <= 0:
        raise MergeError("Full score parts contain 0 measures")

    for p in parts:
        for m in p.findall("measure"):
            times = m.findall("./attributes/time")
            if len(times) > 1:
                raise MergeError(
                    f"Measure {m.get('number')} in {p.get('id')} has {len(times)} <time> elements"
                )

    for mi in range(total_m):
        declared: list[tuple[str, str]] = []
        for p in parts:
            t_el = p.findall("measure")[mi].find("./attributes/time")
            if t_el is not None:
                declared.append(
                    ((t_el.findtext("beats") or "").strip(), (t_el.findtext("beat-type") or "").strip())
                )
        if mi == 0 and len(declared) != len(parts):
            raise MergeError(f"Measure 1: missing time signature on some parts ({len(declared)}/{len(parts)})")
        if len(declared) not in (0, len(parts)):
            raise MergeError(
                f"Measure {mi + 1}: partial time signature across parts ({len(declared)}/{len(parts)}): {Counter(declared)}"
            )
        if len(set(declared)) > 1:
            raise MergeError(f"Measure {mi + 1}: conflicting time signatures: {Counter(declared)}")

    allowed_transpose_ids: set[int] = set()
    transposing_parts = 0
    for p in parts:
        m1_transposes: list[ET.Element] = []
        for m1 in p.findall("./measure[@number='1']"):
            m1_transposes.extend(m1.findall("./attributes/transpose"))
        if len(m1_transposes) > 1:
            raise MergeError(f"Part {p.get('id')} has {len(m1_transposes)} <transpose> elements in Measure 1")
        for tr in m1_transposes:
            allowed_transpose_ids.add(id(tr))
            transposing_parts += 1
            _validate_transpose_element(tr, p.get("id") or "?")

    for tr in root.findall(".//transpose"):
        if id(tr) not in allowed_transpose_ids:
            raise MergeError("Unexpected <transpose> element outside Measure 1 attributes")

    for ch_el in root.findall(".//midi-channel"):
        txt = (ch_el.text or "").strip()
        try:
            ch_val = int(txt)
        except ValueError as exc:
            raise MergeError(f"Invalid non-integer MIDI channel {txt!r}") from exc
        if not (1 <= ch_val <= 16):
            raise MergeError(f"Out-of-range MIDI channel {ch_val} (must be 1..16)")

    note_counts = [sum(1 for n in p.findall(".//note") if n.find("rest") is None) for p in parts]
    summary = {
        "parts": len(parts),
        "measures": total_m,
        "non_empty_parts": sum(1 for c in note_counts if c > 0),
        "total_notes": sum(note_counts),
        "transposing_parts": transposing_parts,
    }
    log.info(
        "Verified all 6 full-score XML invariants (%d parts x %d measures, %d notes across %d non-empty parts, %d transposing)",
        summary["parts"],
        summary["measures"],
        summary["total_notes"],
        summary["non_empty_parts"],
        summary["transposing_parts"],
    )
    return summary


def merge_fullscore_pages(
    pages: Sequence[tuple[int, ET.Element, Optional[PageSidecar]]],
    title: str,
    *,
    keep_line_breaks: bool = False,
) -> ET.Element:
    """Align per page, reconcile varying staff counts, merge sequentially, and verify invariants."""
    if not pages:
        raise MergeError("No full-score pages to merge")

    aligned_pages: list[tuple[int, ET.Element, Optional[PageSidecar]]] = []
    inherited_ts: Optional[tuple[str, str]] = None
    for pno, root, sidecar in pages:
        aligned_root, ending_ts = align_page_score(copy.deepcopy(root), pno, inherited_time_sig=inherited_ts)
        inherited_ts = ending_ts
        aligned_pages.append((pno, aligned_root, sidecar))

    reconciled_roots, labels = reconcile_pages_to_master(aligned_pages)
    is_transposed_score, _ = detect_score_transposition_mode(reconciled_roots, labels)

    page_nums = [pno for pno, _root, _sc in aligned_pages]
    harmonized_parts = 0
    for pno, rec_root in zip(page_nums, reconciled_roots):
        harmonized_parts += harmonize_page_key_signatures(
            rec_root,
            labels,
            is_transposed_score=is_transposed_score,
            page_num=pno,
        )
    if harmonized_parts:
        log.info("Harmonized %d OMR key-signature outlier part(s) across full score", harmonized_parts)

    merged = copy.deepcopy(reconciled_roots[0])
    base_parts = merged.findall("part")
    for p in base_parts:
        for m in list(p.findall("measure")):
            p.remove(m)

    counters = [0] * len(base_parts)
    part_states: list[dict[str, object]] = [
        {"divisions": None, "fifths": None, "time": None, "clef": None}
        for _ in base_parts
    ]
    dedup_removed = 0
    suppressed_rest_clefs = 0
    page_start_measures: list[
        tuple[int, int, Optional[TempoMarking], tuple[PositionedTempo, ...], tuple[PositionedRehearsal, ...]]
    ] = []
    for page_idx, (root, (_pno, _aroot, sc)) in enumerate(zip(reconciled_roots, aligned_pages), start=1):
        start_idx = counters[0]
        parts = root.findall("part")
        page_m_count = len(parts[0].findall("measure")) if parts else 0
        page_start_measures.append(
            (
                start_idx,
                page_m_count,
                sc.tempo if sc is not None else None,
                sc.positioned_tempos if sc is not None else (),
                sc.rehearsal_marks if sc is not None else (),
            )
        )
        for pi, part in enumerate(parts):
            part_measures = part.findall("measure")
            is_perc_part = pi < len(labels) and is_percussion_instrument(labels[pi].full_name)
            suppress_rest_clef = (
                page_idx > 1
                and not is_perc_part
                and part_states[pi]["clef"] is not None
                and bool(part_measures)
                and all(_is_rest_only_measure(pm) for pm in part_measures)
            )
            for mi, m in enumerate(part_measures):
                mc = copy.deepcopy(m)
                if suppress_rest_clef:
                    for attrs in list(mc.findall("attributes")):
                        for old_c in list(attrs.findall("clef")):
                            attrs.remove(old_c)
                            suppressed_rest_clefs += 1
                        if len(attrs) == 0:
                            mc.remove(attrs)
                dedup_removed += dedup_boundary_attributes(
                    mc,
                    is_later_page_start=(page_idx > 1 and mi == 0),
                    state=part_states[pi],
                )
                counters[pi] += 1
                mc.set("number", str(counters[pi]))
                base_parts[pi].append(mc)

    if suppressed_rest_clefs:
        log.info("Suppressed %d spurious clef change(s) on all-rest pitched pages", suppressed_rest_clefs)
    log.info("Deduplicated %d redundant page-boundary attributes (divisions/key/time/clef)", dedup_removed)
    apply_conductor_metadata(
        merged,
        title,
        labels,
        keep_line_breaks=keep_line_breaks,
        is_transposed_score=is_transposed_score,
    )
    add_courtesy_accidentals(merged)
    last_tempo_key: Optional[tuple[str, Optional[int], str, bool]] = None
    for start_idx, page_m_count, p_tempo, p_pos_tempos, p_marks in page_start_measures:
        if p_pos_tempos:
            _cnt, last_tempo_key = map_range_tempos_to_measures(
                merged,
                p_pos_tempos,
                start_index=start_idx,
                measure_count=page_m_count,
                last_tempo_key=last_tempo_key,
            )
        elif p_tempo is not None:
            t_key = (p_tempo.text.lower(), p_tempo.bpm, p_tempo.beat_unit, p_tempo.beat_unit_dot)
            if t_key != last_tempo_key and apply_tempo_marking(merged, p_tempo, measure_index=start_idx):
                last_tempo_key = t_key
        if p_marks:
            map_range_rehearsal_marks_to_measures(
                merged,
                p_marks,
                start_index=start_idx,
                measure_count=page_m_count,
            )
    verify_fullscore_invariants(merged)
    return merged


def merge_fullscore_files(
    files: Sequence[Path],
    out: Path,
    title: Optional[str] = None,
    options: Optional[ConversionOptions] = None,
) -> Path:
    """Read per-page MusicXML + sidecar files, merge in Full Score Mode, and write ``out``."""
    ordered_files = sorted(Path(f) for f in files)
    if not ordered_files:
        raise MergeError("No full-score MusicXML files to merge")

    pages: list[tuple[int, ET.Element, Optional[PageSidecar]]] = []
    for idx, xml_path in enumerate(ordered_files, start=1):
        root = read_score(xml_path)
        sidecar = read_page_sidecar(xml_path)
        pno = sidecar.page_number if sidecar is not None else idx
        pages.append((pno, root, sidecar))

    keep_breaks = bool(options.keep_line_breaks) if options is not None else False
    score_title = title or Path(out).stem
    merged = merge_fullscore_pages(pages, score_title, keep_line_breaks=keep_breaks)
    return write_musicxml(merged, Path(out))
