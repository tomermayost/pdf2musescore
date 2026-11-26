"""Read MusicXML (.mxl / .musicxml), sanitize tuplets/tremolos, merge pages, and apply score post-processing."""

from __future__ import annotations

import copy
import logging
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from .options import ConversionOptions

if TYPE_CHECKING:  # pragma: no cover
    from .fullscore.instruments import InstrumentPlaybackProfile

log = logging.getLogger(__name__)

_PITCHED_MIDI_CHANNELS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16)


class MergeError(ValueError):
    pass


_STEP_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SEMITONE_TO_STEP_ALTER: dict[int, tuple[str, int]] = {
    0: ("C", 0),
    1: ("C", 1),
    2: ("D", 0),
    3: ("E", -1),
    4: ("E", 0),
    5: ("F", 0),
    6: ("F", 1),
    7: ("G", 0),
    8: ("A", -1),
    9: ("A", 0),
    10: ("B", -1),
    11: ("B", 0),
}
_CLEF_MAP: dict[str, tuple[str, str]] = {
    "treble": ("G", "2"),
    "bass": ("F", "4"),
    "alto": ("C", "3"),
    "tenor": ("C", "4"),
}
_SEMITONE_FIFTHS_SHIFT = (0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5)


def sanitize_slurs(root: ET.Element) -> ET.Element:
    """Pair slurs cleanly per (staff, voice) and remove orphan stops or unclosed starts for MuseScore 4."""
    for part in root.findall("part"):
        open_slur: dict[tuple[str, str], tuple[ET.Element, ET.Element]] = {}
        for measure in part.findall("measure"):
            for note in measure.findall("note"):
                key = (note.findtext("staff", "1"), note.findtext("voice", "1"))
                for nt in note.findall("notations"):
                    sls = list(nt.findall("slur"))
                    sls.sort(key=lambda s: 0 if s.get("type") == "stop" else 1)
                    if len(sls) > 1:
                        for idx, sl in zip(
                            [i for i, ch in enumerate(nt) if ch.tag == "slur"], sls
                        ):
                            nt[idx] = sl
                    for sl in sls:
                        sl_type = sl.get("type")
                        if sl_type == "stop":
                            if key in open_slur:
                                open_slur.pop(key)
                                sl.set("number", "1")
                            else:
                                nt.remove(sl)
                        elif sl_type == "start":
                            if key in open_slur:
                                prev_nt, prev_sl = open_slur.pop(key)
                                prev_nt.remove(prev_sl)
                            sl.set("number", "1")
                            open_slur[key] = (nt, sl)
        for prev_nt, prev_sl in open_slur.values():
            prev_nt.remove(prev_sl)
    return root


def sanitize_score(root: ET.Element) -> ET.Element:
    """Fix unclosed tuplets, invalid tremolos, and unpaired slurs so MuseScore 4 imports cleanly."""
    for orn in root.findall(".//ornaments"):
        for trem in list(orn.findall("tremolo")):
            orn.remove(trem)

    for measure in root.findall(".//measure"):
        run: list[ET.Element] = []
        for note in measure.findall("note"):
            for nt in note.findall("notations"):
                for tup in list(nt.findall("tuplet")):
                    nt.remove(tup)
            if note.find("time-modification") is not None:
                run.append(note)
            else:
                if run:
                    _close_tuplet_run(run)
                    run = []
        if run:
            _close_tuplet_run(run)
    return sanitize_slurs(root)


def _close_tuplet_run(run: list[ET.Element]) -> None:
    for i in range(0, len(run), 3):
        grp = run[i : i + 3]
        if len(grp) == 3:
            nt0 = grp[0].find("notations")
            if nt0 is None:
                nt0 = ET.SubElement(grp[0], "notations")
            ET.SubElement(nt0, "tuplet", type="start")
            nt2 = grp[2].find("notations")
            if nt2 is None:
                nt2 = ET.SubElement(grp[2], "notations")
            ET.SubElement(nt2, "tuplet", type="stop")
        else:
            for note in grp:
                tm = note.find("time-modification")
                if tm is not None:
                    note.remove(tm)
                dur = note.find("duration")
                if dur is not None and dur.text == "8":
                    dur.text = "12"


def strip_layout_breaks(root: ET.Element) -> ET.Element:
    """Remove forced <print new-system=... / new-page=...> breaks so MuseScore flows lines cleanly."""
    for measure in root.findall(".//measure"):
        for pr in list(measure.findall("print")):
            pr.attrib.pop("new-system", None)
            pr.attrib.pop("new-page", None)
            if not pr.attrib and len(pr) == 0:
                measure.remove(pr)
    return root


def read_score(path: Path) -> ET.Element:
    """Return the sanitized <score-partwise> root of a .mxl or .musicxml/.xml file."""
    path = Path(path)
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as z:
            container = ET.fromstring(z.read("META-INF/container.xml"))
            rootfile = container.find(".//{*}rootfile")
            if rootfile is None:
                raise MergeError(f"{path}: no rootfile in container.xml")
            root = ET.fromstring(z.read(rootfile.get("full-path")))
    else:
        root = ET.parse(path).getroot()
    if root.tag != "score-partwise":
        raise MergeError(f"{path}: unsupported root <{root.tag}> (only score-partwise)")
    return sanitize_score(root)


def _movement_title(root: ET.Element, index: int) -> str:
    for xp in ("movement-title", "work/work-title"):
        el = root.find(xp)
        if el is not None and (el.text or "").strip():
            return el.text.strip()
    return f"Page {index}"


def _mark_movement_start(measure: ET.Element, title: str, new_page: bool) -> None:
    pos = 0
    if new_page:
        measure.insert(0, ET.Element("print", {"new-page": "yes"}))
        pos = 1
    for i, child in enumerate(list(measure)):
        if child.tag == "attributes":
            pos = i + 1
            break
    direction = ET.Element("direction", {"placement": "above"})
    words = ET.SubElement(ET.SubElement(direction, "direction-type"), "words",
                          {"font-weight": "bold", "font-size": "14"})
    words.text = title
    measure.insert(pos, direction)


def transpose_score(root: ET.Element, semitones: int) -> ET.Element:
    """Shift all <pitch> elements and <key><fifths> signatures by ``semitones``."""
    if semitones == 0:
        return root

    for pitch in root.findall(".//note/pitch"):
        step_el = pitch.find("step")
        oct_el = pitch.find("octave")
        if step_el is None or oct_el is None or not step_el.text or not oct_el.text:
            continue
        step = step_el.text.strip().upper()
        if step not in _STEP_TO_SEMITONE:
            continue
        octave = int(oct_el.text.strip())
        alter_el = pitch.find("alter")
        alter = int(round(float(alter_el.text.strip()))) if (alter_el is not None and alter_el.text) else 0

        midi = (octave + 1) * 12 + _STEP_TO_SEMITONE[step] + alter + semitones
        new_octave = (midi // 12) - 1
        new_step, new_alter = _SEMITONE_TO_STEP_ALTER[midi % 12]
        step_el.text = new_step
        oct_el.text = str(new_octave)
        if new_alter == 0:
            if alter_el is not None:
                pitch.remove(alter_el)
        else:
            if alter_el is None:
                alter_el = ET.Element("alter")
                # MusicXML order inside <pitch>: step, alter, octave
                idx = list(pitch).index(step_el) + 1
                pitch.insert(idx, alter_el)
            alter_el.text = str(new_alter)

    shift_fifths = _SEMITONE_FIFTHS_SHIFT[semitones % 12]
    if shift_fifths != 0:
        for fifths_el in root.findall(".//attributes/key/fifths"):
            if fifths_el.text and fifths_el.text.strip().lstrip("+-").isdigit():
                val = int(fifths_el.text.strip()) + shift_fifths
                while val > 6:
                    val -= 12
                while val < -6:
                    val += 12
                fifths_el.text = str(val)
    return root


def apply_clef_override(root: ET.Element, clef_override: str) -> ET.Element:
    """Force the primary staff clef across the score ('treble', 'bass', 'alto', 'tenor')."""
    key = (clef_override or "auto").strip().lower()
    if key == "auto" or key not in _CLEF_MAP:
        return root
    sign_val, line_val = _CLEF_MAP[key]
    clefs = root.findall(".//attributes/clef")
    if not clefs:
        first_attr = root.find(".//measure/attributes")
        if first_attr is not None:
            c = ET.SubElement(first_attr, "clef")
            ET.SubElement(c, "sign").text = sign_val
            ET.SubElement(c, "line").text = line_val
        return root
    for clef in clefs:
        num = clef.get("number")
        if num not in (None, "1"):
            continue
        sign_el = clef.find("sign")
        if sign_el is None:
            sign_el = ET.SubElement(clef, "sign")
        sign_el.text = sign_val
        line_el = clef.find("line")
        if line_el is None:
            line_el = ET.SubElement(clef, "line")
        line_el.text = line_val
    return root


def extract_part_initial_clef(part: Optional[ET.Element]) -> Optional[tuple[str, str]]:
    """Return the first primary ``(sign, line)`` clef declared in ``part``, or ``None``."""
    if part is None:
        return None
    for attr in part.findall("./measure/attributes"):
        for c_el in attr.findall("clef"):
            if c_el.get("number") in (None, "1"):
                sign = (c_el.findtext("sign") or "").strip()
                line = (c_el.findtext("line") or "").strip()
                if sign and line:
                    return (sign, line)
    return None


def _extract_first_clef(root: ET.Element, part_id: str) -> Optional[tuple[str, str]]:
    part = root.find(f"./part[@id='{part_id}']")
    return extract_part_initial_clef(part)


def set_score_part_instrument(
    score_part: ET.Element,
    instrument_name: str,
    *,
    abbreviation: Optional[str] = None,
    clef: Optional[tuple[str, str]] = None,
    pitched_channel_index: int = 0,
) -> tuple[InstrumentPlaybackProfile, bool]:
    """Write consistent ``<part-name>``, ``<part-abbreviation>``, ``<score-instrument>``, and ``<midi-instrument>``."""
    from .fullscore.instruments import (
        abbreviate_instrument_name,
        infer_instrument_profile,
        is_percussion_instrument,
    )

    part_id = score_part.get("id", "P1")
    inst = instrument_name.strip()
    profile = infer_instrument_profile(inst, clef=clef)
    is_perc = is_percussion_instrument(inst)
    abbr = abbreviation.strip() if (abbreviation and abbreviation.strip()) else abbreviate_instrument_name(inst)

    pn = score_part.find("part-name")
    if pn is None:
        pn = ET.Element("part-name")
        score_part.insert(0, pn)
    pn.text = inst

    for old_pa in list(score_part.findall("part-abbreviation")):
        score_part.remove(old_pa)
    pa = ET.Element("part-abbreviation")
    pa.text = abbr
    score_part.insert(list(score_part).index(pn) + 1, pa)

    si = score_part.find("score-instrument")
    if si is None:
        si = ET.SubElement(score_part, "score-instrument", {"id": f"{part_id}-I1"})
    iname = si.find("instrument-name")
    if iname is None:
        iname = ET.SubElement(si, "instrument-name")
    iname.text = inst
    isound = si.find("instrument-sound")
    if isound is None:
        isound = ET.SubElement(si, "instrument-sound")
    isound.text = profile.sound_id

    mi_el = score_part.find("midi-instrument")
    if mi_el is None:
        mi_el = ET.SubElement(score_part, "midi-instrument", {"id": f"{part_id}-I1"})
    ch = mi_el.find("midi-channel")
    if ch is None:
        ch = ET.SubElement(mi_el, "midi-channel")
    if is_perc:
        ch.text = "10"
    else:
        ch.text = str(_PITCHED_MIDI_CHANNELS[pitched_channel_index % len(_PITCHED_MIDI_CHANNELS)])

    prog_el = mi_el.find("midi-program")
    if is_perc or profile.midi_program <= 0:
        if prog_el is not None:
            mi_el.remove(prog_el)
    else:
        if prog_el is None:
            prog_el = ET.SubElement(mi_el, "midi-program")
        prog_el.text = str(profile.midi_program)

    return profile, is_perc


def is_homr_default_voice(score_part: ET.Element) -> bool:
    """Return True if ``score_part`` is homr's uncustomized single-staff ``Voice`` placeholder."""
    part_name = (score_part.findtext("part-name") or "").strip().lower()
    if part_name != "voice":
        return False
    sound_id = (score_part.findtext("./score-instrument/instrument-sound") or "").strip().lower()
    midi_prog = (score_part.findtext("./midi-instrument/midi-program") or "").strip()
    return sound_id in ("", "voice") and midi_prog in ("", "54")


def normalize_default_instruments(root: ET.Element) -> int:
    """Replace any uncustomized homr ``Voice`` placeholder parts with ``Piano`` (program 1)."""
    from .fullscore.instruments import is_percussion_instrument

    normalized = 0
    pitched_i = 0
    for sp in root.findall("./part-list/score-part"):
        part_id = sp.get("id", "")
        if is_homr_default_voice(sp):
            clef = _extract_first_clef(root, part_id) if part_id else None
            set_score_part_instrument(
                sp,
                "Piano",
                abbreviation="Pno.",
                clef=clef,
                pitched_channel_index=pitched_i,
            )
            normalized += 1
            pitched_i += 1
        else:
            pname = (sp.findtext("part-name") or "").strip()
            if not is_percussion_instrument(pname):
                pitched_i += 1
    if normalized:
        log.info("Replaced %d uncustomized default 'Voice' part(s) with 'Piano' (keyboard.piano)", normalized)
    return normalized


def apply_instrument_override(root: ET.Element, instrument_name: str) -> ET.Element:
    """Set ``<part-name>``, ``<part-abbreviation>``, ``<instrument-name>``, ``<instrument-sound>``, and MIDI."""
    inst = (instrument_name or "auto").strip()
    if not inst or inst.lower() == "auto":
        return root
    pitched_i = 0
    for idx, sp in enumerate(root.findall("./part-list/score-part")):
        part_id = sp.get("id", f"P{idx + 1}")
        clef = _extract_first_clef(root, part_id)
        profile, is_perc = set_score_part_instrument(
            sp,
            inst,
            clef=clef,
            pitched_channel_index=pitched_i,
        )
        if not is_perc:
            pitched_i += 1
        log.info(
            "Applied instrument override on %s: name=%r sound=%s midi_program=%s",
            part_id,
            inst,
            profile.sound_id,
            "perc-ch10" if is_perc else profile.midi_program,
        )
    return root


def apply_score_options(root: ET.Element, options: Optional[ConversionOptions]) -> ET.Element:
    """Apply Section 3 post-processing options (transpose, clef, instrument, layout) to ``root``."""
    if options is None:
        normalize_default_instruments(root)
        return root
    if options.transpose_semitones != 0:
        transpose_score(root, options.transpose_semitones)
    if options.clef_override != "auto":
        apply_clef_override(root, options.clef_override)
    if options.instrument_name.lower() != "auto":
        apply_instrument_override(root, options.instrument_name)
    else:
        normalize_default_instruments(root)
    if not options.keep_line_breaks:
        strip_layout_breaks(root)
    return root


def merge_scores(roots: Sequence[ET.Element], title: str | None = None,
                 options: Optional[ConversionOptions] = None) -> ET.Element:
    """Concatenate pages/movements part-by-part into a single score-partwise tree."""
    if not roots:
        raise MergeError("No scores to merge")
    merged = copy.deepcopy(roots[0])
    base_parts = merged.findall("part")
    if not base_parts:
        raise MergeError("First score has no parts")

    for p in base_parts:
        for m in p.findall("measure"):
            p.remove(m)

    keep_breaks = bool(options.keep_line_breaks) if options is not None else False
    counters = [0] * len(base_parts)
    for idx, root in enumerate(roots, start=1):
        mtitle = _movement_title(root, idx)
        parts = root.findall("part")
        if len(parts) != len(base_parts):
            log.warning("Section %d (%s) has %d part(s), expected %d - extra parts dropped, missing left empty",
                        idx, mtitle, len(parts), len(base_parts))
        for pi, part in enumerate(parts[:len(base_parts)]):
            measures = part.findall("measure")
            for mi, m in enumerate(measures):
                m = copy.deepcopy(m)
                counters[pi] += 1
                m.set("number", str(counters[pi]))
                if mi == 0 and pi == 0:
                    _mark_movement_start(m, mtitle, new_page=(keep_breaks and idx > 1))
                base_parts[pi].append(m)
        log.info("Merged section %d: %s (%d measures)", idx, mtitle,
                 len(parts[0].findall("measure")) if parts else 0)

    if title:
        for xp in ("movement-title",):
            el = merged.find(xp)
            if el is not None:
                merged.remove(el)
        work = merged.find("work")
        if work is None:
            work = ET.Element("work")
            merged.insert(0, work)
        wt = work.find("work-title")
        if wt is None:
            wt = ET.SubElement(work, "work-title")
        wt.text = title

    apply_score_options(merged, options)
    if options is None:
        strip_layout_breaks(merged)
    return merged


def write_musicxml(root: ET.Element, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
    log.info("Wrote MusicXML %s", path)
    return path


def merge_files(files: Sequence[Path], out: Path, title: str | None = None,
                options: Optional[ConversionOptions] = None) -> Path:
    files = sorted(Path(f) for f in files)
    log.info("Merging %d file(s) into %s", len(files), out)
    return write_musicxml(merge_scores([read_score(f) for f in files], title, options=options), out)
