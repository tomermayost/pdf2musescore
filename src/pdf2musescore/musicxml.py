"""Read MusicXML (.mxl / .musicxml), sanitize tuplets/tremolos, merge pages, and apply score post-processing."""

from __future__ import annotations

import copy
import logging
import zipfile
import xml.etree.ElementTree as ET
from fractions import Fraction
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
_SHARP_ORDER: tuple[str, ...] = ("F", "C", "G", "D", "A", "E", "B")
_FLAT_ORDER: tuple[str, ...] = ("B", "E", "A", "D", "G", "C", "F")
_ALTER_TO_ACCIDENTAL: dict[int, str] = {
    -2: "flat-flat",
    -1: "flat",
    0: "natural",
    1: "sharp",
    2: "double-sharp",
}
_NOTE_AFTER_ACCIDENTAL = frozenset(
    {"time-modification", "stem", "notehead", "notehead-text", "staff", "beam", "notations", "lyric", "play"}
)
_ATTR_BEFORE_TIME = frozenset({"footnote", "level", "divisions", "key"})
_ATTR_AFTER_TRANSPOSE = frozenset({"directive", "measure-style"})
_SECTION_BAR_STYLES = frozenset({"light-light", "heavy-heavy", "light-heavy"})
TYPE_TO_QUARTERS: dict[str, Fraction] = {
    "breve": Fraction(8),
    "whole": Fraction(4),
    "half": Fraction(2),
    "quarter": Fraction(1),
    "eighth": Fraction(1, 2),
    "16th": Fraction(1, 4),
    "32nd": Fraction(1, 8),
    "64th": Fraction(1, 16),
}
_STANDARD_DYNAMIC_TAGS = frozenset(
    {
        "p", "pp", "ppp", "pppp", "ppppp", "pppppp",
        "f", "ff", "fff", "ffff", "fffff", "ffffff",
        "mp", "mf",
        "sf", "sfp", "sfpp", "fp", "rf", "rfz", "sfz", "sffz", "fz", "n", "pf", "sfzp",
    }
)


def _prune_colliding_dynamics_children(dyn: ET.Element) -> int:
    """Keep at most one dynamic symbol inside ``<dynamics>``, preferring standard tags over ``<other-dynamics>``."""
    children = list(dyn)
    if len(children) <= 1:
        if len(children) == 1 and children[0].tag == "other-dynamics" and not (children[0].text or "").strip():
            dyn.remove(children[0])
            return 1
        return 0
    std_children = [ch for ch in children if ch.tag in _STANDARD_DYNAMIC_TAGS]
    if std_children:
        keep = std_children[0]
    else:
        non_empty = [ch for ch in children if ch.tag != "other-dynamics" or (ch.text or "").strip()]
        keep = non_empty[0] if non_empty else children[0]
    removed = 0
    for ch in children:
        if ch is not keep:
            dyn.remove(ch)
            removed += 1
    if keep.tag == "other-dynamics" and not (keep.text or "").strip():
        dyn.remove(keep)
        removed += 1
    return removed


def _is_pure_dynamics_direction(direction: ET.Element) -> tuple[bool, bool]:
    """Return ``(is_pure_dynamics, has_standard_tag)`` for a ``<direction>`` element."""
    dts = direction.findall("direction-type")
    if not dts:
        return False, False
    has_dyn = False
    has_std = False
    for dt in dts:
        if len(dt) == 0:
            continue
        for ch in dt:
            if ch.tag != "dynamics":
                return False, False
            has_dyn = True
            if any(d_ch.tag in _STANDARD_DYNAMIC_TAGS for d_ch in ch):
                has_std = True
    return has_dyn, has_std


def sanitize_dynamics(root: ET.Element) -> ET.Element:
    """Remove colliding children inside ``<dynamics>`` and same-beat duplicate dynamic directions."""
    removed_children = 0
    removed_dirs = 0

    for measure in root.findall(".//measure"):
        for direction in list(measure.findall("direction")):
            for dt in list(direction.findall("direction-type")):
                for dyn in list(dt.findall("dynamics")):
                    removed_children += _prune_colliding_dynamics_children(dyn)
                    if len(dyn) == 0:
                        dt.remove(dyn)
                if len(dt) == 0:
                    direction.remove(dt)
            if direction.find("direction-type") is None and direction.find("sound") is None:
                measure.remove(direction)
                removed_dirs += 1

        # Drop consecutive same-beat pure-dynamics directions on the same staff
        beat_dyn_by_staff: dict[str, tuple[ET.Element, bool]] = {}
        for child in list(measure):
            if child.tag in ("note", "forward", "backup"):
                if child.tag == "note" and child.find("grace") is not None:
                    continue
                beat_dyn_by_staff.clear()
            elif child.tag == "direction":
                is_pure, has_std = _is_pure_dynamics_direction(child)
                if not is_pure:
                    continue
                staff_id = (child.findtext("staff") or "1").strip() or "1"
                prev = beat_dyn_by_staff.get(staff_id)
                if prev is None:
                    beat_dyn_by_staff[staff_id] = (child, has_std)
                else:
                    prev_el, prev_std = prev
                    if has_std and not prev_std:
                        measure.remove(prev_el)
                        beat_dyn_by_staff[staff_id] = (child, has_std)
                    else:
                        measure.remove(child)
                    removed_dirs += 1

    if removed_children or removed_dirs:
        log.info(
            "Sanitized colliding dynamics: removed %d extra symbol(s) and %d duplicate/empty direction(s)",
            removed_children,
            removed_dirs,
        )
    return root


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
    sanitize_dynamics(root)
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
                actual = normal = 0
                if tm is not None:
                    an_txt = (tm.findtext("actual-notes") or "").strip()
                    nn_txt = (tm.findtext("normal-notes") or "").strip()
                    if an_txt.isdigit() and nn_txt.isdigit():
                        actual, normal = int(an_txt), int(nn_txt)
                    note.remove(tm)
                dur = note.find("duration")
                if dur is not None and (dur.text or "").strip().isdigit():
                    cur_dur = int(dur.text.strip())
                    if actual > 0 and normal > 0 and (cur_dur * actual) % normal == 0:
                        dur.text = str((cur_dur * actual) // normal)
                    elif cur_dur == 8:
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


def sync_part_transpose(
    part: Optional[ET.Element],
    profile: InstrumentPlaybackProfile,
) -> bool:
    """Write or clear ``<transpose>`` in ``part``'s first measure ``<attributes>`` for playback."""
    if part is None:
        return False
    m1 = part.find("measure")
    if m1 is None:
        return False

    for attrs in list(m1.findall("attributes")):
        for old_tr in list(attrs.findall("transpose")):
            attrs.remove(old_tr)

    if profile.diatonic == 0 and profile.chromatic == 0 and profile.octave_change == 0:
        return False

    attrs = m1.find("attributes")
    if attrs is None:
        attrs = ET.Element("attributes")
        ins_idx = 0
        for idx, ch in enumerate(list(m1)):
            if ch.tag == "print":
                ins_idx = idx + 1
            else:
                break
        m1.insert(ins_idx, attrs)

    tr = ET.Element("transpose")
    ET.SubElement(tr, "diatonic").text = str(profile.diatonic)
    ET.SubElement(tr, "chromatic").text = str(profile.chromatic)
    if profile.octave_change != 0:
        ET.SubElement(tr, "octave-change").text = str(profile.octave_change)

    pos = len(attrs)
    for idx, ch in enumerate(list(attrs)):
        if ch.tag in _ATTR_AFTER_TRANSPOSE:
            pos = idx
            break
    attrs.insert(pos, tr)
    log.debug(
        "Synced <transpose> on %s: diatonic=%d chromatic=%d octave=%d",
        part.get("id", "P1"),
        profile.diatonic,
        profile.chromatic,
        profile.octave_change,
    )
    return True


def normalize_default_instruments(root: ET.Element) -> int:
    """Replace any uncustomized homr ``Voice`` placeholder parts with ``Piano`` (program 1)."""
    from .fullscore.instruments import is_percussion_instrument

    normalized = 0
    pitched_i = 0
    for sp in root.findall("./part-list/score-part"):
        part_id = sp.get("id", "")
        if is_homr_default_voice(sp):
            clef = _extract_first_clef(root, part_id) if part_id else None
            profile, _ = set_score_part_instrument(
                sp,
                "Piano",
                abbreviation="Pno.",
                clef=clef,
                pitched_channel_index=pitched_i,
            )
            if part_id:
                sync_part_transpose(root.find(f"./part[@id='{part_id}']"), profile)
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
        part_el = root.find(f"./part[@id='{part_id}']")
        sync_part_transpose(part_el, profile)
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


def _fifths_to_key_alters(fifths: int) -> dict[str, int]:
    """Return diatonic step alterations (``-1``, ``0``, ``+1``) implied by ``fifths``."""
    alters = {step: 0 for step in "CDEFGAB"}
    wf = ((fifths + 6) % 12) - 6
    if wf > 0:
        for step in _SHARP_ORDER[: min(7, wf)]:
            alters[step] = 1
    elif wf < 0:
        for step in _FLAT_ORDER[: min(7, -wf)]:
            alters[step] = -1
    return alters


def add_courtesy_accidentals(root: ET.Element) -> int:
    """Emit cautionary ``<accidental>`` tags when a note returns to the key signature after an alteration."""
    added = 0
    for part in root.findall("part"):
        key_alters = _fifths_to_key_alters(0)
        prev_altered: set[tuple[str, str]] = set()
        for mi, measure in enumerate(part.findall("measure")):
            fifths_txt = measure.findtext("./attributes/key/fifths")
            if fifths_txt is not None and fifths_txt.strip().lstrip("+-").isdigit():
                key_alters = _fifths_to_key_alters(int(fifths_txt.strip()))
                if mi > 0:
                    prev_altered.clear()

            cur_altered: set[tuple[str, str]] = set()
            seen_in_bar: set[tuple[str, str]] = set()
            for note in measure.findall("note"):
                if note.find("rest") is not None:
                    continue
                pitch = note.find("pitch")
                if pitch is None:
                    continue
                step = (pitch.findtext("step") or "").strip().upper()
                octave = (pitch.findtext("octave") or "").strip()
                if step not in key_alters or not octave:
                    continue
                alter_txt = (pitch.findtext("alter") or "").strip()
                try:
                    alter = int(round(float(alter_txt))) if alter_txt else 0
                except ValueError:
                    alter = 0

                key = (step, octave)
                expected = key_alters[step]
                is_tie_stop = any(t.get("type") == "stop" for t in note.findall("tie"))
                if alter != expected:
                    cur_altered.add(key)
                elif key in prev_altered and key not in seen_in_bar and not is_tie_stop and note.find("accidental") is None:
                    acc_name = _ALTER_TO_ACCIDENTAL.get(alter)
                    if acc_name:
                        acc_el = ET.Element("accidental", {"cautionary": "yes"})
                        acc_el.text = acc_name
                        ins_idx = len(note)
                        for idx, ch in enumerate(list(note)):
                            if ch.tag in _NOTE_AFTER_ACCIDENTAL:
                                ins_idx = idx
                                break
                        note.insert(ins_idx, acc_el)
                        added += 1
                seen_in_bar.add(key)
            prev_altered = cur_altered
    if added:
        log.info("Added %d courtesy <accidental cautionary='yes'> tag(s) across score", added)
    return added


def _plain_visual_ticks(note: ET.Element, div: int) -> Optional[int]:
    """Compute untupled tick duration from explicit ``<type>`` and ``<dot>`` elements."""
    note_type = (note.findtext("type") or "").strip()
    base_q = TYPE_TO_QUARTERS.get(note_type)
    if base_q is None:
        return None
    dots = len(note.findall("dot"))
    if dots == 0:
        dot_mult = Fraction(1)
    elif dots == 1:
        dot_mult = Fraction(3, 2)
    elif dots == 2:
        dot_mult = Fraction(7, 4)
    else:
        return None
    vis_frac = base_q * dot_mult * max(1, div)
    if vis_frac.denominator != 1 or vis_frac <= 0:
        return None
    return int(vis_frac)


def repair_spurious_tuplets(
    measure: ET.Element,
    expected_dur: int,
    div: int,
    time_sig: tuple[str, str],
) -> bool:
    """Remove spurious OMR ``3:2`` tuplets when un-tupling brings an underfull voice to bar length.

    Legitimate full-measure tuplets (``cur_span >= expected_dur``) are never modified.
    """
    if expected_dur <= 0:
        return False
    notes = measure.findall("note")
    if not notes or all(n.find("rest") is not None for n in notes):
        return False
    if not any(n.find("time-modification") is not None for n in notes):
        return False

    b_str, bt_str = time_sig
    is_compound = (
        b_str.isdigit()
        and bt_str.isdigit()
        and int(bt_str) == 8
        and int(b_str) >= 6
        and int(b_str) % 3 == 0
    )

    voice_groups: dict[str, list[list[ET.Element]]] = {}
    for note in notes:
        if note.find("grace") is not None:
            continue
        voice = (note.findtext("voice") or "1").strip() or "1"
        if note.find("chord") is not None:
            groups = voice_groups.get(voice)
            if groups:
                groups[-1].append(note)
        else:
            voice_groups.setdefault(voice, []).append([note])

    changed = False
    old_span_by_voice: dict[str, int] = {}
    delta_by_voice: dict[str, int] = {}
    m_num = measure.get("number", "?")

    for voice, groups in voice_groups.items():
        cur_span = 0
        untupled_span = 0
        candidates: list[tuple[list[ET.Element], int]] = []

        for grp in groups:
            prim = grp[0]
            dur_txt = (prim.findtext("duration") or "").strip()
            cur_dur = int(dur_txt) if dur_txt.isdigit() else 0
            cur_span += cur_dur

            tm = prim.find("time-modification")
            new_dur: Optional[int] = None
            if tm is not None:
                an_txt = (tm.findtext("actual-notes") or "").strip()
                nn_txt = (tm.findtext("normal-notes") or "").strip()
                if an_txt.isdigit() and nn_txt.isdigit():
                    actual, normal = int(an_txt), int(nn_txt)
                    if actual > normal > 0:
                        vis = _plain_visual_ticks(prim, div)
                        if vis is not None and vis > cur_dur:
                            new_dur = vis
                        elif cur_dur > 0 and (cur_dur * actual) % normal == 0:
                            scaled = (cur_dur * actual) // normal
                            if scaled > cur_dur:
                                new_dur = scaled
            if new_dur is not None:
                untupled_span += new_dur
                candidates.append((grp, new_dur))
            else:
                untupled_span += cur_dur

        old_span_by_voice[voice] = cur_span
        if not candidates or cur_span >= expected_dur:
            continue

        case_a = cur_span < untupled_span <= expected_dur
        case_b = (
            is_compound
            and untupled_span > expected_dur
            and abs(untupled_span - expected_dur) < abs(cur_span - expected_dur)
            and (untupled_span - expected_dur) <= max(1, div)
        )
        if not (case_a or case_b):
            continue

        repaired_notes = 0
        for grp, new_dur in candidates:
            for n in grp:
                for tm in list(n.findall("time-modification")):
                    n.remove(tm)
                for nt in list(n.findall("notations")):
                    for tup in list(nt.findall("tuplet")):
                        nt.remove(tup)
                    if len(nt) == 0:
                        n.remove(nt)
                dur_el = n.find("duration")
                if dur_el is not None:
                    dur_el.text = str(new_dur)
                repaired_notes += 1

        delta_by_voice[voice] = untupled_span - cur_span
        changed = True
        log.debug(
            "Repaired %d spurious tuplet note(s) in measure %s voice %s (%d -> %d ticks, expected=%d, ts=%s/%s)",
            repaired_notes,
            m_num,
            voice,
            cur_span,
            untupled_span,
            expected_dur,
            b_str,
            bt_str,
        )

    if changed and delta_by_voice and measure.find("backup") is not None:
        # Keep a <backup> element aligned if it backs up directly over a repaired voice span
        last_voice: Optional[str] = None
        for ch in measure:
            if ch.tag == "note" and ch.find("grace") is None:
                last_voice = (ch.findtext("voice") or "1").strip() or "1"
            elif ch.tag == "backup" and last_voice in delta_by_voice:
                b_dur_el = ch.find("duration")
                if b_dur_el is not None and (b_dur_el.text or "").strip().isdigit():
                    b_val = int(b_dur_el.text.strip())
                    if b_val == old_span_by_voice[last_voice]:
                        b_dur_el.text = str(b_val + delta_by_voice[last_voice])

    return changed


def _quarter_duration_to_time_sig(
    q: Fraction,
    preferred_beat_type: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    if preferred_beat_type == "8" and (q * 2).denominator == 1 and int(q * 2) in (3, 5, 6, 7, 9, 12):
        return (str(int(q * 2)), "8")
    if preferred_beat_type == "2" and (q / 2).denominator == 1 and int(q / 2) in (2, 3, 4):
        return (str(int(q / 2)), "2")
    if q.denominator == 1 and 2 <= int(q) <= 6:
        return (str(int(q)), "4")
    if q.denominator == 2 and int(q * 2) in (3, 5, 7, 9):
        return (str(int(q * 2)), "8")
    return None


def _measure_voice1_quarters(measure: ET.Element, div: int) -> tuple[Optional[Fraction], bool]:
    notes = measure.findall("note")
    is_rest_only = (not notes) or all(n.find("rest") is not None for n in notes)
    if is_rest_only:
        return None, True
    first_voice: Optional[str] = None
    total = 0
    for note in notes:
        if note.find("chord") is not None or note.find("grace") is not None:
            continue
        v = (note.findtext("voice") or "1").strip() or "1"
        if first_voice is None:
            first_voice = v if any((n.findtext("voice") or "1").strip() == "1" for n in notes) else v
            if first_voice != "1":
                first_voice = v
        target_v = "1" if first_voice == "1" else first_voice
        if v != target_v:
            continue
        dur_txt = (note.findtext("duration") or "").strip()
        if dur_txt.isdigit():
            total += int(dur_txt)
    if total <= 0:
        return None, False
    return Fraction(total, max(1, div)), False


def _has_section_boundary_before(measures: Sequence[ET.Element], idx: int) -> bool:
    if idx == 0:
        return True
    cur = measures[idx]
    if cur.find("./attributes/key") is not None or cur.find("./attributes/time") is not None:
        return True
    prev = measures[idx - 1]
    for bl in prev.findall("barline"):
        if bl.find("repeat") is not None:
            return True
        style = (bl.findtext("bar-style") or "").strip().lower()
        if style in _SECTION_BAR_STYLES:
            return True
    return False


def reconcile_standard_time_signatures(root: ET.Element) -> int:
    """Infer piecewise-constant meter regimes from actual measure durations and repair wrong/missing ``<time>``."""
    changed_parts = 0
    for part in root.findall("part"):
        measures = part.findall("measure")
        n_bars = len(measures)
        if n_bars < 2:
            continue

        divs: list[int] = []
        quarters: list[Optional[Fraction]] = []
        is_rest: list[bool] = []
        declared_ts: list[Optional[tuple[str, str]]] = []
        boundary: list[bool] = []

        cur_div = 4
        for i, m in enumerate(measures):
            for a in m.findall("attributes"):
                d_txt = (a.findtext("divisions") or "").strip()
                if d_txt.isdigit():
                    cur_div = max(1, int(d_txt))
            divs.append(cur_div)
            q_val, rest_only = _measure_voice1_quarters(m, cur_div)
            quarters.append(q_val)
            is_rest.append(rest_only)

            ts_found: Optional[tuple[str, str]] = None
            for t in m.findall("./attributes/time"):
                b = (t.findtext("beats") or "").strip()
                bt = (t.findtext("beat-type") or "").strip()
                if b.isdigit() and bt.isdigit() and int(b) > 0 and int(bt) > 0:
                    ts_found = (b, bt)
            declared_ts.append(ts_found)
            boundary.append(_has_section_boundary_before(measures, i))

        cands: list[Optional[tuple[str, str]]] = []
        active_decl: Optional[tuple[str, str]] = None
        for i in range(n_bars):
            if declared_ts[i] is not None:
                active_decl = declared_ts[i]
            if is_rest[i] or quarters[i] is None:
                cands.append(None)
                continue
            pref_bt = active_decl[1] if active_decl is not None else None
            if active_decl is not None and quarters[i] == Fraction(int(active_decl[0]) * 4, int(active_decl[1])):
                cands.append(active_decl)
            else:
                cands.append(_quarter_duration_to_time_sig(quarters[i], preferred_beat_type=pref_bt))

        # Group contiguous non-rest measures with identical candidate meter
        raw_runs: list[tuple[tuple[str, str], list[int]]] = []
        for i, c in enumerate(cands):
            if c is None:
                continue
            if raw_runs and raw_runs[-1][0] == c:
                raw_runs[-1][1].append(i)
            else:
                raw_runs.append((c, [i]))

        valid_runs: list[list[object]] = []
        for ts, idxs in raw_runs:
            if len(idxs) >= 2 or (len(idxs) == 1 and declared_ts[idxs[0]] == ts):
                if valid_runs and valid_runs[-1][0] == ts:
                    valid_runs[-1][2] = idxs[-1]
                else:
                    valid_runs.append([ts, idxs[0], idxs[-1]])

        if not valid_runs:
            continue

        inferred: list[tuple[str, str]] = [valid_runs[0][0]] * n_bars  # type: ignore[list-item]
        for k, (ts, start_i, end_i) in enumerate(valid_runs):
            s_i, e_i = int(start_i), int(end_i)
            for j in range(s_i, e_i + 1):
                inferred[j] = ts  # type: ignore[assignment]
            if k + 1 < len(valid_runs):
                next_ts, next_s_raw, _ = valid_runs[k + 1]
                next_s = int(next_s_raw)
                split_idx = next_s
                for g in range(e_i + 1, next_s + 1):
                    if boundary[g]:
                        split_idx = g
                        break
                for j in range(e_i + 1, split_idx):
                    inferred[j] = ts  # type: ignore[assignment]
                for j in range(split_idx, next_s):
                    inferred[j] = next_ts  # type: ignore[assignment]
            else:
                for j in range(e_i + 1, n_bars):
                    inferred[j] = ts  # type: ignore[assignment]

        # Compare inferred timeline against currently declared timeline
        existing_timeline: list[Optional[tuple[str, str]]] = []
        cur_ts_decl: Optional[tuple[str, str]] = None
        for ts in declared_ts:
            if ts is not None:
                cur_ts_decl = ts
            existing_timeline.append(cur_ts_decl)

        if existing_timeline != list(inferred):
            for mi, m in enumerate(measures):
                b, bt = inferred[mi]
                need_time = (mi == 0) or (inferred[mi] != inferred[mi - 1])
                for attrs in m.findall("attributes"):
                    for old_t in list(attrs.findall("time")):
                        attrs.remove(old_t)
                if need_time:
                    attrs = m.find("attributes")
                    if attrs is None:
                        attrs = ET.Element("attributes")
                        m.insert(0, attrs)
                    t_el = ET.Element("time")
                    ET.SubElement(t_el, "beats").text = b
                    ET.SubElement(t_el, "beat-type").text = bt
                    ins_idx = 0
                    for idx, ch in enumerate(list(attrs)):
                        if ch.tag in _ATTR_BEFORE_TIME:
                            ins_idx = idx + 1
                    attrs.insert(ins_idx, t_el)

                if is_rest[mi]:
                    mr_note = None
                    for n in m.findall("note"):
                        r_el = n.find("rest")
                        if r_el is not None and r_el.get("measure") == "yes":
                            mr_note = n
                            break
                    if mr_note is not None:
                        expected_dur = max(1, int(round(divs[mi] * int(b) * 4 / max(1, int(bt)))))
                        dur_el = mr_note.find("duration")
                        if dur_el is not None and dur_el.text != str(expected_dur):
                            dur_el.text = str(expected_dur)
                            for stale_tag in ("type", "dot"):
                                for stale in list(mr_note.findall(stale_tag)):
                                    mr_note.remove(stale)

            transitions = [
                f"M{mi + 1}={b}/{bt}"
                for mi, (b, bt) in enumerate(inferred)
                if mi == 0 or inferred[mi] != inferred[mi - 1]
            ]
            log.info(
                "Reconciled time signatures on %s (%d bars): %s",
                part.get("id", "P1"),
                n_bars,
                ", ".join(transitions),
            )
            changed_parts += 1

        repaired_tup_bars = 0
        for mi, m in enumerate(measures):
            if is_rest[mi]:
                continue
            b, bt = inferred[mi]
            expected_dur = max(1, int(round(divs[mi] * int(b) * 4 / max(1, int(bt)))))
            if repair_spurious_tuplets(m, expected_dur, divs[mi], (b, bt)):
                repaired_tup_bars += 1
        if repaired_tup_bars:
            log.info(
                "Repaired spurious tuplets in %d measure(s) on %s",
                repaired_tup_bars,
                part.get("id", "P1"),
            )
    return changed_parts


def apply_score_options(root: ET.Element, options: Optional[ConversionOptions]) -> ET.Element:
    """Apply Section 3 post-processing options (transpose, clef, instrument, layout, courtesy accidentals) to ``root``."""
    if options is None:
        normalize_default_instruments(root)
        add_courtesy_accidentals(root)
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
    add_courtesy_accidentals(root)
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
    part_names_lower = {
        (sp.findtext("part-name") or "").strip().lower()
        for sp in merged.findall("./part-list/score-part")
        if (sp.findtext("part-name") or "").strip()
    }
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
                if idx > 1 and mi == 0:
                    for attrs in list(m.findall("attributes")):
                        for dup_tr in list(attrs.findall("transpose")):
                            attrs.remove(dup_tr)
                        if len(attrs) == 0:
                            m.remove(attrs)
                counters[pi] += 1
                m.set("number", str(counters[pi]))
                if len(roots) > 1 and mi == 0 and pi == 0:
                    has_tempo_dir = (
                        m.find("./direction/direction-type/metronome") is not None
                        or m.find("./direction/sound[@tempo]") is not None
                    )
                    is_running_part_hdr = idx > 1 and mtitle.lower() in part_names_lower
                    skip_title_overlap = idx == 1 and bool(title) and has_tempo_dir
                    if is_running_part_hdr or skip_title_overlap:
                        if keep_breaks and idx > 1:
                            m.insert(0, ET.Element("print", {"new-page": "yes"}))
                    else:
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

    reconcile_standard_time_signatures(merged)
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
