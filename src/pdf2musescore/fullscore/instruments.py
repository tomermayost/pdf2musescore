"""Left-margin OCR instrument extraction, OCR cleanup, abbreviation/expansion, and percussion classification."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstrumentPlaybackProfile:
    """Playback sound ID and written-to-sounding transposition parameters."""

    sound_id: str
    diatonic: int = 0
    chromatic: int = 0
    octave_change: int = 0
    fifths_shift: int = 0

    @property
    def is_transposing(self) -> bool:
        """Return True if this instrument sounds at a different pitch than written."""
        return self.diatonic != 0 or self.chromatic != 0 or self.octave_change != 0


_UNPITCHED_SOUND_RULES: tuple[tuple[str, str], ...] = (
    (r"\bsnare\b|\bs\.d\.", "drum.snare-drum"),
    (r"\bbass\s+drum\b|\bb\.d\.", "drum.bass-drum"),
    (r"\bcymbal|\bcym\b", "metal.cymbal.crash"),
    (r"\btriangle\b|\btri\b", "metal.triangle"),
    (r"\btambourine\b|\btamb\b", "drum.tambourine"),
    (r"\bdrum\s*set\b|\bdr\.\s*set\b", "drum.group.set"),
)

_PITCHED_PERCUSSION_RULES: tuple[tuple[str, str], ...] = (
    (r"\btimpani\b|\btimp\b", "drum.timpani"),
    (r"\bglockenspiel\b|\bglock\b", "pitched-percussion.glockenspiel"),
    (r"\bxylophone\b|\bxyl\b", "pitched-percussion.xylophone"),
    (r"\bvibraphone\b|\bvib\b", "pitched-percussion.vibraphone"),
    (r"\bmarimba\b|\bmallet|\bmall\b", "pitched-percussion.marimba"),
    (r"\bcelesta\b|\bcel\b", "keyboard.celesta"),
    (r"\btubular\s+bells\b|\bchimes\b|\bbells\b|\bcrotales\b", "pitched-percussion.tubular-bells"),
)

_OTHER_INST_RULES: tuple[tuple[str, InstrumentPlaybackProfile], ...] = (
    (r"\bviolin\b|\bvln\b", InstrumentPlaybackProfile("strings.violin")),
    (r"\bviola\b|\bvla\b", InstrumentPlaybackProfile("strings.viola")),
    (r"\bvioloncello\b|\bcello\b|\bvc\b", InstrumentPlaybackProfile("strings.cello")),
    (r"\bpiano\b|\bpno\b", InstrumentPlaybackProfile("keyboard.piano")),
    (r"\borgan\b|\borg\b", InstrumentPlaybackProfile("keyboard.organ")),
    (r"\bharp\b|\bhp\b", InstrumentPlaybackProfile("pluck.harp")),
    (r"\bguitar\b|\bgtr\b", InstrumentPlaybackProfile("pluck.guitar", 0, 0, -1, 0)),
)


@dataclass(frozen=True)
class InstrumentLabel:
    """Normalized instrument name and score-part abbreviation for a staff."""

    full_name: str
    abbreviation: str
    raw_ocr: str = ""
    is_fallback: bool = False
    is_abbreviated_source: bool = False


_ABBREV_RULES: tuple[tuple[str, str], ...] = (
    (r"\bEnglish\s+Horn\b", "E. Hn."),
    (r"\bCor\s+Anglais\b", "C.A."),
    (r"\bBass\s+Trombone\b", "B. Tbn."),
    (r"\bBass\s+Clarinet\b", "B. Cl."),
    (r"\bAlto\s+Clarinet\b", "A. Cl."),
    (r"\bSoprano\s+Saxophone\b", "S. Sax."),
    (r"\bSoprano\s+Sax\.?(?=\s|$|/)", "S. Sax."),
    (r"\bAlto\s+Saxophone\b", "A. Sax."),
    (r"\bAlto\s+Sax\.?(?=\s|$|/)", "A. Sax."),
    (r"\bTenor\s+Saxophone\b", "T. Sax."),
    (r"\bTenor\s+Sax\.?(?=\s|$|/)", "T. Sax."),
    (r"\bBaritone\s+Saxophone\b", "Bar. Sax."),
    (r"\bBaritone\s+Sax\.?(?=\s|$|/)", "Bar. Sax."),
    (r"\bMallet\s+Percussion\b", "Mall. Perc."),
    (r"\bString\s+Bass\b", "Str. Bass"),
    (r"\bDouble\s+Bass\b", "Db. Bass"),
    (r"\bElectric\s+Bass\b", "E. Bass"),
    (r"\bFrench\s+Horn\b", "Hn."),
    (r"\bSnare\s+Drum\b", "S.D."),
    (r"\bBass\s+Drum\b", "B.D."),
    (r"\bDrum\s+Set\b", "Dr. Set"),
    (r"\bTubular\s+Bells\b", "Chimes"),
    (r"\bContrabassoon\b", "Cbsn."),
    (r"\bPiccolo\b", "Picc."),
    (r"\bFlute\b", "Fl."),
    (r"\bOboe\b", "Ob."),
    (r"\bBassoon\b", "Bsn."),
    (r"\bClarinet\b", "Cl."),
    (r"\bSaxophone\b", "Sax."),
    (r"\bTrumpet\b", "Tpt."),
    (r"\bCornet\b", "Cnt."),
    (r"\bFlugelhorn\b", "Flgh."),
    (r"\bHorn\b", "Hn."),
    (r"\bTrombone\b", "Tbn."),
    (r"\bEuphonium\b", "Euph."),
    (r"\bBaritone\b", "Bar."),
    (r"\bTuba\b", "Tba."),
    (r"\bTimpani\b", "Timp."),
    (r"\bPercussion\b", "Perc."),
    (r"\bGlockenspiel\b", "Glock."),
    (r"\bXylophone\b", "Xyl."),
    (r"\bVibraphone\b", "Vib."),
    (r"\bMarimba\b", "Mar."),
    (r"\bCelesta\b", "Cel."),
    (r"\bCymbals?\b", "Cym."),
    (r"\bTriangle\b", "Tri."),
    (r"\bTambourine\b", "Tamb."),
    (r"\bViolin\b", "Vln."),
    (r"\bViola\b", "Vla."),
    (r"\bVioloncello\b", "Vc."),
    (r"\bCello\b", "Vc."),
    (r"\bContrabass\b", "Cb."),
    (r"\bGuitar\b", "Gtr."),
    (r"\bPiano\b", "Pno."),
    (r"\bOrgan\b", "Org."),
    (r"\bHarp\b", "Hp."),
)

_EXPAND_MAP: tuple[tuple[str, str], ...] = (
    (r"Mall\.\s*Perc\.?", "Mallet Percussion"),
    (r"Mallet\s+Perc\.?", "Mallet Percussion"),
    (r"Str\.\s*Bass\.?", "String Bass"),
    (r"Db\.\s*Bass\.?", "Double Bass"),
    (r"E\.\s*Bass\.?", "Electric Bass"),
    (r"E\.\s*Hn\.?", "English Horn"),
    (r"B\.\s*Tbn\.?", "Bass Trombone"),
    (r"Bs\.\s*Cl\.?", "Bass Clarinet"),
    (r"B\.\s*Cl\.?", "Bass Clarinet"),
    (r"A\.\s*Cl\.?", "Alto Clarinet"),
    (r"S\.\s*Sax\.?", "Soprano Sax."),
    (r"A\.\s*Sax\.?", "Alto Sax."),
    (r"T\.\s*Sax\.?", "Tenor Sax."),
    (r"Bar\.\s*Sax\.?", "Baritone Sax."),
    (r"B\.\s*Sax\.?", "Baritone Sax."),
    (r"Cbsn\.?", "Contrabassoon"),
    (r"Picc\.?", "Piccolo"),
    (r"Fl\.?", "Flute"),
    (r"Ob\.?", "Oboe"),
    (r"Bsn\.?", "Bassoon"),
    (r"Cl\.?", "Clarinet"),
    (r"Tpt\.?", "Trumpet"),
    (r"Cnt\.?", "Cornet"),
    (r"Flgh\.?", "Flugelhorn"),
    (r"Hn\.?", "Horn"),
    (r"Tbn\.?", "Trombone"),
    (r"Euph\.?", "Euphonium"),
    (r"Bar\.?", "Baritone"),
    (r"Bass\.", "Basses"),
    (r"Tba\.?", "Tuba"),
    (r"Timp\.?", "Timpani"),
    (r"Perc\.?", "Percussion"),
    (r"Glock\.?", "Glockenspiel"),
    (r"Xyl\.?", "Xylophone"),
    (r"Vib\.?", "Vibraphone"),
    (r"Mar\.?", "Marimba"),
    (r"Cel\.?", "Celesta"),
    (r"Vln\.?", "Violin"),
    (r"Vla\.?", "Viola"),
    (r"Vc\.?", "Cello"),
    (r"Cb\.?", "Contrabass"),
    (r"Gtr\.?", "Guitar"),
    (r"Pno\.?", "Piano"),
    (r"Org\.?", "Organ"),
    (r"Hp\.?", "Harp"),
)

_EXPAND_COMBINED_RE = re.compile(
    r"\b(" + "|".join(pat for pat, _ in _EXPAND_MAP) + r")(?=\s|$|/|,)",
    re.IGNORECASE,
)
_EXPAND_COMPILED = [(re.compile(f"^{pat}$", re.IGNORECASE), repl) for pat, repl in _EXPAND_MAP]

_BOILERPLATE_TOKENS: tuple[str, ...] = (
    "full score",
    "conductor",
    "copyright",
    "all rights reserved",
    "printed in",
    "montfortstraat",
    "beriato",
    "concert pitch",
    "transposed score",
)

_PITCHED_PERCUSSION_RE = re.compile(
    r"\b(?:timpani|timp\.?|mallet|mall\.?|glockenspiel|glock\.?|xylophone|xyl\.?|"
    r"vibraphone|vib\.?|marimba|mar\.?|celesta|cel\.?|chimes|tubular\s+bells|bells|crotales)\b",
    re.IGNORECASE,
)
_UNPITCHED_PERCUSSION_RE = re.compile(
    r"\b(?:percussion|perc\.?|drum\s*set|dr\.\s*set|snare(?:\s+drum)?|bass\s+drum|"
    r"cymbals?|cym\.?|triangle|tri\.?|tambourine|tamb\.?|tam-?tam|gong|woodblock|"
    r"claves|maracas|castanets|hi-?hat|toms?|bongos?|congas?|shaker)\b|s\.d\.|b\.d\.",
    re.IGNORECASE,
)

_KNOWN_INSTRUMENT_WORD_RE = re.compile(
    r"\b(?:piccolo|flute|oboe|bassoon|contrabassoon|clarinet|sax|saxophone|trumpet|cornet|"
    r"flugelhorn|horn|trombone|euphonium|baritone|tuba|bass|basses|timpani|mallet|percussion|"
    r"glockenspiel|xylophone|vibraphone|marimba|celesta|chimes|violin|viola|cello|violoncello|"
    r"contrabass|piano|organ|harp|guitar|snare|cymbal|triangle|tambourine|drum|"
    r"picc|fl|ob|bsn|cbsn|cl|tpt|cnt|flgh|hn|tbn|euph|bar|tba|timp|mall|perc|glock|xyl|vib|mar|"
    r"vln|vla|vc|cb|pno|org|hp|gtr|str|cym|tri|tamb)\b",
    re.IGNORECASE,
)

_FULL_WORD_INSTRUMENT_RE = re.compile(
    r"\b(?:piccolo|flute|oboe|bassoon|contrabassoon|clarinet|saxophone|trumpet|cornet|"
    r"flugelhorn|horn|trombone|euphonium|baritone|tuba|basses|timpani|percussion|"
    r"glockenspiel|xylophone|vibraphone|marimba|celesta|violin|viola|cello|violoncello|"
    r"contrabass|piano|organ|harp|guitar)\b",
    re.IGNORECASE,
)


def clean_instrument_label(raw: str) -> str:
    """Normalize common RapidOCR misreads in conductor-score left-margin instrument labels."""
    txt = re.sub(r"\s+", " ", raw).strip()
    txt = re.sub(r"^Bup\s*t\s*(\d+)", r"Bb Trumpet \1", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bH[lI]orn\b", "Horn", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bFI\.", "Fl.", txt)
    txt = re.sub(r"\bCI\.", "Cl.", txt)
    txt = re.sub(r"\bPice\.", "Picc.", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bPere\.", "Perc.", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bCrnt\.", "Cnt.", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bSx\.", "Sax.", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bBs\.\s*Cl\.", "B. Cl.", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\b([A-Za-z]{2,5})\.(\d+)\b", r"\1. \2", txt)
    txt = re.sub(r"^E[>›»\-–\.]\s*", "Eb ", txt)
    txt = re.sub(r"^B[\.\-–i]\s+", "Bb ", txt)
    txt = re.sub(
        r"\bB\s+(Clarinet|Trumpet|Cornet|Tenor|Bass|Cl|Tpt|Cnt|Bs|B|T)(?=\b|\.)",
        r"Bb \1",
        txt,
    )
    txt = re.sub(r"\bE\s+(Alto|Baritone|Clarinet|A|B|Cl)(?=\b|\.)", r"Eb \1", txt)
    txt = re.sub(r"^F\s+(Baritone\s+Sax\.?|B\.\s*Sax\.?)", r"Eb \1", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*/\s*", " / ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def abbreviate_instrument_name(full_name: str) -> str:
    """Convert a full instrument label into a compact MusicXML <part-abbreviation>."""
    txt = re.sub(r"\([^)]*\)", "", full_name).strip()
    for pattern, repl in _ABBREV_RULES:
        txt = re.sub(pattern, repl, txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"\s*/\s*", " / ", txt)
    return txt.strip() or full_name


def expand_abbreviated_name(label: str) -> str:
    """Expand standard score abbreviations (e.g. ``'Ob. 1, 2'``) into full instrument names."""
    txt = re.sub(r"\s+", " ", label).strip()
    txt = re.sub(r"\(opt\.?\)", "(optional)", txt, flags=re.IGNORECASE)

    def _replace_match(m: re.Match[str]) -> str:
        token = m.group(1)
        for rx, repl in _EXPAND_COMPILED:
            if rx.match(token):
                return repl
        return token  # pragma: no cover

    txt = _EXPAND_COMBINED_RE.sub(_replace_match, txt)
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"\s*/\s*", " / ", txt)
    return txt.strip() or label


def looks_abbreviated_label(label: str) -> bool:
    """Return True if ``label`` appears to use abbreviated instrument tokens rather than full names."""
    if not label:
        return False
    if _FULL_WORD_INSTRUMENT_RE.search(label):
        return False
    return "." in label or bool(re.search(r"\b(?:Mall|Str|Perc|Timp|Picc|Bsn|Ob|Fl|Cl|Tpt|Cnt|Hn|Tbn|Bar)\b", label, re.IGNORECASE))


def looks_low_quality_label(label: str) -> bool:
    """Return True if OCR output looks garbled (e.g. ``'DG'``, ``'B2'``, ``'B13 / B1 3'``)."""
    txt = (label or "").strip()
    if not txt or re.match(r"^Part\s+\d+$", txt, re.IGNORECASE):
        return True
    if _KNOWN_INSTRUMENT_WORD_RE.search(txt):
        return False
    alpha_chars = sum(1 for ch in txt if ch.isalpha())
    return alpha_chars < 4


def is_percussion_instrument(name: str) -> bool:
    """Return True ONLY for unpitched percussion (Channel 10); pitched percussion (Timpani/Mallets) stays False."""
    if not name:
        return False
    # Check unpitched markers first ONLY if not exclusively pitched; e.g. "Mallet Percussion" is pitched.
    if _PITCHED_PERCUSSION_RE.search(name):
        return False
    return bool(_UNPITCHED_PERCUSSION_RE.search(name))


def _infer_woodwind_profile(txt: str, clef: Optional[tuple[str, str]]) -> Optional[InstrumentPlaybackProfile]:
    has_flute = bool(re.search(r"\bflute|\bfl\b", txt))
    has_picc = bool(re.search(r"\bpiccolo\b|\bpicc\b", txt))
    if has_flute and has_picc:
        return InstrumentPlaybackProfile("wind.flutes.flute")
    if has_picc:
        return InstrumentPlaybackProfile("wind.flutes.flute.piccolo", 0, 0, 1, 0)
    if re.search(r"\balto\s+(?:flute\b|fl\b)", txt):
        return InstrumentPlaybackProfile("wind.flutes.flute.alto", -3, -5, 0, -1)
    if has_flute:
        return InstrumentPlaybackProfile("wind.flutes.flute")
    if re.search(r"\benglish\s+horn\b|\bcor\s+anglais\b|\be\.\s*hn\b", txt):
        return InstrumentPlaybackProfile("wind.reed.english-horn", -4, -7, 0, 1)
    if re.search(r"\boboe\b|\bob\b", txt):
        return InstrumentPlaybackProfile("wind.reed.oboe")
    if re.search(r"\bcontrabassoon\b|\bcbsn\b", txt):
        return InstrumentPlaybackProfile("wind.reed.contrabassoon", 0, 0, -1, 0)
    if re.search(r"\bbassoon\b|\bbsn\b", txt):
        return InstrumentPlaybackProfile("wind.reed.bassoon")

    if re.search(r"\bclarinet|\bcl\b", txt):
        if re.search(r"\bcontrabass\b", txt):
            if re.search(r"\beb\b|\bin\s+eb\b", txt):
                return InstrumentPlaybackProfile("wind.reed.clarinet.contrabass", -5, -9, -1, 3)
            return InstrumentPlaybackProfile("wind.reed.clarinet.contrabass", -1, -2, -2, 2)
        if re.search(r"\bbass\b|\bb\.\s*cl\b", txt):
            oct_chg = 0 if clef == ("F", "4") else -1
            return InstrumentPlaybackProfile("wind.reed.clarinet.bass", -1, -2, oct_chg, 2)
        if re.search(r"\balto\b|\ba\.\s*cl\b", txt):
            return InstrumentPlaybackProfile("wind.reed.clarinet.alto", -5, -9, 0, 3)
        if re.search(r"\beb\s+(?:clarinet|cl)\b|\b(?:clarinet|cl)\s+in\s+eb\b", txt):
            return InstrumentPlaybackProfile("wind.reed.clarinet.eflat", 2, 3, 0, 3)
        if re.search(r"\ba\s+(?:clarinet|cl)\b|\b(?:clarinet|cl)\s+in\s+a\b", txt):
            return InstrumentPlaybackProfile("wind.reed.clarinet.a", -2, -3, 0, -3)
        if re.search(r"\bc\s+(?:clarinet|cl)\b|\b(?:clarinet|cl)\s+in\s+c\b", txt):
            return InstrumentPlaybackProfile("wind.reed.clarinet")
        return InstrumentPlaybackProfile("wind.reed.clarinet.bflat", -1, -2, 0, 2)

    if re.search(r"\bsax(?:ophone)?\b", txt):
        if re.search(r"\bsoprano\b|\bs\.\s*sax\b", txt):
            return InstrumentPlaybackProfile("wind.reed.saxophone.soprano", -1, -2, 0, 2)
        if re.search(r"\balto\b|\ba\.\s*sax\b", txt):
            return InstrumentPlaybackProfile("wind.reed.saxophone.alto", -5, -9, 0, 3)
        if re.search(r"\btenor\b|\bt\.\s*sax\b", txt):
            return InstrumentPlaybackProfile("wind.reed.saxophone.tenor", -1, -2, -1, 2)
        if re.search(r"\bbaritone\b|\bbar\.\s*sax\b", txt):
            return InstrumentPlaybackProfile("wind.reed.saxophone.baritone", -5, -9, -1, 3)
    return None


def _infer_brass_profile(txt: str, clef: Optional[tuple[str, str]]) -> Optional[InstrumentPlaybackProfile]:
    if re.search(r"\b(?:trumpet|tpt|cornet|cnt|flugelhorn|flgh)\b", txt):
        if re.search(r"\bd\s+(?:trumpet|tpt)\b|\b(?:trumpet|tpt)\s+in\s+d\b", txt):
            return InstrumentPlaybackProfile("brass.trumpet.d", 1, 2, 0, -2)
        if re.search(r"\bc\s+(?:trumpet|tpt)\b|\b(?:trumpet|tpt)\s+in\s+c\b", txt):
            return InstrumentPlaybackProfile("brass.trumpet.c")
        if re.search(r"\beb\s+(?:trumpet|tpt|cornet|cnt)\b|\b(?:trumpet|tpt|cornet|cnt)\s+in\s+eb\b", txt):
            return InstrumentPlaybackProfile("brass.trumpet", 2, 3, 0, 3)
        if re.search(r"\ba\s+(?:trumpet|tpt)\b|\b(?:trumpet|tpt)\s+in\s+a\b", txt):
            return InstrumentPlaybackProfile("brass.trumpet", -2, -3, 0, -3)
        if re.search(r"\bflugelhorn\b|\bflgh\b", txt):
            sound_id = "brass.flugelhorn"
        elif re.search(r"\bcornet\b|\bcnt\b", txt) and not re.search(r"\btrumpet\b|\btpt\b", txt):
            sound_id = "brass.cornet"
        else:
            sound_id = "brass.trumpet.bflat"
        return InstrumentPlaybackProfile(sound_id, -1, -2, 0, 2)

    if re.search(r"\b(?:eb\s+horn|alto\s+horn|tenor\s+horn)\b", txt):
        return InstrumentPlaybackProfile("brass.alto-horn", -5, -9, 0, 3)
    if re.search(r"\b(?:french\s+horn|f\s+horn|horn|hn)\b", txt):
        return InstrumentPlaybackProfile("brass.french-horn", -4, -7, 0, 1)
    if re.search(r"\bbass\s+trombone\b|\bb\.?\s*tbn\b", txt):
        return InstrumentPlaybackProfile("brass.trombone.bass")

    explicit_bb_treble = bool(re.search(r"\bbb\b|\bin\s+bb\b", txt)) and clef == ("G", "2")
    if re.search(r"\btrombone\b|\btbn\b", txt):
        if explicit_bb_treble:
            return InstrumentPlaybackProfile("brass.trombone", -1, -2, -1, 2)
        return InstrumentPlaybackProfile("brass.trombone")
    if re.search(r"\b(?:baritone|euphonium|bar|euph)\b", txt):
        if explicit_bb_treble:
            return InstrumentPlaybackProfile("brass.euphonium", -1, -2, -1, 2)
        return InstrumentPlaybackProfile("brass.euphonium")
    if re.search(r"\b(?:tuba|tba|basses)\b", txt):
        return InstrumentPlaybackProfile("brass.tuba")
    return None


def infer_instrument_profile(
    name: str,
    clef: Optional[tuple[str, str]] = None,
) -> InstrumentPlaybackProfile:
    """Infer MusicXML ``<instrument-sound>`` and transposition intervals from a part name."""
    txt = (name or "").replace("♭", "b").replace("♯", "#")
    txt = re.sub(r"\s+", " ", txt).strip().lower()
    if not txt:
        return InstrumentPlaybackProfile("voice")

    if is_percussion_instrument(name):
        for pat, sid in _UNPITCHED_SOUND_RULES:
            if re.search(pat, txt):
                return InstrumentPlaybackProfile(sid)
        return InstrumentPlaybackProfile("drum.group")

    for pat, sid in _PITCHED_PERCUSSION_RULES:
        if re.search(pat, txt):
            return InstrumentPlaybackProfile(sid)

    ww = _infer_woodwind_profile(txt, clef)
    if ww is not None:
        return ww

    # String/Double/Electric Bass must be checked BEFORE Tuba / "Basses"
    if re.search(r"\b(?:string|double)\s+bass\b|\bcontrabass\b|\b(?:str|db)\.?\s*bass\b|\bcb\b", txt):
        return InstrumentPlaybackProfile("strings.contrabass", 0, 0, -1, 0)
    if re.search(r"\belectric\s+bass\b|\be\.\s*bass\b", txt):
        return InstrumentPlaybackProfile("pluck.bass.electric", 0, 0, -1, 0)

    br = _infer_brass_profile(txt, clef)
    if br is not None:
        return br

    for pat, prof in _OTHER_INST_RULES:
        if re.search(pat, txt):
            return prof

    return InstrumentPlaybackProfile("voice")


def _label_from_cleaned_text(cleaned: str, raw_joined: str) -> InstrumentLabel:
    if looks_abbreviated_label(cleaned):
        expanded = expand_abbreviated_name(cleaned)
        abbr = abbreviate_instrument_name(expanded)
        return InstrumentLabel(
            full_name=expanded,
            abbreviation=abbr,
            raw_ocr=raw_joined,
            is_fallback=False,
            is_abbreviated_source=True,
        )
    abbr = abbreviate_instrument_name(cleaned)
    return InstrumentLabel(
        full_name=cleaned,
        abbreviation=abbr,
        raw_ocr=raw_joined,
        is_fallback=False,
        is_abbreviated_source=False,
    )


def extract_instrument_labels(
    png_path: Path,
    staff_cys: Sequence[float],
    crop_x: int,
    median_gap: float,
    y_lo: float,
    y_hi: float,
    *,
    ocr_reader: Any = None,
    ocr_dir: Optional[Path] = None,
    ocr_runner: Optional[Callable[[np.ndarray], tuple[Sequence[Any], Sequence[str]]]] = None,
) -> list[InstrumentLabel]:
    """Run left-margin OCR on ``png_path`` and map labels cleanly to each staff."""
    if not staff_cys:
        return []

    fallbacks = [
        InstrumentLabel(full_name=f"Part {i + 1}", abbreviation=f"P{i + 1}", is_fallback=True)
        for i in range(len(staff_cys))
    ]

    img = cv2.imread(str(png_path))
    if img is None:
        log.warning("Could not read %s for instrument OCR; using fallback part names", png_path)
        return fallbacks

    safe_crop_x = max(1, min(int(crop_x), img.shape[1]))
    margin = img[:, :safe_crop_x]
    log.info("Running left-margin OCR on %s (0..%d px, median_gap=%.1f px)", Path(png_path).name, safe_crop_x, median_gap)

    if ocr_runner is not None:
        boxes_raw, txts_raw = ocr_runner(margin)
    else:
        reader = ocr_reader
        if reader is None:
            from rapidocr import RapidOCR

            params = {"Global.model_root_dir": str(ocr_dir)} if ocr_dir is not None else None
            reader = RapidOCR(params=params) if params is not None else RapidOCR()
        res = reader(margin)
        txts_raw = list(res.txts) if getattr(res, "txts", None) is not None else []
        boxes_raw = list(res.boxes) if getattr(res, "boxes", None) is not None else []

    assigned: list[list[tuple[float, str]]] = [[] for _ in staff_cys]
    for box, text in zip(boxes_raw, txts_raw):
        clean_t = str(text).strip()
        if not clean_t:
            continue
        cy = float(sum(float(pt[1]) for pt in box)) / max(1, len(box))
        if cy < y_lo or cy > y_hi:
            log.debug("OCR skip outside Y band (y=%.1f): %r", cy, clean_t)
            continue
        if len(clean_t) == 1 and clean_t.isdigit():
            log.debug("OCR skip single-digit stray (y=%.1f): %r", cy, clean_t)
            continue
        low = clean_t.lower()
        if any(tok in low for tok in _BOILERPLATE_TOKENS):
            log.debug("OCR skip boilerplate (y=%.1f): %r", cy, clean_t)
            continue

        dists = [abs(cy - scy) for scy in staff_cys]
        best_idx = int(np.argmin(dists))
        dist = dists[best_idx]
        if dist <= 0.65 * max(1.0, float(median_gap)):
            assigned[best_idx].append((cy, clean_t))
        else:
            log.debug("OCR skip unmatched (y=%.1f, min_dist=%.1f): %r", cy, dist, clean_t)

    labels: list[InstrumentLabel] = []
    missing = 0
    for idx, items in enumerate(assigned):
        if not items:
            missing += 1
            lbl = fallbacks[idx]
            labels.append(lbl)
            log.debug("  staff[%02d] (cy=%.1f): <NONE> -> fallback %r / %r", idx, staff_cys[idx], lbl.full_name, lbl.abbreviation)
            continue
        items_sorted = sorted(items, key=lambda x: x[0])
        parts_txt = [clean_instrument_label(t) for _, t in items_sorted]
        combined = parts_txt[0]
        for nxt in parts_txt[1:]:
            if nxt.startswith("(") or combined.lower() in ("mallet", "mall."):
                combined = f"{combined} {nxt}"
            else:
                combined = f"{combined} / {nxt}"
        raw_joined = " | ".join(t for _, t in items_sorted)
        lbl = _label_from_cleaned_text(combined, raw_joined)
        labels.append(lbl)
        log.debug(
            "  staff[%02d] (cy=%.1f): raw=%r -> full=%r | abbr=%r%s",
            idx,
            staff_cys[idx],
            raw_joined,
            lbl.full_name,
            lbl.abbreviation,
            " (expanded)" if lbl.is_abbreviated_source else "",
        )

    log.info("Instrument OCR summary: %d/%d matched (%d fallback)", len(labels) - missing, len(labels), missing)
    return labels


def _label_quality_score(lbl: InstrumentLabel) -> tuple[int, int, int, int]:
    """Higher tuple means better label quality."""
    is_valid = 0 if lbl.is_fallback else 1
    is_good = 0 if looks_low_quality_label(lbl.full_name) else 1
    is_full_src = 0 if lbl.is_abbreviated_source else 1
    return (is_valid, is_good, is_full_src, len(lbl.full_name))


def consolidate_page_labels(pages_labels: Sequence[list[InstrumentLabel]]) -> list[InstrumentLabel]:
    """Select the richest label page and repair any missing/garbled slots from sibling pages."""
    if not pages_labels:
        return []
    part_count = len(pages_labels[0])
    compatible = [pl for pl in pages_labels if len(pl) == part_count]
    if not compatible:
        return list(pages_labels[0])

    def _page_score(idx_pl: tuple[int, list[InstrumentLabel]]) -> tuple[int, int, int, int]:
        page_idx, pl = idx_pl
        good_count = sum(1 for lbl in pl if not lbl.is_fallback and not looks_low_quality_label(lbl.full_name))
        full_src_count = sum(1 for lbl in pl if not lbl.is_fallback and not lbl.is_abbreviated_source)
        total_len = sum(len(lbl.full_name) for lbl in pl if not lbl.is_fallback)
        return (good_count, full_src_count, total_len, -page_idx)

    best_idx, best_page = max(enumerate(compatible), key=_page_score)
    result = list(best_page)

    repaired = 0
    for slot in range(part_count):
        cur = result[slot]
        if not cur.is_fallback and not looks_low_quality_label(cur.full_name):
            continue
        candidates = [pl[slot] for pl in compatible]
        best_cand = max(candidates, key=_label_quality_score)
        if _label_quality_score(best_cand) > _label_quality_score(cur):
            result[slot] = best_cand
            repaired += 1
            log.info(
                "Repaired slot[%02d] label from %r to %r using sibling page",
                slot,
                cur.full_name,
                best_cand.full_name,
            )

    log.info(
        "Consolidated instrument labels from page #%d/%d (%d slot(s) repaired)",
        best_idx + 1,
        len(compatible),
        repaired,
    )
    return result
