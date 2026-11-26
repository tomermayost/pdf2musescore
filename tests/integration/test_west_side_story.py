"""Integration: West Side Story (Baritone TC) -> homr AI + OCR + MuseScore 4."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from pdf2musescore.config import AppConfig
from pdf2musescore.deps import MUSESCORE, find_tool
from pdf2musescore.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
WSS_PDF = Path(r"C:\Users\tmayo\Desktop\תווים\west side story\west side story - bariton tc.pdf")
LOCAL_MODELS = Path(r"C:\Users\tmayo\AppData\Local\pdf2musescore\models")
OUT = ROOT / "tmp" / "integration" / "west_side_story"

pytestmark = pytest.mark.integration


def test_west_side_story_baritone_time_sigs_tempos_and_courtesy_sharp():
    if not WSS_PDF.is_file():
        pytest.skip(f"Test PDF missing: {WSS_PDF}")
    if not find_tool(MUSESCORE):
        pytest.skip("MuseScore required")

    work_dir = OUT / "work"
    shutil.rmtree(work_dir, ignore_errors=True)
    models_dir = LOCAL_MODELS if LOCAL_MODELS.is_dir() else (ROOT / "tmp" / "models")
    cfg = AppConfig(work_dir=work_dir, models_dir=models_dir, log_level="DEBUG")
    out_mscz = OUT / "west_side_story_baritone.mscz"
    out_mscz.parent.mkdir(parents=True, exist_ok=True)

    result = Pipeline(cfg).convert(WSS_PDF, out_mscz)
    assert result == out_mscz and zipfile.is_zipfile(out_mscz)

    staged_jobs = sorted(work_dir.glob("job-*/score.musicxml"), key=lambda p: p.stat().st_mtime)
    assert staged_jobs
    xml_root = ET.parse(staged_jobs[-1]).getroot()

    # 1) Instrument must be BARITONE (not Voice) with Bb treble-clef (-14) transposition in M1 only
    sp = xml_root.find("./part-list/score-part")
    assert sp is not None
    assert "baritone" in (sp.findtext("part-name") or "").lower()
    assert sp.findtext("./score-instrument/instrument-sound") == "brass.euphonium"
    assert len(xml_root.findall(".//transpose")) == 1
    assert xml_root.findtext("./part/measure[@number='1']/attributes/transpose/diatonic") == "-1"
    assert xml_root.findtext("./part/measure[@number='1']/attributes/transpose/chromatic") == "-2"
    assert xml_root.findtext("./part/measure[@number='1']/attributes/transpose/octave-change") == "-1"

    # 2) Time signatures: 4/4 at M1 -> 3/4 at rehearsal [1] -> 4/4 at rehearsal [5]
    xml_ms = xml_root.findall("./part/measure")
    assert len(xml_ms) >= 75
    ts_changes = [
        (i + 1, m.findtext("./attributes/time/beats"), m.findtext("./attributes/time/beat-type"))
        for i, m in enumerate(xml_ms)
        if m.find("./attributes/time") is not None
    ]
    assert ts_changes[0] == (1, "4", "4")
    assert len(ts_changes) >= 3
    assert ts_changes[1][1:] == ("3", "4") and ts_changes[1][0] in (6, 7)
    assert ts_changes[2][1:] == ("4", "4") and ts_changes[2][0] >= 73

    # 3) Courtesy sharp on Bar 5 F#4 (after Bar 4 F-natural)
    bar5_acc = xml_ms[4].find("./note/accidental")
    assert bar5_acc is not None
    assert bar5_acc.text == "sharp"
    assert bar5_acc.get("cautionary") == "yes"

    # 4) All 3 tempos detected and single-page title not stamped on Bar 1
    tempo_words = [
        (w.text or "").strip()
        for w in xml_root.findall("./part/measure/direction/direction-type/words")
    ]
    assert "WEST SIDE STORY" not in tempo_words
    joined_tempos = " | ".join(tempo_words).lower()
    assert "andante maestoso" in joined_tempos
    assert "allegro" in joined_tempos and "spagnola" in joined_tempos
    assert "andante moderato" in joined_tempos

    sound_bpms = [s.get("tempo") for s in xml_root.findall("./part/measure/direction/sound[@tempo]")]
    assert "72" in sound_bpms
    assert "84" in sound_bpms

    # 4b) Colliding dynamics removed on Bar 77 & expression markings tranq. (Bar 76) and rit. (Bar 77)
    assert xml_root.findall(".//other-dynamics") == []
    bar76_words = [
        (w.text or "").strip().lower()
        for w in xml_ms[75].findall("./direction/direction-type/words[@font-style='italic']")
    ]
    bar77_words = [
        (w.text or "").strip().lower()
        for w in xml_ms[76].findall("./direction/direction-type/words[@font-style='italic']")
    ]
    assert "tranq." in bar76_words
    assert "rit." in bar77_words

    # 4c) Boxed rehearsal marks [1]..[5] on Page 1 mapped cleanly and zero false-positive unboxed marks
    reh_bars = [
        (i + 1, (r.text or "").strip(), r.get("enclosure"))
        for i, m in enumerate(xml_ms)
        for r in m.findall("./direction/direction-type/rehearsal")
    ]
    assert all(enc == "square" for _, _, enc in reh_bars), f"Unexpected unboxed marks: {reh_bars}"
    reh_labels = {lbl: (bar, enc) for bar, lbl, enc in reh_bars}
    for expected_bar, expected_lbl in ((6, "1"), (25, "2"), (41, "3"), (57, "4"), (74, "5")):
        assert expected_lbl in reh_labels, f"Missing rehearsal mark {expected_lbl} (found: {reh_bars})"
        assert reh_labels[expected_lbl][0] == expected_bar
        assert reh_labels[expected_lbl][1] == "square"

    # 5) MuseScore 4 (.mscx) verification
    with zipfile.ZipFile(out_mscz) as z:
        mscx_name = next(n for n in z.namelist() if n.endswith(".mscx"))
        mscx_root = ET.fromstring(z.read(mscx_name))

    assert mscx_root.findtext("./Score/Part/Instrument/instrumentId") != "voice.vocals"
    assert mscx_root.findtext("./Score/Part/Instrument/transposeDiatonic") == "-8"
    assert mscx_root.findtext("./Score/Part/Instrument/transposeChromatic") == "-14"
    mscx_ms = mscx_root.findall("./Score/Staff/Measure")
    page1_pitches = [int(n.findtext("pitch") or "0") for m in mscx_ms[:77] for n in m.findall(".//Note")]
    assert all(40 <= p <= 70 for p in page1_pitches), f"Out-of-range pitches on p1: {[p for p in page1_pitches if not (40 <= p <= 70)]}"
    # Bars 1-5 and final 4 bars must have no overfull len mismatch
    for idx in (0, 1, 2, 3, 4, len(mscx_ms) - 4, len(mscx_ms) - 3, len(mscx_ms) - 2, len(mscx_ms) - 1):
        assert mscx_ms[idx].get("len") is None, f"Measure {idx + 1} still has overfull len={mscx_ms[idx].get('len')}"

    # Bar 5 has visible accidentalSharp in MuseScore
    assert mscx_ms[4].findtext(".//Note/Accidental/subtype") == "accidentalSharp"

    # Bar 77 dynamic has only 'mp' (no colliding 'più') and tranq./rit. appear on bars 76/77
    all_dyn_subtypes = [d.findtext("subtype") for d in mscx_root.findall(".//Dynamic")]
    assert not any("pi" in (s or "").lower() for s in all_dyn_subtypes)
    tranq_idx = next(
        i for i, m in enumerate(mscx_ms)
        if "tranq." in "".join("".join(st.itertext()) for st in m.findall(".//StaffText/text")).lower()
    )
    rit_m = mscx_ms[tranq_idx + 1]
    assert "rit." in "".join("".join(st.itertext()) for st in rit_m.findall(".//StaffText/text")).lower()
    assert "mp" in [d.findtext("subtype") for d in rit_m.findall(".//Dynamic")]

    # All 3 <Tempo> elements imported into MuseScore with spacing before metronome symbol
    mscx_tempos = mscx_root.findall(".//Tempo")
    mscx_tempo_texts = ["".join(t.itertext()).lower() for t in mscx_tempos]
    assert any("andante maestoso" in t for t in mscx_tempo_texts)
    assert any("spagnola" in t for t in mscx_tempo_texts)
    assert any("andante moderato" in t for t in mscx_tempo_texts)
    moderato_xml = ET.tostring(next(t for t in mscx_tempos if "andante moderato" in "".join(t.itertext()).lower()), encoding="unicode")
    assert "Andante moderato " in moderato_xml

    # Rehearsal marks [1]..[5] imported as real MuseScore <RehearsalMark> elements
    mscx_reh = ["".join(r.itertext()).strip() for r in mscx_root.findall(".//RehearsalMark/text")]
    for lbl in ("1", "2", "3", "4", "5"):
        assert lbl in mscx_reh, f"Expected RehearsalMark {lbl!r} in .mscx (found {mscx_reh})"
