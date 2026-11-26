from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from pdf2musescore.tempo import (
    PositionedTempo,
    TempoMarking,
    apply_tempo_marking,
    extract_tempo_marking,
    extract_tempo_markings_in_band,
    map_system_tempos_to_measures,
    parse_tempo_text,
)


@pytest.mark.parametrize(
    "raw,exp_text,exp_bpm,exp_unit,exp_dot",
    [
        ("Adagio affetuoso= 72", "Adagio affetuoso", 72, "quarter", False),
        ("Andante= 84", "Andante", 84, "quarter", False),
        ("Andante moderato (J=s4)", "Andante moderato", 84, "quarter", False),
        ("Andante moderato (J=,84)", "Andante moderato", 84, "quarter", False),
        ("5 Andante moderato (J= 84)", "Andante moderato", 84, "quarter", False),
        ("Allegro (q = 120)", "Allegro", 120, "quarter", False),
        ("Moderato ♩. = 66", "Moderato", 66, "quarter", True),
        ("Andante h = 58", "Andante", 58, "half", False),
        ("Presto e = 180", "Presto", 180, "eighth", False),
        ("= 120", "", 120, "quarter", False),
        ("BPM 108", "", 108, "quarter", False),
        ("120 BPM", "", 120, "quarter", False),
        ("M.M. 96", "", 96, "quarter", False),
        ("ca. = 104", "", 104, "quarter", False),
        ("With energy = 132", "With energy", 132, "quarter", False),
        ("Allegro ma non troppo", "Allegro ma non troppo", None, "quarter", False),
        ("Allegro (in t) alla Spagnola", "Allegro (in 1) alla Spagnola", None, "quarter", False),
        ("Allegro (in 1) alla Spagnola", "Allegro (in 1) alla Spagnola", None, "quarter", False),
        ("Andante moderato (J=??)", "Andante moderato", None, "quarter", False),
        ("Andante cantabile", "Andante cantabile", None, "quarter", False),
        ("Largo", "Largo", None, "quarter", False),
        ("Slowly", "Slowly", None, "quarter", False),
    ],
)
def test_parse_tempo_text_valid(raw, exp_text, exp_bpm, exp_unit, exp_dot):
    res = parse_tempo_text(raw)
    assert res is not None
    assert res.text == exp_text
    assert res.bpm == exp_bpm
    assert res.beat_unit == exp_unit
    assert res.beat_unit_dot is exp_dot


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "Prélude.",
        "SUITE II.",
        "Air",
        "dolce",
        "molto cantabile",
        "Bert APPERMONT",
        "Full score CONCERT BAND",
        "Copyright 2005 Edition",
        "64",
        "= 5",
        "= 15",
        "= 999",
        "= abc",
        "BPM 300",
        "Allegro ma non troppo con molto fuoco e passione sempre accelerando",
        "Allegro 123",
        "Allegro ma non troppo davvero molto subito",
    ],
)
def test_parse_tempo_text_rejects_non_tempo(raw):
    assert parse_tempo_text(raw) is None


def test_tempo_marking_quarter_bpm_and_serialization():
    assert TempoMarking(text="Adagio affetuoso", bpm=72).quarter_bpm() == 72
    assert TempoMarking(bpm=66, beat_unit="quarter", beat_unit_dot=True).quarter_bpm() == 99
    assert TempoMarking(bpm=58, beat_unit="half").quarter_bpm() == 116
    assert TempoMarking(bpm=60, beat_unit="half", beat_unit_dot=True).quarter_bpm() == 180
    assert TempoMarking(bpm=180, beat_unit="eighth").quarter_bpm() == 90
    assert TempoMarking(bpm=40, beat_unit="whole").quarter_bpm() == 160
    assert TempoMarking(text="Allegro ma non troppo").quarter_bpm() == 132
    assert TempoMarking(text="Andante moderato").quarter_bpm() == 84
    assert TempoMarking(text="A tempo").quarter_bpm() is None
    assert TempoMarking().quarter_bpm() is None

    tm = TempoMarking(text="Allegro", bpm=120, beat_unit="quarter", beat_unit_dot=False, raw_ocr="Allegro = 120")
    d = tm.to_dict()
    assert TempoMarking.from_dict(d) == tm
    assert TempoMarking.from_dict(None) is None
    assert TempoMarking.from_dict({}) is None
    assert TempoMarking.from_dict({"text": "Largo", "bpm": "bad", "beat_unit": "weird"}) == TempoMarking(
        text="Largo", bpm=None, beat_unit="quarter", beat_unit_dot=False, raw_ocr=""
    )


def test_extract_tempo_marking_horizontal_merge_and_ranking(tmp_path):
    assert extract_tempo_marking(tmp_path / "missing.png", 10.0) is None
    assert extract_tempo_marking(tmp_path / "missing.png", 150.0) is None

    png = tmp_path / "page_001.png"
    Image.new("RGB", (1000, 1200), (255, 255, 255)).save(png)

    # Empty OCR output returns None
    assert extract_tempo_marking(png, 250.0, ocr_runner=lambda _c: ([], [])) is None

    # Non-tempo text only returns None
    assert (
        extract_tempo_marking(
            png,
            250.0,
            ocr_runner=lambda _c: ([[[300, 40], [500, 40], [500, 80], [300, 80]]], ["SUITE II."]),
        )
        is None
    )

    # Horizontally split "Allegro" + "= 120" merges cleanly and outranks distant word-only text
    def fake_split_ocr(_crop):
        boxes = [
            [[150, 120], [260, 120], [260, 150], [150, 150]],  # Slow (higher up)
            [[160, 200], [270, 200], [270, 232], [160, 232]],  # Allegro
            [[282, 202], [360, 202], [360, 234], [282, 234]],  # = 120
            [[100, 50], [120, 50], [120, 70], [100, 70]],      # blank
        ]
        txts = ["Lento", "Allegro", "= 120", "  "]
        return boxes, txts

    found = extract_tempo_marking(png, 260.0, staff_left_x=150.0, ocr_runner=fake_split_ocr)
    assert found is not None
    assert found.text == "Allegro"
    assert found.bpm == 120

    # Also works with numpy boxes via ocr_reader
    class _FakeRes:
        boxes = np.array([[[120, 180], [380, 180], [380, 210], [120, 210]]])
        txts = ["Adagio affetuoso= 72"]

    found2 = extract_tempo_marking(png, 260.0, ocr_reader=lambda _c: _FakeRes())
    assert found2 is not None
    assert found2.text == "Adagio affetuoso"
    assert found2.bpm == 72


def test_apply_tempo_marking_xml_structure_and_guards():
    root = ET.fromstring(
        "<score-partwise><part id='P1'>"
        "<measure number='1'><attributes><divisions>4</divisions></attributes><note/></measure>"
        "<measure number='2'><print new-system='yes'/><note/></measure>"
        "</part></score-partwise>"
    )
    assert apply_tempo_marking(root, None) is False
    assert apply_tempo_marking(root, TempoMarking()) is False
    assert apply_tempo_marking(ET.Element("score-partwise"), TempoMarking(text="Largo")) is False
    assert apply_tempo_marking(root, TempoMarking(text="Largo"), measure_index=99) is False

    tm1 = TempoMarking(text="Adagio affetuoso", bpm=72, beat_unit="quarter", beat_unit_dot=True)
    assert apply_tempo_marking(root, tm1, measure_index=0) is True
    # Idempotent: second call on same measure returns False
    assert apply_tempo_marking(root, tm1, measure_index=0) is False

    m1 = root.find("./part/measure[@number='1']")
    assert [c.tag for c in m1] == ["attributes", "direction", "note"]
    d = m1.find("direction")
    assert d.get("placement") == "above"
    assert d.findtext("./direction-type/words") == "Adagio affetuoso"
    assert d.findtext("./direction-type/metronome/beat-unit") == "quarter"
    assert d.find("./direction-type/metronome/beat-unit-dot") is not None
    assert d.findtext("./direction-type/metronome/per-minute") == "72"
    assert d.find("sound").get("tempo") == "108"

    # Measure 2 has <print> without <attributes>, and word-only tempo without explicit BPM
    assert apply_tempo_marking(root, TempoMarking(text="Allegro"), measure_index=1) is True
    m2 = root.find("./part/measure[@number='2']")
    assert [c.tag for c in m2] == ["print", "direction", "note"]
    assert m2.findtext("./direction/direction-type/words") == "Allegro"
    assert m2.find("./direction/direction-type/metronome") is None
    assert m2.find("./direction/sound").get("tempo") == "132"


def test_extract_tempo_markings_in_band_and_map_to_measures():
    img = np.full((600, 1200, 3), 255, dtype=np.uint8)
    assert extract_tempo_markings_in_band(img, 100, 105) == []
    assert extract_tempo_markings_in_band(img, 100, 200, ocr_runner=lambda _c: ([], [])) == []

    # Returns both left and right markings on the same system, deduplicating overlapping candidates
    def _band_ocr(_c):
        boxes = [
            [[140, 20], [420, 20], [420, 50], [140, 50]],   # Andante maestoso (J=72)
            [[145, 5], [380, 5], [380, 22], [145, 22]],     # overlapping word-only "Andante" (dropped in favor of BPM)
            [[820, 20], [1080, 20], [1080, 50], [820, 50]], # Allegro (in t) alla Spagnola
            [[450, 10], [650, 10], [650, 35], [450, 35]],   # non-tempo title
        ]
        txts = ["Andante maestoso (J=72)", "Andante", "Allegro (in t) alla Spagnola", "WEST SIDE STORY"]
        return boxes, txts

    pts = extract_tempo_markings_in_band(
        img,
        10,
        100,
        sys_min_x=100.0,
        sys_max_x=1100.0,
        ocr_runner=_band_ocr,
        source_name="p1.png",
    )
    assert len(pts) == 2
    assert pts[0].tempo.text == "Andante maestoso" and pts[0].tempo.bpm == 72 and pts[0].u_norm < 0.12
    assert pts[1].tempo.text == "Allegro (in 1) alla Spagnola" and pts[1].u_norm > 0.65

    # map_system_tempos_to_measures snaps mid-system tempo to double-barline/key-change boundary
    xml = """<score-partwise><part id="P1">
      <measure number="1"><attributes><key><fifths>2</fifths></key></attributes><note/></measure>
      <measure number="2"><note/></measure>
      <measure number="3"><note/></measure>
      <measure number="4"><note/></measure>
      <measure number="5"><note/><barline location="right"><bar-style>light-light</bar-style></barline></measure>
      <measure number="6"><attributes><key><fifths>-2</fifths></key></attributes><note/></measure>
      <measure number="7"><note/></measure>
      <measure number="8"><print new-system="yes"/><note/></measure>
      <measure number="9"><note/></measure>
      <measure number="10"><note/></measure>
    </part></score-partwise>"""
    score_root = ET.fromstring(xml)
    sys1_right = PositionedTempo(TempoMarking(text="Andante moderato", bpm=84), 700, 950, 200, 230, u_norm=0.58)
    assert map_system_tempos_to_measures(ET.Element("score-partwise"), [pts]) == 0
    assert map_system_tempos_to_measures(ET.fromstring("<score-partwise><part id='P1'/></score-partwise>"), [pts]) == 0

    injected = map_system_tempos_to_measures(score_root, [pts, [sys1_right], []])
    assert injected == 3
    ms = score_root.findall("./part/measure")
    assert ms[0].findtext("./direction/direction-type/words") == "Andante maestoso"
    assert ms[5].findtext("./direction/direction-type/words") == "Allegro (in 1) alla Spagnola"
    assert ms[8].findtext("./direction/direction-type/words") == "Andante moderato"
    assert ms[8].find("./direction/sound").get("tempo") == "84"
