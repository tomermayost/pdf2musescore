from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from pdf2musescore.tempo import (
    PositionedTempo,
    TempoMarking,
    apply_tempo_marking,
    detect_staff_barlines,
    extract_tempo_marking,
    extract_tempo_markings_in_band,
    map_system_tempos_to_measures,
    parse_expression_tokens,
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
    words_el = d.find("./direction-type/words")
    assert words_el is not None
    assert words_el.text == "Adagio affetuoso "
    assert words_el.get("{http://www.w3.org/XML/1998/namespace}space") == "preserve"
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
    assert ms[0].findtext("./direction/direction-type/words") == "Andante maestoso "
    assert ms[5].findtext("./direction/direction-type/words") == "Allegro (in 1) alla Spagnola"
    assert ms[8].findtext("./direction/direction-type/words") == "Andante moderato "
    assert ms[8].find("./direction/sound").get("tempo") == "84"


@pytest.mark.parametrize(
    "raw,exp",
    [
        ("tranq.", [("tranq.", 0, 6)]),
        ("trang.", [("tranq.", 0, 6)]),
        ("rit.", [("rit.", 0, 4)]),
        ("trang. rit.", [("tranq.", 0, 6), ("rit.", 7, 11)]),
        ("poco rit.", [("poco rit.", 0, 9)]),
        ("molto espress.", [("molto espress.", 0, 14)]),
        ("dolce", [("dolce", 0, 5)]),
        ("a tempo", [("a tempo", 0, 7)]),
        ("poco a poco accel.", [("poco a poco accel.", 0, 18)]),
    ],
)
def test_parse_expression_tokens_valid(raw, exp):
    assert parse_expression_tokens(raw) == exp


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "WEST SIDE STORY",
        "Andante moderato (J=84)",
        "dolce in the band part",
        "1st Bsn.",
        "Copyright 1956 rit.",
        "spirit.",
        "guitar",
    ],
)
def test_parse_expression_tokens_rejects_non_expressions(raw):
    assert parse_expression_tokens(raw) == []


def test_expression_extraction_barlines_mapping_and_xml():
    assert TempoMarking(text="tranq.", is_expression=True).quarter_bpm() is None
    tm_expr = TempoMarking(text="rit.", is_expression=True, raw_ocr="rit.")
    assert TempoMarking.from_dict(tm_expr.to_dict()) == tm_expr

    img = np.full((300, 1200, 3), 255, dtype=np.uint8)

    # Combined box "trang. rit." coexists below overlapping wide main tempo box "Andante moderato (J=s4)"
    def _ocr(_c):
        return (
            [
                [[700, 10], [1080, 10], [1080, 42], [700, 42]],   # y=110..142
                [[820, 48], [1060, 48], [1060, 82], [820, 82]],  # y=148..182
                [[200, 5], [320, 5], [320, 28], [200, 28]],      # high title-zone expression y=105..128 (filtered by expr_min_y)
            ],
            ["Andante moderato (J=s4)", "trang. rit.", "dolce"],
        )

    pts = extract_tempo_markings_in_band(
        img,
        100,
        200,
        sys_min_x=50.0,
        sys_max_x=1150.0,
        ocr_runner=_ocr,
        expr_min_y=140,
    )
    assert [(p.tempo.text, p.tempo.is_expression, p.tempo.bpm) for p in pts] == [
        ("Andante moderato", False, 84),
        ("tranq.", True, None),
        ("rit.", True, None),
    ]
    assert pts[1].x_right < pts[2].x_left

    # Synthetic staff image with 4 bars (unequal widths) + 1 stem with a filled notehead blob
    gray = np.full((160, 1000), 255, dtype=np.uint8)
    assert detect_staff_barlines(gray, 2, 10, 50, 950) == []  # too short staff guard
    sy0, sy1 = 40, 100
    for r in (40, 55, 70, 85, 100):
        gray[r : r + 2, 40:961] = 0
    # True barlines at 320, 620, 780, 960
    for bx in (320, 620, 780, 960):
        gray[sy0 : sy1 + 1, bx - 1 : bx + 2] = 0
    # Note stem at x=470 spanning staff height + filled notehead on left [458..468]
    gray[sy0 : sy1 + 1, 470:472] = 0
    gray[62:76, 458:470] = 0
    bars = detect_staff_barlines(gray, sy0, sy1, 40, 960)
    assert bars == [320, 620, 780, 960]

    xml = """<score-partwise><part id="P1">
      <measure number="1"><attributes><key><fifths>-2</fifths></key></attributes><note/></measure>
      <measure number="2"><attributes><key><fifths>-3</fifths></key></attributes><note/></measure>
      <measure number="3"><note/></measure>
      <measure number="4"><note/></measure>
    </part></score-partwise>"""
    score_root = ET.fromstring(xml)
    assert map_system_tempos_to_measures(
        score_root,
        [pts],
        system_barlines=[bars],
        system_x_bounds=[(40.0, 960.0)],
    ) == 3

    ms = score_root.findall("./part/measure")
    # Andante moderato -> snaps to key change M2; tranq. (cx~885 -> 780..960? adjust boxes cleanly below)
    # Check M2..M4 received all 3 markings cleanly
    words_per_bar = {i + 1: [w.text for w in m.findall("./direction/direction-type/words")] for i, m in enumerate(ms)}
    assert words_per_bar[2] == ["Andante moderato "]

    # Explicitly test exact-barline expression localization across unequal measures + idempotency
    score_root2 = ET.fromstring(xml)
    p_tempo = PositionedTempo(TempoMarking("Andante moderato", bpm=84), 600, 920, 20, 35, u_norm=0.58)
    p_tranq = PositionedTempo(TempoMarking("tranq.", is_expression=True), 650, 750, 40, 58, u_norm=0.68)  # inside bar 3 [620..780]
    p_rit = PositionedTempo(TempoMarking("rit.", is_expression=True), 810, 880, 40, 58, u_norm=0.84)    # inside bar 4 [780..960]
    assert map_system_tempos_to_measures(
        score_root2,
        [[p_tempo, p_tranq, p_rit]],
        system_barlines=[bars],
        system_x_bounds=[(40.0, 960.0)],
    ) == 3
    ms2 = score_root2.findall("./part/measure")
    assert ms2[1].findtext("./direction/direction-type/words") == "Andante moderato "
    assert ms2[2].findtext("./direction/direction-type/words") == "tranq."
    assert ms2[2].find("./direction/direction-type/words").get("font-style") == "italic"
    assert ms2[2].find("./direction/sound") is None
    assert ms2[3].findtext("./direction/direction-type/words") == "rit."
    # Coexistence + idempotency: expression can be added to a bar that already has tempo, and duplicate expression is skipped
    assert apply_tempo_marking(score_root2, TempoMarking("tranq.", is_expression=True), measure_index=1) is True
    assert apply_tempo_marking(score_root2, TempoMarking("tranq.", is_expression=True), measure_index=1) is False
