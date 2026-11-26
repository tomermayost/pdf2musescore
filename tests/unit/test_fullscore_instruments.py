from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pdf2musescore.fullscore.instruments import (
    InstrumentLabel,
    InstrumentPlaybackProfile,
    abbreviate_instrument_name,
    clean_instrument_label,
    consolidate_page_labels,
    expand_abbreviated_name,
    extract_header_instrument_label,
    extract_instrument_labels,
    infer_instrument_profile,
    is_percussion_instrument,
    is_recognized_instrument_label,
    looks_abbreviated_label,
    looks_low_quality_label,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Bup t 1", "Bb Trumpet 1"),
        ("E> Alto Clarinet", "Eb Alto Clarinet"),
        ("B. Bass Clarinet", "Bb Bass Clarinet"),
        ("B- Tenor Sax.", "Bb Tenor Sax."),
        ("F Baritone Sax.", "Eb Baritone Sax."),
        ("F Hlorn 2", "F Horn 2"),
        ("F HIorn 2", "F Horn 2"),
        ("FI. 1, 2 / Pice.", "Fl. 1, 2 / Picc."),
        ("B CI. 1", "Bb Cl. 1"),
        ("Pere. 2", "Perc. 2"),
        ("Perc.1", "Perc. 1"),
        ("B Tpt. 1 / B Crnt. 1", "Bb Tpt. 1 / Bb Cnt. 1"),
        ("E A. Sx. 1", "Eb A. Sax. 1"),
    ],
)
def test_clean_instrument_label(raw, expected):
    assert clean_instrument_label(raw) == expected


@pytest.mark.parametrize(
    "full_name,expected_abbr",
    [
        ("Flute 1, 2 / Piccolo", "Fl. 1, 2 / Picc."),
        ("Oboe 1, 2", "Ob. 1, 2"),
        ("Bassoon", "Bsn."),
        ("Eb Alto Clarinet", "Eb A. Cl."),
        ("Bb Bass Clarinet", "Bb B. Cl."),
        ("Eb Baritone Saxophone", "Eb Bar. Sax."),
        ("French Horn 1, 3", "Hn. 1, 3"),
        ("C Bass Trombone", "C B. Tbn."),
        ("String Bass (optional)", "Str. Bass"),
        ("Mallet Percussion", "Mall. Perc."),
        ("Timpani", "Timp."),
        ("Percussion 1", "Perc. 1"),
        ("Snare Drum", "S.D."),
    ],
)
def test_abbreviate_instrument_name(full_name, expected_abbr):
    assert abbreviate_instrument_name(full_name) == expected_abbr


@pytest.mark.parametrize(
    "abbr,expected_full",
    [
        ("Fl. 1, 2 / Picc.", "Flute 1, 2 / Piccolo"),
        ("Ob. 1, 2", "Oboe 1, 2"),
        ("Bsn.", "Bassoon"),
        ("Bb Cl. 1", "Bb Clarinet 1"),
        ("Eb A. Sax. 1", "Eb Alto Sax. 1"),
        ("Bb T. Sax.", "Bb Tenor Sax."),
        ("Eb Bar. Sax.", "Eb Baritone Sax."),
        ("Bb Tpt. 1 / Bb Cnt. 1", "Bb Trumpet 1 / Bb Cornet 1"),
        ("F Hn. 1, 3", "F Horn 1, 3"),
        ("C Tbn. 1", "C Trombone 1"),
        ("C B. Tbn.", "C Bass Trombone"),
        ("C Bar.", "C Baritone"),
        ("Str. Bass (opt.)", "String Bass (optional)"),
        ("Timp.", "Timpani"),
        ("Mall. Perc.", "Mallet Percussion"),
        ("Perc. 1", "Percussion 1"),
    ],
)
def test_expand_abbreviated_name(abbr, expected_full):
    assert expand_abbreviated_name(abbr) == expected_full


def test_percussion_and_quality_helpers():
    assert is_percussion_instrument("Percussion 1") is True
    assert is_percussion_instrument("Snare Drum") is True
    assert is_percussion_instrument("Cymbals") is True
    # Pitched percussion must NOT be assigned MIDI channel 10
    assert is_percussion_instrument("Timpani") is False
    assert is_percussion_instrument("Mallet Percussion") is False
    assert is_percussion_instrument("Glockenspiel") is False
    assert is_percussion_instrument("Marimba") is False
    assert is_percussion_instrument("Flute") is False
    assert is_percussion_instrument("") is False

    assert looks_abbreviated_label("Ob. 1, 2") is True
    assert looks_abbreviated_label("Oboe 1, 2") is False
    assert looks_abbreviated_label("") is False

    assert looks_low_quality_label("DG") is True
    assert looks_low_quality_label("Part 9") is True
    assert looks_low_quality_label("B2") is True
    assert looks_low_quality_label("Fl. 1") is False

    assert is_recognized_instrument_label("Violin I") is True
    assert is_recognized_instrument_label("Bb Clarinet 1") is True
    assert is_recognized_instrument_label("Voice") is True
    assert is_recognized_instrument_label("Recorder") is True
    assert is_recognized_instrument_label("Prélude.") is False
    assert is_recognized_instrument_label("Prélude. / 262") is False
    assert is_recognized_instrument_label("Part 1") is False
    assert is_recognized_instrument_label("DG") is False
    assert is_recognized_instrument_label("Allegro") is False
    assert is_recognized_instrument_label("") is False


def test_extract_instrument_labels_full_and_abbreviated(tmp_path, caplog):
    assert extract_instrument_labels(tmp_path / "missing.png", [], 200, 120.0, 0.0, 500.0) == []

    missing_img = extract_instrument_labels(tmp_path / "missing.png", [100.0, 250.0], 200, 150.0, 0.0, 400.0)
    assert [l.full_name for l in missing_img] == ["Part 1", "Part 2"]
    assert all(l.is_fallback for l in missing_img)

    png = tmp_path / "page_004.png"
    Image.new("RGB", (800, 1200), (255, 255, 255)).save(png)

    def fake_ocr(_margin):
        boxes = [
            [[0, 20], [50, 20], [50, 40], [0, 40]],      # out of Y band
            [[0, 190], [50, 190], [50, 210], [0, 210]],  # Full score boilerplate
            [[0, 195], [20, 195], [20, 215], [0, 215]],  # single digit '1'
            [[0, 200], [40, 200], [40, 220], [0, 220]],  # multi-digit plate/page number '262'
            [[0, 190], [60, 190], [60, 210], [0, 210]],  # Flute 1, 2
            [[0, 220], [60, 220], [60, 240], [0, 240]],  # Piccolo (same staff)
            [[0, 355], [60, 355], [60, 375], [0, 375]],  # Mallet
            [[0, 380], [60, 380], [60, 400], [0, 400]],  # Percussion
            [[0, 620], [60, 620], [60, 640], [0, 640]],  # too far from any staff
        ]
        txts = ["", "Full score", "1", "262", "Flute 1, 2", "Piccolo", "Mallet", "Percussion", "Bassoon"]
        # Need first item non-empty for Y-band test
        txts[0] = "Header"
        return boxes, txts

    caplog.clear()
    with caplog.at_level("WARNING", logger="pdf2musescore.fullscore.instruments"):
        labels = extract_instrument_labels(
            png,
            staff_cys=[210.0, 370.0, 520.0],
            crop_x=250,
            median_gap=150.0,
            y_lo=100.0,
            y_hi=700.0,
            ocr_runner=fake_ocr,
        )
    assert caplog.records == []
    assert len(labels) == 3
    assert labels[0].full_name == "Flute 1, 2 / Piccolo"
    assert labels[0].abbreviation == "Fl. 1, 2 / Picc."
    assert labels[1].full_name == "Mallet Percussion"
    assert labels[1].abbreviation == "Mall. Perc."
    assert labels[2].is_fallback is True

    # Abbreviated source page expands for full_name
    def fake_abbr_ocr(_margin):
        return (
            [
                [[0, 200], [60, 200], [60, 220], [0, 220]],
                [[0, 360], [60, 360], [60, 380], [0, 380]],
            ],
            ["Ob. 1, 2", "Timp."],
        )

    abbr_labels = extract_instrument_labels(
        png,
        staff_cys=[210.0, 370.0],
        crop_x=250,
        median_gap=150.0,
        y_lo=100.0,
        y_hi=500.0,
        ocr_runner=fake_abbr_ocr,
    )
    assert abbr_labels[0].full_name == "Oboe 1, 2"
    assert abbr_labels[0].abbreviation == "Ob. 1, 2"
    assert abbr_labels[0].is_abbreviated_source is True
    assert abbr_labels[1].full_name == "Timpani"
    assert abbr_labels[1].abbreviation == "Timp."


def test_consolidate_page_labels_repairs_gaps_and_prefers_full_names():
    assert consolidate_page_labels([]) == []

    p1_abbr = [
        InstrumentLabel("Oboe 1, 2", "Ob. 1, 2", is_abbreviated_source=True),
        InstrumentLabel("Part 2", "P2", is_fallback=True),
        InstrumentLabel("DG", "DG"),
    ]
    p2_full = [
        InstrumentLabel("Oboe 1, 2", "Ob. 1, 2", is_abbreviated_source=False),
        InstrumentLabel("Bassoon", "Bsn."),
        InstrumentLabel("Eb Clarinet", "Eb Cl."),
    ]
    consolidated = consolidate_page_labels([p1_abbr, p2_full])
    assert [l.full_name for l in consolidated] == ["Oboe 1, 2", "Bassoon", "Eb Clarinet"]

    # Repair slot in richest page using sibling page
    p_a = [
        InstrumentLabel("Flute", "Fl."),
        InstrumentLabel("Part 2", "P2", is_fallback=True),
    ]
    p_b = [
        InstrumentLabel("Part 1", "P1", is_fallback=True),
        InstrumentLabel("Oboe", "Ob."),
    ]
    repaired = consolidate_page_labels([p_a, p_b])
    assert [l.full_name for l in repaired] == ["Flute", "Oboe"]


@pytest.mark.parametrize(
    "name,clef,expected",
    [
        ("Percussion 1", None, InstrumentPlaybackProfile("drum.group", midi_program=0)),
        ("Snare Drum", None, InstrumentPlaybackProfile("drum.snare-drum", midi_program=0)),
        ("Bass Drum", None, InstrumentPlaybackProfile("drum.bass-drum", midi_program=0)),
        ("Crash Cymbals", None, InstrumentPlaybackProfile("metal.cymbal.crash", midi_program=0)),
        ("Triangle", None, InstrumentPlaybackProfile("metal.triangle", midi_program=0)),
        ("Tambourine", None, InstrumentPlaybackProfile("drum.tambourine", midi_program=0)),
        ("Drum Set", None, InstrumentPlaybackProfile("drum.group.set", midi_program=0)),
        ("Timpani", None, InstrumentPlaybackProfile("drum.timpani", midi_program=48)),
        ("Mallet Percussion", None, InstrumentPlaybackProfile("pitched-percussion.marimba", midi_program=13)),
        ("Glockenspiel", None, InstrumentPlaybackProfile("pitched-percussion.glockenspiel", midi_program=10)),
        ("Xylophone", None, InstrumentPlaybackProfile("pitched-percussion.xylophone", midi_program=14)),
        ("Vibraphone", None, InstrumentPlaybackProfile("pitched-percussion.vibraphone", midi_program=12)),
        ("Celesta", None, InstrumentPlaybackProfile("keyboard.celesta", midi_program=9)),
        ("Tubular Bells", None, InstrumentPlaybackProfile("pitched-percussion.tubular-bells", midi_program=15)),
        ("Flute 1, 2 / Piccolo", None, InstrumentPlaybackProfile("wind.flutes.flute", midi_program=74)),
        ("Piccolo", None, InstrumentPlaybackProfile("wind.flutes.flute.piccolo", 0, 0, 1, 0, midi_program=73)),
        ("Alto Flute", None, InstrumentPlaybackProfile("wind.flutes.flute.alto", -3, -5, 0, -1, midi_program=74)),
        ("Flute", None, InstrumentPlaybackProfile("wind.flutes.flute", midi_program=74)),
        ("English Horn", None, InstrumentPlaybackProfile("wind.reed.english-horn", -4, -7, 0, 1, midi_program=70)),
        ("Oboe 1, 2", None, InstrumentPlaybackProfile("wind.reed.oboe", midi_program=69)),
        ("Contrabassoon", None, InstrumentPlaybackProfile("wind.reed.contrabassoon", 0, 0, -1, 0, midi_program=71)),
        ("Bassoon", None, InstrumentPlaybackProfile("wind.reed.bassoon", midi_program=71)),
        ("Eb Contrabass Clarinet", None, InstrumentPlaybackProfile("wind.reed.clarinet.contrabass", -5, -9, -1, 3, midi_program=72)),
        ("Bb Contrabass Clarinet", None, InstrumentPlaybackProfile("wind.reed.clarinet.contrabass", -1, -2, -2, 2, midi_program=72)),
        ("B♭ Bass Clarinet", ("G", "2"), InstrumentPlaybackProfile("wind.reed.clarinet.bass", -1, -2, -1, 2, midi_program=72)),
        ("Bb Bass Clarinet", ("F", "4"), InstrumentPlaybackProfile("wind.reed.clarinet.bass", -1, -2, 0, 2, midi_program=72)),
        ("Eb Alto Clarinet", None, InstrumentPlaybackProfile("wind.reed.clarinet.alto", -5, -9, 0, 3, midi_program=72)),
        ("Eb Clarinet", None, InstrumentPlaybackProfile("wind.reed.clarinet.eflat", 2, 3, 0, 3, midi_program=72)),
        ("A Clarinet", None, InstrumentPlaybackProfile("wind.reed.clarinet.a", -2, -3, 0, -3, midi_program=72)),
        ("C Clarinet", None, InstrumentPlaybackProfile("wind.reed.clarinet", midi_program=72)),
        ("Bb Clarinet 1", None, InstrumentPlaybackProfile("wind.reed.clarinet.bflat", -1, -2, 0, 2, midi_program=72)),
        ("Bb Soprano Sax.", None, InstrumentPlaybackProfile("wind.reed.saxophone.soprano", -1, -2, 0, 2, midi_program=65)),
        ("Eb Alto Sax. 1", None, InstrumentPlaybackProfile("wind.reed.saxophone.alto", -5, -9, 0, 3, midi_program=66)),
        ("Bb Tenor Sax.", None, InstrumentPlaybackProfile("wind.reed.saxophone.tenor", -1, -2, -1, 2, midi_program=67)),
        ("Eb Baritone Sax.", None, InstrumentPlaybackProfile("wind.reed.saxophone.baritone", -5, -9, -1, 3, midi_program=68)),
        ("Saxophone", None, InstrumentPlaybackProfile("wind.reed.saxophone.alto", -5, -9, 0, 3, midi_program=66)),
        ("String Bass (optional)", None, InstrumentPlaybackProfile("strings.contrabass", 0, 0, -1, 0, midi_program=44)),
        ("Bass", None, InstrumentPlaybackProfile("strings.contrabass", 0, 0, -1, 0, midi_program=44)),
        ("Electric Bass", None, InstrumentPlaybackProfile("pluck.bass.electric", 0, 0, -1, 0, midi_program=34)),
        ("D Trumpet", None, InstrumentPlaybackProfile("brass.trumpet.d", 1, 2, 0, -2, midi_program=57)),
        ("C Trumpet", None, InstrumentPlaybackProfile("brass.trumpet.c", midi_program=57)),
        ("Eb Trumpet", None, InstrumentPlaybackProfile("brass.trumpet", 2, 3, 0, 3, midi_program=57)),
        ("A Trumpet", None, InstrumentPlaybackProfile("brass.trumpet", -2, -3, 0, -3, midi_program=57)),
        ("Flugelhorn", None, InstrumentPlaybackProfile("brass.flugelhorn", -1, -2, 0, 2, midi_program=57)),
        ("Bb Cornet 1", None, InstrumentPlaybackProfile("brass.cornet", -1, -2, 0, 2, midi_program=57)),
        ("Bb Trumpet 1 / Bb Cornet 1", None, InstrumentPlaybackProfile("brass.trumpet.bflat", -1, -2, 0, 2, midi_program=57)),
        ("Eb Alto Horn", None, InstrumentPlaybackProfile("brass.alto-horn", -5, -9, 0, 3, midi_program=61)),
        ("F Horn 1, 3", None, InstrumentPlaybackProfile("brass.french-horn", -4, -7, 0, 1, midi_program=61)),
        ("C Bass Trombone", None, InstrumentPlaybackProfile("brass.trombone.bass", midi_program=58)),
        ("Bb Trombone", ("G", "2"), InstrumentPlaybackProfile("brass.trombone", -1, -2, -1, 2, midi_program=58)),
        ("C Trombone 1", ("F", "4"), InstrumentPlaybackProfile("brass.trombone", midi_program=58)),
        ("Bb Baritone", ("G", "2"), InstrumentPlaybackProfile("brass.euphonium", -1, -2, -1, 2, midi_program=58)),
        ("C Baritone Solo", ("F", "4"), InstrumentPlaybackProfile("brass.euphonium", midi_program=58)),
        ("C Basses", ("F", "4"), InstrumentPlaybackProfile("brass.tuba", midi_program=59)),
        ("Violin I", None, InstrumentPlaybackProfile("strings.violin", midi_program=41)),
        ("Viola", None, InstrumentPlaybackProfile("strings.viola", midi_program=42)),
        ("Violoncello", None, InstrumentPlaybackProfile("strings.cello", midi_program=43)),
        ("Piano", None, InstrumentPlaybackProfile("keyboard.piano", midi_program=1)),
        ("Organ", None, InstrumentPlaybackProfile("keyboard.organ", midi_program=20)),
        ("Harp", None, InstrumentPlaybackProfile("pluck.harp", midi_program=47)),
        ("Guitar", None, InstrumentPlaybackProfile("pluck.guitar", 0, 0, -1, 0, midi_program=25)),
        ("Soprano", None, InstrumentPlaybackProfile("voice.soprano", midi_program=53)),
        ("Alto", None, InstrumentPlaybackProfile("voice.alto", midi_program=53)),
        ("Tenor", None, InstrumentPlaybackProfile("voice.tenor", midi_program=53)),
        ("Bass Voice", None, InstrumentPlaybackProfile("voice.bass", midi_program=53)),
        ("Voice", None, InstrumentPlaybackProfile("voice.vocals", midi_program=53)),
        ("Choir", None, InstrumentPlaybackProfile("voice.vocals", midi_program=53)),
        ("Unknown Mystery Part", None, InstrumentPlaybackProfile("keyboard.piano", midi_program=1)),
        ("", None, InstrumentPlaybackProfile("keyboard.piano", midi_program=1)),
    ],
)
def test_infer_instrument_profile(name, clef, expected):
    assert infer_instrument_profile(name, clef=clef) == expected


def test_extract_header_instrument_label(tmp_path):
    assert extract_header_instrument_label(tmp_path / "missing.png", 10.0) is None
    assert extract_header_instrument_label(tmp_path / "missing.png", 120.0) is None

    png = tmp_path / "page_001.png"
    Image.new("RGB", (1000, 1400), (255, 255, 255)).save(png)

    # Rejects titles, boilerplate, tempo markings, and picks left-aligned instrument label
    def fake_header_ocr(_crop):
        boxes = [
            [[350, 40], [650, 40], [650, 80], [350, 80]],     # SUITE II.
            [[80, 100], [260, 100], [260, 130], [80, 130]],   # Allegro = 120
            [[80, 135], [240, 135], [240, 160], [80, 160]],   # Full score
            [[80, 170], [260, 170], [260, 195], [80, 195]],   # Violoncello
            [[80, 200], [600, 200], [600, 220], [80, 220]],   # Too many words (6)
            [[500, 170], [700, 170], [700, 195], [500, 195]], # Right-side Violin (lower priority than left Violoncello)
            [[80, 225], [140, 225], [140, 240], [80, 240]],   # Empty / noise
        ]
        txts = [
            "SUITE II.",
            "Allegro = 120",
            "Full score",
            "Violoncello",
            "Arranged for Piano Violin Viola Cello",
            "Violin",
            "",
        ]
        return boxes, txts

    lbl = extract_header_instrument_label(png, 260.0, ocr_runner=fake_header_ocr)
    assert lbl is not None
    assert lbl.full_name == "Violoncello"
    assert lbl.abbreviation == "Vc."

    # Also works via reader object returning numpy-array boxes (must not evaluate truthiness!)
    class _FakeRes:
        boxes = np.array([[[60, 140], [200, 140], [200, 165], [60, 165]]])
        txts = ["Bb Cl. 1"]

    lbl2 = extract_header_instrument_label(png, 250.0, ocr_reader=lambda _crop: _FakeRes())
    assert lbl2 is not None
    assert lbl2.full_name == "Bb Clarinet 1"
    assert lbl2.abbreviation == "Bb Cl. 1"

    # Returns None when only non-instrument header text is present
    assert (
        extract_header_instrument_label(
            png,
            250.0,
            ocr_runner=lambda _crop: ([[[300, 50], [600, 50], [600, 90], [300, 90]]], ["Prélude."]),
        )
        is None
    )
