from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pdf2musescore.fullscore.instruments import (
    InstrumentLabel,
    abbreviate_instrument_name,
    clean_instrument_label,
    consolidate_page_labels,
    expand_abbreviated_name,
    extract_instrument_labels,
    is_percussion_instrument,
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
            [[0, 190], [60, 190], [60, 210], [0, 210]],  # Flute 1, 2
            [[0, 220], [60, 220], [60, 240], [0, 240]],  # Piccolo (same staff)
            [[0, 355], [60, 355], [60, 375], [0, 375]],  # Mallet
            [[0, 380], [60, 380], [60, 400], [0, 400]],  # Percussion
            [[0, 620], [60, 620], [60, 640], [0, 640]],  # too far from any staff
        ]
        txts = ["", "Full score", "1", "Flute 1, 2", "Piccolo", "Mallet", "Percussion", "Bassoon"]
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
