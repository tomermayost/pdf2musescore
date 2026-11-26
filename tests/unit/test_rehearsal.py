from __future__ import annotations

import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from pdf2musescore.rehearsal import (
    PositionedRehearsal,
    RehearsalMark,
    apply_rehearsal_mark,
    assign_boxed_rehearsal_marks_to_systems,
    detect_boxed_rehearsal_marks_in_band,
    detect_unboxed_rehearsal_marks_in_band,
    extract_rehearsal_marks_in_band,
    map_range_rehearsal_marks_to_measures,
    map_system_rehearsal_marks_to_measures,
)


def _make_boxed_image(
    width: int = 1200,
    height: int = 200,
    boxes: list[tuple[int, int, int, int, str]] | None = None,
) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    for x, y, w, h, label in (boxes or []):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), thickness=3)
        if label:
            cv2.putText(
                img,
                label,
                (x + max(6, w // 4), y + int(h * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
    return img


def test_rehearsal_mark_and_positioned_serialization():
    rm = RehearsalMark(label="5", enclosure="square", raw_ocr="5")
    assert RehearsalMark.from_dict(rm.to_dict()) == rm
    assert RehearsalMark.from_dict(None) is None
    assert RehearsalMark.from_dict({"label": ""}) is None
    assert RehearsalMark.from_dict({"label": "A", "enclosure": "bogus"}) == RehearsalMark("A", "square", "")
    assert RehearsalMark.from_dict({"label": "B", "enclosure": "none"}) == RehearsalMark("B", "none", "")

    pr = PositionedRehearsal(rm, 100.0, 150.0, 20.0, 70.0, u_norm=0.45)
    assert PositionedRehearsal.from_dict(pr.to_dict()) == pr
    assert PositionedRehearsal.from_dict("bad") is None
    assert PositionedRehearsal.from_dict({"mark": {"label": ""}}) is None
    assert PositionedRehearsal.from_dict({"mark": {"label": "1"}, "x_left": "bad"}) is None


def test_detect_boxed_rehearsal_marks_in_band():
    img = _make_boxed_image(
        boxes=[
            (200, 30, 52, 52, "1"),
            (650, 32, 58, 52, "12"),
            (850, 40, 48, 48, ""),       # empty box -> rejected (fill < 0.04)
            (940, 40, 12, 12, "9"),      # tiny box -> rejected
            (1000, 40, 220, 48, "AB"),   # too wide -> rejected
            (1080, 110, 44, 44, "2"),    # will be painted solid black below
        ]
    )
    cv2.rectangle(img, (1080, 110), (1124, 154), (0, 0, 0), thickness=-1)  # solid black -> fill > 0.60

    def _box_ocr(crop):
        # Return different mock texts by brightness or call count
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        mean_val = float(gray.mean())
        return ([], ["1"] if mean_val > 220 else ["12"])

    assert detect_boxed_rehearsal_marks_in_band(img, 0, 10) == []

    # Alternate deterministic OCR callback keyed by horizontal position
    labels_iter = iter(["[1]", "lowercase", "12"])
    img_multi = _make_boxed_image(
        boxes=[
            (200, 30, 52, 52, "1"),
            (420, 32, 52, 52, "x"),
            (650, 32, 58, 52, "12"),
            (850, 40, 48, 48, ""),
        ]
    )
    # Also draw an open L-bracket (missing top/right borders) -> border_cov < 0.85
    cv2.line(img_multi, (960, 40), (960, 88), (0, 0, 0), 3)
    cv2.line(img_multi, (960, 88), (1008, 88), (0, 0, 0), 3)
    cv2.putText(img_multi, "7", (974, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    hits = detect_boxed_rehearsal_marks_in_band(
        img_multi,
        0,
        180,
        sys_min_x=100.0,
        sys_max_x=1100.0,
        ocr_runner=lambda _c: ([], [next(labels_iter, "")]),
    )
    assert [h.mark.label for h in hits] == ["1", "12"]
    assert all(h.mark.enclosure == "square" for h in hits)
    assert hits[0].u_norm < hits[1].u_norm

    # Also exercises grayscale crop + upscaling on small 28x28 box + empty OCR response
    gray_small = cv2.cvtColor(
        _make_boxed_image(width=800, height=120, boxes=[(100, 30, 28, 28, "2"), (300, 30, 28, 28, "3")]),
        cv2.COLOR_BGR2GRAY,
    )
    small_iter = iter(["2", ""])
    small_hits = detect_boxed_rehearsal_marks_in_band(
        gray_small,
        0,
        120,
        ocr_runner=lambda _c: ([], [next(small_iter, "")]),
    )
    assert [h.mark.label for h in small_hits] == ["2"]

    # Boundary snapping in map_range_rehearsal_marks_to_measures + proportional fallback
    rng_root = ET.fromstring(
        "<score-partwise><part id='P1'>"
        "<measure number='1'><attributes><key><fifths>0</fifths></key></attributes><note/></measure>"
        "<measure number='2'><note/><barline location='right'><bar-style>light-light</bar-style></barline></measure>"
        "<measure number='3'><attributes><key><fifths>1</fifths></key></attributes><note/></measure>"
        "<measure number='4'><note/></measure>"
        "<measure number='5'><note/></measure>"
        "</part></score-partwise>"
    )
    assert (
        map_range_rehearsal_marks_to_measures(
            rng_root,
            [
                PositionedRehearsal(RehearsalMark("A", "none"), 80, 110, 10, 40, u_norm=0.05),
                PositionedRehearsal(RehearsalMark("B", "none"), 460, 490, 10, 40, u_norm=0.45),
                PositionedRehearsal(RehearsalMark("C", "none"), 780, 810, 10, 40, u_norm=0.80),
            ],
            start_index=0,
            measure_count=5,
        )
        == 3
    )
    rng_ms = rng_root.findall("./part/measure")
    assert rng_ms[0].findtext("./direction/direction-type/rehearsal") == "A"
    assert rng_ms[2].findtext("./direction/direction-type/rehearsal") == "B"
    assert rng_ms[4].findtext("./direction/direction-type/rehearsal") == "C"


def test_detect_unboxed_and_combined_rehearsal_marks():
    img = np.full((240, 1800, 3), 255, dtype=np.uint8)
    # Paint a boxed [3] at x=900..954, y=60..114
    cv2.rectangle(img, (900, 60), (954, 114), (0, 0, 0), thickness=3)
    cv2.putText(img, "3", (916, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)

    assert detect_unboxed_rehearsal_marks_in_band(img, 0, 10) == []
    assert detect_unboxed_rehearsal_marks_in_band(img, 0, 200, ocr_runner=lambda _c: ([], [])) == []

    def _band_ocr(_c):
        boxes = [
            [[240, 60], [270, 60], [270, 96], [240, 96]],      # Valid tall unboxed 'A'
            [[240, 140], [260, 140], [260, 158], [240, 158]],   # Too short 'B' (h=18)
            [[320, 60], [342, 60], [342, 92], [320, 92]],       # Left of min_y when tested, or normal 'C'
            [[400, 60], [428, 60], [428, 94], [400, 94]],       # 'A' followed by 'tempo' -> blocked!
            [[438, 62], [540, 62], [540, 92], [438, 92]],       # 'tempo'
            [[600, 60], [680, 60], [680, 92], [600, 92]],       # 'PIANO'
            [[692, 60], [722, 60], [722, 96], [692, 96]],       # 'D' immediately preceded by word -> blocked!
            [[904, 64], [948, 64], [948, 108], [904, 108]],     # 'B' overlapping boxed [3] -> suppressed
            [[100, 60], [130, 60], [130, 96], [100, 96]],       # In left margin < sys_min_x - 0.05*span -> rejected
            [[1100, 60], [1120, 60], [1120, 96], [1100, 96]],   # Blank/lower 'p'
            [[1200, 10], [1350, 10], [1350, 195], [1200, 195]], # Too tall 'M' (h=185 > max_h) -> rejected
            [[1400, 145], [1428, 145], [1428, 185], [1400, 185]], # Below unboxed_max_y=165 (accent touching staff) -> rejected
        ]
        txts = ["A", "B", " ", "A", "tempo", "PIANO", "D", "B", "E", "p", "M", "V"]
        return boxes, txts

    res = extract_rehearsal_marks_in_band(
        img,
        0,
        200,
        sys_min_x=220.0,
        sys_max_x=1720.0,
        unboxed_min_y=85,
        unboxed_max_y=165,
        box_ocr_runner=lambda _c: ([], ["3"]),
        band_ocr_runner=_band_ocr,
        source_name="p6.png",
    )
    assert [(r.mark.label, r.mark.enclosure) for r in res] == [("A", "none"), ("3", "square")]

    # Full-page boxed assignment attaches boxed mark to first system below within max_dist
    img_sys = _make_boxed_image(width=1200, height=500, boxes=[(400, 80, 50, 50, "1"), (500, 290, 50, 50, "2")])
    assert assign_boxed_rehearsal_marks_to_systems(img_sys, [], avg_h=60.0) == []
    lbls = iter(["1", "2"])
    assigned = assign_boxed_rehearsal_marks_to_systems(
        img_sys,
        [(80.0, 1100.0, 150.0, 210.0), (80.0, 1100.0, 360.0, 420.0)],
        avg_h=60.0,
        ocr_runner=lambda _c: ([], [next(lbls, "")]),
        source_name="p1.png",
    )
    assert [[m.mark.label for m in s] for s in assigned] == [["1"], ["2"]]


def test_apply_and_map_rehearsal_marks():
    root = ET.fromstring(
        "<score-partwise><part id='P1'>"
        "<measure number='1'><attributes><key><fifths>0</fifths></key></attributes><note/></measure>"
        "<measure number='2'><note/><barline location='right'><bar-style>light-light</bar-style></barline></measure>"
        "<measure number='3'><attributes><key><fifths>-2</fifths></key></attributes><note/></measure>"
        "<measure number='4'><note/></measure>"
        "<measure number='5'><print new-system='yes'/><note/></measure>"
        "<measure number='6'><note/></measure>"
        "</part></score-partwise>"
    )

    assert apply_rehearsal_mark(root, None) is False
    assert apply_rehearsal_mark(root, RehearsalMark("   ")) is False
    assert apply_rehearsal_mark(ET.Element("score-partwise"), RehearsalMark("1")) is False
    assert apply_rehearsal_mark(root, RehearsalMark("1"), measure_index=99) is False

    # Guards in mappers
    assert map_system_rehearsal_marks_to_measures(ET.Element("score-partwise"), []) == 0
    assert map_system_rehearsal_marks_to_measures(ET.fromstring("<score-partwise><part id='P1'/></score-partwise>"), []) == 0
    assert map_range_rehearsal_marks_to_measures(ET.Element("score-partwise"), [], start_index=0, measure_count=2) == 0
    assert map_range_rehearsal_marks_to_measures(ET.Element("score-partwise"), [PositionedRehearsal(RehearsalMark("1"), 0, 10, 0, 10)], start_index=0, measure_count=2) == 0
    assert map_range_rehearsal_marks_to_measures(ET.fromstring("<score-partwise><part id='P1'/></score-partwise>"), [PositionedRehearsal(RehearsalMark("1"), 0, 10, 0, 10)], start_index=0, measure_count=2) == 0
    assert map_range_rehearsal_marks_to_measures(root, [PositionedRehearsal(RehearsalMark("1"), 0, 10, 0, 10)], start_index=99, measure_count=2) == 0

    # Sys 0 with exact barlines [250, 480, 720, 960] -> mark at x=478 snaps to M3 (barline 480);
    # 'square' beats 'none' even when 'none' appears earlier on the same bar
    m_sys0_ub = PositionedRehearsal(RehearsalMark("V", "none"), 460.0, 485.0, 10.0, 50.0, u_norm=0.48)
    m_sys0 = PositionedRehearsal(RehearsalMark("1", "square"), 460.0, 496.0, 10.0, 50.0, u_norm=0.48)
    m_sys1 = PositionedRehearsal(RehearsalMark("A", "none"), 60.0, 90.0, 200.0, 240.0, u_norm=0.04)
    m_sys1_right = PositionedRehearsal(RehearsalMark("B", "none"), 560.0, 590.0, 200.0, 240.0, u_norm=0.60)
    injected = map_system_rehearsal_marks_to_measures(
        root,
        [[m_sys0_ub, m_sys0], [m_sys1, m_sys1_right]],
        system_barlines=[[250, 480, 720, 960]],
        system_x_bounds=[(50.0, 960.0)],
    )
    assert injected == 3
    ms = root.findall("./part/measure")
    assert ms[2].find("./direction/direction-type/rehearsal").text == "1"
    assert ms[2].find("./direction/direction-type/rehearsal").get("enclosure") == "square"
    assert ms[4].find("./direction/direction-type/rehearsal").text == "A"
    assert ms[4].find("./direction/direction-type/rehearsal").get("enclosure") == "none"
    assert ms[5].find("./direction/direction-type/rehearsal").text == "B"
    # Idempotent
    assert apply_rehearsal_mark(root, RehearsalMark("1"), measure_index=2) is False
