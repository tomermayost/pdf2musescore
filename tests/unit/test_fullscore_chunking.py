from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from pdf2musescore.engines.homr_engine import CancelledError
from pdf2musescore.fullscore.chunking import (
    StaffChunkPlan,
    _has_cross_barline_arc,
    detect_system_barlines,
    detect_system_barlines_with_skew,
    extract_clef_key_header,
    parse_staff_image_chunked,
    plan_staff_chunks,
    should_chunk_staff,
    staff_barlines_from_system,
    stitch_chunk_symbols,
)


@dataclass
class _FakeStaffPoint:
    x: float
    y: list[float]
    angle: float = 0.0


class _FakeBarLine:
    def __init__(self, cx: float, cy: float) -> None:
        self.center = (cx, cy)


@dataclass
class _FakeStaff:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    average_unit_size: float = 12.0
    grid: list[_FakeStaffPoint] = field(default_factory=list)
    n_notes: int = 0
    barlines: list[_FakeBarLine] = field(default_factory=list)

    def get_number_of_notes(self) -> int:
        return self.n_notes

    def get_bar_lines(self) -> list[_FakeBarLine]:
        return self.barlines


@dataclass
class _FakeSym:
    rhythm: str
    pitch: str = "_"
    lift: str = "_"
    articulation: str = "_"
    slur: str = "_"
    position: str = "upper"
    coordinates: tuple[float, float] | None = None
    image_coordinates: tuple[float, float] | None = None

    def add_slurs(self, slurs: list[str]) -> "_FakeSym":
        cur = [s for s in self.slur.split("_") if s and s != "."]
        cur.extend(slurs)
        return _FakeSym(
            self.rhythm,
            self.pitch,
            self.lift,
            self.articulation,
            "_".join(sorted(cur)),
            self.position,
            self.coordinates,
            self.image_coordinates,
        )


def _draw_synthetic_score(
    *,
    width: int = 1600,
    height: int = 560,
    staff_cys: list[int] | None = None,
    unit: int = 12,
    barlines_x: list[int] | None = None,
    choir_break: bool = False,
    double_bar_at: int | None = None,
    skew_dx: int = 0,
    opening_timesig_x: int | None = None,
    courtesy_stub_x: int | None = None,
) -> tuple[np.ndarray, list[_FakeStaff]]:
    img = np.full((height, width), 255, dtype=np.uint8)
    cys = staff_cys or [90, 210, 340, 460]
    bls = barlines_x or [160, 480, 760, 1040, 1300, 1520]
    y_mid = 0.5 * (cys[0] + cys[-1])
    y_span = max(1, cys[-1] - cys[0])

    staves: list[_FakeStaff] = []
    for idx, cy in enumerate(cys):
        y0 = cy - 2 * unit
        y1 = cy + 2 * unit
        for k in range(5):
            ly = y0 + k * unit
            cv2.line(img, (bls[0], ly), (bls[-1], ly), 0, 1)

        shift = int(round(skew_dx * ((cy - y_mid) / y_span))) if len(cys) > 1 else 0
        for bx in bls:
            # Skip interior barlines for half of the choirs when choir_break=True
            if choir_break and idx >= len(cys) // 2 and bx not in (bls[0], bls[-1]):
                continue
            xb = bx + shift
            cv2.line(img, (xb, y0), (xb, y1), 0, 2)
            if double_bar_at is not None and bx == double_bar_at:
                cv2.line(img, (xb + max(4, unit // 2), y0), (xb + max(4, unit // 2), y1), 0, 2)

        if opening_timesig_x is not None:
            cv2.line(img, (opening_timesig_x + shift, y0 + 2), (opening_timesig_x + shift, y1 - 2), 0, 3)
        if courtesy_stub_x is not None:
            cv2.line(img, (courtesy_stub_x + shift, y0 + 2), (courtesy_stub_x + shift, y1 - 2), 0, 2)

        grid = [
            _FakeStaffPoint(float(px), [float(y0 + k * unit) for k in range(5)])
            for px in range(bls[0], bls[-1] + 1, 200)
        ]
        staves.append(
            _FakeStaff(
                min_x=float(bls[0] + shift),
                max_x=float(bls[-1] + shift),
                min_y=float(y0),
                max_y=float(y1),
                average_unit_size=float(unit),
                grid=grid,
            )
        )
    return img, staves


def test_detect_system_barlines_consensus_broken_skewed_and_edge_cases():
    # Degenerate inputs -> []
    assert detect_system_barlines(None, []) == []
    assert detect_system_barlines(np.zeros((10, 300), dtype=np.uint8), [_FakeStaff(0, 100, 10, 50)]) == []

    # 1. Clean 5-measure system with opening time-sig stroke, double barline merge, & trailing courtesy stub
    bls = [160, 480, 760, 1040, 1300, 1520]
    img, staves = _draw_synthetic_score(
        barlines_x=bls,
        double_bar_at=760,
        opening_timesig_x=160 + 20,   # < 4.5 * 48 = 216px from left -> filtered
        courtesy_stub_x=1520 - 30,    # < 3.5 * 48 = 168px from right -> filtered
    )
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    detected = detect_system_barlines(img_bgr, staves)
    assert len(detected) == 6
    for got, exp in zip(detected, [160, 480, 763, 1040, 1300, 1520]):
        assert abs(got - exp) <= 5

    # 2. Choir-broken interior barlines (only top 2/4 staves carry interior barlines) + SegNet BarLine fusion
    img_brk, st_brk = _draw_synthetic_score(barlines_x=bls, choir_break=True)
    # Without SegNet BarLine fusion, min_vote_ratio=0.45 finds all 6 bars (2/4 = 0.50)
    assert len(detect_system_barlines(img_brk, st_brk, min_vote_ratio=0.45)) == 6
    # Adding a SegNet BarLine on staff 2 at x=760 boosts consensus even under default ratio
    st_brk[2].barlines.append(_FakeBarLine(760.0, st_brk[2].min_y + 24.0))
    assert any(abs(x - 760) <= 6 for x in detect_system_barlines(img_brk, st_brk, min_vote_ratio=0.70))

    # 3. Skewed page (-30px horizontal drift top-to-bottom)
    img_sk, st_sk = _draw_synthetic_score(barlines_x=bls, skew_dx=-30)
    pts_sk, y_mid, slope = detect_system_barlines_with_skew(img_sk, st_sk)
    assert len(pts_sk) == 6
    assert slope < -0.02
    top_shifted = staff_barlines_from_system(pts_sk, st_sk[0], y_mid, slope)
    bot_shifted = staff_barlines_from_system(pts_sk, st_sk[-1], y_mid, slope)
    assert bot_shifted[2] < top_shifted[2] - 10

    # 4. Fewer than 3 barlines -> disabled ([])
    img_1bar, st_1bar = _draw_synthetic_score(barlines_x=[160, 1500])
    assert detect_system_barlines(img_1bar, st_1bar) == []


def test_should_chunk_staff_bypass_and_activation_gates():
    img, staves = _draw_synthetic_score(width=3840, height=560, barlines_x=[400, 1100, 1800, 2500, 3200, 3760])
    s = staves[0]
    s_wide = _FakeStaff(400, 1100, s.min_y, s.max_y, average_unit_size=12.0, grid=s.grid, n_notes=10)
    reg_solo = np.array([400, 20, 1100, 160])  # width=700 -> canvas_w=1280 -> scaled > 36px!
    ok_solo, reason_solo = should_chunk_staff(s_wide, img, reg_solo, [400, 1100, 1800])
    assert not ok_solo and "native_scale_ok" in reason_solo

    reg_full = np.array([380, 20, 3780, 160])  # width=3400 -> canvas_w=1280 -> 5-line ~ 18px < 36px
    # Rest-only empty staff (only tiny whole-bar rest blocks inside measures) -> bypassed!
    cy = int(round(0.5 * (s.min_y + s.max_y)))
    for bx in (750, 1450, 2150, 2850, 3480):
        cv2.rectangle(img, (bx - 4, cy - 3), (bx + 4, cy + 2), 0, -1)
    s.n_notes = 0
    ok_empty, reason_empty = should_chunk_staff(s, img, reg_full, [400, 1100, 1800, 2500, 3200, 3760])
    assert not ok_empty and "empty_rest_only" in reason_empty

    # Draw actual dense noteheads & stems in Bar 2 -> activated even if SegNet missed note count!
    for nx in range(1300, 1650, 45):
        cv2.circle(img, (nx, cy), 8, 0, -1)
        cv2.line(img, (nx + 7, cy), (nx + 7, cy - 35), 0, 2)
    ok_ink, reason_ink = should_chunk_staff(s, img, reg_full, [400, 1100, 1800, 2500, 3200, 3760])
    assert ok_ink and "compressed" in reason_ink

    # SegNet notes present -> activated immediately
    s.n_notes = 12
    ok_notes, _ = should_chunk_staff(s, img, reg_full, [400, 1100, 1800, 2500, 3200, 3760])
    assert ok_notes

    # Invalid inputs
    assert not should_chunk_staff(s, None, reg_full, [400, 1100, 1800])[0]
    assert not should_chunk_staff(s, img, reg_full, [400, 1100])[0]


def test_tie_detection_and_tie_aware_chunk_planning():
    # Build 5-bar dewarped strip where balanced 2-bar windows fit >= 42px AND a tie crosses barline 2
    h, w, unit = 190, 3250, 12
    strip = np.full((h, w), 255, dtype=np.uint8)
    y0, y1 = 71, 71 + 4 * unit
    for k in range(5):
        cv2.line(strip, (0, y0 + k * unit), (w - 1, y0 + k * unit), 0, 1)
    bls = [10, 680, 1300, 1920, 2540, 3160]
    for bx in bls:
        cv2.line(strip, (bx, y0), (bx, y1), 0, 2)

    # Draw thin horizontal tie arc bridging across barline 2 (1300)
    cv2.ellipse(strip, (1300, y0 + 28), (18, 6), 0, 0, 180, 0, 2)
    assert _has_cross_barline_arc(strip, 1300, y0, y1, float(unit)) is True
    assert _has_cross_barline_arc(strip, 1920, y0, y1, float(unit)) is False
    # Out-of-bounds barline X returns False cleanly
    assert _has_cross_barline_arc(strip, 1, y0, y1, float(unit)) is False

    tie_flags = [_has_cross_barline_arc(strip, bx, y0, y1, float(unit)) for bx in bls]
    plans = plan_staff_chunks(bls, h, 4.0 * unit, strip_width=w, unit=float(unit), tie_barline_flags=tie_flags)
    # 5 balanced bars split into 3 chunks ([0..1, 1..3, 3..5]), avoiding the tie at boundary 2!
    assert len(plans) == 3
    assert plans[0].measure_start == 0
    assert plans[-1].measure_end == 5
    boundaries = [p.measure_start for p in plans[1:]]
    assert 2 not in boundaries

    # Edge case: huge single measures -> falls back cleanly to 1-bar chunks
    huge_plans = plan_staff_chunks([0, 2000, 4000], 190, 48.0, strip_width=4010, unit=12.0)
    assert [(p.measure_start, p.measure_end) for p in huge_plans] == [(0, 1), (1, 2)]
    assert plan_staff_chunks([0, 100], 190, 48.0) == []


def test_chunking_edge_branches_bgr_header_and_fallbacks():
    # Cover BGR image header with 0 flats + missing coordinates + no-grid staff consensus
    h, w, unit = 190, 1200, 12
    bgr = np.full((h, w, 3), 255, dtype=np.uint8)
    y0, y1 = 70, 70 + 4 * unit
    for k in range(5):
        cv2.line(bgr, (0, y0 + k * unit), (w - 1, y0 + k * unit), (0, 0, 0), 1)
    cv2.line(bgr, (10, y0), (10, y1), (0, 0, 0), 2)
    cv2.line(bgr, (600, y0), (600, y1), (0, 0, 0), 2)
    cv2.rectangle(bgr, (20, y0 + 2), (44, y1 - 2), (0, 0, 0), -1)  # clef only
    cv2.circle(bgr, (120, y0 + 24), 7, (0, 0, 0), -1)  # first note

    hdr_bgr, _ = extract_clef_key_header(
        bgr,
        10,
        600,
        y0,
        y1,
        float(unit),
        [_FakeSym("clef_G2"), _FakeSym("note_4", "G4")],
        [None, None],
    )
    assert hdr_bgr.ndim == 3 and hdr_bgr.shape[1] < 120

    # No-grid staves in detect_system_barlines & un-skewed staff_barlines_from_system
    assert staff_barlines_from_system([10, 500, 1000], _FakeStaff(10, 1000, 50, 98), 100.0, 0.0) == [10, 500, 1000]
    img_nogrid, st_nogrid = _draw_synthetic_score(barlines_x=[120, 450, 780, 1100, 1450])
    for st in st_nogrid:
        st.grid = []
    assert len(detect_system_barlines(img_nogrid, st_nogrid)) == 5

    # stitch empty / pitch-mismatch tie / missing calc_region / single-chunk fallback
    assert stitch_chunk_symbols([], [], 12.0) == []
    p0 = StaffChunkPlan(0, 1, 0, 500, 48.0, False)
    p1 = StaffChunkPlan(1, 2, 505, 1000, 48.0, True)
    diff_pitch = stitch_chunk_symbols(
        [
            (p0, [_FakeSym("note_1", "G2"), _FakeSym("barline")], [200.0, 500.0], 80),
            (p1, [_FakeSym("clef_F4"), _FakeSym("barline"), _FakeSym("note_1", "A2"), _FakeSym("barline")], [450.0, 501.0, 700.0, 1000.0], 80),
        ],
        [10, 500, 1000],
        unit=12.0,
    )
    assert diff_pitch[0].slur == "_" and diff_pitch[2].slur == "_"

    fb_calls: list[int] = []
    orig_fn = lambda *a, **k: fb_calls.append(1) or [_FakeSym("fb")]
    assert [
        s.rhythm
        for s in parse_staff_image_chunked(
            None,
            0,
            st_nogrid[0],
            img_nogrid,
            None,
            None,
            lambda pt: pt,
            barlines_x=[120, 450],
            orig_parse_staff_image=orig_fn,
            homr_staff_parsing=SimpleNamespace(),
        )
    ] == ["fb"]
    assert len(fb_calls) == 1


def test_extract_clef_key_header_excludes_timesig_notes_and_dynamic_bleed():
    h, w, unit = 190, 1200, 12
    strip = np.full((h, w), 255, dtype=np.uint8)
    y0 = 70
    y1 = y0 + 4 * unit
    for k in range(5):
        cv2.line(strip, (0, y0 + k * unit), (w - 1, y0 + k * unit), 0, 1)
    b0, b1 = 10, 620
    cv2.line(strip, (b0, y0), (b0, y1), 0, 2)
    cv2.line(strip, (b1, y0), (b1, y1), 0, 2)

    # Clef [22..46], Key 2 flats [54..63, 68..77], TimeSig [92..112], First note [135..148]
    cv2.rectangle(strip, (22, y0 + 2), (46, y1 - 2), 0, -1)
    cv2.rectangle(strip, (54, y0 + 8), (63, y0 + 26), 0, -1)
    cv2.rectangle(strip, (68, y0 + 18), (77, y0 + 36), 0, -1)
    cv2.rectangle(strip, (92, y0 + 4), (112, y1 - 4), 0, -1)
    cv2.circle(strip, (140, y0 + 24), 7, 0, -1)
    # Leaked dynamic 'mf' below staff at x=30..65, y=155..175
    cv2.rectangle(strip, (30, 155), (65, 175), 0, -1)

    chunk0_syms = [
        _FakeSym("clef_F4"),
        _FakeSym("keySignature_-2"),
        _FakeSym("timeSignature/4"),
        _FakeSym("note_8", "F3"),
    ]
    chunk0_xs = [34.0, 65.0, 102.0, 140.0]

    hdr, hdr_w = extract_clef_key_header(strip, b0, b1, y0, y1, float(unit), chunk0_syms, chunk0_xs)
    cut_x = hdr_w - max(6, int(round(1.1 * unit)))
    # Cut falls AFTER the 2nd flat (77) and strictly BEFORE time signature (92)!
    assert 78 <= cut_x <= 90
    # Leaked 'mf' below staff (y=155..175) was scrubbed back to white background
    assert np.all(hdr[155:175, :] == 255)
    # 5 staff lines extend smoothly across appended spacer
    assert hdr[y0, -2] == 0 and hdr[y0 + 3, -2] == 255


def test_stitch_chunk_symbols_strips_headers_keeps_mid_changes_and_bridges_ties():
    p0 = StaffChunkPlan(0, 2, 0, 1370, 50.0, False)
    p1 = StaffChunkPlan(2, 4, 1375, 2610, 50.0, True)
    p2 = StaffChunkPlan(4, 5, 2615, 3330, 48.0, False)

    c0_syms = [
        _FakeSym("clef_F4"),
        _FakeSym("keySignature_-2"),
        _FakeSym("timeSignature/8"),
        _FakeSym("dynamic_mf"),
        _FakeSym("note_1.", "G2"),
        _FakeSym("barline"),
        _FakeSym("note_1.", "G2"),  # cross-chunk tie source (no slur from TrOMR)
        _FakeSym("barline"),
        _FakeSym("note_4", "A2"),  # hallucinated past closing barline 2 -> trimmed!
    ]
    c0_xs = [30.0, 60.0, 90.0, 120.0, 300.0, 750.0, 1000.0, 1360.0, 1400.0]

    c1_syms = [
        _FakeSym("clef_F4"),
        _FakeSym("keySignature_-2"),
        _FakeSym("barline"),       # duplicate boundary barline -> dropped
        _FakeSym("note_1.", "G2"),  # cross-chunk tie target (same pitch!)
        _FakeSym("barline"),
        _FakeSym("timeSignature/4"),  # genuine mid-chunk meter change!
        _FakeSym("note_1", "B2"),
        _FakeSym("barline"),
    ]
    c1_xs = [1100.0, 1140.0, 1362.0, 1600.0, 1980.0, 2040.0, 2250.0, 2600.0]

    c2_syms = [
        _FakeSym("clef_F4"),
        _FakeSym("keySignature_-2"),
        _FakeSym("repeatStart"),   # special boundary barline upgrades previous barline!
        _FakeSym("note_1", "C3"),
        _FakeSym("barline"),
    ]
    c2_xs = [2300.0, 2340.0, 2602.0, 2900.0, 3320.0]

    stitched = stitch_chunk_symbols(
        [(p0, c0_syms, c0_xs, 120), (p1, c1_syms, c1_xs, 120), (p2, c2_syms, c2_xs, 120)],
        [10, 750, 1360, 1980, 2600, 3320],
        unit=12.0,
    )
    rhythms = [s.rhythm for s in stitched]
    assert rhythms == [
        "clef_F4",
        "keySignature_-2",
        "timeSignature/8",
        "dynamic_mf",
        "note_1.",
        "barline",
        "note_1.",
        "barline",
        "note_1.",
        "barline",
        "timeSignature/4",
        "note_1",
        "repeatStart",
        "note_1",
        "barline",
    ]
    # Verify cross-chunk tie bridged G2 (bar 2) -> G2 (bar 3)
    assert "slurStart" in stitched[6].slur
    assert "slurStop" in stitched[8].slur

    # Beat-1 dynamic at chunk start is preserved AND missing closing barline on prev chunk is auto-inserted
    no_bar_p0 = StaffChunkPlan(0, 1, 0, 500, 48.0, False)
    dyn_p1 = StaffChunkPlan(1, 2, 505, 1000, 48.0, False)
    with_dyn = stitch_chunk_symbols(
        [
            (no_bar_p0, [_FakeSym("clef_F4"), _FakeSym("note_1", "G2")], [30.0, 250.0], 80),
            (
                dyn_p1,
                [
                    _FakeSym("clef_F4"),
                    _FakeSym("dynamic_f"),   # synthetic/header zone or Beat-1? put inside chunk body (> 509)!
                    _FakeSym("note_1", "A2"),
                    _FakeSym("barline"),
                ],
                [460.0, 540.0, 620.0, 990.0],
                80,
            ),
        ],
        [10, 500, 1000],
        unit=12.0,
    )
    assert [s.rhythm for s in with_dyn] == ["clef_F4", "note_1", "barline", "dynamic_f", "note_1", "barline"]


def test_parse_staff_image_chunked_orchestration_and_fallbacks():
    img, staves = _draw_synthetic_score(
        width=3600,
        height=300,
        staff_cys=[70, 200],
        unit=12,
        barlines_x=[120, 820, 1500, 2200, 2900, 3520],
    )
    bls = [120, 820, 1500, 2200, 2900, 3520]
    staves[0].n_notes = 15

    # Draw Clef/Key at bar 0
    cv2.rectangle(img, (135, 50), (155, 90), 0, -1)

    class _IdentityDewarp:
        def dewarp(self, im: np.ndarray) -> np.ndarray:
            return im

        def dewarp_point(self, pt: tuple[float, float]) -> tuple[float, float]:
            return pt

        def undewarp_point(self, pt: tuple[float, float]) -> tuple[float, float]:
            return pt

    tromr_calls: list[tuple[int, int]] = []

    def _fake_tromr(staff_image: np.ndarray, staff: Any, config: Any) -> list[_FakeSym]:
        tromr_calls.append(staff_image.shape[:2])
        if len(tromr_calls) == 1:
            return [
                _FakeSym("clef_F4", coordinates=(20.0, 128.0)),
                _FakeSym("note_8", "F3", coordinates=(220.0, 128.0)),
                _FakeSym("barline", coordinates=(400.0, 128.0)),
                _FakeSym("note_8", "G3", coordinates=(700.0, 128.0)),
                _FakeSym("barline", coordinates=(1200.0, 128.0)),
            ]
        return [
            _FakeSym("clef_F4", coordinates=(20.0, 128.0)),
            _FakeSym("note_8", "A3", coordinates=(350.0, 128.0)),
            _FakeSym("barline", coordinates=(750.0, 128.0)),
            _FakeSym("note_8", "B3", coordinates=(1000.0, 128.0)),
            _FakeSym("barline", coordinates=(1250.0, 128.0)),
        ]

    fake_hsp = SimpleNamespace(
        _calculate_region=lambda st, regs: np.array([100, 20, 3540, 120]),
        crop_image_and_return_new_top=lambda im, x1, y1, x2, y2: (
            im[max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)],
            np.array([max(0, int(x1)), max(0, int(y1))]),
        ),
        _dewarp_staff=lambda st, dw, tl, sc: st,
        dewarp_staff_image=lambda im, st, idx, dbg: _IdentityDewarp(),
        remove_black_contours_at_edges_of_image=lambda im, u: im,
        add_image_into_tr_omr_canvas=lambda im: np.zeros((256, 1280), dtype=np.uint8),
        get_tr_omr_canvas_size=lambda shp: np.array([1280, max(1, int(round(shp[0] * 1280.0 / max(1, shp[1]))))]),
        parse_staff_tromr=_fake_tromr,
        _get_symbol_center=lambda sym: sym.coordinates,
        undo_crop=lambda x, y: (lambda pt: (pt[0] + x, pt[1] + y)),
        chain=lambda *fns: (lambda pt: [pt := fn(pt) for fn in fns][-1]),
    )
    fallback_calls: list[int] = []
    orig_parse = lambda *a, **k: fallback_calls.append(1) or [_FakeSym("fallback")]

    out = parse_staff_image_chunked(
        None,
        0,
        staves[0],
        img,
        None,
        None,
        lambda pt: (pt[0] * 2.0, pt[1] * 2.0),
        barlines_x=bls,
        orig_parse_staff_image=orig_parse,
        homr_staff_parsing=fake_hsp,
    )
    assert not fallback_calls
    assert len(tromr_calls) >= 2
    assert [s.rhythm for s in out][:3] == ["clef_F4", "note_8", "barline"]
    # Check image_coordinates were mapped through page_to_input_image (2x scale here)
    assert out[1].image_coordinates is not None and out[1].image_coordinates[0] > 0

    # Exception inside chunking falls back cleanly to orig_parse_staff_image
    fake_hsp.parse_staff_tromr = lambda **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    fb_out = parse_staff_image_chunked(
        None, 0, staves[0], img, None, None, lambda pt: pt,
        barlines_x=bls, orig_parse_staff_image=orig_parse, homr_staff_parsing=fake_hsp,
    )
    assert [s.rhythm for s in fb_out] == ["fallback"]

    # CancelledError propagates directly
    with pytest.raises(CancelledError):
        parse_staff_image_chunked(
            None, 0, staves[0], img, None, None, lambda pt: pt,
            barlines_x=bls, orig_parse_staff_image=orig_parse, homr_staff_parsing=fake_hsp,
            is_cancelled=lambda: True,
        )
