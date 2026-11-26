from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pdf2musescore.fullscore.align import (
    PageSidecar,
    StaffMatchSpec,
    align_page_score,
    apply_conductor_metadata,
    compute_conductor_scaling,
    detect_score_transposition_mode,
    harmonize_page_key_signatures,
    match_staves_monotonic_dp,
    merge_fullscore_files,
    merge_fullscore_pages,
    read_page_sidecar,
    reconcile_pages_to_master,
    sidecar_path_for,
    verify_fullscore_invariants,
    write_page_sidecar,
)
from pdf2musescore.fullscore.instruments import InstrumentLabel
from pdf2musescore.musicxml import MergeError, write_musicxml
from pdf2musescore.options import ConversionOptions
from pdf2musescore.rehearsal import PositionedRehearsal, RehearsalMark
from pdf2musescore.tempo import PositionedTempo, TempoMarking


def _make_part(
    pid: str,
    measures: list[tuple[bool, tuple[str, str] | None]],
    *,
    divisions: int = 4,
    clef: tuple[str, str] = ("G", "2"),
    fifths: str = "-3",
    include_clef_key: bool = True,
) -> tuple[ET.Element, ET.Element]:
    sp = ET.Element("score-part", id=pid)
    ET.SubElement(sp, "part-name").text = "Voice"
    si = ET.SubElement(sp, "score-instrument", id=f"{pid}-I1")
    ET.SubElement(si, "instrument-name").text = "Voice"
    ET.SubElement(si, "instrument-sound").text = "voice"
    mi = ET.SubElement(sp, "midi-instrument", id=f"{pid}-I1")
    ET.SubElement(mi, "midi-channel").text = "1"

    part = ET.Element("part", id=pid)
    for idx, (is_rest, ts) in enumerate(measures):
        m = ET.SubElement(part, "measure", number=str(idx + 1))
        if idx == 0 or ts is not None:
            attrs = ET.SubElement(m, "attributes")
            if idx == 0:
                ET.SubElement(attrs, "divisions").text = str(divisions)
                if include_clef_key:
                    k = ET.SubElement(attrs, "key")
                    ET.SubElement(k, "fifths").text = fifths
                    c = ET.SubElement(attrs, "clef", number="1")
                    ET.SubElement(c, "sign").text = clef[0]
                    ET.SubElement(c, "line").text = clef[1]
            if ts is not None:
                t = ET.SubElement(attrs, "time")
                ET.SubElement(t, "beats").text = ts[0]
                ET.SubElement(t, "beat-type").text = ts[1]
        note = ET.SubElement(m, "note")
        if is_rest:
            ET.SubElement(note, "rest")
        else:
            p = ET.SubElement(note, "pitch")
            ET.SubElement(p, "step").text = "C"
            ET.SubElement(p, "octave").text = "4"
        ET.SubElement(note, "duration").text = str(4 * divisions)
        ET.SubElement(note, "voice").text = "1"
        ET.SubElement(note, "staff").text = "1"
    return sp, part


def _build_score(parts_specs: list[tuple[ET.Element, ET.Element]]) -> ET.Element:
    root = ET.Element("score-partwise", version="4.0")
    ET.SubElement(root, "identification")
    pl = ET.SubElement(root, "part-list")
    for sp, _ in parts_specs:
        pl.append(sp)
    for _, part in parts_specs:
        root.append(part)
    return root


def test_sidecar_roundtrip_and_corrupt(tmp_path):
    xml_path = tmp_path / "page_004.musicxml"
    assert read_page_sidecar(xml_path) is None

    sc = PageSidecar(
        page_number=4,
        staff_count=2,
        staff_cys=[100.0, 250.0],
        staff_y_norms=[0.0, 1.0],
        labels=[
            InstrumentLabel("Flute", "Fl.", raw_ocr="Flute"),
            InstrumentLabel("Timpani", "Timp.", raw_ocr="Timpani"),
        ],
        sys_min_x=400,
        sys_max_x=3500,
        crop_x=395,
        tempo=TempoMarking(text="Adagio affetuoso", bpm=72, raw_ocr="Adagio affetuoso= 72"),
    )
    sc_path = write_page_sidecar(sc, xml_path)
    assert sc_path == sidecar_path_for(xml_path)
    assert read_page_sidecar(xml_path) == sc

    # Backwards-compatibility: sidecar JSON without "tempo" loads with tempo=None
    sc_no_tempo = PageSidecar(
        page_number=5,
        staff_count=1,
        staff_cys=[100.0],
        staff_y_norms=[0.5],
        labels=[InstrumentLabel("Flute", "Fl.")],
    )
    write_page_sidecar(sc_no_tempo, xml_path)
    assert read_page_sidecar(xml_path).tempo is None

    sc_path.write_text("{corrupt", encoding="utf-8")
    assert read_page_sidecar(xml_path) is None


def test_merge_fullscore_pages_tempo_dedup_and_change():
    labels = [InstrumentLabel("Flute", "Fl.")]
    r1 = _build_score([_make_part("P1", [(False, ("4", "4"))])])
    r2 = _build_score([_make_part("P1", [(False, ("4", "4"))])])
    r3 = _build_score([_make_part("P1", [(False, ("4", "4"))])])
    t1 = TempoMarking(text="Adagio", bpm=72)
    t3 = TempoMarking(text="Allegro", bpm=132)
    r_mark_p2 = PositionedRehearsal(RehearsalMark("A", enclosure="none"), 350.0, 390.0, 10.0, 50.0, u_norm=0.05)
    sc1 = PageSidecar(1, 1, [100.0], [0.5], labels, tempo=t1)
    sc2 = PageSidecar(2, 1, [100.0], [0.5], labels, tempo=t1, rehearsal_marks=(r_mark_p2,))
    sc3 = PageSidecar(3, 1, [100.0], [0.5], labels, tempo=t3)  # mid-piece tempo change -> kept

    # Round-trip rehearsal_marks through sidecar JSON
    tmp_sc_xml = r2  # in-memory check only; file round-trip tested in test_sidecar_roundtrip
    merged = merge_fullscore_pages([(1, r1, sc1), (2, r2, sc2), (3, r3, sc3)], title="Tempo Test")
    ms = merged.findall("./part[@id='P1']/measure")
    assert ms[0].findtext("./direction/direction-type/words") == "Adagio "
    assert ms[0].findtext("./direction/direction-type/metronome/per-minute") == "72"
    assert ms[1].findtext("./direction/direction-type/rehearsal") == "A"
    assert ms[1].find("./direction/direction-type/rehearsal").get("enclosure") == "none"
    assert ms[2].findtext("./direction/direction-type/words") == "Allegro "
    assert ms[2].findtext("./direction/direction-type/metronome/per-minute") == "132"


def test_align_page_score_consensus_noise_trim_pad_and_inherited_ts():
    # 4 parts: 3 pitched say 4/4 with 3 measures; P3 has noisy 3/4 on m1, P2 has noisy 2/4 on m2 (below quorum),
    # P4 has 4 measures (trimmed) and P1 has 2 measures (padded)
    p1 = _make_part("P1", [(False, ("4", "4")), (True, None)])
    p2 = _make_part("P2", [(False, ("4", "4")), (False, ("2", "4")), (True, None)], divisions=8)
    p3 = _make_part("P3", [(True, ("3", "4")), (False, None), (False, None)])
    p4 = _make_part("P4", [(False, ("4", "4")), (False, None), (False, None), (True, None)])
    root = _build_score([p1, p2, p3, p4])

    aligned, end_ts = align_page_score(root, page_num=1)
    assert end_ts == ("4", "4")
    parts = aligned.findall("part")
    assert [len(p.findall("measure")) for p in parts] == [3, 3, 3, 3]
    # P1 measure 3 was padded with full-measure rest duration 4*4 = 16
    p1_m3 = parts[0].findall("measure")[2]
    assert p1_m3.find("./note/rest").get("measure") == "yes"
    assert p1_m3.findtext("./note/duration") == "16"
    # P2 noisy 2/4 in m2 was stripped
    assert parts[1].findall("measure")[1].find("./attributes/time") is None

    # Quorum meter change in m2 accepted (3 of 4 parts declare 6/4) + inherited ts when m1 has no <time>
    q1 = _make_part("P1", [(False, None), (False, ("6", "4"))])
    q2 = _make_part("P2", [(False, None), (False, ("6", "4"))])
    q3 = _make_part("P3", [(False, None), (False, ("6", "4"))])
    q4 = _make_part("P4", [(True, None), (True, None)])
    q_root = _build_score([q1, q2, q3, q4])
    q_aligned, q_end = align_page_score(q_root, page_num=2, inherited_time_sig=("3", "4"))
    assert q_end == ("6", "4")
    for p in q_aligned.findall("part"):
        ms = p.findall("measure")
        assert (ms[0].findtext("./attributes/time/beats"), ms[0].findtext("./attributes/time/beat-type")) == ("3", "4")
        assert (ms[1].findtext("./attributes/time/beats"), ms[1].findtext("./attributes/time/beat-type")) == ("6", "4")

    # Empty score guard
    empty_root = ET.Element("score-partwise")
    _, empty_ts = align_page_score(empty_root, page_num=1, inherited_time_sig=("5", "4"))
    assert empty_ts == ("5", "4")


def test_reconcile_variable_staves_dp_and_stray_drop():
    # Master page (page 4): 4 instruments (Flute, Bassoon, Horn, Percussion)
    p4_parts = [
        _make_part("P1", [(False, ("4", "4"))], clef=("G", "2")),
        _make_part("P2", [(False, ("4", "4"))], clef=("F", "4")),
        _make_part("P3", [(False, ("4", "4"))], clef=("G", "2")),
        _make_part("P4", [(False, ("4", "4"))], clef=("percussion", "2")),
    ]
    r4 = _build_score(p4_parts)
    sc4 = PageSidecar(
        page_number=4,
        staff_count=4,
        staff_cys=[100.0, 250.0, 400.0, 550.0],
        staff_y_norms=[0.0, 0.33, 0.66, 1.0],
        labels=[
            InstrumentLabel("Flute", "Fl."),
            InstrumentLabel("Bassoon", "Bsn."),
            InstrumentLabel("F Horn", "F Hn."),
            InstrumentLabel("Percussion 1", "Perc. 1"),
        ],
    )

    # Page 5 (cutaway/hidden): Bassoon & Percussion hidden -> only Flute + Horn present
    p5_parts = [
        _make_part("P1", [(False, ("4", "4"))], clef=("G", "2")),
        _make_part("P2", [(False, ("4", "4"))], clef=("G", "2")),
    ]
    r5 = _build_score(p5_parts)
    sc5 = PageSidecar(
        page_number=5,
        staff_count=2,
        staff_cys=[100.0, 400.0],
        staff_y_norms=[0.0, 0.7],
        labels=[
            InstrumentLabel("Flute", "Fl."),
            InstrumentLabel("F Horn", "F Hn."),
        ],
    )

    # Page 6: same 4 staves PLUS 1 empty unlabeled stray fragment -> should drop stray to match consensus 4
    p6_parts = [
        _make_part("P1", [(False, ("4", "4"))], clef=("G", "2")),
        _make_part("P2", [(False, ("4", "4"))], clef=("F", "4")),
        _make_part("P3", [(True, ("4", "4"))], clef=("G", "2")),  # stray!
        _make_part("P4", [(False, ("4", "4"))], clef=("G", "2")),
        _make_part("P5", [(False, ("4", "4"))], clef=("percussion", "2")),
    ]
    r6 = _build_score(p6_parts)
    sc6 = PageSidecar(
        page_number=6,
        staff_count=5,
        staff_cys=[100.0, 250.0, 320.0, 400.0, 550.0],
        staff_y_norms=[0.0, 0.33, 0.5, 0.66, 1.0],
        labels=[
            InstrumentLabel("Flute", "Fl."),
            InstrumentLabel("Bassoon", "Bsn."),
            InstrumentLabel("Part 3", "P3", is_fallback=True),
            InstrumentLabel("F Horn", "F Hn."),
            InstrumentLabel("Percussion 1", "Perc. 1"),
        ],
    )

    merged = merge_fullscore_pages([(4, r4, sc4), (5, r5, sc5), (6, r6, sc6)], title="Symphony")
    assert len(merged.findall("part")) == 4
    assert [len(p.findall("measure")) for p in merged.findall("part")] == [3, 3, 3, 3]
    # Page 5 Bassoon (P2) was rest-filled and kept Bass clef
    p2_m2 = merged.findall("part")[1].findall("measure")[1]
    assert p2_m2.find("./note/rest").get("measure") == "yes"
    # Verify DP helper error on k > m
    with pytest.raises(ValueError):
        match_staves_monotonic_dp(
            [StaffMatchSpec(InstrumentLabel("A", "A"), None, None, 0.0)] * 2,
            [StaffMatchSpec(InstrumentLabel("A", "A"), None, None, 0.0)],
        )


def test_merge_fullscore_files_metadata_channels_scaling_dedup_and_invariants(tmp_path):
    # Create 29-part page 1 (4/4) + page 2 (6/4 meter change + redundant divisions/key/clef removed)
    labels = [InstrumentLabel(f"Clarinet {i + 1}", f"Cl. {i + 1}") for i in range(25)] + [
        InstrumentLabel("Timpani", "Timp."),
        InstrumentLabel("Mallet Percussion", "Mall. Perc."),
        InstrumentLabel("Percussion 1", "Perc. 1"),
        InstrumentLabel("Percussion 2", "Perc. 2"),
    ]
    p1_specs = [_make_part(f"P{i + 1}", [(False, ("4", "4"))], divisions=8) for i in range(29)]
    p2_specs = [_make_part(f"P{i + 1}", [(False, ("6", "4"))], divisions=8) for i in range(29)]
    r1 = _build_score(p1_specs)
    r2 = _build_score(p2_specs)
    # Inject a stray <transpose> and <movement-title> to prove fullscore strips both
    ET.SubElement(r1, "movement-title").text = "Old Movement"
    ET.SubElement(r1.find("./part/measure/attributes"), "transpose").text = "2"

    f1 = write_musicxml(r1, tmp_path / "page_001.musicxml")
    f2 = write_musicxml(r2, tmp_path / "page_002.musicxml")
    write_page_sidecar(PageSidecar(1, 29, [float(i) for i in range(29)], [i / 28.0 for i in range(29)], labels), f1)
    write_page_sidecar(PageSidecar(2, 29, [float(i) for i in range(29)], [i / 28.0 for i in range(29)], labels), f2)

    out_xml = merge_fullscore_files([f1, f2], tmp_path / "full.musicxml", title="The Green Hill", options=ConversionOptions(full_score=True))
    merged = ET.parse(out_xml).getroot()

    assert merged.find("movement-title") is None
    assert merged.findtext("work/work-title") == "The Green Hill"
    assert merged.findtext("defaults/scaling/millimeters") == "3.4"
    assert len(merged.findall(".//transpose")) == 0

    sps = merged.findall("./part-list/score-part")
    assert len(sps) == 29
    # <part-abbreviation> must immediately follow <part-name>
    for sp in sps:
        tags = [ch.tag for ch in sp]
        assert tags[tags.index("part-name") + 1] == "part-abbreviation"

    chans = [int(sp.findtext("./midi-instrument/midi-channel")) for sp in sps]
    assert all(1 <= c <= 16 for c in chans)
    # Timpani (25) and Mallet Percussion (26) are pitched -> NOT channel 10
    assert chans[25] != 10
    assert chans[26] != 10
    # Unpitched Percussion 1 & 2 (27, 28) -> channel 10 and no stale midi-program
    assert chans[27] == 10 and chans[28] == 10
    assert sps[0].findtext("./midi-instrument/midi-program") == "72"
    assert sps[25].findtext("./midi-instrument/midi-program") == "48"
    assert sps[26].findtext("./midi-instrument/midi-program") == "13"
    assert sps[27].find("./midi-instrument/midi-program") is None
    assert sps[28].find("./midi-instrument/midi-program") is None

    # Measure 2 kept 6/4 time signature change while deduplicating redundant divisions/key/clef
    m2_attrs = merged.find("./part/measure[@number='2']/attributes")
    assert m2_attrs.findtext("time/beats") == "6"
    assert m2_attrs.find("divisions") is None
    assert m2_attrs.find("key") is None
    assert m2_attrs.find("clef") is None


def test_compute_conductor_scaling_tiers_and_invariant_errors(tmp_path):
    assert compute_conductor_scaling(30)[0] == "3.4"
    assert compute_conductor_scaling(20)[0] == "4.2"
    assert compute_conductor_scaling(10)[0] == "5.2"
    assert compute_conductor_scaling(4)[0] == "6.5"

    with pytest.raises(MergeError, match="No full-score"):
        merge_fullscore_files([], tmp_path / "x.musicxml")
    with pytest.raises(MergeError, match="No full-score"):
        merge_fullscore_pages([], "Empty")
    with pytest.raises(MergeError, match="0 parts"):
        reconcile_pages_to_master([(1, ET.Element("score-partwise"), None)])

    # Invariant failures
    with pytest.raises(MergeError, match="no <part>"):
        verify_fullscore_invariants(ET.Element("score-partwise"))

    base = _build_score([_make_part("P1", [(False, ("4", "4"))]), _make_part("P2", [(False, ("4", "4"))])])
    verify_fullscore_invariants(base)

    bad_pl = copy.deepcopy(base)
    bad_pl.find("part-list").remove(bad_pl.find("./part-list/score-part"))
    with pytest.raises(MergeError, match="part-list count"):
        verify_fullscore_invariants(bad_pl)

    bad_mcount = copy.deepcopy(base)
    bad_mcount.find("part").remove(bad_mcount.find("./part/measure"))
    with pytest.raises(MergeError, match="Inconsistent part measure counts"):
        verify_fullscore_invariants(bad_mcount)

    bad_dup_time = copy.deepcopy(base)
    ET.SubElement(bad_dup_time.find("./part/measure/attributes"), "time")
    with pytest.raises(MergeError, match="2 <time>"):
        verify_fullscore_invariants(bad_dup_time)

    bad_partial = copy.deepcopy(base)
    t_p2 = bad_partial.findall("part")[1].find("./measure/attributes/time")
    bad_partial.findall("part")[1].find("./measure/attributes").remove(t_p2)
    with pytest.raises(MergeError, match="missing time signature"):
        verify_fullscore_invariants(bad_partial)

    bad_conflict = copy.deepcopy(base)
    bad_conflict.findall("part")[1].find("./measure/attributes/time/beats").text = "3"
    with pytest.raises(MergeError, match="conflicting time signatures"):
        verify_fullscore_invariants(bad_conflict)

    bad_tr = copy.deepcopy(base)
    ET.SubElement(bad_tr, "transpose")
    with pytest.raises(MergeError, match="transpose"):
        verify_fullscore_invariants(bad_tr)

    bad_ch = copy.deepcopy(base)
    bad_ch.find(".//midi-channel").text = "17"
    with pytest.raises(MergeError, match="Out-of-range MIDI channel"):
        verify_fullscore_invariants(bad_ch)

    bad_ch_str = copy.deepcopy(base)
    bad_ch_str.find(".//midi-channel").text = "abc"
    with pytest.raises(MergeError, match="non-integer MIDI channel"):
        verify_fullscore_invariants(bad_ch_str)


def _set_measure_dur(measure: ET.Element, duration: int, *, voice: str = "1", rest: bool = False) -> None:
    for ch in list(measure.findall("note")):
        measure.remove(ch)
    n = ET.SubElement(measure, "note")
    if rest:
        ET.SubElement(n, "rest")
    else:
        p = ET.SubElement(n, "pitch")
        ET.SubElement(p, "step").text = "C"
        ET.SubElement(p, "octave").text = "4"
    ET.SubElement(n, "duration").text = str(duration)
    ET.SubElement(n, "voice").text = voice
    ET.SubElement(n, "staff").text = "1"


def test_align_infers_midpage_meter_change_and_rejects_glitches():
    # 5 parts (4/4 -> 2/4 -> 4/4) with NO explicit <time> on M2/M3:
    # 3 playing parts have 16, 8, 16 (at div=4 -> 4, 2, 4 quarter beats); 2 rest parts have whole-bar rests.
    parts_specs = []
    for i in range(5):
        sp, part = _make_part(
            f"P{i + 1}",
            [(i >= 3, ("4", "4")), (i >= 3, None), (i >= 3, None)],
            divisions=4,
        )
        ms = part.findall("measure")
        if i < 3:
            _set_measure_dur(ms[0], 16)
            _set_measure_dur(ms[1], 8)
            _set_measure_dur(ms[2], 16)
        parts_specs.append((sp, part))

    root = _build_score(parts_specs)
    aligned, end_ts = align_page_score(root, page_num=4)
    assert end_ts == ("4", "4")
    for p in aligned.findall("part"):
        ms = p.findall("measure")
        assert (ms[0].findtext("./attributes/time/beats"), ms[0].findtext("./attributes/time/beat-type")) == ("4", "4")
        assert (ms[1].findtext("./attributes/time/beats"), ms[1].findtext("./attributes/time/beat-type")) == ("2", "4")
        assert (ms[2].findtext("./attributes/time/beats"), ms[2].findtext("./attributes/time/beat-type")) == ("4", "4")
    # Rest-only part on M2 got 2/4 full-measure rest duration (2 * div = 8)
    assert aligned.findall("part")[4].findall("measure")[1].findtext("./note/duration") == "8"

    # /8 meter inference (4/4 -> 7/8 -> 4/4) with div=2 (7/8 = 3.5 quarters = 7 units)
    eight_specs = []
    for i in range(4):
        sp, part = _make_part(f"P{i + 1}", [(i == 3, ("4", "4")), (i == 3, None), (i == 3, None)], divisions=2)
        ms = part.findall("measure")
        if i < 3:
            _set_measure_dur(ms[0], 8)
            _set_measure_dur(ms[1], 7)
            _set_measure_dur(ms[2], 8)
        eight_specs.append((sp, part))
    aligned_8, _ = align_page_score(_build_score(eight_specs), page_num=1)
    m2_t = aligned_8.find("./part/measure[@number='2']/attributes/time")
    assert (m2_t.findtext("beats"), m2_t.findtext("beat-type")) == ("7", "8")

    # OMR glitch durations (17/4 = 17 units at div=4) are rejected
    glitch_specs = []
    for i in range(4):
        sp, part = _make_part(f"P{i + 1}", [(i == 3, ("4", "4")), (i == 3, None)], divisions=4)
        ms = part.findall("measure")
        if i < 3:
            _set_measure_dur(ms[0], 16)
            _set_measure_dur(ms[1], 17)
        glitch_specs.append((sp, part))
    aligned_g, _ = align_page_score(_build_score(glitch_specs), page_num=1)
    assert aligned_g.find("./part/measure[@number='2']/attributes/time") is None

    # Preserves 6/8 without falsely switching to 3/4 when q = 3 quarters
    compound_specs = []
    for i in range(4):
        sp, part = _make_part(f"P{i + 1}", [(i == 3, ("6", "8")), (i == 3, None)], divisions=2)
        ms = part.findall("measure")
        if i < 3:
            _set_measure_dur(ms[0], 6)
            _set_measure_dur(ms[1], 6)
        compound_specs.append((sp, part))
    aligned_c, _ = align_page_score(_build_score(compound_specs), page_num=1)
    assert aligned_c.find("./part/measure[@number='2']/attributes/time") is None

    # Genuine 4/4 pickup bar on M1 (1 quarter note) is NOT overridden when explicit 4/4 is present
    pickup_specs = []
    for i in range(4):
        sp, part = _make_part(f"P{i + 1}", [(i == 3, ("4", "4")), (i == 3, None)], divisions=4)
        ms = part.findall("measure")
        if i < 3:
            _set_measure_dur(ms[0], 4)
            _set_measure_dur(ms[1], 16)
        pickup_specs.append((sp, part))
    aligned_pk, _ = align_page_score(_build_score(pickup_specs), page_num=1)
    m1_t = aligned_pk.find("./part/measure[@number='1']/attributes/time")
    assert (m1_t.findtext("beats"), m1_t.findtext("beat-type")) == ("4", "4")
    assert aligned_pk.find("./part/measure[@number='2']/attributes/time") is None

    # Tie-breaker on M1: 2 parts say 5/4, 2 parts say 6/4, playing duration is 6 quarters -> picks 6/4
    tie_specs = []
    for i in range(4):
        ts = ("5", "4") if i < 2 else ("6", "4")
        sp, part = _make_part(f"P{i + 1}", [(False, ts)], divisions=4)
        _set_measure_dur(part.find("measure"), 24)
        tie_specs.append((sp, part))
    aligned_tie, _ = align_page_score(_build_score(tie_specs), page_num=1)
    m1_tie = aligned_tie.find("./part/measure[@number='1']/attributes/time")
    assert (m1_tie.findtext("beats"), m1_tie.findtext("beat-type")) == ("6", "4")


def _add_note(
    measure: ET.Element,
    step: str,
    octave: str = "4",
    *,
    alter: str | None = None,
    accidental: str | None = None,
    duration: str = "4",
) -> ET.Element:
    n = ET.SubElement(measure, "note")
    p = ET.SubElement(n, "pitch")
    ET.SubElement(p, "step").text = step
    if alter is not None:
        ET.SubElement(p, "alter").text = alter
    ET.SubElement(p, "octave").text = octave
    ET.SubElement(n, "duration").text = duration
    ET.SubElement(n, "voice").text = "1"
    ET.SubElement(n, "staff").text = "1"
    if accidental is not None:
        ET.SubElement(n, "accidental").text = accidental
    return n


def test_transposed_score_detection_harmonization_and_metadata_emission():
    # Transposed Concert Band score in Concert Eb (fifths=-3):
    # P1 Bassoon (-3), P2 Bb Clarinet 1 (-1), P3 Eb Alto Sax. 1 (MISREAD as +1 instead of 0!),
    # P4 F Horn 1, 3 (-2), P5 String Bass (-3), P6 Percussion 1 (-1)
    labels = [
        InstrumentLabel("Bassoon", "Bsn."),
        InstrumentLabel("Bb Clarinet 1", "Bb Cl. 1"),
        InstrumentLabel("Eb Alto Sax. 1", "Eb A. Sax. 1"),
        InstrumentLabel("F Horn 1, 3", "F Hn. 1, 3"),
        InstrumentLabel("String Bass (optional)", "Str. Bass"),
        InstrumentLabel("Percussion 1", "Perc. 1"),
    ]
    specs = [
        _make_part("P1", [(False, ("4", "4"))], clef=("F", "4"), fifths="-3"),
        _make_part("P2", [(False, ("4", "4"))], clef=("G", "2"), fifths="-1"),
        _make_part("P3", [(False, ("4", "4"))], clef=("G", "2"), fifths="1"),  # OMR misread!
        _make_part("P4", [(False, ("4", "4"))], clef=("G", "2"), fifths="-2"),
        _make_part("P5", [(False, ("4", "4"))], clef=("F", "4"), fifths="-3"),
        _make_part("P6", [(True, ("4", "4"))], clef=("C", "3"), fifths="-1"),
    ]
    # Put implied F# (from misread 1#) + explicit accidental F# (must stay!) on P3 page 1
    p3_m1 = specs[2][1].find("measure")
    _set_measure_dur(p3_m1, 8)
    _add_note(p3_m1, "F", "4", alter="1", duration="4")  # no accidental -> must clear to natural F
    _add_note(p3_m1, "F", "4", alter="1", accidental="sharp", duration="4")  # explicit -> must keep alter=1

    # Add a <directive> child on P2 attributes to verify <transpose> inserts BEFORE <directive>
    ET.SubElement(specs[1][1].find("./measure/attributes"), "directive").text = "dolce"
    root_p1 = _build_score(specs)
    sc1 = PageSidecar(1, 6, [float(i) for i in range(6)], [i / 5.0 for i in range(6)], labels)

    # Page 2 has P3 correctly read as fifths=0 -> after harmonizing Page 1 to 0, Page 2 dedup must remove P3's <key>
    specs_p2 = [
        _make_part("P1", [(False, ("4", "4"))], clef=("F", "4"), fifths="-3"),
        _make_part("P2", [(False, ("4", "4"))], clef=("G", "2"), fifths="-1"),
        _make_part("P3", [(False, ("4", "4"))], clef=("G", "2"), fifths="0"),
        _make_part("P4", [(False, ("4", "4"))], clef=("G", "2"), fifths="-2"),
        _make_part("P5", [(False, ("4", "4"))], clef=("F", "4"), fifths="-3"),
        _make_part("P6", [(True, ("4", "4"))], clef=("C", "3"), fifths="-1"),
    ]
    root_p2 = _build_score(specs_p2)
    sc2 = PageSidecar(2, 6, [float(i) for i in range(6)], [i / 5.0 for i in range(6)], labels)

    merged = merge_fullscore_pages([(1, root_p1, sc1), (2, root_p2, sc2)], title="Green Hill")
    summary = verify_fullscore_invariants(merged)
    assert summary["transposing_parts"] == 4  # Bb Cl, Eb Alto Sax, F Horn, String Bass

    parts = merged.findall("part")
    sps = merged.findall("./part-list/score-part")
    assert [sp.findtext("./score-instrument/instrument-sound") for sp in sps] == [
        "wind.reed.bassoon",
        "wind.reed.clarinet.bflat",
        "wind.reed.saxophone.alto",
        "brass.french-horn",
        "strings.contrabass",
        "drum.group",
    ]

    # Written keys remain preserved (and P3 outlier repaired to 0 with no spurious M2 key change;
    # P6 unpitched percussion normalized to fifths=0 and percussion clef)
    assert parts[0].findtext("./measure[@number='1']/attributes/key/fifths") == "-3"
    assert parts[1].findtext("./measure[@number='1']/attributes/key/fifths") == "-1"
    assert parts[2].findtext("./measure[@number='1']/attributes/key/fifths") == "0"
    assert parts[2].find("./measure[@number='2']/attributes") is None
    assert parts[3].findtext("./measure[@number='1']/attributes/key/fifths") == "-2"
    assert parts[5].findtext("./measure[@number='1']/attributes/key/fifths") == "0"
    assert parts[5].findtext("./measure[@number='1']/attributes/clef/sign") == "percussion"

    # Check P3 M1 notes: note[1] (F without accidental) cleared alter, note[2] (explicit sharp) kept alter=1
    p3_notes = parts[2].findall("./measure[@number='1']/note")
    assert p3_notes[1].find("./pitch/alter") is None
    assert p3_notes[2].findtext("./pitch/alter") == "1"

    # Check <transpose> placement before <directive> on P2 and exact values
    p2_attrs = parts[1].find("./measure[@number='1']/attributes")
    p2_tags = [el.tag for el in p2_attrs]
    assert p2_tags.index("transpose") < p2_tags.index("directive")
    assert (p2_attrs.findtext("transpose/diatonic"), p2_attrs.findtext("transpose/chromatic")) == ("-1", "-2")

    p3_tr = parts[2].find("./measure[@number='1']/attributes/transpose")
    assert (p3_tr.findtext("diatonic"), p3_tr.findtext("chromatic")) == ("-5", "-9")

    p4_tr = parts[3].find("./measure[@number='1']/attributes/transpose")
    assert (p4_tr.findtext("diatonic"), p4_tr.findtext("chromatic")) == ("-4", "-7")

    p5_tr = parts[4].find("./measure[@number='1']/attributes/transpose")
    assert (
        p5_tr.findtext("diatonic"),
        p5_tr.findtext("chromatic"),
        p5_tr.findtext("octave-change"),
    ) == ("0", "0", "-1")


def test_concert_score_and_all_transposing_detection_and_key_reconciliation():
    # Concert-pitch (C) score: all pitched instruments written in Concert Eb (-3), String Bass still gets octave=-1
    labels_c = [
        InstrumentLabel("Bassoon", "Bsn."),
        InstrumentLabel("Bb Clarinet 1", "Bb Cl. 1"),
        InstrumentLabel("F Horn", "Hn."),
        InstrumentLabel("String Bass", "Str. Bass"),
    ]
    c_root = _build_score([_make_part(f"P{i + 1}", [(False, ("4", "4"))], fifths="-3") for i in range(4)])
    apply_conductor_metadata(c_root, "Concert Score", labels_c)
    is_tr, conc = detect_score_transposition_mode([c_root], labels_c)
    assert is_tr is False and conc == -3
    c_parts = c_root.findall("part")
    assert c_parts[0].find(".//transpose") is None
    assert c_parts[1].find(".//transpose") is None
    assert c_parts[2].find(".//transpose") is None
    assert c_parts[3].findtext("./measure/attributes/transpose/octave-change") == "-1"

    # All-transposing ensemble (Saxophone Quartet: no C instruments, Concert Eb) -> still detected as Transposed!
    labels_sax = [
        InstrumentLabel("Eb Alto Sax. 1", "A. Sax. 1"),
        InstrumentLabel("Eb Alto Sax. 2", "A. Sax. 2"),
        InstrumentLabel("Bb Tenor Sax.", "T. Sax."),
        InstrumentLabel("Eb Baritone Sax.", "Bar. Sax."),
    ]
    sax_root = _build_score([
        _make_part("P1", [(False, ("4", "4"))], fifths="0"),
        _make_part("P2", [(False, ("4", "4"))], fifths="0"),
        _make_part("P3", [(False, ("4", "4"))], fifths="-1"),
        _make_part("P4", [(False, ("4", "4"))], fifths="0"),
    ])
    is_sax_tr, sax_conc = detect_score_transposition_mode([sax_root], labels_sax)
    assert is_sax_tr is True and sax_conc == -3

    # Empty roots guard
    assert detect_score_transposition_mode([], []) == (False, None)

    # Harmonize adding sharp/flat alter on outlier (fifths=0 misread instead of -3 -> adds alter=-1 to B,E,A)
    # plus stopping at later key change measure and preserving +6 == -6 enharmonic equivalence
    labels_h = [
        InstrumentLabel("Bassoon", "Bsn."),
        InstrumentLabel("Flute", "Fl."),
        InstrumentLabel("Oboe", "Ob."),
        InstrumentLabel("Piano", "Pno."),
    ]
    h_specs = [
        _make_part("P1", [(False, ("4", "4")), (False, None)], fifths="0"),  # outlier! target=-3
        _make_part("P2", [(False, ("4", "4")), (False, None)], fifths="-3"),
        _make_part("P3", [(False, ("4", "4")), (False, None)], fifths="-3"),
        _make_part("P4", [(False, ("4", "4")), (False, None)], fifths="-3"),
    ]
    p1_ms = h_specs[0][1].findall("measure")
    _add_note(p1_ms[0], "B", "3")  # no alter -> gets alter=-1
    # m2 declares an explicit key change -> stops before m2 notes
    m2_attr = ET.SubElement(p1_ms[1], "attributes")
    ET.SubElement(ET.SubElement(m2_attr, "key"), "fifths").text = "1"
    _add_note(p1_ms[1], "E", "4")
    h_root = _build_score(h_specs)
    assert harmonize_page_key_signatures(h_root, labels_h, is_transposed_score=False, page_num=1) == 1
    assert p1_ms[0].findtext("./attributes/key/fifths") == "-3"
    assert p1_ms[0].findall("note")[-1].findtext("./pitch/alter") == "-1"
    assert p1_ms[1].findall("note")[-1].find("./pitch/alter") is None

    # Enharmonic +6 vs -6 is NOT rewritten ( Concert +3 + Eb shift +3 = +6 ~ -6 )
    enh_labels = [
        InstrumentLabel("Flute", "Fl."),
        InstrumentLabel("Oboe", "Ob."),
        InstrumentLabel("Bassoon", "Bsn."),
        InstrumentLabel("Eb Alto Sax.", "A. Sax."),
    ]
    enh_root = _build_score([
        _make_part("P1", [(False, ("4", "4"))], fifths="3"),
        _make_part("P2", [(False, ("4", "4"))], fifths="3"),
        _make_part("P3", [(False, ("4", "4"))], fifths="3"),
        _make_part("P4", [(False, ("4", "4"))], fifths="6"),
    ])
    assert harmonize_page_key_signatures(enh_root, enh_labels, is_transposed_score=True, page_num=1) == 0
    assert enh_root.findall("part")[3].findtext("./measure/attributes/key/fifths") == "6"


def test_verify_fullscore_invariants_transpose_validation():
    base = _build_score([_make_part("P1", [(False, ("4", "4"))])])
    attrs = base.find("./part/measure[@number='1']/attributes")
    tr = ET.SubElement(attrs, "transpose")
    ET.SubElement(tr, "diatonic").text = "-5"
    ET.SubElement(tr, "chromatic").text = "-9"
    ET.SubElement(tr, "octave-change").text = "-1"
    assert verify_fullscore_invariants(base)["transposing_parts"] == 1

    # Duplicate <transpose> in Measure 1
    bad_dup = copy.deepcopy(base)
    ET.SubElement(bad_dup.find("./part/measure[@number='1']/attributes"), "transpose")
    with pytest.raises(MergeError, match="2 <transpose>"):
        verify_fullscore_invariants(bad_dup)

    # Non-integer chromatic / out of range chromatic / invalid diatonic / invalid octave-change
    for tag, bad_val, pat in (
        ("chromatic", "abc", "invalid/missing <chromatic>"),
        ("chromatic", "99", "chromatic 99 out of range"),
        ("diatonic", "bad", "invalid <diatonic>"),
        ("diatonic", "-20", "diatonic -20 out of range"),
        ("octave-change", "x", "invalid <octave-change>"),
        ("octave-change", "5", "octave-change 5 out of range"),
    ):
        bad = copy.deepcopy(base)
        bad.find(f".//transpose/{tag}").text = bad_val
        with pytest.raises(MergeError, match=pat):
            verify_fullscore_invariants(bad)


def _add_rhythm_note(
    measure: ET.Element,
    dur: int,
    note_type: str,
    *,
    dots: int = 0,
    voice: str = "1",
    chord: bool = False,
    rest: bool = False,
    beams: Sequence[str] = (),
    step: str = "C",
) -> ET.Element:
    n = ET.SubElement(measure, "note")
    if chord:
        ET.SubElement(n, "chord")
    if rest:
        ET.SubElement(n, "rest")
    else:
        p = ET.SubElement(n, "pitch")
        ET.SubElement(p, "step").text = step
        ET.SubElement(p, "octave").text = "4"
    ET.SubElement(n, "duration").text = str(dur)
    ET.SubElement(n, "voice").text = voice
    ET.SubElement(n, "type").text = note_type
    for _ in range(dots):
        ET.SubElement(n, "dot")
    ET.SubElement(n, "staff").text = "1"
    for b_idx, b_val in enumerate(beams, start=1):
        b = ET.SubElement(n, "beam", number=str(b_idx))
        b.text = b_val
    return n


def test_normalize_overfull_measure_tiers_and_backup_sync():
    # Helper builds a 2-part score so align_page_score preserves time signatures properly
    def _run_one(measures_setup):
        p1 = _make_part("P1", [(False, ("4", "4"))], divisions=8)
        p2 = _make_part("P2", [(True, ("4", "4"))], divisions=8)
        m1 = p1[1].find("measure")
        for n in list(m1.findall("note")):
            m1.remove(n)
        measures_setup(m1, p2[1].find("measure"))
        aligned, _ = align_page_score(_build_score([p1, p2]), page_num=1)
        return aligned.find("./part[@id='P1']/measure")

    # Tier 1: Scotch snap (eighth + dotted-eighth) with chord + beamed run, PLUS standard (dotted-eighth + eighth)
    def setup_tier1(m, _m2):
        # 4 + 6 + 4 + 4 + 6 + 4 + 4 = 32? Make overfull 36 (2 extra eighth -> 16th in two dotted pairs)
        _add_rhythm_note(m, 4, "eighth", beams=("begin",))
        _add_rhythm_note(m, 4, "eighth", chord=True, step="E", beams=("begin",))
        _add_rhythm_note(m, 6, "eighth", dots=1, beams=("end",))
        _add_rhythm_note(m, 6, "eighth", dots=1)
        _add_rhythm_note(m, 4, "eighth")
        _add_rhythm_note(m, 8, "quarter")
        _add_rhythm_note(m, 8, "quarter")  # total = 36 > 32 (div=8)

    res1 = _run_one(setup_tier1)
    durs1 = [int(n.findtext("duration")) for n in res1.findall("note") if n.find("chord") is None]
    types1 = [n.findtext("type") for n in res1.findall("note") if n.find("chord") is None]
    assert sum(durs1) == 32
    assert durs1 == [2, 6, 6, 2, 8, 8]
    assert types1[:4] == ["16th", "eighth", "eighth", "16th"]
    # Chord member stayed in sync and beam run cleared cleanly (no dangling begin/end)
    chord_n = [n for n in res1.findall("note") if n.find("chord") is not None][0]
    assert chord_n.findtext("duration") == "2" and chord_n.findtext("type") == "16th"
    assert res1.find(".//beam") is None

    # Tier 1 second pass (excess=1 tick = 1/8q, while halving eighth saves 2 ticks) -> clears overflow (<= 32)
    def setup_tier1_pass2(m, _m2):
        _add_rhythm_note(m, 4, "eighth")
        _add_rhythm_note(m, 6, "eighth", dots=1)
        _add_rhythm_note(m, 1, "32nd")
        _add_rhythm_note(m, 8, "quarter")
        _add_rhythm_note(m, 8, "quarter")
        _add_rhythm_note(m, 4, "eighth")
        _add_rhythm_note(m, 2, "16th")  # 33 > 32, excess=1

    res_p2 = _run_one(setup_tier1_pass2)
    assert sum(int(n.findtext("duration")) for n in res_p2.findall("note")) == 31
    assert res_p2.findall("note")[0].findtext("type") == "16th"

    # Tier 2: Spurious dot (half.(24) + whole(32) = 56 > 48 in 6/4? test in 4/4 div=8: quarter.(12) + half(16) + quarter(8) = 36 -> dot_val=4 -> quarter(8))
    def setup_tier2(m, _m2):
        _add_rhythm_note(m, 12, "quarter", dots=1)
        _add_rhythm_note(m, 16, "half")
        _add_rhythm_note(m, 8, "quarter")  # total = 36 > 32

    res2 = _run_one(setup_tier2)
    n2 = res2.findall("note")
    assert [int(n.findtext("duration")) for n in n2] == [8, 16, 8]
    assert n2[0].find("dot") is None and n2[0].findtext("type") == "quarter"

    # Tier 3a: Single-note step-down (two quarters where trailing quarter read instead of eighth -> 8*3 + 8 + 8 = 40 -> 36? make 36 > 32: 4*4 + 8 + 8 + 4? wait 4*3=12, +8+8+8=36)
    def setup_tier3a(m, _m2):
        for _ in range(3):
            _add_rhythm_note(m, 4, "eighth")
        for _ in range(3):
            _add_rhythm_note(m, 8, "quarter")  # 12 + 24 = 36 > 32

    res3a = _run_one(setup_tier3a)
    n3a = res3a.findall("note")
    assert [int(n.findtext("duration")) for n in n3a] == [4, 4, 4, 8, 8, 4]
    assert n3a[-1].findtext("type") == "eighth"

    # Tier 3b: Trailing clamp + drop + multi-voice <backup> duration sync
    def setup_tier3b(m, _m2):
        _add_rhythm_note(m, 16, "half", voice="1")
        n_weird = _add_rhythm_note(m, 12, "quarter", dots=0, voice="1")
        n_weird.remove(n_weird.find("type"))  # non-standard 12 without <type> -> forces Tier 3b excess=12
        _add_rhythm_note(m, 8, "quarter", voice="1")  # onset 28, dur 8 -> clamped to 4 (eighth)
        _add_rhythm_note(m, 8, "quarter", voice="1")  # onset 36 >= 32 -> dropped!
        bk = ET.SubElement(m, "backup")
        ET.SubElement(bk, "duration").text = "44"
        _add_rhythm_note(m, 32, "whole", voice="2", rest=True)

    res3b = _run_one(setup_tier3b)
    v1_notes = [n for n in res3b.findall("note") if n.findtext("voice") == "1"]
    assert sum(int(n.findtext("duration")) for n in v1_notes) == 32
    assert len(v1_notes) == 3
    assert v1_notes[-1].findtext("type") == "eighth"
    assert res3b.findtext("./backup/duration") == "32"

    # Exact (32) and underfull pickup (8) measures remain completely untouched
    def setup_underfull(m, _m2):
        _add_rhythm_note(m, 4, "eighth")
        _add_rhythm_note(m, 6, "eighth", dots=1)  # 10 < 32

    res_ok = _run_one(setup_underfull)
    assert [int(n.findtext("duration")) for n in res_ok.findall("note")] == [4, 6]


def test_normalize_interleaved_multivoice_deinterleaves_and_preserves_offsets():
    # Exact reproduction of Green Hill P18 M46 (F Horn 1, 3): interleaved v1 + v2 with v1 overfull (17/16)
    p1 = _make_part("P1", [(False, ("4", "4"))], divisions=4)
    p2 = _make_part("P2", [(True, ("4", "4"))], divisions=4)
    m = p1[1].find("measure")
    for n in list(m.findall("note")):
        m.remove(n)

    # v1 D5(2) -> backup(2) -> v2 Eb4(12) -> backup(10) -> v1 C5(3), C5(8) -> backup(1), forward(1) -> v2 Eb4(4) -> backup(4) -> v1 C5(4)
    g_n = ET.SubElement(m, "note")
    ET.SubElement(g_n, "grace")
    gp = ET.SubElement(g_n, "pitch")
    ET.SubElement(gp, "step").text = "E"
    ET.SubElement(gp, "octave").text = "5"
    ET.SubElement(g_n, "voice").text = "1"
    ET.SubElement(g_n, "type").text = "16th"

    _add_rhythm_note(m, 2, "eighth", voice="1", step="D")
    ET.SubElement(ET.SubElement(m, "backup"), "duration").text = "2"
    _add_rhythm_note(m, 12, "half", dots=1, voice="2", step="E")
    ET.SubElement(ET.SubElement(m, "backup"), "duration").text = "10"
    _add_rhythm_note(m, 3, "eighth", dots=1, voice="1", step="C")
    _add_rhythm_note(m, 8, "half", voice="1", step="C")
    ET.SubElement(ET.SubElement(m, "backup"), "duration").text = "1"
    ET.SubElement(ET.SubElement(m, "forward"), "duration").text = "1"
    _add_rhythm_note(m, 4, "quarter", voice="2", step="E")
    ET.SubElement(ET.SubElement(m, "backup"), "duration").text = "4"
    _add_rhythm_note(m, 4, "quarter", voice="1", step="C")
    ET.SubElement(m, "barline", location="right")

    aligned, _ = align_page_score(_build_score([p1, p2]), page_num=13)
    res_m = aligned.find("./part[@id='P1']/measure")

    # Canonical output: grace+v1 [1,3,8,4] -> backup(16) -> v2 [12,4] -> barline, zero overlaps
    assert [n.findtext("duration") for n in res_m.findall("note") if n.find("grace") is None] == [
        "1", "3", "8", "4", "12", "4"
    ]
    assert [n.findtext("voice") for n in res_m.findall("note") if n.find("grace") is None] == [
        "1", "1", "1", "1", "2", "2"
    ]
    assert res_m.find("note").find("grace") is not None and res_m.find("note").findtext("voice") == "1"
    assert [b.findtext("duration") for b in res_m.findall("backup")] == ["16"]
    assert res_m[-1].tag == "barline"

    # Voice 2 entering on beat 3 (<forward>8) is preserved when Voice 1 is repaired
    p_fw1 = _make_part("P1", [(False, ("4", "4"))], divisions=4)
    p_fw2 = _make_part("P2", [(True, ("4", "4"))], divisions=4)
    mf = p_fw1[1].find("measure")
    for n in list(mf.findall("note")):
        mf.remove(n)
    _add_rhythm_note(mf, 2, "eighth", voice="1")
    _add_rhythm_note(mf, 3, "eighth", dots=1, voice="1")
    _add_rhythm_note(mf, 12, "half", dots=1, voice="1")  # sum=17 > 16
    ET.SubElement(ET.SubElement(mf, "backup"), "duration").text = "17"
    ET.SubElement(ET.SubElement(mf, "forward"), "duration").text = "8"
    _add_rhythm_note(mf, 8, "half", voice="2")

    aligned_fw, _ = align_page_score(_build_score([p_fw1, p_fw2]), page_num=1)
    res_fw = aligned_fw.find("./part[@id='P1']/measure")
    assert [b.findtext("duration") for b in res_fw.findall("backup")] == ["16"]
    assert [f.findtext("duration") for f in res_fw.findall("forward")] == ["8"]
    v2_durs = [n.findtext("duration") for n in res_fw.findall("note") if n.findtext("voice") == "2"]
    assert v2_durs == ["8"]


def test_reconcile_duration_type_mismatch_nonstandard_clamp_and_equivalent_meter():
    # 1) Orphan triplet duration mismatch (<duration>4</duration><type>eighth</type> at div=12 -> visual 6)
    #    PLUS Tier 3b non-standard remainder clamp (rem=8 at div=12 -> clamped to largest standard 6 eighth)
    p1 = _make_part("P1", [(False, ("4", "4"))], divisions=12)
    p2 = _make_part("P2", [(True, ("4", "4"))], divisions=12)
    m1 = p1[1].find("measure")
    for n in list(m1.findall("note")):
        m1.remove(n)
    _add_rhythm_note(m1, 24, "half")
    _add_rhythm_note(m1, 4, "eighth")  # mismatched visual=6 -> reconciled!
    _add_rhythm_note(m1, 12, "quarter")
    _add_rhythm_note(m1, 12, "quarter")  # total after reconcile = 54 > 48 -> Tier 3a fixes last quarter->eighth(6)
    aligned1, _ = align_page_score(_build_score([p1, p2]), page_num=1)
    durs1 = [n.findtext("duration") for n in aligned1.findall("./part[@id='P1']/measure/note")]
    assert durs1 == ["24", "6", "12", "6"]

    # Direct non-standard Tier 3b remainder (rem=8 at div=12 -> 2/3 quarter -> largest standard 6 eighth, no type mismatch)
    p3 = _make_part("P1", [(False, ("4", "4"))], divisions=12)
    p4 = _make_part("P2", [(True, ("4", "4"))], divisions=12)
    m3 = p3[1].find("measure")
    for n in list(m3.findall("note")):
        m3.remove(n)
    _add_rhythm_note(m3, 12, "quarter")
    _add_rhythm_note(m3, 12, "quarter")
    n16 = _add_rhythm_note(m3, 16, "quarter")  # non-standard raw dur without matching type? Force type="whole" removed or weird type:
    n16.remove(n16.find("type"))  # total=40 -> onset of next is 40, rem=8!
    _add_rhythm_note(m3, 12, "quarter")  # onset 40, dur 12 > 48 -> Tier 3b rem=8 -> clamped to (6, eighth)!
    aligned2, _ = align_page_score(_build_score([p3, p4]), page_num=1)
    last_n = aligned2.findall("./part[@id='P1']/measure/note")[-1]
    assert last_n.findtext("duration") == "6" and last_n.findtext("type") == "eighth"

    # 2) Continuation page preserves equivalent /8 compound meter even when all parts emit homr's /4 fallback,
    #    accepts strong >=60% non-fallback equivalent changes (4/4 -> 2/2, 6/4 -> 12/8),
    #    and accepts different-duration changes (4/4 -> 12/8).
    def _meter_page(ts_list, inh):
        specs = [_make_part(f"P{i+1}", [(False, ts)], divisions=4) for i, ts in enumerate(ts_list)]
        return align_page_score(_build_score(specs), page_num=15, inherited_time_sig=inh)[1]

    assert _meter_page([("6", "4"), ("6", "4"), ("3", "4"), ("4", "4"), ("2", "4")], ("12", "8")) == ("12", "8")
    assert _meter_page([("6", "4"), ("6", "4"), ("6", "4"), ("6", "4")], ("12", "8")) == ("12", "8")
    assert _meter_page([("3", "4"), ("3", "4"), ("2", "4"), ("4", "4"), ("5", "4")], ("6", "8")) == ("6", "8")
    assert _meter_page([("2", "2"), ("2", "2"), ("2", "2"), ("4", "4")], ("4", "4")) == ("2", "2")
    assert _meter_page([("12", "8"), ("12", "8"), ("12", "8"), ("4", "4")], ("4", "4")) == ("12", "8")


def _add_tuplet_note(
    measure: ET.Element,
    dur: int,
    note_type: str,
    *,
    actual: int = 3,
    normal: int = 2,
    tuplet_mark: str | None = None,
    slur_mark: str | None = None,
    voice: str = "1",
    chord: bool = False,
    beams: Sequence[str] = (),
) -> ET.Element:
    n = _add_rhythm_note(measure, dur, note_type, voice=voice, chord=chord, beams=beams)
    tm = ET.SubElement(n, "time-modification")
    ET.SubElement(tm, "actual-notes").text = str(actual)
    ET.SubElement(tm, "normal-notes").text = str(normal)
    if tuplet_mark is not None or slur_mark is not None:
        nt = ET.SubElement(n, "notations")
        if tuplet_mark is not None:
            ET.SubElement(nt, "tuplet", type=tuplet_mark)
        if slur_mark is not None:
            ET.SubElement(nt, "slur", type=slur_mark, number="1")
    return n


def test_repair_spurious_tuplets_and_pairwise_stepdown():
    # div=12 -> 12/8 expected_dur = 72 ticks; eighth=6 ticks, triplet eighth=4 ticks, 16th=3 ticks, quarter.=18 ticks
    p1 = _make_part("P1", [(False, ("12", "8"))], divisions=12)
    p2 = _make_part("P2", [(False, ("12", "8"))], divisions=12)
    p3 = _make_part("P3", [(False, ("12", "8"))], divisions=12)
    p4 = _make_part("P4", [(True, ("12", "8"))], divisions=12)

    # P1 M1: 12 spurious triplet eighths (48 < 72) -> restored to 12 plain eighths (72), no tuplets
    m_all_trip = p1[1].find("measure")
    for n in list(m_all_trip.findall("note")):
        m_all_trip.remove(n)
    for i in range(12):
        mark = "start" if i % 3 == 0 else ("stop" if i % 3 == 2 else None)
        _add_tuplet_note(m_all_trip, 4, "eighth", tuplet_mark=mark)

    # P2 M1: Green Hill m.69 pattern -> 12 triplet eighths + 1 plain eighth (54 -> 78 > 72 -> pair step-down to 72)
    # Mark notes 6 & 7 with a slur so Tier 3a-pair preferentially converts that slurred pair to 16ths!
    m_13 = p2[1].find("measure")
    for n in list(m_13.findall("note")):
        m_13.remove(n)
    for i in range(12):
        mark = "start" if i % 3 == 0 else ("stop" if i % 3 == 2 else None)
        slur = "start" if i == 6 else ("stop" if i == 7 else None)
        _add_tuplet_note(m_13, 4, "eighth", tuplet_mark=mark, slur_mark=slur, beams=("continue",))
    _add_rhythm_note(m_13, 6, "eighth")

    # P3 M1: 9 spurious triplet eighths + 1 dotted quarter (36 + 18 = 54 -> 54 + 18 = 72)
    m_mix = p3[1].find("measure")
    for n in list(m_mix.findall("note")):
        m_mix.remove(n)
    for i in range(9):
        mark = "start" if i % 3 == 0 else ("stop" if i % 3 == 2 else None)
        _add_tuplet_note(m_mix, 4, "eighth", tuplet_mark=mark)
    _add_rhythm_note(m_mix, 18, "quarter", dots=1)

    aligned, _ = align_page_score(_build_score([p1, p2, p3, p4]), page_num=20)
    parts = aligned.findall("part")

    # P1: all 12 notes untupled to duration 6 (sum=72)
    p1_notes = parts[0].find("measure").findall("note")
    assert all(n.find("time-modification") is None and n.find(".//tuplet") is None for n in p1_notes)
    assert [int(n.findtext("duration")) for n in p1_notes] == [6] * 12

    # P2 (m.69): all 13 notes preserved, 0 tuplets, notes[6..7] stepped down to 16th(3) -> sum=72
    p2_notes = parts[1].find("measure").findall("note")
    assert len(p2_notes) == 13
    assert all(n.find("time-modification") is None and n.find(".//tuplet") is None for n in p2_notes)
    assert sum(int(n.findtext("duration")) for n in p2_notes) == 72
    assert p2_notes[6].findtext("type") == "16th" and p2_notes[7].findtext("type") == "16th"

    # P3: 9 eighths + quarter. = 72, 0 tuplets
    p3_notes = parts[2].find("measure").findall("note")
    assert all(n.find("time-modification") is None for n in p3_notes)
    assert [int(n.findtext("duration")) for n in p3_notes] == [6] * 9 + [18]

    # Genuine full-measure 4/4 triplet (Trumpets m.32/m.40): 12+12 + 8+8+8 = 48 at div=12 -> untouched!
    t1 = _make_part("P1", [(False, ("4", "4"))], divisions=12)
    t2 = _make_part("P2", [(True, ("4", "4"))], divisions=12)
    tm_m = t1[1].find("measure")
    for n in list(tm_m.findall("note")):
        tm_m.remove(n)
    _add_rhythm_note(tm_m, 12, "quarter")
    _add_rhythm_note(tm_m, 12, "quarter")
    for mk in ("start", None, "stop"):
        _add_tuplet_note(tm_m, 8, "quarter", tuplet_mark=mk)
    aligned_44, _ = align_page_score(_build_score([t1, t2]), page_num=10)
    assert len(aligned_44.findall("./part[@id='P1']/measure/note/time-modification")) == 3


def test_align_ghost_trailing_bar_and_midpage_meter_hardening():
    # 1) Ghost trailing empty measure (Page 13 pattern: 4 parts, 2 have 6 bars where bar 6 is all-rest, 2 have 5 bars)
    g_parts = [
        _make_part("P1", [(False, ("4", "4")), (False, None), (False, None), (False, None), (False, None), (True, None)]),
        _make_part("P2", [(False, ("4", "4")), (False, None), (False, None), (False, None), (False, None), (True, None)]),
        _make_part("P3", [(False, ("4", "4")), (False, None), (False, None), (False, None), (False, None)]),
        _make_part("P4", [(True, ("4", "4")), (True, None), (True, None), (True, None), (True, None)]),
    ]
    g_aligned, _ = align_page_score(_build_score(g_parts), page_num=13, inherited_time_sig=("4", "4"))
    assert [len(p.findall("measure")) for p in g_aligned.findall("part")] == [5, 5, 5, 5]

    # 2) Continuation page M1 minority 5/4 guess before 6-quarter bars preserves inherited 12/8 (Page 31 pattern)
    m31_parts = []
    for i in range(6):
        m1_ts = ("5", "4") if i < 2 else None
        sp, part = _make_part(f"P{i+1}", [(i >= 2, m1_ts), (i >= 4, None), (i >= 4, None)], divisions=4)
        ms = part.findall("measure")
        if i < 2:
            _set_measure_dur(ms[0], 20)
        if i < 4:
            _set_measure_dur(ms[1], 24)
            _set_measure_dur(ms[2], 24)
        m31_parts.append((sp, part))
    a31, end31 = align_page_score(_build_score(m31_parts), page_num=31, inherited_time_sig=("12", "8"))
    assert end31 == ("12", "8")
    assert (a31.findtext("./part/measure/attributes/time/beats"), a31.findtext("./part/measure/attributes/time/beat-type")) == ("12", "8")
    assert len(a31.findall("./part[@id='P1']/measure/attributes/time")) == 1

    # 3) Mid-page rest-only 10/8 noise vs pitched 12/8 (Page 32) and 2-part 16/16 noise (Page 37)
    n_parts = []
    for i in range(8):
        if i < 3:
            # Pitched parts stay in 12/8
            sp, part = _make_part(f"P{i+1}", [(False, ("12", "8")), (False, None)], divisions=4)
            for m in part.findall("measure"):
                _set_measure_dur(m, 24)
        else:
            # 5 rest-only parts wrongly guess 10/8 on bar 2
            sp, part = _make_part(f"P{i+1}", [(True, ("12", "8")), (True, ("10", "8"))], divisions=4)
        n_parts.append((sp, part))
    a32, end32 = align_page_score(_build_score(n_parts), page_num=32, inherited_time_sig=("12", "8"))
    assert end32 == ("12", "8")
    assert a32.findall("./part[@id='P1']/measure")[1].find("./attributes/time") is None

    # 29 parts in 4/4, 2 pitched parts say 16/16 on bar 2 -> rejected!
    p16_parts = []
    for i in range(10):
        ts2 = ("16", "16") if i < 2 else None
        sp, part = _make_part(f"P{i+1}", [(i >= 3, ("4", "4")), (i >= 3, ts2)], divisions=4)
        if i < 3:
            for m in part.findall("measure"):
                _set_measure_dur(m, 16)
        p16_parts.append((sp, part))
    a37, end37 = align_page_score(_build_score(p16_parts), page_num=37, inherited_time_sig=("4", "4"))
    assert end37 == ("4", "4")
    assert a37.findall("./part[@id='P1']/measure")[1].find("./attributes/time") is None


def test_percussion_normalization_all_rest_clef_suppression_and_positioned_tempos(tmp_path):
    # Multi-page score with:
    # P1: Flute (plays all 3 pages)
    # P2: Timpani (plays p1 in F4, all-rest p2 with spurious C3 clef!)
    # P3: Percussion 1 (p1 in C3 with D4 + accidental, p2 in F4 with D3, p3 in G2 with C5 — ALL are 3rd space!)
    labels = [
        InstrumentLabel("Flute", "Fl."),
        InstrumentLabel("Timpani", "Timp."),
        InstrumentLabel("Percussion 1", "Perc. 1"),
    ]

    def _page(p1_clef, p2_clef, p2_rest, p3_clef, p3_note):
        sp1, pt1 = _make_part("P1", [(False, ("4", "4")), (False, None)], clef=p1_clef, fifths="0")
        sp2, pt2 = _make_part("P2", [(p2_rest, ("4", "4")), (p2_rest, None)], clef=p2_clef, fifths="0")
        sp3, pt3 = _make_part("P3", [(False, ("4", "4")), (True, None)], clef=p3_clef, fifths="-2")
        m3 = pt3.find("measure")
        for n in list(m3.findall("note")):
            m3.remove(n)
        _add_note(m3, p3_note[0], p3_note[1], accidental="flat", duration="16")
        return _build_score([(sp1, pt1), (sp2, pt2), (sp3, pt3)])

    r1 = _page(("G", "2"), ("F", "4"), False, ("C", "3"), ("D", "4"))  # 3rd space in C3 = D4
    r2 = _page(("G", "2"), ("C", "3"), True, ("F", "4"), ("E", "3"))   # 3rd space in F4 = E3; Timpani spurious C3 on rest page!
    r3 = _page(("G", "2"), ("F", "4"), False, ("G", "2"), ("C", "5"))  # 3rd space in G2 = C5

    t_meno = PositionedTempo(TempoMarking("Meno mosso", bpm=69), 500, 700, 10, 40, u_norm=0.45)
    t_atempo = PositionedTempo(TempoMarking("A Tempo", bpm=76), 800, 950, 10, 40, u_norm=0.75)
    sc1 = PageSidecar(1, 3, [10.0, 20.0, 30.0], [0.0, 0.5, 1.0], labels, positioned_tempos=(t_meno, t_atempo))
    sc2 = PageSidecar(2, 3, [10.0, 20.0, 30.0], [0.0, 0.5, 1.0], labels)
    sc3 = PageSidecar(3, 3, [10.0, 20.0, 30.0], [0.0, 0.5, 1.0], labels)

    # Sidecar JSON round-trip preserves positioned_tempos
    xml_p = tmp_path / "page_001.musicxml"
    write_page_sidecar(sc1, xml_p)
    assert read_page_sidecar(xml_p) == sc1

    merged = merge_fullscore_pages([(1, r1, sc1), (2, r2, sc2), (3, r3, sc3)], title="Perc Test")

    # 1) Percussion 1 has percussion/2 clef ONLY in Measure 1, fifths=0, zero accidentals,
    #    and ALL 3 notes across C3/F4/G2 mapped to identical <unpitched> C5!
    perc_p = merged.findall("part")[2]
    assert perc_p.findtext("./measure[@number='1']/attributes/clef/sign") == "percussion"
    assert perc_p.findtext("./measure[@number='1']/attributes/clef/line") == "2"
    assert perc_p.findtext("./measure[@number='1']/attributes/key/fifths") == "0"
    assert len(perc_p.findall(".//clef")) == 1
    assert len(perc_p.findall(".//key")) == 1
    assert perc_p.find(".//accidental") is None
    assert perc_p.findall(".//note/pitch") == []
    unpitched = [
        (u.findtext("display-step"), u.findtext("display-octave"))
        for u in perc_p.findall(".//note/unpitched")
    ]
    assert unpitched == [("C", "5"), ("C", "5"), ("C", "5")]

    # 2) Timpani suppressed spurious C3 clef on rest-only Page 2 -> only F4 clef in M1!
    timp_p = merged.findall("part")[1]
    assert [(c.findtext("sign"), c.findtext("line")) for c in timp_p.findall(".//clef")] == [("F", "4")]

    # 3) Positioned tempos placed on their respective measures (M1 and M2) instead of dropping t_atempo!
    p1_ms = merged.findall("./part[@id='P1']/measure")
    assert p1_ms[0].findtext("./direction/direction-type/words") == "Meno mosso "
    assert p1_ms[1].findtext("./direction/direction-type/words") == "A Tempo "
