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
        ET.SubElement(note, "duration").text = "8"
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
    )
    sc_path = write_page_sidecar(sc, xml_path)
    assert sc_path == sidecar_path_for(xml_path)
    assert read_page_sidecar(xml_path) == sc

    sc_path.write_text("{corrupt", encoding="utf-8")
    assert read_page_sidecar(xml_path) is None


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

    out_xml = merge_fullscore_files([f1, f2], tmp_path / "full.musicxml", title="The Green Hill", options=ConversionOptions(keep_line_breaks=False))
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
    # Unpitched Percussion 1 & 2 (27, 28) -> channel 10
    assert chans[27] == 10 and chans[28] == 10

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
