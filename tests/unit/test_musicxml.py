import zipfile
import xml.etree.ElementTree as ET

import pytest

from pdf2musescore.musicxml import (
    MergeError,
    add_courtesy_accidentals,
    apply_clef_override,
    apply_instrument_override,
    apply_score_options,
    is_homr_default_voice,
    merge_files,
    merge_scores,
    normalize_default_instruments,
    read_score,
    reconcile_standard_time_signatures,
    sanitize_dynamics,
    sanitize_score,
    sanitize_slurs,
    set_score_part_instrument,
    strip_layout_breaks,
    sync_part_transpose,
    transpose_score,
    write_musicxml,
)
from pdf2musescore.options import ConversionOptions


def score(n_measures=2, parts=1, mtitle=None, wtitle=None):
    root = ET.Element("score-partwise", version="4.0")
    if wtitle:
        ET.SubElement(ET.SubElement(root, "work"), "work-title").text = wtitle
    if mtitle:
        ET.SubElement(root, "movement-title").text = mtitle
    pl = ET.SubElement(root, "part-list")
    for p in range(parts):
        ET.SubElement(pl, "score-part", id=f"P{p+1}")
        part = ET.SubElement(root, "part", id=f"P{p+1}")
        for i in range(n_measures):
            m = ET.SubElement(part, "measure", number=str(i + 1))
            if i == 0:
                ET.SubElement(m, "attributes")
            ET.SubElement(m, "note")
    return root


def write_mxl(root, path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("META-INF/container.xml",
                   '<container><rootfiles><rootfile full-path="s.xml"/></rootfiles></container>')
        z.writestr("s.xml", ET.tostring(root))
    return path


def test_read_mxl_and_xml(tmp_path):
    assert read_score(write_mxl(score(), tmp_path / "a.mxl")).tag == "score-partwise"
    p = write_musicxml(score(), tmp_path / "a.musicxml")
    assert len(read_score(p).findall(".//measure")) == 2


def test_sanitize_tuplets_and_tremolos():
    xml = """<score-partwise><part id="P1"><measure number="1">
      <note><duration>48</duration><notations><ornaments><tremolo type="start">3</tremolo></ornaments></notations></note>
      <note><duration>8</duration><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tuplet type="start"/></notations></note>
      <note><duration>8</duration><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
      <note><duration>8</duration><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations/></note>
      <note><duration>12</duration></note>
      <note><duration>8</duration><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
      <note><duration>4</duration><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
      <note><duration>12</duration></note>
      <note><duration>2</duration><type>16th</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
    </measure></part></score-partwise>"""
    root = sanitize_score(ET.fromstring(xml))
    assert root.find(".//tremolo") is None
    tuplets = [(t.get("type")) for t in root.findall(".//tuplet")]
    assert tuplets == ["start", "stop"]
    notes = root.findall(".//note")
    orphans = [notes[5], notes[6], notes[8]]
    assert all(n.find("time-modification") is None for n in orphans)
    assert [n.findtext("duration") for n in orphans] == ["12", "6", "3"]


def test_transpose_clef_instrument_and_score_options():
    xml = """<score-partwise>
      <part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>
      <part id="P1"><measure number="1">
        <attributes>
          <key><fifths>5</fifths></key>
          <clef number="1"><sign>G</sign><line>2</line></clef>
          <clef number="2"><sign>F</sign><line>4</line></clef>
        </attributes>
        <note><pitch><step>C</step><octave>4</octave></pitch></note>
        <note><pitch><step>C</step><alter>1</alter><octave>4</octave></pitch></note>
        <note><pitch><step>B</step><octave>3</octave></pitch></note>
        <note><pitch><step>X</step><octave>4</octave></pitch></note>
        <note><pitch><step>C</step></pitch></note>
      </measure></part>
    </score-partwise>"""
    root = ET.fromstring(xml)
    assert transpose_score(root, 0) is root

    # Transpose +2 semitones: C4 -> D4, C#4 -> Eb4 (alter -1), B3 -> C#4 (alter 1), fifths 5+2=7 -> -5
    transpose_score(root, 2)
    pitches = root.findall(".//note/pitch")
    assert (pitches[0].find("step").text, pitches[0].find("octave").text, pitches[0].find("alter")) == ("D", "4", None)
    assert (pitches[1].find("step").text, pitches[1].find("alter").text) == ("E", "-1")
    assert (pitches[2].find("step").text, pitches[2].find("alter").text, pitches[2].find("octave").text) == ("C", "1", "4")
    assert root.find(".//attributes/key/fifths").text == "-5"

    # Transpose -4 semitones so fifths wraps around <-6
    root.find(".//attributes/key/fifths").text = "-5"
    transpose_score(root, 3)  # shift_fifths = -3 -> -8 -> +4
    assert root.find(".//attributes/key/fifths").text == "4"

    # Clef override on existing clef + no-op on auto
    assert apply_clef_override(root, "auto") is root
    apply_clef_override(root, "tenor")
    c1 = root.find(".//attributes/clef[@number='1']")
    assert (c1.find("sign").text, c1.find("line").text) == ("C", "4")

    # Clef override when no <clef> exists yet and when <clef> has empty children
    s_no_clef = score(1)
    apply_clef_override(s_no_clef, "bass")
    assert s_no_clef.find(".//attributes/clef/sign").text == "F"

    s_empty_clef = score(1)
    ET.SubElement(s_empty_clef.find(".//attributes"), "clef")
    apply_clef_override(s_empty_clef, "alto")
    assert s_empty_clef.find(".//attributes/clef/sign").text == "C"

    # Instrument override updates part-name, abbreviation, sound, midi-channel, and midi-program
    assert apply_instrument_override(root, "auto") is root
    apply_instrument_override(root, "Trombone")
    assert root.findtext("./part-list/score-part/part-name") == "Trombone"
    assert root.findtext("./part-list/score-part/part-abbreviation") == "Tbn."
    assert root.findtext("./part-list/score-part/score-instrument/instrument-name") == "Trombone"
    assert root.findtext("./part-list/score-part/score-instrument/instrument-sound") == "brass.trombone"
    assert root.findtext("./part-list/score-part/midi-instrument/midi-program") == "58"
    assert root.findtext("./part-list/score-part/midi-instrument/midi-channel") == "1"

    # apply_score_options end-to-end
    s_opts = score(1)
    assert apply_score_options(s_opts, None) is s_opts
    opts = ConversionOptions(
        transpose_semitones=2,
        clef_override="bass",
        instrument_name="Trombone",
    )
    apply_score_options(s_opts, opts)
    assert s_opts.find("./part-list/score-part/part-name").text == "Trombone"
    assert s_opts.find("./part-list/score-part/score-instrument/instrument-sound").text == "brass.trombone"
    assert s_opts.find("./part-list/score-part/midi-instrument/midi-program").text == "58"
    assert s_opts.find(".//attributes/clef/sign").text == "F"


def test_normalize_default_voice_and_percussion_midi():
    homr_xml = """<score-partwise>
      <part-list>
        <score-part id="P1">
          <part-name>Voice</part-name>
          <score-instrument id="P1-I1">
            <instrument-name>Voice</instrument-name>
            <instrument-sound>voice</instrument-sound>
          </score-instrument>
          <midi-instrument id="P1-I1">
            <midi-channel>1</midi-channel>
            <midi-program>54</midi-program>
          </midi-instrument>
        </score-part>
        <score-part id="P2">
          <part-name>Voice</part-name>
          <score-instrument id="P2-I1">
            <instrument-name>Voice</instrument-name>
            <instrument-sound>voice.vocals</instrument-sound>
          </score-instrument>
          <midi-instrument id="P2-I1">
            <midi-channel>2</midi-channel>
            <midi-program>53</midi-program>
          </midi-instrument>
        </score-part>
      </part-list>
      <part id="P1"><measure number="1"><attributes/><note/></measure></part>
      <part id="P2"><measure number="1"><attributes/><note/></measure></part>
    </score-partwise>"""
    root = ET.fromstring(homr_xml)
    sps = root.findall("./part-list/score-part")
    assert is_homr_default_voice(sps[0]) is True
    assert is_homr_default_voice(sps[1]) is False

    # Normalizing replaces dummy Voice (P1) with Piano, while preserving real Voice (P2)
    assert normalize_default_instruments(root) == 1
    assert sps[0].findtext("part-name") == "Piano"
    assert sps[0].findtext("part-abbreviation") == "Pno."
    assert sps[0].findtext("./score-instrument/instrument-sound") == "keyboard.piano"
    assert sps[0].findtext("./midi-instrument/midi-program") == "1"
    assert sps[1].findtext("part-name") == "Voice"
    assert sps[1].findtext("./score-instrument/instrument-sound") == "voice.vocals"
    assert sps[1].findtext("./midi-instrument/midi-program") == "53"

    # Unpitched percussion removes stale midi-program and assigns channel 10
    set_score_part_instrument(sps[1], "Snare Drum")
    assert sps[1].findtext("part-name") == "Snare Drum"
    assert sps[1].findtext("./score-instrument/instrument-sound") == "drum.snare-drum"
    assert sps[1].findtext("./midi-instrument/midi-channel") == "10"
    assert sps[1].find("./midi-instrument/midi-program") is None

    # merge_scores normalizes dummy Voice automatically
    merged = merge_scores([ET.fromstring(homr_xml)])
    assert merged.findtext("./part-list/score-part[@id='P1']/part-name") == "Piano"


def test_read_errors(tmp_path):
    bad = tmp_path / "b.mxl"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("META-INF/container.xml", "<container/>")
    with pytest.raises(MergeError, match="rootfile"):
        read_score(bad)
    tw = tmp_path / "t.xml"
    tw.write_text("<score-timewise/>")
    with pytest.raises(MergeError, match="unsupported"):
        read_score(tw)


def test_sanitize_slurs():
    xml = """<score-partwise><part id="P1">
      <measure number="1">
        <note><staff>1</staff><voice>1</voice><notations><slur type="stop" number="1"/></notations></note>
        <note><staff>1</staff><voice>1</voice><notations><slur type="start" number="2"/></notations></note>
        <note><staff>1</staff><voice>1</voice><notations><slur type="start" number="3"/></notations></note>
        <note><staff>2</staff><voice>5</voice><notations><slur type="start" number="2"/></notations></note>
        <note><staff>1</staff><voice>1</voice><notations><slur type="start" number="1"/><slur type="stop" number="1"/></notations></note>
        <note><staff>2</staff><voice>5</voice><notations><slur type="stop" number="2"/></notations></note>
        <note><staff>1</staff><voice>1</voice><notations><slur type="stop" number="1"/></notations></note>
        <note><staff>1</staff><voice>1</voice><notations><slur type="other" number="1"/></notations></note>
      </measure>
      <measure number="2">
        <note><staff>1</staff><voice>1</voice><notations><slur type="start" number="1"/></notations></note>
      </measure>
    </part></score-partwise>"""
    root = sanitize_slurs(ET.fromstring(xml))
    notes = root.findall(".//note")
    # Note 0: orphan leading stop removed
    assert notes[0].findall("notations/slur") == []
    # Note 1: unclosed start dropped when Note 2 opens another start
    assert notes[1].findall("notations/slur") == []
    # Note 2: paired start kept with number="1"
    assert [(s.get("type"), s.get("number")) for s in notes[2].findall("notations/slur")] == [("start", "1")]
    # Note 3 (staff 2, voice 5): independent start kept
    assert [(s.get("type"), s.get("number")) for s in notes[3].findall("notations/slur")] == [("start", "1")]
    # Note 4: same-note slurStart_slurStop ordered stop then start cleanly
    assert [(s.get("type"), s.get("number")) for s in notes[4].findall("notations/slur")] == [("stop", "1"), ("start", "1")]
    # Note 5 (staff 2, voice 5): matched stop kept
    assert [(s.get("type"), s.get("number")) for s in notes[5].findall("notations/slur")] == [("stop", "1")]
    # Note 6 (staff 1, voice 1): matched stop kept
    assert [(s.get("type"), s.get("number")) for s in notes[6].findall("notations/slur")] == [("stop", "1")]
    # Note 8 (measure 2): trailing unclosed start at end of part removed
    assert notes[8].findall("notations/slur") == []


def test_strip_layout_breaks():
    xml = """<score-partwise><part id="P1">
      <measure number="1"><print new-system="yes"/></measure>
      <measure number="2"><print new-page="yes" page-number="2"/></measure>
      <measure number="3"><print new-system="yes"><page-layout/></print></measure>
    </part></score-partwise>"""
    root = strip_layout_breaks(ET.fromstring(xml))
    measures = root.findall(".//measure")
    assert measures[0].find("print") is None
    assert measures[1].find("print") is not None
    assert measures[1].find("print").attrib == {"page-number": "2"}
    assert measures[2].find("print") is not None
    assert measures[2].find("print").attrib == {}


def test_merge_basic():
    s1 = score(2, mtitle="Prelude")
    ET.SubElement(s1.findall(".//measure")[1], "print", {"new-system": "yes"})
    merged = merge_scores([s1, score(3, wtitle="W"), score(1)])
    measures = merged.findall("part/measure")
    assert [m.get("number") for m in measures] == [str(i) for i in range(1, 7)]
    words = [w.text for w in merged.iter("words")]
    assert words == ["Prelude", "W", "Page 3"]
    assert len(merged.findall(".//print[@new-page='yes']")) == 0
    assert len(merged.findall(".//print[@new-system='yes']")) == 0
    first = measures[0]
    assert [c.tag for c in first][:2] == ["attributes", "direction"]

    kept = merge_scores(
        [s1, score(3, wtitle="W"), score(1)],
        options=ConversionOptions(keep_line_breaks=True),
    )
    assert len(kept.findall(".//print[@new-page='yes']")) == 2
    assert len(kept.findall(".//print[@new-system='yes']")) == 1


def test_merge_title_replaces_movement_title():
    merged = merge_scores([score(mtitle="x"), score()], title="My Title")
    assert merged.find("movement-title") is None
    assert merged.find("work/work-title").text == "My Title"
    merged2 = merge_scores([score(wtitle="old")], title="New")
    assert merged2.find("work/work-title").text == "New"
    w = score()
    ET.SubElement(w, "work")
    assert merge_scores([w], title="T").find("work/work-title").text == "T"


def test_merge_part_mismatch_warns(caplog):
    with caplog.at_level("WARNING"):
        merged = merge_scores([score(parts=2), score(parts=1), score(parts=3)])
    assert "expected 2" in caplog.text
    assert len(merged.findall("part")) == 2


def test_merge_errors():
    with pytest.raises(MergeError):
        merge_scores([])
    with pytest.raises(MergeError):
        merge_scores([ET.Element("score-partwise")])


def test_merge_movement_without_parts():
    merged = merge_scores([score(), ET.Element("score-partwise")])
    assert len(merged.findall("part/measure")) == 2


def test_merge_files_sorted(tmp_path):
    write_mxl(score(1, mtitle="B"), tmp_path / "x.mvt2.mxl")
    write_mxl(score(1, mtitle="A"), tmp_path / "x.mvt1.mxl")
    out = merge_files(list(tmp_path.glob("*.mxl")), tmp_path / "o" / "x.musicxml")
    assert [w.text for w in read_score(out).iter("words")] == ["A", "B"]


def test_single_page_merge_omits_title_words():
    s = score(2, mtitle="WEST SIDE STORY")
    merged = merge_scores([s], title="west side story")
    assert list(merged.iter("words")) == []

    # Multi-page score with tempo on M1 and running instrument headers suppresses overlapping titles
    p1 = score(1, mtitle="WEST SIDE STORY")
    p1.find("./part-list/score-part").append(ET.fromstring("<part-name>BARITONE</part-name>"))
    d = ET.SubElement(p1.find("./part/measure"), "direction")
    ET.SubElement(ET.SubElement(d, "direction-type"), "words").text = "Andante maestoso"
    ET.SubElement(d, "sound", tempo="72")
    p2 = score(1, mtitle="BARITONE")
    merged_multi = merge_scores([p1, p2], title="West Side Story", options=ConversionOptions(keep_line_breaks=True))
    assert [w.text for w in merged_multi.iter("words")] == ["Andante maestoso"]
    assert len(merged_multi.findall(".//print[@new-page='yes']")) == 1


def test_reconcile_standard_time_signatures():
    # Empty / single measure -> no-op
    assert reconcile_standard_time_signatures(score(1)) == 0

    def _m(num: int, q_dur: float, *, ts: tuple[str, str] | None = None, rest_meas: bool = False, key: str | None = None, double_bar: bool = False) -> ET.Element:
        m = ET.Element("measure", number=str(num))
        if ts or key or num == 1:
            a = ET.SubElement(m, "attributes")
            if num == 1:
                ET.SubElement(a, "divisions").text = "4"
            if key is not None:
                ET.SubElement(ET.SubElement(a, "key"), "fifths").text = key
            if ts is not None:
                t = ET.SubElement(a, "time")
                ET.SubElement(t, "beats").text = ts[0]
                ET.SubElement(t, "beat-type").text = ts[1]
        n = ET.SubElement(m, "note")
        if rest_meas:
            ET.SubElement(n, "rest", measure="yes")
            ET.SubElement(n, "duration").text = str(int(round(q_dur * 4)))
            ET.SubElement(n, "type").text = "whole"
        else:
            p = ET.SubElement(n, "pitch")
            ET.SubElement(p, "step").text = "C"
            ET.SubElement(p, "octave").text = "4"
            ET.SubElement(n, "duration").text = str(int(round(q_dur * 4)))
        if double_bar:
            ET.SubElement(ET.SubElement(m, "barline", location="right"), "bar-style").text = "light-light"
        return m

    # West Side Story pattern: M1 wrongly labeled 3/4, M1-5 are 4/4 (double bar at M5),
    # M6 is key-change rest bar (3/4), M7-14 are 3/4 (double bar at M14), M15-18 are 4/4.
    root = ET.Element("score-partwise")
    ET.SubElement(ET.SubElement(root, "part-list"), "score-part", id="P1")
    part = ET.SubElement(root, "part", id="P1")
    for i in range(1, 6):
        part.append(_m(i, 4.0, ts=("3", "4") if i == 1 else None, double_bar=(i == 5)))
    part.append(_m(6, 4.0, rest_meas=True, key="-2"))  # stale duration=16 -> should sync to 12 for 3/4
    for i in range(7, 15):
        part.append(_m(i, 3.0, double_bar=(i == 14)))
    for i in range(15, 19):
        part.append(_m(i, 4.0))

    assert reconcile_standard_time_signatures(root) == 1
    ms = part.findall("measure")
    times = [(i + 1, m.findtext("./attributes/time/beats"), m.findtext("./attributes/time/beat-type")) for i, m in enumerate(ms) if m.find("./attributes/time") is not None]
    assert times == [(1, "4", "4"), (6, "3", "4"), (15, "4", "4")]
    assert ms[5].findtext("./note/duration") == "12"
    assert ms[5].find("./note/type") is None

    # Idempotent when re-run
    assert reconcile_standard_time_signatures(root) == 0

    # Preserves 6/8 compound meter and 1-bar pickup without breaking them
    root_68 = ET.Element("score-partwise")
    ET.SubElement(ET.SubElement(root_68, "part-list"), "score-part", id="P1")
    p68 = ET.SubElement(root_68, "part", id="P1")
    p68.append(_m(1, 0.5, ts=("6", "8")))  # 1-eighth pickup
    for i in range(2, 6):
        p68.append(_m(i, 3.0))  # 6/8 = 3 quarters
    assert reconcile_standard_time_signatures(root_68) == 0
    assert (p68.findtext("./measure[@number='1']/attributes/time/beats"), p68.findtext("./measure[@number='1']/attributes/time/beat-type")) == ("6", "8")

    # Standard mode repairs spurious 3:2 tuplets in underfull compound-meter bars (e.g. 6/8 with 6 triplet eighths -> 6 plain eighths)
    from pdf2musescore.musicxml import repair_spurious_tuplets

    root_tup = ET.Element("score-partwise")
    ET.SubElement(ET.SubElement(root_tup, "part-list"), "score-part", id="P1")
    pt = ET.SubElement(root_tup, "part", id="P1")
    for m_no in (1, 2, 3):
        m = ET.SubElement(pt, "measure", number=str(m_no))
        if m_no == 1:
            a = ET.SubElement(m, "attributes")
            ET.SubElement(a, "divisions").text = "12"
            t = ET.SubElement(a, "time")
            ET.SubElement(t, "beats").text = "6"
            ET.SubElement(t, "beat-type").text = "8"
        if m_no in (1, 2):
            for _ in range(6):
                n = ET.SubElement(m, "note")
                ET.SubElement(ET.SubElement(n, "pitch"), "step").text = "C"
                ET.SubElement(n, "duration").text = "6"
                ET.SubElement(n, "voice").text = "1"
                ET.SubElement(n, "type").text = "eighth"
        else:
            # M3 has 6 spurious triplet eighths (dur=4 each -> 24 < 36)
            for idx in range(6):
                n = ET.SubElement(m, "note")
                ET.SubElement(ET.SubElement(n, "pitch"), "step").text = "C"
                ET.SubElement(n, "duration").text = "4"
                ET.SubElement(n, "voice").text = "1"
                ET.SubElement(n, "type").text = "eighth"
                tm = ET.SubElement(n, "time-modification")
                ET.SubElement(tm, "actual-notes").text = "3"
                ET.SubElement(tm, "normal-notes").text = "2"
                if idx in (0, 2, 3, 5):
                    ET.SubElement(ET.SubElement(n, "notations"), "tuplet", type="start" if idx in (0, 3) else "stop")
    reconcile_standard_time_signatures(root_tup)
    m3_notes = pt.find("./measure[@number='3']").findall("note")
    assert [n.findtext("duration") for n in m3_notes] == ["6"] * 6
    assert all(n.find("time-modification") is None and n.find(".//tuplet") is None for n in m3_notes)
    # Multi-voice <backup> alignment when spurious tuplet is repaired
    mv_m = ET.fromstring(
        "<measure number='1'>"
        "<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type>"
        "<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>"
        "<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type>"
        "<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>"
        "<note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type>"
        "<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>"
        "<note><pitch><step>F</step><octave>4</octave></pitch><duration>18</duration><voice>1</voice><type>quarter</type><dot/></note>"
        "<backup><duration>30</duration></backup>"
        "<note><pitch><step>C</step><octave>3</octave></pitch><duration>36</duration><voice>2</voice><type>half</type><dot/></note>"
        "</measure>"
    )
    assert repair_spurious_tuplets(mv_m, 36, 12, ("6", "8")) is True
    assert mv_m.findtext("./backup/duration") == "36"


def test_add_courtesy_accidentals_and_transposed():
    xml = """<score-partwise><part-list><score-part id="P1"/></part-list><part id="P1">
      <measure number="1">
        <attributes><key><fifths>2</fifths></key></attributes>
        <note><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration><stem>up</stem></note>
        <note><pitch><step>C</step><alter>0</alter><octave>5</octave></pitch><duration>4</duration><stem>down</stem></note>
        <note><rest/><duration>4</duration></note>
      </measure>
      <measure number="2">
        <note><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch><duration>8</duration><stem>up</stem></note>
        <note><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch><duration>4</duration><stem>up</stem></note>
        <note><pitch><step>C</step><alter>1</alter><octave>5</octave></pitch><duration>4</duration><tie type="stop"/><stem>down</stem></note>
      </measure>
      <measure number="3">
        <attributes><key><fifths>-1</fifths></key></attributes>
        <note><pitch><step>B</step><alter>0</alter><octave>4</octave></pitch><duration>4</duration><stem>down</stem></note>
      </measure>
      <measure number="4">
        <note><pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch><duration>4</duration><stem>down</stem></note>
      </measure>
    </part></score-partwise>"""
    root = ET.fromstring(xml)
    # M1 alters F4 (0 vs +1) and C5 (0 vs +1).
    # M2: first F#4 gets cautionary sharp before <stem>; second F#4 does not; tied C#5 stop does not!
    # M3 has key change to -1 (clears prev_altered), alters B4 (natural vs -1).
    # M4: Bb4 gets cautionary flat!
    assert add_courtesy_accidentals(root) == 2

    m2_notes = root.findall("./part/measure[@number='2']/note")
    acc0 = m2_notes[0].find("accidental")
    assert acc0 is not None and acc0.text == "sharp" and acc0.get("cautionary") == "yes"
    assert [c.tag for c in m2_notes[0]].index("accidental") < [c.tag for c in m2_notes[0]].index("stem")
    assert m2_notes[1].find("accidental") is None
    assert m2_notes[2].find("accidental") is None

    m4_note = root.find("./part/measure[@number='4']/note")
    acc_b = m4_note.find("accidental")
    assert acc_b is not None and acc_b.text == "flat" and acc_b.get("cautionary") == "yes"

    # Idempotent
    assert add_courtesy_accidentals(root) == 0


def test_sanitize_dynamics_colliding_and_same_beat_duplicates():
    xml = """<score-partwise><part-list><score-part id="P1"/></part-list><part id="P1">
      <measure number="1">
        <direction placement="below">
          <direction-type><dynamics><mp/><other-dynamics>più</other-dynamics></dynamics></direction-type>
          <staff>1</staff>
        </direction>
        <direction placement="below">
          <direction-type><dynamics><other-dynamics/></dynamics></direction-type>
          <staff>1</staff>
        </direction>
        <direction placement="below">
          <direction-type><dynamics><other-dynamics>molto</other-dynamics><other-dynamics>p</other-dynamics></dynamics></direction-type>
          <staff>2</staff>
        </direction>
        <note><grace/><pitch><step>C</step><octave>4</octave></pitch></note>
        <direction placement="below">
          <direction-type><dynamics><f/></dynamics></direction-type>
          <staff>2</staff>
        </direction>
        <direction placement="above">
          <direction-type><words>Allegro</words></direction-type>
        </direction>
        <direction placement="below">
          <direction-type><dynamics><other-dynamics>dolce</other-dynamics></dynamics></direction-type>
          <staff>1</staff>
        </direction>
        <note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration></note>
        <direction placement="below">
          <direction-type><dynamics><ff/></dynamics></direction-type>
          <staff>1</staff>
        </direction>
      </measure>
    </part></score-partwise>"""
    root = sanitize_dynamics(ET.fromstring(xml))
    m1 = root.find("./part/measure[@number='1']")
    dyns = m1.findall(".//dynamics")
    # Remaining: staff 1 beat 1 <mp/>, staff 2 beat 1 <f/> (replaces <other-dynamics>molto</other-dynamics>), staff 1 beat 2 <ff/>
    assert len(dyns) == 3
    assert [ch.tag for ch in dyns[0]] == ["mp"]
    assert [ch.tag for ch in dyns[1]] == ["f"]
    assert [ch.tag for ch in dyns[2]] == ["ff"]
    assert root.find(".//other-dynamics") is None

    # Idempotent
    assert len(sanitize_dynamics(root).findall(".//dynamics")) == 3

    # Preserves single valid <other-dynamics> alone on a beat, and removes empty directions cleanly
    xml2 = """<score-partwise><part id="P1"><measure number="1">
      <direction><direction-type/><direction-type><dynamics><other-dynamics>subito</other-dynamics></dynamics></direction-type></direction>
      <direction><direction-type><dynamics><other-dynamics/></dynamics></direction-type></direction>
      <direction><direction-type/></direction>
    </measure></part></score-partwise>"""
    r2 = sanitize_dynamics(ET.fromstring(xml2))
    assert [el.findtext(".//other-dynamics") for el in r2.findall(".//measure/direction")] == ["subito"]


def test_sync_part_transpose_and_multi_page_merge_dedup():
    from pdf2musescore.fullscore.instruments import InstrumentPlaybackProfile

    assert sync_part_transpose(None, InstrumentPlaybackProfile("brass.euphonium", -1, -2, -1, 2, 58)) is False
    empty_part = ET.Element("part", id="P1")
    assert sync_part_transpose(empty_part, InstrumentPlaybackProfile("brass.euphonium", -1, -2, -1, 2, 58)) is False

    # Creates <attributes> after <print> when missing and inserts <transpose> before <directive>
    xml = """<score-partwise>
      <part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
      <part id="P1">
        <measure number="1">
          <attributes><divisions>4</divisions><clef><sign>G</sign><line>2</line></clef><directive>Solo</directive></attributes>
          <note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration></note>
        </measure>
      </part>
    </score-partwise>"""
    root = ET.fromstring(xml)
    apply_instrument_override(root, "BARITONE")
    m1_attrs = root.find("./part/measure[@number='1']/attributes")
    tags = [ch.tag for ch in m1_attrs]
    assert tags.index("transpose") < tags.index("directive")
    assert (
        m1_attrs.findtext("transpose/diatonic"),
        m1_attrs.findtext("transpose/chromatic"),
        m1_attrs.findtext("transpose/octave-change"),
    ) == ("-1", "-2", "-1")

    # Switching back to concert instrument removes <transpose>
    apply_instrument_override(root, "Piano")
    assert root.find(".//transpose") is None

    # Measure without <attributes> still gets <attributes><transpose> after <print>
    no_attr_part = ET.fromstring("<part id='P1'><measure number='1'><print/><note/></measure></part>")
    assert sync_part_transpose(no_attr_part, InstrumentPlaybackProfile("brass.euphonium", -1, -2, -1, 2, 58)) is True
    assert [c.tag for c in no_attr_part.find("measure")] == ["print", "attributes", "note"]

    # Multi-page merge strips duplicate <transpose> from page 2 measure 1
    p1 = copy_score = ET.fromstring(xml)
    p2 = ET.fromstring(xml)
    apply_instrument_override(p1, "BARITONE")
    apply_instrument_override(p2, "BARITONE")
    merged = merge_scores([p1, p2], title="Baritone Part")
    assert len(merged.findall(".//transpose")) == 1
    assert merged.find("./part/measure[@number='1']/attributes/transpose") is not None
    assert merged.find("./part/measure[@number='2']/attributes/transpose") is None

    # Percussion normalization cleans multiple <attributes> inside Measure 1 (e.g. homr split attributes/clef)
    perc_multi_attr = ET.fromstring(
        "<score-partwise><part-list><score-part id='P1'><part-name>Percussion 1</part-name></score-part></part-list>"
        "<part id='P1'><measure number='1'>"
        "<attributes><divisions>4</divisions><key><fifths>-2</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
        "<attributes><clef number='1'><sign>F</sign><line>4</line></clef></attributes>"
        "<note><pitch><step>E</step><octave>3</octave></pitch><duration>16</duration></note>"
        "</measure></part></score-partwise>"
    )
    apply_instrument_override(perc_multi_attr, "Percussion 1")
    m1_p = perc_multi_attr.find("./part/measure[@number='1']")
    assert [(c.findtext("sign"), c.findtext("line")) for c in m1_p.findall(".//clef")] == [("percussion", "2")]
    assert [k.findtext("fifths") for k in m1_p.findall(".//key")] == ["0"]
    assert m1_p.findtext("./note/unpitched/display-step") == "C"
    assert m1_p.findtext("./note/unpitched/display-octave") == "5"
