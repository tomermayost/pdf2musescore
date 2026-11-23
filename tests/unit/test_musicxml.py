import zipfile
import xml.etree.ElementTree as ET

import pytest

from pdf2musescore.musicxml import (
    MergeError,
    apply_clef_override,
    apply_instrument_override,
    apply_score_options,
    merge_files,
    merge_scores,
    read_score,
    sanitize_score,
    sanitize_slurs,
    strip_layout_breaks,
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
      <note><duration>8</duration><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
    </measure></part></score-partwise>"""
    root = sanitize_score(ET.fromstring(xml))
    assert root.find(".//tremolo") is None
    tuplets = [(t.get("type")) for t in root.findall(".//tuplet")]
    assert tuplets == ["start", "stop"]
    notes = root.findall(".//note")
    assert notes[-1].find("time-modification") is None
    assert notes[-1].find("duration").text == "12"


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

    # Instrument override
    assert apply_instrument_override(root, "auto") is root
    apply_instrument_override(root, "Trombone")
    assert root.find("./part-list/score-part/part-name").text == "Trombone"
    assert root.find("./part-list/score-part/score-instrument/instrument-name").text == "Trombone"

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
    assert s_opts.find(".//attributes/clef/sign").text == "F"


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
