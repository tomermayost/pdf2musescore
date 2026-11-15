import zipfile
import xml.etree.ElementTree as ET

import pytest

from pdf2musescore.musicxml import MergeError, merge_files, merge_scores, read_score, write_musicxml


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


def test_merge_basic():
    merged = merge_scores([score(2, mtitle="Prelude"), score(3, wtitle="W"), score(1)])
    measures = merged.findall("part/measure")
    assert [m.get("number") for m in measures] == [str(i) for i in range(1, 7)]
    words = [w.text for w in merged.iter("words")]
    assert words == ["Prelude", "W", "Movement 3"]
    assert len(merged.findall(".//print[@new-page='yes']")) == 2
    first = measures[0]
    assert [c.tag for c in first][:2] == ["attributes", "direction"]


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
