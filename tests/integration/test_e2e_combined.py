"""Integration: Bach PDF -> homr AI -> combined MusicXML -> .mscz (real tools)."""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from pdf2musescore.deps import MUSESCORE, find_tool
from pdf2musescore.engines.homr_engine import HomrEngine
from pdf2musescore.exporters.musescore import MuseScoreExporter
from pdf2musescore.musicxml import merge_files, read_score, set_score_part_instrument, write_musicxml
from pdf2musescore.tempo import TempoMarking, apply_tempo_marking

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.pdf"
OUT = ROOT / "tmp" / "integration" / "e2e_combined"

pytestmark = pytest.mark.integration


def test_pdf_to_combined_mscz():
    ms = find_tool(MUSESCORE)
    if not ms:
        pytest.skip("MuseScore required")
    pages = HomrEngine(dpi=200).recognize(FIXTURE, OUT / "omr")
    xml = merge_files(pages, OUT / "bach.musicxml", title="Bach - Cello Suite No. 2")
    root = read_score(xml)
    assert len(root.findall(".//measure")) > 50
    assert len(list(root.iter("words"))) >= len(pages)
    assert root.findall(".//rehearsal") == []
    assert len(root.findall(".//slur")) > 0, "Expected homr 0.7+ to preserve slurs"
    assert root.findall(".//print[@new-system]") == []
    assert root.findall(".//print[@new-page]") == []
    assert root.findtext("./part-list/score-part/part-name") == "Piano"
    assert root.findtext("./part-list/score-part/score-instrument/instrument-sound") == "keyboard.piano"
    assert root.findtext("./part-list/score-part/midi-instrument/midi-program") == "1"

    mscz = MuseScoreExporter(ms).export(xml, OUT / "bach.mscz")
    assert zipfile.is_zipfile(mscz)
    with zipfile.ZipFile(mscz) as z:
        mscx_files = [n for n in z.namelist() if n.endswith(".mscx")]
        assert mscx_files
        mscx_txt = z.read(mscx_files[0]).decode("utf-8", errors="replace")
        assert mscx_txt.count("<Slur") > 50, "Expected slurs to survive MuseScore 4 import"
        assert mscx_txt.count("<LayoutBreak") == 0, "Expected zero forced system/page breaks in .mscx"
        assert "<trackName>Voice</trackName>" not in mscx_txt
        assert "voice.vocals" not in mscx_txt
        assert "<trackName>Piano</trackName>" in mscx_txt


def test_musescore_instrument_mixer_names():
    ms = find_tool(MUSESCORE)
    if not ms:
        pytest.skip("MuseScore required")

    raw_xml = """<score-partwise version="4.0">
      <part-list>
        <score-part id="P1"><part-name>Voice</part-name><score-instrument id="P1-I1"><instrument-name>Voice</instrument-name><instrument-sound>voice</instrument-sound></score-instrument><midi-instrument id="P1-I1"><midi-channel>1</midi-channel><midi-program>54</midi-program></midi-instrument></score-part>
        <score-part id="P2"><part-name>Voice</part-name><score-instrument id="P2-I1"><instrument-name>Voice</instrument-name><instrument-sound>voice</instrument-sound></score-instrument><midi-instrument id="P2-I1"><midi-channel>2</midi-channel><midi-program>54</midi-program></midi-instrument></score-part>
        <score-part id="P3"><part-name>Voice</part-name><score-instrument id="P3-I1"><instrument-name>Voice</instrument-name><instrument-sound>voice</instrument-sound></score-instrument><midi-instrument id="P3-I1"><midi-channel>3</midi-channel><midi-program>54</midi-program></midi-instrument></score-part>
      </part-list>
      <part id="P1"><measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes><note><pitch><step>G</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice><type>whole</type></note></measure></part>
      <part id="P2"><measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>C</sign><line>3</line></clef></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice><type>whole</type></note></measure></part>
      <part id="P3"><measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>F</sign><line>4</line></clef></attributes><note><pitch><step>C</step><octave>3</octave></pitch><duration>16</duration><voice>1</voice><type>whole</type></note></measure></part>
    </score-partwise>"""
    root = ET.fromstring(raw_xml)
    for idx, (sp, inst) in enumerate(zip(root.findall("./part-list/score-part"), ("Violin I", "Viola", "Violoncello"))):
        set_score_part_instrument(sp, inst, pitched_channel_index=idx)

    xml_path = write_musicxml(root, OUT / "strings_mixer.musicxml")
    mscz = MuseScoreExporter(ms).export(xml_path, OUT / "strings_mixer.mscz")
    with zipfile.ZipFile(mscz) as z:
        mscx_name = next(n for n in z.namelist() if n.endswith(".mscx"))
        mscx_root = ET.fromstring(z.read(mscx_name))

    track_names = [p.findtext("trackName") for p in mscx_root.findall("./Score/Part")]
    inst_ids = [p.findtext("./Instrument/instrumentId") for p in mscx_root.findall("./Score/Part")]
    assert track_names == ["Violin I", "Viola", "Violoncello"]
    assert inst_ids == ["strings.violin", "strings.viola", "strings.cello"]
    assert all("voice" not in (t or "").lower() for t in track_names + inst_ids)


def test_musescore_tempo_import():
    ms = find_tool(MUSESCORE)
    if not ms:
        pytest.skip("MuseScore required")

    raw_xml = """<score-partwise version="4.0">
      <part-list>
        <score-part id="P1"><part-name>Piano</part-name><score-instrument id="P1-I1"><instrument-name>Piano</instrument-name><instrument-sound>keyboard.piano</instrument-sound></score-instrument><midi-instrument id="P1-I1"><midi-channel>1</midi-channel><midi-program>1</midi-program></midi-instrument></score-part>
      </part-list>
      <part id="P1"><measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
        <direction placement="below"><direction-type><dynamics><mp/><other-dynamics>più</other-dynamics></dynamics></direction-type><staff>1</staff></direction>
        <note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice><type>whole</type></note>
      </measure></part>
    </score-partwise>"""
    from pdf2musescore.musicxml import sanitize_dynamics

    root = sanitize_dynamics(ET.fromstring(raw_xml))
    assert apply_tempo_marking(root, TempoMarking(text="Adagio affetuoso", bpm=72)) is True
    assert apply_tempo_marking(root, TempoMarking(text="tranq.", is_expression=True)) is True

    xml_path = write_musicxml(root, OUT / "tempo_import.musicxml")
    mscz = MuseScoreExporter(ms).export(xml_path, OUT / "tempo_import.mscz")
    with zipfile.ZipFile(mscz) as z:
        mscx_name = next(n for n in z.namelist() if n.endswith(".mscx"))
        mscx_root = ET.fromstring(z.read(mscx_name))

    tempos = mscx_root.findall(".//Tempo")
    assert tempos, "Expected MuseScore 4 to import <Tempo> element"
    assert abs(float(tempos[0].findtext("tempo") or "0") - 1.2) < 0.05
    tempo_xml = ET.tostring(tempos[0], encoding="unicode")
    assert "Adagio affetuoso " in tempo_xml
    assert "tranq." in "".join("".join(st.itertext()) for st in mscx_root.findall(".//StaffText/text"))
    dyn_subtypes = [d.findtext("subtype") for d in mscx_root.findall(".//Dynamic")]
    assert dyn_subtypes == ["mp"]


def test_musescore_courtesy_accidental_import():
    ms = find_tool(MUSESCORE)
    if not ms:
        pytest.skip("MuseScore required")

    from pdf2musescore.musicxml import add_courtesy_accidentals

    raw_xml = """<score-partwise version="4.0">
      <part-list><score-part id="P1"><part-name>Baritone</part-name></score-part></part-list>
      <part id="P1">
        <measure number="1">
          <attributes><divisions>4</divisions><key><fifths>2</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
          <note><pitch><step>F</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice><type>whole</type></note>
        </measure>
        <measure number="2">
          <note><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch><duration>16</duration><voice>1</voice><type>whole</type></note>
        </measure>
      </part>
    </score-partwise>"""
    root = ET.fromstring(raw_xml)
    assert add_courtesy_accidentals(root) == 1
    xml_path = write_musicxml(root, OUT / "courtesy_acc.musicxml")
    mscz = MuseScoreExporter(ms).export(xml_path, OUT / "courtesy_acc.mscz")
    with zipfile.ZipFile(mscz) as z:
        mscx_name = next(n for n in z.namelist() if n.endswith(".mscx"))
        mscx_root = ET.fromstring(z.read(mscx_name))

    m2_note = mscx_root.findall("./Score/Staff/Measure")[1].find(".//Note")
    assert m2_note is not None
    assert m2_note.findtext("./Accidental/subtype") == "accidentalSharp"


def test_musescore_rehearsal_mark_import():
    ms = find_tool(MUSESCORE)
    if not ms:
        pytest.skip("MuseScore required")

    from pdf2musescore.rehearsal import RehearsalMark, apply_rehearsal_mark

    raw_xml = """<score-partwise version="4.0">
      <part-list><score-part id="P1"><part-name>Baritone</part-name></score-part></part-list>
      <part id="P1">
        <measure number="1">
          <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
          <note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice><type>whole</type></note>
        </measure>
        <measure number="2">
          <note><pitch><step>D</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice><type>whole</type></note>
        </measure>
      </part>
    </score-partwise>"""
    root = ET.fromstring(raw_xml)
    assert apply_rehearsal_mark(root, RehearsalMark("1", enclosure="square"), measure_index=0) is True
    assert apply_rehearsal_mark(root, RehearsalMark("A", enclosure="none"), measure_index=1) is True

    xml_path = write_musicxml(root, OUT / "rehearsal_import.musicxml")
    mscz = MuseScoreExporter(ms).export(xml_path, OUT / "rehearsal_import.mscz")
    with zipfile.ZipFile(mscz) as z:
        mscx_name = next(n for n in z.namelist() if n.endswith(".mscx"))
        mscx_root = ET.fromstring(z.read(mscx_name))

    reh_texts = ["".join(r.itertext()).strip() for r in mscx_root.findall(".//RehearsalMark/text")]
    assert reh_texts == ["1", "A"]
