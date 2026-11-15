"""Integration: Bach PDF -> homr AI -> combined MusicXML -> .mscz (real tools)."""

import zipfile
from pathlib import Path

import pytest

from pdf2musescore.deps import MUSESCORE, find_tool
from pdf2musescore.engines.homr_engine import HomrEngine
from pdf2musescore.exporters.musescore import MuseScoreExporter
from pdf2musescore.musicxml import merge_files, read_score

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

    mscz = MuseScoreExporter(ms).export(xml, OUT / "bach.mscz")
    assert zipfile.is_zipfile(mscz)
    with zipfile.ZipFile(mscz) as z:
        assert any(n.endswith(".mscx") for n in z.namelist())
