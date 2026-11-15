"""Integration: Bach PDF -> Audiveris -> combined MusicXML -> .mscz (real tools)."""

import zipfile
from pathlib import Path

import pytest

from pdf2musescore.deps import AUDIVERIS, MUSESCORE, find_tool
from pdf2musescore.engines.audiveris import AudiverisEngine
from pdf2musescore.exporters.musescore import MuseScoreExporter
from pdf2musescore.musicxml import merge_files, read_score

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.pdf"
OUT = ROOT / "tmp" / "integration" / "e2e_combined"

pytestmark = pytest.mark.integration


def test_pdf_to_combined_mscz():
    aud, ms = find_tool(AUDIVERIS), find_tool(MUSESCORE)
    if not (aud and ms):
        pytest.skip("Audiveris and MuseScore required")
    movements = AudiverisEngine(aud).recognize(FIXTURE, OUT / "omr")
    xml = merge_files(movements, OUT / "bach.musicxml", title="Bach - Cello Suite No. 2")
    root = read_score(xml)
    assert len(root.findall(".//measure")) > 100
    assert len(list(root.iter("words"))) >= len(movements)

    mscz = MuseScoreExporter(ms).export(xml, OUT / "bach.mscz")
    assert zipfile.is_zipfile(mscz)
    with zipfile.ZipFile(mscz) as z:
        assert any(n.endswith(".mscx") for n in z.namelist())
