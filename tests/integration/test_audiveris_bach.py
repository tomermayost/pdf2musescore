"""Integration: real Audiveris on the user-supplied Bach Cello Suite 2 PDF.

Skipped automatically if Audiveris is not installed. Output is kept in tmp/integration for inspection.
"""

import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pdf2musescore.deps import AUDIVERIS, find_tool
from pdf2musescore.engines.audiveris import AudiverisEngine

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.pdf"
OUT = ROOT / "tmp" / "integration" / "bach_cello_suite_2"

pytestmark = pytest.mark.integration


def _read_musicxml(mxl: Path) -> ET.Element:
    with zipfile.ZipFile(mxl) as z:
        container = ET.fromstring(z.read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile").get("full-path")
        return ET.fromstring(z.read(rootfile))


@pytest.fixture(scope="module")
def audiveris():
    exe = find_tool(AUDIVERIS)
    if not exe:
        pytest.skip("Audiveris not installed")
    return exe


def test_bach_cello_suite_to_musicxml(audiveris):
    shutil.rmtree(OUT, ignore_errors=True)
    outputs = AudiverisEngine(audiveris, timeout_sec=1800).recognize(FIXTURE, OUT)
    assert outputs
    total_measures = total_notes = 0
    for mxl in outputs:
        root = _read_musicxml(mxl)
        assert root.tag in ("score-partwise", "score-timewise")
        total_measures += len(root.findall(".//measure"))
        total_notes += len(root.findall(".//note"))
    print(f"\nOutputs: {[str(p) for p in outputs]}\nMeasures: {total_measures}  Notes: {total_notes}")
    assert total_measures > 10
    assert total_notes > 100
