"""Integration: full Pipeline on the Bach PDF with real Audiveris + MuseScore."""

import zipfile
from pathlib import Path

import pytest

from pdf2musescore.config import AppConfig
from pdf2musescore.deps import AUDIVERIS, MUSESCORE, find_tool
from pdf2musescore.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.pdf"
OUT = ROOT / "tmp" / "integration" / "pipeline"

pytestmark = pytest.mark.integration


def test_pipeline_mscz():
    if not (find_tool(AUDIVERIS) and find_tool(MUSESCORE)):
        pytest.skip("Audiveris and MuseScore required")
    steps = []
    out = Pipeline(AppConfig(work_dir=OUT / "work"), progress=lambda p, m: steps.append(p)) \
        .convert(FIXTURE, OUT / "bach.mscz")
    assert zipfile.is_zipfile(out)
    assert steps[-1] == 100
