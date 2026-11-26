"""Integration: full Pipeline on the Bach PDF with homr AI + MuseScore."""

import shutil
import zipfile
from pathlib import Path

import pytest

from pdf2musescore.config import AppConfig
from pdf2musescore.deps import MUSESCORE, find_tool
from pdf2musescore.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.pdf"
OUT = ROOT / "tmp" / "integration" / "pipeline"

pytestmark = pytest.mark.integration


def test_pipeline_mscz():
    if not find_tool(MUSESCORE):
        pytest.skip("MuseScore required")
    steps = []
    heb_in = OUT / "תווים" / "מנחם" / "bach cello suite 2.pdf"
    heb_in.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURE, heb_in)
    heb_out = OUT / "תווים" / "מנחם" / "bach cello suite 2.mscz"
    out = Pipeline(AppConfig(work_dir=OUT / "work", render_dpi=200),
                   progress=lambda p, m: steps.append(p)).convert(heb_in, heb_out)
    assert zipfile.is_zipfile(out)
    assert steps[-1] == 100
    with zipfile.ZipFile(out) as z:
        mscx_name = next(n for n in z.namelist() if n.endswith(".mscx"))
        mscx_txt = z.read(mscx_name).decode("utf-8", errors="replace")
        assert "<trackName>Piano</trackName>" in mscx_txt
        assert "<trackName>Voice</trackName>" not in mscx_txt
        assert "voice.vocals" not in mscx_txt
