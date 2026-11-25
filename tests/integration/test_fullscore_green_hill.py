"""Integration: Full Score Mode E2E on The Green Hill (homr AI + RapidOCR + MuseScore 4)."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import homr.resize as homr_resize
import pytest

from pdf2musescore.config import AppConfig
from pdf2musescore.deps import MUSESCORE, find_tool
from pdf2musescore.fullscore.align import verify_fullscore_invariants
from pdf2musescore.fullscore.instruments import is_percussion_instrument
from pdf2musescore.options import ConversionOptions
from pdf2musescore.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
GREEN_HILL_PDF = Path(r"C:\Users\tmayo\Desktop\תווים\green hill\the-green-hill-b-appermont.pdf")
LOCAL_MODELS = Path(r"C:\Users\tmayo\AppData\Local\pdf2musescore\models")
OUT = ROOT / "tmp" / "integration" / "fullscore"

pytestmark = pytest.mark.integration


def _make_config(subdir: str) -> AppConfig:
    work_dir = OUT / subdir / "work"
    shutil.rmtree(work_dir, ignore_errors=True)
    models_dir = LOCAL_MODELS if LOCAL_MODELS.is_dir() else (ROOT / "tmp" / "models")
    return AppConfig(work_dir=work_dir, models_dir=models_dir, log_level="DEBUG")


def _newest_staged_xml(work_dir: Path) -> Path:
    jobs = sorted(work_dir.glob("job-*/score.musicxml"), key=lambda p: p.stat().st_mtime)
    assert jobs, f"Expected staged score.musicxml under {work_dir}"
    return jobs[-1]


def _require_prereqs() -> None:
    if not GREEN_HILL_PDF.is_file():
        pytest.skip(f"Test score PDF missing: {GREEN_HILL_PDF}")
    if not find_tool(MUSESCORE):
        pytest.skip("MuseScore required")


def test_fullscore_green_hill_opening_pages():
    _require_prereqs()
    cfg = _make_config("opening")
    steps: list[tuple[int, str]] = []
    out_mscz = OUT / "opening" / "green_hill_p4_6.mscz"
    out_mscz.parent.mkdir(parents=True, exist_ok=True)

    result = Pipeline(cfg, progress=lambda p, m: steps.append((p, m))).convert(
        GREEN_HILL_PDF,
        out_mscz,
        pages_spec="4-6",
        options=ConversionOptions(render_dpi=500, full_score=True),
    )
    assert result == out_mscz and zipfile.is_zipfile(out_mscz)
    assert steps[-1][0] == 100
    assert any("staff " in msg for _, msg in steps)

    # Verify homr resize hook was strictly restored after Full Score Mode finished
    assert homr_resize.calc_target_image_size(4000, 5000)[0] == 1920

    with zipfile.ZipFile(out_mscz) as z:
        mscx_names = [n for n in z.namelist() if n.endswith(".mscx")]
        assert mscx_names
        assert len(z.read(mscx_names[0])) > 10_000

    staged_xml = _newest_staged_xml(cfg.work_dir)
    root = ET.parse(staged_xml).getroot()
    summary = verify_fullscore_invariants(root)
    assert summary["parts"] == 29
    assert summary["measures"] == 18
    assert summary["total_notes"] > 400
    assert root.findtext("defaults/scaling/millimeters") == "3.4"

    # Page 4 opening bars must capture the 4/4 -> 2/4 -> 4/4 meter change uniformly across all 29 parts
    p1_ms = root.findall("./part[@id='P1']/measure")
    assert (p1_ms[0].findtext("./attributes/time/beats"), p1_ms[0].findtext("./attributes/time/beat-type")) == ("4", "4")
    assert (p1_ms[1].findtext("./attributes/time/beats"), p1_ms[1].findtext("./attributes/time/beat-type")) == ("2", "4")
    assert (p1_ms[2].findtext("./attributes/time/beats"), p1_ms[2].findtext("./attributes/time/beat-type")) == ("4", "4")

    sps = root.findall("./part-list/score-part")
    names = [sp.findtext("part-name") or "" for sp in sps]
    abbrs = [sp.findtext("part-abbreviation") or "" for sp in sps]
    assert all(abbrs)
    non_fallback = [n for n in names if not n.startswith("Part ")]
    assert len(non_fallback) >= 25
    joined = " | ".join(names)
    for expected_kw in ("Flute", "Oboe", "Bassoon", "Clarinet", "Timpani", "Percussion"):
        assert expected_kw in joined

    for sp in sps:
        nm = sp.findtext("part-name") or ""
        ch = int(sp.findtext("./midi-instrument/midi-channel") or "0")
        assert 1 <= ch <= 16
        if is_percussion_instrument(nm):
            assert ch == 10
        else:
            assert ch != 10


def test_fullscore_green_hill_midscore_and_meter_change():
    _require_prereqs()
    cfg = _make_config("midscore")
    out_mscz = OUT / "midscore" / "green_hill_p11_29_30.mscz"
    out_mscz.parent.mkdir(parents=True, exist_ok=True)

    result = Pipeline(cfg).convert(
        GREEN_HILL_PDF,
        out_mscz,
        pages_spec="11, 29, 30",
        options=ConversionOptions(render_dpi=500, full_score=True),
    )
    assert result == out_mscz and zipfile.is_zipfile(out_mscz)
    assert homr_resize.calc_target_image_size(4000, 5000)[0] == 1920

    staged_xml = _newest_staged_xml(cfg.work_dir)
    root = ET.parse(staged_xml).getroot()
    summary = verify_fullscore_invariants(root)
    assert summary["parts"] == 29
    assert summary["measures"] == 16

    # Page 11 is 4/4, pages 29-30 switch to 6/4
    p1_times = [
        (t.findtext("beats"), t.findtext("beat-type"))
        for t in root.findall("./part[@id='P1']/measure/attributes/time")
    ]
    assert ("4", "4") in p1_times
    assert ("6", "4") in p1_times


def test_fullscore_cover_page_skipped_cleanly():
    _require_prereqs()
    cfg = _make_config("cover_skip")
    out_mscz = OUT / "cover_skip" / "green_hill_p3_4.mscz"
    out_mscz.parent.mkdir(parents=True, exist_ok=True)

    result = Pipeline(cfg).convert(
        GREEN_HILL_PDF,
        out_mscz,
        pages_spec="3-4",
        options=ConversionOptions(render_dpi=500, full_score=True),
    )
    assert result == out_mscz and zipfile.is_zipfile(out_mscz)

    staged_xml = _newest_staged_xml(cfg.work_dir)
    root = ET.parse(staged_xml).getroot()
    summary = verify_fullscore_invariants(root)
    assert summary["parts"] == 29
    assert summary["measures"] == 6
