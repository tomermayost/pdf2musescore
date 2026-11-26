"""Integration: Full Score Mode E2E on The Green Hill (homr AI + RapidOCR + MuseScore 4)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction
from pathlib import Path

import homr.resize as homr_resize
import pytest

from pdf2musescore.config import AppConfig
from pdf2musescore.deps import MUSESCORE, find_tool
from pdf2musescore.engines.base import OmrError
from pdf2musescore.fullscore import runner as fs_runner_mod
from pdf2musescore.fullscore.align import verify_fullscore_invariants
from pdf2musescore.fullscore.instruments import is_percussion_instrument
from pdf2musescore.options import ConversionOptions
from pdf2musescore.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
GREEN_HILL_PDF = Path(r"C:\Users\tmayo\Desktop\תווים\green hill\the-green-hill-b-appermont.pdf")
LOCAL_MODELS = Path(r"C:\Users\tmayo\AppData\Local\pdf2musescore\models")
OUT = ROOT / "tmp" / "integration" / "fullscore"
_OMR_CACHE_ROOT = OUT / "_omr_cache"
_OMR_SOURCE_FILES: tuple[Path, ...] = (
    ROOT / "src" / "pdf2musescore" / "fullscore" / "runner.py",
    ROOT / "src" / "pdf2musescore" / "fullscore" / "staves.py",
    ROOT / "src" / "pdf2musescore" / "fullscore" / "homr_patch.py",
    ROOT / "src" / "pdf2musescore" / "fullscore" / "instruments.py",
    ROOT / "src" / "pdf2musescore" / "engines" / "homr_engine.py",
    ROOT / "src" / "pdf2musescore" / "tempo.py",
)

pytestmark = pytest.mark.integration


def _omr_cache_fingerprint(
    pdf_path: Path,
    *,
    render_dpi: int,
    target_width: int,
    segnet_step_size: int,
    min_notehead_size: int,
    preprocess_mode: str,
) -> str:
    h = hashlib.sha256()
    if pdf_path.is_file():
        st = pdf_path.stat()
        h.update(f"pdf:{st.st_size}:{st.st_mtime_ns}\n".encode("utf-8"))
    h.update(
        f"opts:{render_dpi}:{target_width}:{segnet_step_size}:{min_notehead_size}:{preprocess_mode}\n".encode("utf-8")
    )
    h.update(f"homr:{importlib.metadata.version('homr')}\n".encode("utf-8"))
    for src in _OMR_SOURCE_FILES:
        h.update(src.name.encode("utf-8") + b"\0")
        h.update(src.read_bytes())
    return h.hexdigest()[:16]


@pytest.fixture(autouse=True)
def _cached_fullscore_omr(monkeypatch):
    """Reuse per-page OMR XML/sidecar outputs when OMR code & inputs are unchanged.

    Keeps full ``Pipeline``, ``align.py`` merge/normalization, and ``MuseScore4.exe`` export live
    while cutting warm integration runtime from ~7 minutes down to seconds.
    Set ``PDF2MUSESCORE_NO_OMR_CACHE=1`` to force live CPU OMR on every page.
    """
    real_run = fs_runner_mod.run_fullscore_homr

    def _run_with_cache(image_paths, **kwargs):
        no_cache = os.environ.get("PDF2MUSESCORE_NO_OMR_CACHE", "").strip().lower() in ("1", "true", "yes")
        if no_cache:
            return real_run(image_paths, **kwargs)

        fp = _omr_cache_fingerprint(
            GREEN_HILL_PDF,
            render_dpi=500,
            target_width=int(kwargs.get("target_width", 3840)),
            segnet_step_size=int(kwargs.get("segnet_step_size", 320)),
            min_notehead_size=int(kwargs.get("min_notehead_size", 4)),
            preprocess_mode=str(kwargs.get("preprocess_mode", "auto")),
        )
        cache_dir = _OMR_CACHE_ROOT / fp
        total = len(image_paths)
        page_cb = kwargs.get("page_cb")
        staff_cb = kwargs.get("staff_cb")

        outputs: list[Path] = []
        missing: list[Path] = []

        for idx, img in enumerate(image_paths, start=1):
            img_path = Path(img)
            pno = fs_runner_mod._parse_page_number(img_path, idx)
            c_xml = cache_dir / f"page_{pno:03d}.musicxml"
            c_sc = cache_dir / f"page_{pno:03d}.fullscore.json"
            if c_xml.is_file() and c_sc.is_file():
                dst_xml = img_path.with_suffix(".musicxml")
                dst_sc = img_path.with_suffix(".fullscore.json")
                shutil.copy2(c_xml, dst_xml)
                shutil.copy2(c_sc, dst_sc)
                if page_cb is not None:
                    page_cb(idx, total)
                if staff_cb is not None:
                    sc_data = json.loads(c_sc.read_text(encoding="utf-8"))
                    n_staves = int(sc_data.get("staff_count", 0))
                    if n_staves > 0:
                        staff_cb(idx, total, n_staves, n_staves)
                outputs.append(dst_xml)
            else:
                missing.append(img_path)

        if missing:
            try:
                fresh = real_run(missing, **kwargs)
            except OmrError as exc:
                if outputs and "no music staves found" in str(exc).lower():
                    fresh = []
                else:
                    raise
            if fresh:
                cache_dir.mkdir(parents=True, exist_ok=True)
                for f_idx, f_xml in enumerate(fresh, start=1):
                    pno = fs_runner_mod._parse_page_number(f_xml, f_idx)
                    shutil.copy2(f_xml, cache_dir / f"page_{pno:03d}.musicxml")
                    f_sc = f_xml.with_suffix(".fullscore.json")
                    if f_sc.is_file():
                        shutil.copy2(f_sc, cache_dir / f"page_{pno:03d}.fullscore.json")
                    outputs.append(f_xml)

        return sorted(outputs)

    monkeypatch.setattr(fs_runner_mod, "run_fullscore_homr", _run_with_cache)


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


def _assert_no_overfull_measures(root: ET.Element) -> None:
    parts = root.findall("part")
    total_m = len(parts[0].findall("measure"))
    ts = (4, 4)
    for mi in range(total_m):
        m0 = parts[0].findall("measure")[mi]
        t = m0.find("./attributes/time")
        if t is not None:
            ts = (int(t.findtext("beats")), int(t.findtext("beat-type")))
        exp_q = Fraction(ts[0] * 4, ts[1])
        for p in parts:
            div = 4
            for m in p.findall("measure")[: mi + 1]:
                d = m.findtext("./attributes/divisions")
                if d and d.strip().isdigit():
                    div = int(d.strip())
            m = p.findall("measure")[mi]
            pos = max_pos = prev_pos = 0
            for ch in m:
                if ch.tag == "note" and ch.find("grace") is None:
                    dur = int(ch.findtext("duration") or "0")
                    if ch.find("chord") is not None:
                        max_pos = max(max_pos, prev_pos + dur)
                    else:
                        prev_pos = pos
                        pos += dur
                        max_pos = max(max_pos, pos)
                elif ch.tag == "backup":
                    pos = max(0, pos - int(ch.findtext("duration") or "0"))
                elif ch.tag == "forward":
                    pos += int(ch.findtext("duration") or "0")
                    max_pos = max(max_pos, pos)
            assert Fraction(max_pos, div) <= exp_q, f"{p.get('id')} M{mi + 1} overflowed {exp_q}q: {Fraction(max_pos, div)}q"


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
        mscx_bytes = z.read(mscx_names[0])
        assert len(mscx_bytes) > 10_000
        mscx_root = ET.fromstring(mscx_bytes)

    staged_xml = _newest_staged_xml(cfg.work_dir)
    root = ET.parse(staged_xml).getroot()
    summary = verify_fullscore_invariants(root)
    assert summary["parts"] == 29
    assert summary["measures"] == 18
    assert summary["total_notes"] > 400
    assert summary["transposing_parts"] >= 15
    assert root.findtext("defaults/scaling/millimeters") == "3.4"
    _assert_no_overfull_measures(root)

    # Page 4 opening bars must capture the 4/4 -> 2/4 -> 4/4 meter change uniformly across all 29 parts
    p1_ms = root.findall("./part[@id='P1']/measure")
    assert (p1_ms[0].findtext("./attributes/time/beats"), p1_ms[0].findtext("./attributes/time/beat-type")) == ("4", "4")
    assert (p1_ms[1].findtext("./attributes/time/beats"), p1_ms[1].findtext("./attributes/time/beat-type")) == ("2", "4")
    assert (p1_ms[2].findtext("./attributes/time/beats"), p1_ms[2].findtext("./attributes/time/beat-type")) == ("4", "4")

    # P1 Scotch-snap bars (M5, M8, M9, M12, M14) must be repaired to 16th + dotted-eighth
    for m_idx in (4, 7, 8, 11, 13):
        notes = [n for n in p1_ms[m_idx].findall("note") if n.find("chord") is None]
        assert notes[0].findtext("type") == "16th"
        assert notes[1].findtext("type") == "eighth" and len(notes[1].findall("dot")) == 1

    # P11 (Eb Alto Sax. 1) OMR key outlier (+1 on p.4) must be harmonized to 0 with no spurious key change on M7
    p11 = root.find("./part[@id='P11']")
    assert p11.findtext("./measure[@number='1']/attributes/key/fifths") == "0"
    assert p11.find("./measure[@number='7']/attributes/key") is None

    # MusicXML Measure 1 <transpose> present for transposing parts and absent for C non-octave parts
    assert root.find("./part[@id='P1']/measure[@number='1']/attributes/transpose") is None
    assert root.find("./part[@id='P4']/measure[@number='1']/attributes/transpose") is None
    assert root.find("./part[@id='P6']/measure[@number='1']/attributes/transpose") is not None
    assert root.find("./part[@id='P11']/measure[@number='1']/attributes/transpose") is not None
    assert root.find("./part[@id='P18']/measure[@number='1']/attributes/transpose") is not None
    assert root.find("./part[@id='P25']/measure[@number='1']/attributes/transpose") is not None

    # Verify MuseScore 4 (.mscx) imported transpositions cleanly without shifting written keys/notes
    mscx_parts = mscx_root.findall("./Score/Part")
    mscx_staves = mscx_root.findall("./Score/Staff")
    assert len(mscx_parts) == 29 and len(mscx_staves) == 29
    part_by_idx = {idx: (mscx_parts[idx - 1], mscx_staves[idx - 1]) for idx in range(1, 30)}

    # Check transposeChromatic in MuseScore (.mscx)
    expected_tc = {
        5: "3",     # Eb Clarinet
        6: "-2",    # Bb Clarinet 1
        9: "-9",    # Eb Alto Clarinet
        10: "-14",  # Bb Bass Clarinet
        11: "-9",   # Eb Alto Sax. 1
        13: "-14",  # Bb Tenor Sax.
        14: "-21",  # Eb Baritone Sax.
        15: "-2",   # Bb Trumpet 1 / Cornet 1
        18: "-7",   # F Horn 1, 3
        25: "-12",  # String Bass
    }
    for idx, exp_chrom in expected_tc.items():
        assert part_by_idx[idx][0].findtext("./Instrument/transposeChromatic") == exp_chrom

    # Check instrumentId mappings (must not fall back to "voice")
    for idx in (1, 14, 23, 24, 25, 27):
        iid = part_by_idx[idx][0].findtext("./Instrument/instrumentId")
        assert iid and iid != "voice"

    # Check written keys & Bar 1 Beat 1 sounding Concert Eb4 (MIDI 63, tpc 11) + written TPCs
    assert part_by_idx[4][1].findtext("./Measure/voice/KeySig/concertKey") == "-3"  # Bassoon written 3b
    assert part_by_idx[6][1].findtext("./Measure/voice/KeySig/actualKey") == "-1"   # Bb Cl. 1 written 1b
    assert part_by_idx[11][1].find("./Measure/voice/KeySig") is None               # Eb A. Sax 1 written 0 flats/sharps
    assert part_by_idx[18][1].findtext("./Measure/voice/KeySig/actualKey") == "-2"  # F Horn 1, 3 written 2b

    for idx, exp_tpc2 in ((4, None), (6, "13"), (7, "13"), (8, "13"), (11, "14"), (18, "12")):
        note1 = part_by_idx[idx][1].find("./Measure/voice/Chord/Note")
        assert note1 is not None
        assert note1.findtext("pitch") == "63"  # Sounding Concert Eb4
        assert note1.findtext("tpc") == "11"    # Sounding Eb
        assert note1.findtext("tpc2") == exp_tpc2  # Written pitch preserved (Bsn=Eb, BbCl=F, EbSax=C, FHn=Bb)

    # Measure 12 in rest-only staves (e.g. Bassoon, Bb Clarinet 2) must contain only full-measure rest (no trailing 16th rest)
    for rest_idx in (4, 7):
        m12 = part_by_idx[rest_idx][1].findall("./Measure")[11]
        assert [r.findtext("durationType") for r in m12.findall(".//Rest")] == ["measure"]

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

    # No leftover homr default midi-program=54 or instrument-sound="voice"
    for sp in sps:
        assert sp.findtext("./score-instrument/instrument-sound") != "voice"
        assert sp.findtext("./midi-instrument/midi-program") != "54"

    # Opening tempo marking ("Adagio affetuoso", BPM 72) detected & exported to MuseScore <Tempo>
    m1_dir = root.find("./part[@id='P1']/measure[@number='1']/direction")
    assert m1_dir is not None
    assert "Adagio" in (m1_dir.findtext(".//words") or "")
    assert m1_dir.findtext(".//metronome/per-minute") == "72"
    assert m1_dir.find("./sound").get("tempo") == "72"
    mscx_tempos = mscx_root.findall(".//Tempo")
    assert mscx_tempos, "Expected MuseScore 4 to import <Tempo> element"
    assert any(abs(float(t.findtext("tempo") or "0") - 1.2) < 0.05 for t in mscx_tempos)


def test_fullscore_green_hill_midscore_and_meter_change():
    _require_prereqs()
    cfg = _make_config("midscore")
    out_mscz = OUT / "midscore" / "green_hill_p11_29.mscz"
    out_mscz.parent.mkdir(parents=True, exist_ok=True)

    result = Pipeline(cfg).convert(
        GREEN_HILL_PDF,
        out_mscz,
        pages_spec="11, 29",
        options=ConversionOptions(render_dpi=500, full_score=True),
    )
    assert result == out_mscz and zipfile.is_zipfile(out_mscz)
    assert homr_resize.calc_target_image_size(4000, 5000)[0] == 1920

    staged_xml = _newest_staged_xml(cfg.work_dir)
    root = ET.parse(staged_xml).getroot()
    summary = verify_fullscore_invariants(root)
    assert summary["parts"] == 29
    assert summary["measures"] == 10
    _assert_no_overfull_measures(root)

    # Page 11 is 4/4, page 29 switches to 6/4
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
