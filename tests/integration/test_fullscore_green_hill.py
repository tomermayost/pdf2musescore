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
_OMR_CACHE_ROOT = ROOT / "tmp" / "integration" / "_raw_omr_cache" / "fullscore"
_OMR_SOURCE_FILES: tuple[Path, ...] = (
    ROOT / "src" / "pdf2musescore" / "fullscore" / "staves.py",
    ROOT / "src" / "pdf2musescore" / "fullscore" / "homr_patch.py",
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
def _cached_fullscore_omr(monkeypatch, shared_rapidocr_reader):
    """Cache raw Full Score neural-net OMR & staff layout while running OCR, sidecar, align, and MuseScore 4 LIVE.

    Changing OCR, tempo, rehearsal, or alignment code never invalidates the 29-staff OMR cache.
    Set ``PDF2MUSESCORE_NO_OMR_CACHE=1`` to force live CPU OMR on every page.
    """
    real_run = fs_runner_mod.run_fullscore_homr
    real_flatten = fs_runner_mod.flatten_filter_and_extend_staves

    def _run_with_cache(image_paths, **kwargs):
        no_cache = os.environ.get("PDF2MUSESCORE_NO_OMR_CACHE", "").strip().lower() in ("1", "true", "yes")
        if no_cache:
            return real_run(image_paths, **kwargs)

        import cv2
        from PIL import Image
        from pdf2musescore.fullscore.align import PageSidecar, write_page_sidecar
        from pdf2musescore.fullscore.instruments import extract_instrument_labels

        ocr_reader, ocr_ver_dir = shared_rapidocr_reader
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
            c_skipped = cache_dir / f"page_{pno:03d}.skipped.json"
            c_raw = cache_dir / f"page_{pno:03d}.raw.musicxml"
            c_layout = cache_dir / f"page_{pno:03d}.layout.json"

            if c_skipped.is_file():
                if page_cb is not None:
                    page_cb(idx, total)
                continue
            if c_raw.is_file() and c_layout.is_file():
                dst_xml = img_path.with_suffix(".musicxml")
                shutil.copy2(c_raw, dst_xml)
                ld = json.loads(c_layout.read_text(encoding="utf-8"))
                labels = extract_instrument_labels(
                    img_path,
                    ld["staff_cys"],
                    int(ld["crop_x"]),
                    float(ld["median_gap"]),
                    float(ld["y_lo"]),
                    float(ld["y_hi"]),
                    ocr_reader=ocr_reader,
                    ocr_dir=ocr_ver_dir,
                )
                with Image.open(img_path) as im_sz:
                    img_w = float(im_sz.width)
                scale_x = img_w / float(kwargs.get("target_width", 3840))
                in_sys_min_x = float(ld["sys_min_x"]) * scale_x
                in_sys_max_x = float(ld["sys_max_x"]) * scale_x
                tempo, positioned_tempos, rehearsal_marks = fs_runner_mod.extract_page_header_markings(
                    img_path,
                    top_staff_y=float(ld["top_staff_y"]),
                    median_gap=float(ld["median_gap"]),
                    sys_min_x=in_sys_min_x,
                    sys_max_x=in_sys_max_x,
                    ocr_reader=ocr_reader,
                    ocr_dir=ocr_ver_dir,
                )
                write_page_sidecar(
                    PageSidecar(
                        page_number=pno,
                        staff_count=len(ld["staff_cys"]),
                        staff_cys=list(ld["staff_cys"]),
                        staff_y_norms=list(ld["staff_y_norms"]),
                        labels=labels,
                        sys_min_x=int(in_sys_min_x),
                        sys_max_x=int(in_sys_max_x),
                        crop_x=int(ld["crop_x"]),
                        tempo=tempo,
                        positioned_tempos=tuple(positioned_tempos),
                        rehearsal_marks=tuple(rehearsal_marks),
                    ),
                    dst_xml,
                )
                if page_cb is not None:
                    page_cb(idx, total)
                n_staves = len(ld["staff_cys"])
                if staff_cb is not None and n_staves > 0:
                    staff_cb(idx, total, n_staves, n_staves)
                outputs.append(dst_xml)
            else:
                missing.append(img_path)

        if missing:
            captured_layouts: list[Any] = []

            def _capturing_flatten(multi_staffs, to_input_image):
                res = real_flatten(multi_staffs, to_input_image)
                if res.kept_staffs:
                    captured_layouts.append(res)
                return res

            monkeypatch.setattr(fs_runner_mod, "flatten_filter_and_extend_staves", _capturing_flatten)
            try:
                fresh = real_run(missing, **kwargs)
            except OmrError as exc:
                if outputs and "no music staves found" in str(exc).lower():
                    fresh = []
                else:
                    raise
            finally:
                monkeypatch.setattr(fs_runner_mod, "flatten_filter_and_extend_staves", real_flatten)

            cache_dir.mkdir(parents=True, exist_ok=True)
            fresh_by_pno = {fs_runner_mod._parse_page_number(f_xml, fi): f_xml for fi, f_xml in enumerate(fresh, start=1)}
            for mp in missing:
                pno = fs_runner_mod._parse_page_number(mp, 0)
                if pno not in fresh_by_pno:
                    (cache_dir / f"page_{pno:03d}.skipped.json").write_text("{\"skipped\": true}\n", encoding="utf-8")

            for layout_idx, f_xml in enumerate(fresh):
                pno = fs_runner_mod._parse_page_number(f_xml, layout_idx + 1)
                shutil.copy2(f_xml, cache_dir / f"page_{pno:03d}.raw.musicxml")
                if layout_idx < len(captured_layouts):
                    ld_obj = captured_layouts[layout_idx]
                    layout_payload = {
                        "staff_cys": list(ld_obj.staff_cys),
                        "staff_y_norms": list(ld_obj.staff_y_norms),
                        "sys_min_x": int(ld_obj.sys_min_x),
                        "sys_max_x": int(ld_obj.sys_max_x),
                        "crop_x": int(ld_obj.crop_x),
                        "median_gap": float(ld_obj.median_gap),
                        "y_lo": float(ld_obj.y_lo),
                        "y_hi": float(ld_obj.y_hi),
                        "top_staff_y": float(ld_obj.top_staff_y),
                        "staff_left_x": float(ld_obj.staff_left_x),
                    }
                    (cache_dir / f"page_{pno:03d}.layout.json").write_text(
                        json.dumps(layout_payload, indent=2) + "\n", encoding="utf-8"
                    )
                outputs.append(f_xml)

        if not outputs:
            raise OmrError(f"No music staves found across {total} selected full-score page(s).")
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

    # Unboxed rehearsal letter 'A' on Page 6 (Measure 13) with zero false positives on pages 4-5
    early_rehs = [
        r.text
        for m in p1_ms[:12]
        for r in m.findall("./direction/direction-type/rehearsal")
    ]
    assert early_rehs == []
    assert p1_ms[12].findtext("./direction/direction-type/rehearsal") == "A"
    mscx_rehs = ["".join(r.itertext()).strip() for r in mscx_root.findall(".//RehearsalMark/text")]
    assert "A" in mscx_rehs


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


def test_fullscore_green_hill_tuplets_meter_rehearsal_and_percussion():
    _require_prereqs()
    cfg = _make_config("regression_pages")
    out_mscz = OUT / "regression_pages" / "green_hill_reg.mscz"
    out_mscz.parent.mkdir(parents=True, exist_ok=True)

    result = Pipeline(cfg).convert(
        GREEN_HILL_PDF,
        out_mscz,
        pages_spec="13, 14, 17, 29, 35",
        options=ConversionOptions(render_dpi=500, full_score=True),
    )
    assert result == out_mscz and zipfile.is_zipfile(out_mscz)

    staged_xml = _newest_staged_xml(cfg.work_dir)
    root = ET.parse(staged_xml).getroot()
    summary = verify_fullscore_invariants(root)
    assert summary["parts"] == 29
    # p.13 (5, ghost m.51 trimmed!) + p.14 (5) + p.17 (5) + p.29 (5) + p.35 (4) = 24
    assert summary["measures"] == 24
    _assert_no_overfull_measures(root)

    p1_ms = root.findall("./part[@id='P1']/measure")
    # p.13: 5 bars, Poco meno mosso bpm=72 (no garbled '₀')
    p13_tempos = [
        (w.text or "", m.findtext("./direction/direction-type/metronome/per-minute"))
        for m in p1_ms[:5]
        for w in m.findall("./direction/direction-type/words")
    ]
    assert any("Poco meno mosso" in t and bpm == "72" for t, bpm in p13_tempos)
    assert not any("₀" in t for t, _ in p13_tempos)

    # p.14 (M6): switches to 12/8, boxed 'E', clean tempo without 'E ' or '•'
    assert (p1_ms[5].findtext("./attributes/time/beats"), p1_ms[5].findtext("./attributes/time/beat-type")) == ("12", "8")
    assert p1_ms[5].findtext("./direction/direction-type/rehearsal") == "E"
    assert p1_ms[5].find("./direction/direction-type/rehearsal").get("enclosure") == "square"
    m6_words = " ".join((w.text or "") for w in p1_ms[5].findall("./direction/direction-type/words"))
    assert "E " not in m6_words and "•" not in m6_words

    # p.17 (M11..M15, Bar. Solo): ZERO spurious tuplets! M13 (5th bar after F) preserves all 13 notes cleanly
    for mi in range(10, 15):
        assert p1_ms[mi].find(".//time-modification") is None, f"Unexpected spurious tuplet in M{mi + 1}"
    m13_notes = [n for n in p1_ms[12].findall("note") if n.find("chord") is None and n.find("rest") is None]
    assert len(m13_notes) == 13

    # p.29 (M16..M20): inherited 12/8 is preserved (does NOT flip to 6/4)
    for mi in range(10, 20):
        assert p1_ms[mi].find("./attributes/time") is None

    # p.35 (M21): switches to 4/4, boxed 'P' (enclosure="square"), 'Andante appassionato' without 'P '
    assert (p1_ms[20].findtext("./attributes/time/beats"), p1_ms[20].findtext("./attributes/time/beat-type")) == ("4", "4")
    assert p1_ms[20].findtext("./direction/direction-type/rehearsal") == "P"
    assert p1_ms[20].find("./direction/direction-type/rehearsal").get("enclosure") == "square"
    m21_words = [w.text or "" for w in p1_ms[20].findall("./direction/direction-type/words")]
    assert any("Andante appassionato" in w and not w.startswith("P ") for w in m21_words)

    with zipfile.ZipFile(out_mscz) as z:
        mscx = [n for n in z.namelist() if n.endswith(".mscx")][0]
        mscx_root = ET.fromstring(z.read(mscx))

    mscx_staves = mscx_root.findall("./Score/Staff")
    # Bar. Solo has 0 <Tuplet> elements across p.17 bars (M11..M15)
    solo_ms = mscx_staves[0].findall("Measure")
    assert all(len(solo_ms[mi].findall(".//Tuplet")) == 0 for mi in range(10, 15))

    # Percussion 1 & 2 (staves 27, 28) have PERC clef only in M1, zero key flats, and no erratic heads
    for perc_idx in (27, 28):
        p_st = mscx_staves[perc_idx]
        clefs = [(c.findtext("concertClefType"), c.findtext("transposingClefType")) for c in p_st.findall(".//Clef")]
        assert clefs == [("PERC", "PERC")]
        assert p_st.findall(".//KeySig") == []
        bad_heads = {
            n.findtext("head")
            for n in p_st.findall(".//Note")
            if n.findtext("head") in ("ti", "diamond", "slashed2", "triangle-down", "custom")
        }
        assert not bad_heads


def test_fullscore_green_hill_header_ocr_fast(shared_rapidocr_reader):
    if not GREEN_HILL_PDF.is_file():
        pytest.skip(f"Test score PDF missing: {GREEN_HILL_PDF}")
    import cv2
    import numpy as np
    import pypdfium2 as pdfium
    from pdf2musescore.rehearsal import extract_rehearsal_marks_in_band
    from pdf2musescore.tempo import extract_tempo_markings_in_band

    ocr_reader, ocr_ver_dir = shared_rapidocr_reader
    doc = pdfium.PdfDocument(str(GREEN_HILL_PDF))
    try:
        results = {}
        for pno in (12, 13, 14, 30, 33, 35, 36):
            bm = doc[pno - 1].render(scale=500 / 72.0)
            img = cv2.cvtColor(np.array(bm.to_pil()), cv2.COLOR_RGB2BGR)
            h_px = img.shape[0]
            y_bot = int(h_px * 0.16)
            rehs = extract_rehearsal_marks_in_band(
                img, 0, y_bot, ocr_reader=ocr_reader, ocr_dir=ocr_ver_dir, source_name=f"p{pno}"
            )
            excl = [(r.x_left, r.y_top, r.x_right, r.y_bottom) for r in rehs]
            tempos = extract_tempo_markings_in_band(
                img, 0, y_bot, ocr_reader=ocr_reader, ocr_dir=ocr_ver_dir, exclude_boxes=excl, source_name=f"p{pno}"
            )
            results[pno] = (rehs, tempos)
    finally:
        doc.close()

    # p.12: Meno mosso = 69, A Tempo = 76, rit.
    p12_t = [(t.tempo.text, t.tempo.bpm, t.tempo.is_expression) for t in results[12][1]]
    assert ("Meno mosso", 69, False) in p12_t
    assert ("A Tempo", 76, False) in p12_t
    assert any(txt == "rit." and is_e for txt, _, is_e in p12_t)

    # p.13: Poco meno mosso = 72
    p13_t = [(t.tempo.text, t.tempo.bpm) for t in results[13][1] if not t.tempo.is_expression]
    assert ("Poco meno mosso", 72) in p13_t

    # p.14: boxed 'E', clean tempo without 'E ' or '•'
    assert [(r.mark.label, r.mark.enclosure) for r in results[14][0]] == [("E", "square")]
    assert all(not t.tempo.text.startswith("E ") and "•" not in t.tempo.text for t in results[14][1])

    # p.30: boxed 'N', 'A Tempo (in 2)' without 'N '
    assert [(r.mark.label, r.mark.enclosure) for r in results[30][0]] == [("N", "square")]
    assert any(t.tempo.text == "A Tempo (in 2)" for t in results[30][1])

    # p.33: boxed 'O' (not dropped as '0')
    assert [(r.mark.label, r.mark.enclosure) for r in results[33][0]] == [("O", "square")]

    # p.35: boxed 'P', 'Andante appassionato' bpm=76 without 'P '
    assert [(r.mark.label, r.mark.enclosure) for r in results[35][0]] == [("P", "square")]
    assert any(t.tempo.text == "Andante appassionato" and t.tempo.bpm == 76 for t in results[35][1])

    # p.36: Meno mosso = 65, rit.
    p36_t = [(t.tempo.text, t.tempo.bpm, t.tempo.is_expression) for t in results[36][1]]
    assert ("Meno mosso", 65, False) in p36_t
    assert any(txt == "rit." and is_e for txt, _, is_e in p36_t)


def test_fullscore_musescore_percussion_and_meter_edge_cases():
    ms_exe = find_tool(MUSESCORE)
    if not ms_exe:
        pytest.skip("MuseScore required")

    from pdf2musescore.exporters.musescore import MuseScoreExporter
    from pdf2musescore.fullscore.align import PageSidecar, merge_fullscore_pages
    from pdf2musescore.fullscore.instruments import InstrumentLabel
    from pdf2musescore.musicxml import write_musicxml

    edge_dir = OUT / "edge_cases"
    shutil.rmtree(edge_dir, ignore_errors=True)
    edge_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        InstrumentLabel("Baritone Solo", "Bar. Solo"),
        InstrumentLabel("Flute 1", "Fl. 1"),
        InstrumentLabel("Oboe 1", "Ob. 1"),
        InstrumentLabel("Bassoon", "Bsn."),
        InstrumentLabel("Percussion 1", "Perc. 1"),
    ]

    def _build_page(m_specs, perc_clef, perc_note):
        # m_specs: list of (ts_or_none, q_len_or_none_for_rest)
        root = ET.Element("score-partwise", version="4.0")
        ET.SubElement(root, "identification")
        pl = ET.SubElement(root, "part-list")
        for idx in range(5):
            pid = f"P{idx + 1}"
            sp = ET.SubElement(pl, "score-part", id=pid)
            ET.SubElement(sp, "part-name").text = labels[idx].full_name
            part = ET.SubElement(root, "part", id=pid)
            for mi, (ts, q_len) in enumerate(m_specs):
                m = ET.SubElement(part, "measure", number=str(mi + 1))
                this_ts = ts
                if idx == 4 and mi == 0 and this_ts is None:
                    this_ts = ("6", "4")
                if mi == 0 or this_ts is not None:
                    a = ET.SubElement(m, "attributes")
                    if mi == 0:
                        ET.SubElement(a, "divisions").text = "4"
                        ET.SubElement(ET.SubElement(a, "key"), "fifths").text = "-2"
                        c = ET.SubElement(a, "clef", number="1")
                        cl = perc_clef if idx == 4 else ("F", "4")
                        ET.SubElement(c, "sign").text = cl[0]
                        ET.SubElement(c, "line").text = cl[1]
                    if this_ts is not None:
                        t = ET.SubElement(a, "time")
                        ET.SubElement(t, "beats").text = this_ts[0]
                        ET.SubElement(t, "beat-type").text = this_ts[1]
                n = ET.SubElement(m, "note")
                if idx == 4 and mi == 0 and perc_note is not None:
                    p = ET.SubElement(n, "pitch")
                    ET.SubElement(p, "step").text = perc_note[0]
                    ET.SubElement(p, "octave").text = perc_note[1]
                    ET.SubElement(n, "duration").text = str(int(round((q_len or 6) * 4)))
                elif q_len is None or idx >= 3:
                    ET.SubElement(n, "rest")
                    ET.SubElement(n, "duration").text = str(int(round((q_len or 6) * 4)))
                else:
                    p = ET.SubElement(n, "pitch")
                    ET.SubElement(p, "step").text = "C"
                    ET.SubElement(p, "octave").text = "4"
                    ET.SubElement(n, "duration").text = str(int(round(q_len * 4)))
                ET.SubElement(n, "voice").text = "1"
                ET.SubElement(n, "staff").text = "1"
        return root

    # Page 1: explicit 12/8 (Perc in C3, D4 = 3rd space)
    p1 = _build_page([(("12", "8"), 6), (None, 6)], ("C", "3"), ("D", "4"))
    # Page 2: p.21 pattern (all parts emit 6/4 fallback, Perc in F4, E3 = 3rd space) -> preserves 12/8!
    p2 = _build_page([(("6", "4"), 6), (None, 6)], ("F", "4"), ("E", "3"))
    # Page 3: p.31/p.32 pattern (m1 minority 5/4, m2 rest-only 10/8 while pitched play 6q; Perc in G2, C5 = 3rd space) -> stays 12/8!
    p3 = _build_page([(("5", "4"), 6), (("10", "8"), 6)], ("G", "2"), ("C", "5"))
    # Adjust P3 so only 1 pitched part says 5/4 on m1 and only rest parts say 10/8 on m2
    for idx, part in enumerate(p3.findall("part")):
        ms = part.findall("measure")
        if idx >= 1:
            t1 = ms[0].find("./attributes/time")
            if t1 is not None:
                ms[0].find("attributes").remove(t1)
        if idx < 3:
            t2 = ms[1].find("./attributes/time")
            if t2 is not None:
                ms[1].remove(ms[1].find("attributes"))

    sc = lambda n: PageSidecar(n, 5, [10.0 * i for i in range(5)], [i / 4.0 for i in range(5)], labels)
    merged = merge_fullscore_pages([(1, p1, sc(1)), (2, p2, sc(2)), (3, p3, sc(3))], title="Edge Cases")
    verify_fullscore_invariants(merged)
    _assert_no_overfull_measures(merged)

    # All 6 bars stay 12/8
    assert [
        (t.findtext("beats"), t.findtext("beat-type"))
        for t in merged.findall("./part[@id='P1']/measure/attributes/time")
    ] == [("12", "8")]

    xml_out = write_musicxml(merged, edge_dir / "edges.musicxml")
    mscz_out = MuseScoreExporter(ms_exe).export(xml_out, edge_dir / "edges.mscz")
    with zipfile.ZipFile(mscz_out) as z:
        mscx = [n for n in z.namelist() if n.endswith(".mscx")][0]
        mscx_root = ET.fromstring(z.read(mscx))

    perc_staff = mscx_root.findall("./Score/Staff")[4]
    assert [(c.findtext("concertClefType"), c.findtext("transposingClefType")) for c in perc_staff.findall(".//Clef")] == [
        ("PERC", "PERC")
    ]
    assert perc_staff.findall(".//KeySig") == []
    perc_notes = [(n.findtext("pitch"), n.findtext("head")) for n in perc_staff.findall(".//Note")]
    assert perc_notes == [("38", None), ("38", None), ("38", None)]
