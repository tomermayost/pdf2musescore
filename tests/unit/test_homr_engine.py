import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from pdf2musescore.engines.base import OmrError
from pdf2musescore.engines.homr_engine import (
    CancelledError,
    HomrEngine,
    _default_render_pdf,
    annotate_standard_page_instruments,
    deskew_image,
    get_engine_version,
    get_pdf_page_count,
    parse_page_spec,
    preprocess_page_image,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.pdf"


@pytest.fixture
def pdf(tmp_path):
    p = tmp_path / "in.pdf"
    p.write_bytes(b"%PDF-1.4")
    return p


def test_get_pdf_page_count():
    assert get_pdf_page_count(FIXTURE) == 4


def test_get_engine_version(monkeypatch):
    import importlib.metadata
    ver = get_engine_version()
    assert ver != "unknown" and "." in ver

    def missing(_name):
        raise importlib.metadata.PackageNotFoundError("homr")

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert get_engine_version() == "unknown"


def test_parse_page_spec():
    assert parse_page_spec(None, 4) == [1, 2, 3, 4]
    assert parse_page_spec("", 4) == [1, 2, 3, 4]
    assert parse_page_spec("all", 4) == [1, 2, 3, 4]
    assert parse_page_spec("2-3", 4) == [2, 3]
    assert parse_page_spec("1, 3-4, 2", 4) == [1, 2, 3, 4]
    assert parse_page_spec("2, , 4", 4) == [2, 4]


@pytest.mark.parametrize("spec,total,match", [
    ("1", 0, "no pages"),
    ("abc", 4, "Invalid page range"),
    ("0-2", 4, "start at 1"),
    ("3-2", 4, "in order"),
    ("2-5", 4, "out of range"),
    (", ,", 4, "No pages selected"),
])
def test_parse_page_spec_errors(spec, total, match):
    with pytest.raises(OmrError, match=match):
        parse_page_spec(spec, total)


def test_deskew_and_preprocess_modes():
    blank = np.full((200, 300, 3), 255, dtype=np.uint8)
    # Blank image has no lines -> deskew returns unchanged
    assert deskew_image(blank).shape == blank.shape

    # Draw tilted horizontal lines (+3 degrees) and verify deskew rotates
    tilted = np.full((200, 400, 3), 255, dtype=np.uint8)
    for y in (50, 70, 90, 110, 130):
        cv2.line(tilted, (20, y), (380, y + 18), (0, 0, 0), 2)
    straight = deskew_image(tilted)
    assert straight.shape == tilted.shape

    # Vertical line only -> ignored by +/-10 deg filter
    vert = np.full((200, 400, 3), 255, dtype=np.uint8)
    cv2.line(vert, (100, 10), (100, 190), (0, 0, 0), 3)
    cv2.line(vert, (120, 10), (200, 190), (0, 0, 0), 3)
    assert deskew_image(vert).shape == vert.shape

    # Horizontal line with <0.05 deg tilt -> no rotation needed
    flat = np.full((200, 400, 3), 255, dtype=np.uint8)
    for y in (50, 70, 90):
        cv2.line(flat, (20, y), (380, y), (0, 0, 0), 2)
    assert deskew_image(flat) is flat

    for mode in ("auto", "faded", "binarize", "none", "unknown"):
        res = preprocess_page_image(tilted, mode=mode, deskew=(mode == "faded"))
        assert res.shape == tilted.shape


def test_default_render_pdf_subset_and_cancel(tmp_path):
    pages = _default_render_pdf(FIXTURE, tmp_path, dpi=72, pages_spec="2-3", preprocess_mode="faded", deskew=True)
    assert [p.name for p in pages] == ["page_002.png", "page_003.png"]

    with pytest.raises(CancelledError):
        _default_render_pdf(FIXTURE, tmp_path, dpi=72, is_cancelled=lambda: True)


def test_default_render_empty_pdf(monkeypatch, tmp_path):
    import pypdfium2 as pdfium
    monkeypatch.setattr(pdfium, "PdfDocument", lambda p: [])
    with pytest.raises(OmrError, match="no pages"):
        _default_render_pdf(FIXTURE, tmp_path, dpi=72)


def test_recognize_success(pdf, tmp_path):
    seen_cb = []

    def fake_render(p, out_dir, dpi, pages_spec=None, is_cancelled=None):
        assert pages_spec == "2-3"
        img = out_dir / "page_002.png"
        img.write_text("img")
        return [img]

    def fake_run(pages, cb, is_cancelled=None):
        if cb:
            cb(1, len(pages))
        xml = pages[0].with_suffix(".musicxml")
        xml.write_text("<score-partwise/>")
        return [xml]

    eng = HomrEngine(dpi=200, pages_spec="2-3", renderer=fake_render, runner=fake_run,
                     page_callback=lambda i, t: seen_cb.append((i, t)))
    outputs = eng.recognize(pdf, tmp_path / "out")
    assert [p.name for p in outputs] == ["page_002.musicxml"]
    assert seen_cb == [(1, 1)]


def test_recognize_with_default_render(tmp_path):
    def fake_run(pages, cb, is_cancelled=None):
        xml = pages[0].with_suffix(".musicxml")
        xml.write_text("<score-partwise/>")
        return [xml]

    eng = HomrEngine(dpi=72, pages_spec="1", runner=fake_run)
    outputs = eng.recognize(FIXTURE, tmp_path / "out")
    assert len(outputs) == 1


def test_recognize_cancelled_in_runner(pdf, tmp_path):
    def fake_render(p, out_dir, dpi, pages_spec=None, is_cancelled=None):
        return [out_dir / "page_001.png"]

    def cancel_run(pages, cb, is_cancelled=None):
        raise CancelledError("aborted")

    with pytest.raises(CancelledError):
        HomrEngine(renderer=fake_render, runner=cancel_run).recognize(pdf, tmp_path / "out")


def test_missing_pdf(tmp_path):
    with pytest.raises(OmrError, match="not found"):
        HomrEngine().recognize(tmp_path / "missing.pdf", tmp_path)


def test_render_failure(pdf, tmp_path):
    def bad_render(p, out_dir, dpi, *a):
        raise ValueError("corrupt pdf")

    with pytest.raises(OmrError, match="Failed to render"):
        HomrEngine(renderer=bad_render).recognize(pdf, tmp_path / "out")


def test_render_omr_error_passthrough(pdf, tmp_path):
    def empty_render(p, out_dir, dpi, *a):
        raise OmrError("PDF has no pages")

    with pytest.raises(OmrError, match="no pages"):
        HomrEngine(renderer=empty_render).recognize(pdf, tmp_path / "out")


def test_runner_exception(pdf, tmp_path):
    def fake_render(p, out_dir, dpi, *a):
        return [out_dir / "page_001.png"]

    def bad_run(pages, cb, *a):
        raise RuntimeError("onnx crash")

    with pytest.raises(OmrError, match="recognition failed"):
        HomrEngine(renderer=fake_render, runner=bad_run).recognize(pdf, tmp_path / "out")


def test_no_output(pdf, tmp_path):
    def fake_render(p, out_dir, dpi, *a):
        return [out_dir / "page_001.png"]

    with pytest.raises(OmrError, match="no MusicXML"):
        HomrEngine(renderer=fake_render, runner=lambda pages, cb, *a: []).recognize(pdf, tmp_path / "out")


def test_render_failure_modes(monkeypatch, tmp_path):
    from PIL import Image

    # imwrite returning False raises OmrError instead of silently continuing
    monkeypatch.setattr(cv2, "imwrite", lambda *_a, **_k: False)
    with pytest.raises(OmrError, match="Could not save rendered page 1"):
        _default_render_pdf(FIXTURE, tmp_path, dpi=72, pages_spec="1")

    # Image branch imwrite False
    img_in = tmp_path / "score.png"
    Image.new("RGB", (60, 80), (255, 255, 255)).save(img_in)
    with pytest.raises(OmrError, match="Could not save rendered page 1"):
        _default_render_pdf(img_in, tmp_path, dpi=72)
    monkeypatch.undo()

    # MemoryError during PDF page render gives lower-DPI hint
    import pypdfium2 as pdfium

    class _OomPage:
        def get_size(self):
            return (612.0, 792.0)

        def render(self, scale=1.0):
            raise MemoryError("oom")

    class _OomDoc:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return _OomPage()

        def close(self):
            pass

    monkeypatch.setattr(pdfium, "PdfDocument", lambda _p: _OomDoc())
    with pytest.raises(OmrError, match="try a lower Render DPI"):
        _default_render_pdf(FIXTURE, tmp_path, dpi=300)

    # Generic exception during PDF page render includes exception type
    class _BrokenPage:
        def get_size(self):
            return (612.0, 792.0)

        def render(self, scale=1.0):
            raise RuntimeError("corrupt stream")

    class _BrokenDoc(_OomDoc):
        def __getitem__(self, idx):
            return _BrokenPage()

    monkeypatch.setattr(pdfium, "PdfDocument", lambda _p: _BrokenDoc())
    with pytest.raises(OmrError, match="RuntimeError: corrupt stream"):
        _default_render_pdf(FIXTURE, tmp_path, dpi=72)
    monkeypatch.undo()

    # Image branch MemoryError and generic exception
    import pdf2musescore.engines.homr_engine as he

    monkeypatch.setattr(he, "load_pil_image_rgb", lambda *_a, **_k: (_ for _ in ()).throw(MemoryError("img oom")))
    with pytest.raises(OmrError, match="Out of memory loading image page 1"):
        _default_render_pdf(img_in, tmp_path, dpi=72)

    monkeypatch.setattr(he, "load_pil_image_rgb", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad img")))
    with pytest.raises(OmrError, match="ValueError: bad img"):
        _default_render_pdf(img_in, tmp_path, dpi=72)


def test_run_pages_success_partial_failure_all_fail_and_cancel(tmp_path, caplog):
    from PIL import Image
    from pdf2musescore.engines.homr_engine import run_pages

    p1 = tmp_path / "page_001.png"
    p2 = tmp_path / "page_002.png"
    Image.new("RGB", (40, 50), (255, 255, 255)).save(p1)
    # p2 intentionally missing so _image_dimensions exercises fallback safely
    seen_cb: list[tuple[int, int]] = []

    def ok_proc(img: Path) -> Path:
        out = img.with_suffix(".musicxml")
        out.write_text("<score-partwise/>")
        return out

    with caplog.at_level("INFO", logger="pdf2musescore.engines.homr_engine"):
        res = run_pages([p1, p2], ok_proc, page_cb=lambda i, t: seen_cb.append((i, t)))
    assert len(res) == 2
    assert seen_cb == [(1, 2), (2, 2)]
    assert "40x50 px" in caplog.text

    # Partial failure logs warning summary and returns surviving pages
    caplog.clear()

    def partial_proc(img: Path):
        if img == p1:
            raise ValueError("No staffs found")
        return img.with_suffix(".musicxml")

    with caplog.at_level("WARNING", logger="pdf2musescore.engines.homr_engine"):
        res_partial = run_pages([p1, p2], partial_proc)
    assert res_partial == [p2.with_suffix(".musicxml")]
    assert "1 of 2 page(s) could not be read" in caplog.text
    assert "ValueError: No staffs found" in caplog.text

    # All fail raises OmrError with every page's error details
    def all_fail_proc(img: Path):
        if img == p1:
            raise RuntimeError("No staffs found")
        return None  # NoOutput

    with pytest.raises(OmrError, match="No staffs found") as ei:
        run_pages([p1, p2], all_fail_proc)
    assert "NoOutput" in str(ei.value)

    # OmrError raised by runner passes through HomrEngine.recognize without double-wrapping
    eng = HomrEngine(
        renderer=lambda p, out_dir, dpi, *a: [p1, p2],
        runner=lambda pages, cb, *a: run_pages(pages, all_fail_proc),
    )
    with pytest.raises(OmrError, match=r"^homr could not read any"):
        eng.recognize(FIXTURE, tmp_path / "out")

    # Cancelled propagates immediately
    with pytest.raises(CancelledError):
        run_pages([p1], ok_proc, is_cancelled=lambda: True)

    def cancel_mid(_img: Path):
        raise CancelledError("stop")

    with pytest.raises(CancelledError):
        run_pages([p1], cancel_mid)


def test_homr_engine_full_score_routing_and_isolation(pdf, tmp_path):
    seen_fs: list[dict] = []
    seen_std: list[bool] = []

    def fake_render(p, out_dir, dpi, pages_spec=None, is_cancelled=None):
        img = out_dir / "page_001.png"
        img.write_text("img")
        return [img]

    def fake_std_run(pages, cb, is_cancelled=None):
        seen_std.append(True)
        xml = pages[0].with_suffix(".musicxml")
        xml.write_text("<score-partwise/>")
        return [xml]

    def fake_fs_run(pages, **kwargs):
        seen_fs.append(kwargs)
        xml = pages[0].with_suffix(".musicxml")
        xml.write_text("<score-partwise/>")
        return [xml]

    # Standard mode never calls fullscore_runner
    eng_std = HomrEngine(
        dpi=300,
        renderer=fake_render,
        runner=fake_std_run,
        full_score=False,
        fullscore_runner=fake_fs_run,
    )
    eng_std.recognize(pdf, tmp_path / "std")
    assert seen_std == [True] and seen_fs == []

    # Full Score mode routes exclusively to fullscore_runner with target_width and staff_callback
    staff_cb = lambda *a: None
    eng_fs = HomrEngine(
        dpi=500,
        renderer=fake_render,
        runner=fake_std_run,
        full_score=True,
        full_score_target_width=4096,
        staff_callback=staff_cb,
        fullscore_runner=fake_fs_run,
    )
    eng_fs.recognize(pdf, tmp_path / "fs")
    assert len(seen_fs) == 1
    assert seen_fs[0]["target_width"] == 4096
    assert seen_fs[0]["staff_cb"] is staff_cb


@dataclass
class _StdStaff:
    min_x: float
    max_x: float
    min_y: float
    max_y: float


def test_annotate_standard_page_instruments(tmp_path):
    assert annotate_standard_page_instruments(tmp_path / "p.png", tmp_path / "missing.xml", []) == []

    bad_xml = tmp_path / "bad.musicxml"
    bad_xml.write_text("<broken", encoding="utf-8")
    assert annotate_standard_page_instruments(tmp_path / "p.png", bad_xml, []) == []

    empty_xml = tmp_path / "empty.musicxml"
    empty_xml.write_text("<score-partwise/>", encoding="utf-8")
    assert annotate_standard_page_instruments(tmp_path / "p.png", empty_xml, []) == []

    png = tmp_path / "page_001.png"
    Image.new("RGB", (1000, 1200), (255, 255, 255)).save(png)

    def _homr_xml(part_count: int, staves_per_part: tuple[int, ...] = ()) -> str:
        sps = []
        pts = []
        for i in range(part_count):
            st = staves_per_part[i] if i < len(staves_per_part) else 1
            pname = "Piano" if st == 2 else "Voice"
            isound = "keyboard.piano" if st == 2 else "voice"
            mprog = "1" if st == 2 else "54"
            st_tag = f"<staves>{st}</staves>" if st == 2 else ""
            sps.append(
                f"<score-part id='P{i+1}'><part-name>{pname}</part-name>"
                f"<score-instrument id='P{i+1}-I1'><instrument-name>{pname}</instrument-name>"
                f"<instrument-sound>{isound}</instrument-sound></score-instrument>"
                f"<midi-instrument id='P{i+1}-I1'><midi-channel>{i+1}</midi-channel>"
                f"<midi-program>{mprog}</midi-program></midi-instrument></score-part>"
            )
            pts.append(
                f"<part id='P{i+1}'><measure number='1'><attributes>{st_tag}"
                f"<clef><sign>G</sign><line>2</line></clef></attributes><note/></measure></part>"
            )
        return f"<score-partwise><part-list>{''.join(sps)}</part-list>{''.join(pts)}</score-partwise>"

    # 1) Multi-part string orchestra score: left-margin OCR assigns Violin I + Violoncello
    x1 = tmp_path / "strings.musicxml"
    x1.write_text(_homr_xml(2), encoding="utf-8")
    ms_strings = [SimpleNamespace(staffs=[_StdStaff(180, 950, 260, 320), _StdStaff(180, 950, 440, 500)])]
    names = annotate_standard_page_instruments(
        png,
        x1,
        ms_strings,
        margin_ocr_runner=lambda _m: (
            [[[20, 275], [140, 275], [140, 305], [20, 305]], [[20, 455], [140, 455], [140, 485], [20, 485]]],
            ["Violin I", "Violoncello"],
        ),
    )
    assert names == ["Violin I", "Violoncello"]
    r1 = ET.parse(x1).getroot()
    assert r1.findtext("./part-list/score-part[@id='P1']/score-instrument/instrument-sound") == "strings.violin"
    assert r1.findtext("./part-list/score-part[@id='P1']/midi-instrument/midi-program") == "41"
    assert r1.findtext("./part-list/score-part[@id='P2']/score-instrument/instrument-sound") == "strings.cello"
    assert r1.findtext("./part-list/score-part[@id='P2']/midi-instrument/midi-program") == "43"

    # 2) Single-part with no left-margin name -> detects written instrument in header + tempo marking
    x2 = tmp_path / "solo_header.musicxml"
    x2.write_text(_homr_xml(1), encoding="utf-8")
    ms_solo = [SimpleNamespace(staffs=[_StdStaff(80, 950, 240, 300)])]
    names2 = annotate_standard_page_instruments(
        png,
        x2,
        ms_solo,
        margin_ocr_runner=lambda _m: ([], []),
        header_ocr_runner=lambda _h: ([[[60, 150], [240, 150], [240, 180], [60, 180]]], ["Bb Clarinet"]),
        tempo_ocr_runner=lambda _h: ([[[100, 190], [310, 190], [310, 220], [100, 220]]], ["Andante = 88"]),
    )
    assert names2 == ["Bb Clarinet"]
    r2 = ET.parse(x2).getroot()
    assert r2.findtext("./part-list/score-part/score-instrument/instrument-sound") == "wind.reed.clarinet.bflat"
    assert r2.findtext("./part-list/score-part/midi-instrument/midi-program") == "72"
    assert r2.findtext("./part/measure/direction/direction-type/words") == "Andante"
    assert r2.findtext("./part/measure/direction/direction-type/metronome/per-minute") == "88"

    # 3) Single-part with NO written instrument name anywhere -> falls back from Voice to Piano
    x3 = tmp_path / "solo_noname.musicxml"
    x3.write_text(_homr_xml(1), encoding="utf-8")
    names3 = annotate_standard_page_instruments(
        png,
        x3,
        ms_solo,
        margin_ocr_runner=lambda _m: ([], []),
        header_ocr_runner=lambda _h: ([[[350, 50], [600, 50], [600, 90], [350, 90]]], ["Prélude."]),
    )
    assert names3 == ["Piano"]
    r3 = ET.parse(x3).getroot()
    assert r3.findtext("./part-list/score-part/part-name") == "Piano"
    assert r3.findtext("./part-list/score-part/score-instrument/instrument-sound") == "keyboard.piano"
    assert r3.findtext("./part-list/score-part/midi-instrument/midi-program") == "1"

    # 4) Grand-staff (2 staves) + single staff preserves Piano and detects Percussion 1 (ch 10)
    x4 = tmp_path / "grand_plus_perc.musicxml"
    x4.write_text(_homr_xml(2, staves_per_part=(2, 1)), encoding="utf-8")
    ms_gp = [
        SimpleNamespace(
            staffs=[
                _StdStaff(200, 950, 200, 250),
                _StdStaff(200, 950, 310, 360),
                _StdStaff(200, 950, 460, 510),
            ]
        )
    ]
    names4 = annotate_standard_page_instruments(
        png,
        x4,
        ms_gp,
        margin_ocr_runner=lambda _m: ([[[20, 470], [160, 470], [160, 500], [20, 500]]], ["Snare Drum"]),
    )
    assert names4 == ["Piano", "Snare Drum"]
    r4 = ET.parse(x4).getroot()
    assert r4.findtext("./part-list/score-part[@id='P2']/midi-instrument/midi-channel") == "10"
    assert r4.find("./part-list/score-part[@id='P2']/midi-instrument/midi-program") is None

    # 5) Indented single staff with movement title "Prélude." above & plate "262" in margin -> Piano fallback
    x5_indented = tmp_path / "indented_title.musicxml"
    x5_indented.write_text(_homr_xml(1), encoding="utf-8")
    ms_indented = [SimpleNamespace(staffs=[_StdStaff(180, 950, 260, 320)])]
    names5 = annotate_standard_page_instruments(
        png,
        x5_indented,
        ms_indented,
        margin_ocr_runner=lambda _m: (
            [[[40, 200], [150, 200], [150, 230], [40, 230]], [[20, 275], [65, 275], [65, 300], [20, 300]]],
            ["Prélude.", "262"],
        ),
        header_ocr_runner=lambda _h: ([[[40, 200], [150, 200], [150, 230], [40, 230]]], ["Prélude."]),
    )
    assert names5 == ["Piano"]
    # Also test skipping non-instrument text sitting directly beside the staff (e.g. "Allegro")
    x5_noninst = tmp_path / "beside_noninst.musicxml"
    x5_noninst.write_text(_homr_xml(1), encoding="utf-8")
    assert (
        annotate_standard_page_instruments(
            png,
            x5_noninst,
            ms_indented,
            margin_ocr_runner=lambda _m: ([[[20, 275], [140, 275], [140, 300], [20, 300]]], ["Allegro"]),
            header_ocr_runner=lambda _h: ([], []),
        )
        == ["Piano"]
    )

    # 6) Mismatched staff count & empty multi_staffs still falls back dummy Voice -> Piano cleanly
    x5 = tmp_path / "mismatch.musicxml"
    x5.write_text(_homr_xml(1), encoding="utf-8")
    assert annotate_standard_page_instruments(png, x5, [SimpleNamespace(staffs=[])]) == ["Piano"]

    # 7) Multi-system tempo detection maps System 0 (left + right) and System 1 (right)
    x7 = tmp_path / "multi_sys.musicxml"
    x7.write_text(
        "<score-partwise><part-list><score-part id='P1'><part-name>Voice</part-name>"
        "<score-instrument id='P1-I1'><instrument-name>Voice</instrument-name><instrument-sound>voice</instrument-sound></score-instrument>"
        "<midi-instrument id='P1-I1'><midi-channel>1</midi-channel><midi-program>54</midi-program></midi-instrument>"
        "</score-part></part-list><part id='P1'>"
        "<measure number='1'><attributes><key><fifths>2</fifths></key></attributes><note/></measure>"
        "<measure number='2'><note/><barline location='right'><bar-style>light-light</bar-style></barline></measure>"
        "<measure number='3'><attributes><key><fifths>-2</fifths></key></attributes><note/></measure>"
        "<measure number='4'><print new-system='yes'/><note/></measure>"
        "<measure number='5'><note/></measure>"
        "<measure number='6'><note/></measure>"
        "</part></score-partwise>",
        encoding="utf-8",
    )
    ms_multi = [
        SimpleNamespace(staffs=[_StdStaff(120, 920, 200, 260)]),
        SimpleNamespace(staffs=[_StdStaff(80, 920, 420, 480)]),
    ]
    band_calls = {"n": 0}

    def _multi_tempo_ocr(_crop):
        band_calls["n"] += 1
        if band_calls["n"] == 1:
            return (
                [[[130, 120], [340, 120], [340, 150], [130, 150]], [[660, 120], [880, 120], [880, 150], [660, 150]]],
                ["Andante maestoso (J=72)", "Allegro (in t) alla Spagnola"],
            )
        return ([[[700, 20], [900, 20], [900, 50], [700, 50]]], ["Andante moderato (J=s4)"])

    annotate_standard_page_instruments(
        png,
        x7,
        ms_multi,
        margin_ocr_runner=lambda _m: ([], []),
        header_ocr_runner=lambda _h: ([[[40, 60], [180, 60], [180, 90], [40, 90]]], ["BARITONE"]),
        tempo_ocr_runner=_multi_tempo_ocr,
    )
    r7 = ET.parse(x7).getroot()
    ms7 = r7.findall("./part/measure")
    assert ms7[0].findtext("./direction/direction-type/words") == "Andante maestoso"
    assert ms7[2].findtext("./direction/direction-type/words") == "Allegro (in 1) alla Spagnola"
    assert ms7[5].findtext("./direction/direction-type/words") == "Andante moderato"
    assert ms7[5].find("./direction/sound").get("tempo") == "84"
