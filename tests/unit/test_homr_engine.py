from pathlib import Path

import cv2
import numpy as np
import pytest

from pdf2musescore.engines.base import OmrError
from pdf2musescore.engines.homr_engine import (
    CancelledError,
    HomrEngine,
    _default_render_pdf,
    deskew_image,
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
