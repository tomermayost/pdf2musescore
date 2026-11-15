from pathlib import Path

import pytest

from pdf2musescore.engines.base import OmrError
from pdf2musescore.engines.homr_engine import HomrEngine, _default_render_pdf

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.pdf"


@pytest.fixture
def pdf(tmp_path):
    p = tmp_path / "in.pdf"
    p.write_bytes(b"%PDF-1.4")
    return p


def test_default_render_pdf(tmp_path):
    pages = _default_render_pdf(FIXTURE, tmp_path, dpi=72)
    assert len(pages) == 4
    assert all(p.is_file() for p in pages)


def test_default_render_empty_pdf(monkeypatch, tmp_path):
    import pypdfium2 as pdfium
    monkeypatch.setattr(pdfium, "PdfDocument", lambda p: [])
    with pytest.raises(OmrError, match="no pages"):
        _default_render_pdf(FIXTURE, tmp_path, dpi=72)


def test_recognize_success(pdf, tmp_path):
    seen_cb = []

    def fake_render(p, out_dir, dpi):
        img = out_dir / "page_001.png"
        img.write_text("img")
        return [img]

    def fake_run(pages, cb):
        if cb:
            cb(1, len(pages))
        xml = pages[0].with_suffix(".musicxml")
        xml.write_text("<score-partwise/>")
        return [xml]

    eng = HomrEngine(dpi=200, renderer=fake_render, runner=fake_run,
                     page_callback=lambda i, t: seen_cb.append((i, t)))
    outputs = eng.recognize(pdf, tmp_path / "out")
    assert [p.name for p in outputs] == ["page_001.musicxml"]
    assert seen_cb == [(1, 1)]


def test_missing_pdf(tmp_path):
    with pytest.raises(OmrError, match="not found"):
        HomrEngine().recognize(tmp_path / "missing.pdf", tmp_path)


def test_render_failure(pdf, tmp_path):
    def bad_render(p, out_dir, dpi):
        raise ValueError("corrupt pdf")

    with pytest.raises(OmrError, match="Failed to render"):
        HomrEngine(renderer=bad_render).recognize(pdf, tmp_path / "out")


def test_render_omr_error_passthrough(pdf, tmp_path):
    def empty_render(p, out_dir, dpi):
        raise OmrError("PDF has no pages")

    with pytest.raises(OmrError, match="no pages"):
        HomrEngine(renderer=empty_render).recognize(pdf, tmp_path / "out")


def test_runner_exception(pdf, tmp_path):
    def fake_render(p, out_dir, dpi):
        return [out_dir / "page_001.png"]

    def bad_run(pages, cb):
        raise RuntimeError("onnx crash")

    with pytest.raises(OmrError, match="recognition failed"):
        HomrEngine(renderer=fake_render, runner=bad_run).recognize(pdf, tmp_path / "out")


def test_no_output(pdf, tmp_path):
    def fake_render(p, out_dir, dpi):
        return [out_dir / "page_001.png"]

    with pytest.raises(OmrError, match="no MusicXML"):
        HomrEngine(renderer=fake_render, runner=lambda pages, cb: []).recognize(pdf, tmp_path / "out")
