import gc
import tkinter as tk
from pathlib import Path

import pytest
from PIL import Image

from pdf2musescore.ui_theme import get_theme
from pdf2musescore.ui_widgets import (
    CollapsibleSection,
    PdfPreviewPane,
    ToolTip,
    render_pdf_page_image,
)

FIXTURE_PDF = Path(__file__).resolve().parents[1] / "fixtures" / "bach_cello_suite_2.pdf"


def test_tooltip_delayed_schedule_clamp_and_theme(root):
    btn = tk.Button(root, text="Target")
    btn.pack()
    root.update_idletasks()

    tip = ToolTip(btn, "Scan Quality\nLine 2 details", delay_ms=40)
    tip._schedule()
    assert tip._after_id is not None
    # Leaving cancels pending scheduled tooltip cleanly
    tip.hide()
    assert tip._after_id is None
    assert tip.tip_window is None

    # Delay 0 shows immediately on schedule
    tip.delay_ms = 0
    tip._schedule()
    assert tip.tip_window is not None
    tip.set_palette(get_theme("dark"))
    tip.hide()
    assert tip.tip_window is None

    # No-op when text is empty
    empty_tip = ToolTip(btn, "")
    empty_tip._schedule()
    empty_tip.show()
    assert empty_tip.tip_window is None

    btn.destroy()


def test_collapsible_section_toggle_badge_and_theme(root):
    toggles = []
    sec = CollapsibleSection(root, "Advanced Options", expanded=False, on_toggle=lambda v: toggles.append(v))
    sec.pack(fill="x")
    assert sec.is_expanded is False
    assert not sec.content.winfo_manager()

    sec._on_click()
    assert sec.is_expanded is True
    assert sec.content.winfo_manager()
    assert toggles == [True]

    sec.set_expanded(True)  # idempotent no-op
    assert toggles == [True]

    sec.set_badge("Customized", tone="accent")
    assert sec.badge_lbl.winfo_manager()
    sec.set_badge("1 warning", tone="warn")
    sec.set_badge("1 error", tone="danger")
    sec.set_badge("")
    assert not sec.badge_lbl.winfo_manager()

    sec.apply_theme(get_theme("dark"))
    sec.destroy()


def test_render_pdf_page_image_real_fixture():
    img1 = render_pdf_page_image(FIXTURE_PDF, 1, max_size=(200, 260))
    assert img1.mode == "RGB"
    assert img1.width <= 200 and img1.height <= 260

    img2 = render_pdf_page_image(FIXTURE_PDF, 2, max_size=(200, 260))
    assert img2.mode == "RGB"

    with pytest.raises(ValueError, match="out of range"):
        render_pdf_page_image(FIXTURE_PDF, 99)


def test_render_pdf_page_image_huge_page_bounded(tmp_path):
    import pypdfium2 as pdfium

    huge_pdf = tmp_path / "huge.pdf"
    doc = pdfium.PdfDocument.new()
    try:
        doc.new_page(14400, 14400)
        doc.save(str(huge_pdf))
    finally:
        doc.close()

    img = render_pdf_page_image(huge_pdf, 1, max_size=(226, 296))
    assert img.mode == "RGB"
    assert 1 <= img.width <= 226 and 1 <= img.height <= 296


def test_pdf_preview_pane_lifecycle(root, tmp_path):
    events = {"browse": 0, "picked": []}
    render_calls = []

    def fake_renderer(pdf: Path, page: int, max_size: tuple[int, int]) -> Image.Image:
        render_calls.append((pdf.name, page, max_size))
        if page == 3:
            raise RuntimeError("corrupt page")
        return Image.new("RGB", (120, 160), (255, 255, 255))

    pane = PdfPreviewPane(
        root,
        on_browse=lambda: events.__setitem__("browse", events["browse"] + 1),
        on_toggle_page=lambda p: events["picked"].append(p),
        page_renderer=fake_renderer,
    )
    pane.pack()

    # Empty state: clicking triggers browse
    pane._handle_stage_click()
    assert events["browse"] == 1
    assert str(pane.prev_btn["state"]) == "disabled"
    assert str(pane.pick_btn["state"]) == "disabled"

    # Load 3-page PDF -> renders p1, caches p1
    sample_pdf = tmp_path / "etude.pdf"
    pane.set_pdf(sample_pdf, total_pages=3)
    assert pane.current_page == 1
    assert pane.page_lbl.cget("text") == "Page 1 of 3"
    assert pane._photo_ref is not None
    assert len(render_calls) == 1

    # Boundary prev does nothing
    pane.go_prev()
    assert pane.current_page == 1

    # Navigate next -> p2
    pane.go_next()
    assert pane.current_page == 2
    assert len(render_calls) == 2

    # Navigate back -> hits cache (render_calls still 2)
    pane.go_prev()
    assert pane.current_page == 1
    assert len(render_calls) == 2

    # Navigate to p3 -> renderer raises -> clean text fallback without crash
    pane.goto_page(3)
    assert pane.current_page == 3
    assert pane._photo_ref is None
    assert "etude.pdf" in pane.image_lbl.cget("text")

    # Selection state updates button text
    pane.update_selection_state({3})
    assert "p.3 in custom range" in pane.pick_btn.cget("text")
    pane.pick_btn.invoke()
    assert events["picked"] == [3]

    # Busy disables navigation/pick
    pane.set_busy(True)
    assert str(pane.pick_btn["state"]) == "disabled"
    pane.set_busy(False)

    # Clear PDF & switch theme
    pane.set_pdf(None, 0)
    assert pane.page_lbl.cget("text") == "No PDF loaded"
    pane.apply_theme(get_theme("dark"))
    pane.destroy()
    gc.collect()
