from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pdf2musescore.engines.base import OmrError
from pdf2musescore.engines.homr_engine import CancelledError
from pdf2musescore.fullscore.align import read_page_sidecar
from pdf2musescore.fullscore.runner import run_fullscore_homr


@dataclass
class _FakeStaff:
    min_x: float = 400.0
    max_x: float = 3500.0
    min_y: float = 100.0
    max_y: float = 160.0
    average_unit_size: float = 12.0
    is_grandstaff: bool = False
    symbols: list[str] = field(default_factory=list)

    def extend_to_x_range(self, min_x: int, max_x: int) -> "_FakeStaff":
        return _FakeStaff(min(min_x, self.min_x), max(max_x, self.max_x), self.min_y, self.max_y)


def _build_fake_homr(behavior: str = "ok"):
    orig_calc_size = lambda w, h: (1920, round(h * 1920 / w))
    cleaned_debug_calls: list[str] = []

    class _Debug:
        def clean_debug_files_from_previous_runs(self):
            cleaned_debug_calls.append("cleaned")

    class _ProcessingConfig:
        def __init__(self, **_kw):
            pass

    class _XmlArgs:
        def __init__(self, **_kw):
            pass

    class _MultiStaff:
        def __init__(self, staffs, _extra):
            self.staffs = staffs

    class _Config:
        use_gpu_inference = False
        use_coreml_encoder = False

    def _detect(img_path, _cfg):
        p = Path(img_path)
        if behavior == "skip_p1" and "page_001" in p.name:
            raise RuntimeError("No staffs found")
        if behavior == "empty_staves" and "page_001" in p.name:
            return ([_MultiStaff([], [])], np.zeros((10, 10), dtype=np.uint8), _Debug(), None, 0, lambda pt: pt)
        if behavior == "boom_p1" and "page_001" in p.name:
            raise ValueError("corrupt onnx tensor")
        if behavior == "all_empty":
            raise RuntimeError("No noteheads or clefs found")
        if behavior == "all_boom":
            raise RuntimeError("fatal segnet error")
        staves = [_FakeStaff(min_y=100, max_y=160), _FakeStaff(min_y=260, max_y=320)]
        return ([_MultiStaff(staves, [])], np.zeros((10, 10), dtype=np.uint8), _Debug(), None, 2, lambda pt: pt)

    def _parse_staff_image(_dbg, idx, _staff, _img, _reg, _cfg, page_to_input_image=None):
        return [f"s{idx}"]

    def _parse_staffs(dbg, full_sys, img, config=None, selected_staff=-1, page_to_input_image=None):
        out = []
        for idx, st in enumerate(full_sys[0].staffs):
            out.append(staff_mod.parse_staff_image(dbg, idx, st, img, None, config, page_to_input_image=page_to_input_image))
        return out

    def _generate_xml(_xargs, result_staffs, title=""):
        root = ET.Element("score-partwise", version="4.0")
        ET.SubElement(root, "identification")
        pl = ET.SubElement(root, "part-list")
        for i in range(len(result_staffs)):
            sp = ET.SubElement(pl, "score-part", id=f"P{i + 1}")
            ET.SubElement(sp, "part-name").text = "Voice"
            p = ET.SubElement(root, "part", id=f"P{i + 1}")
            m = ET.SubElement(p, "measure", number="1")
            a = ET.SubElement(m, "attributes")
            ET.SubElement(a, "divisions").text = "4"
            t = ET.SubElement(a, "time")
            ET.SubElement(t, "beats").text = "4"
            ET.SubElement(t, "beat-type").text = "4"
            n = ET.SubElement(m, "note")
            ET.SubElement(n, "rest")
            ET.SubElement(n, "duration").text = "16"
        return root

    resize_mod = SimpleNamespace(calc_target_image_size=orig_calc_size)
    staff_mod = SimpleNamespace(
        _calculate_region=lambda s, r: np.array([0, 0, 100, 100]),
        parse_staff_image=_parse_staff_image,
        parse_staffs=_parse_staffs,
        identity=lambda pt: pt,
    )
    main_mod = SimpleNamespace(
        ProcessingConfig=_ProcessingConfig,
        detect_staffs_in_image=_detect,
        extract=lambda *a, **k: None,
        create_bounding_ellipses=lambda *a, **k: None,
        color_adjust=SimpleNamespace(apply_clahe=lambda x: x),
    )
    xml_mod = SimpleNamespace(XmlGeneratorArguments=_XmlArgs, generate_xml=_generate_xml)

    return {
        "main": main_mod,
        "xml": xml_mod,
        "resize": resize_mod,
        "staff_parsing": staff_mod,
        "MultiStaff": _MultiStaff,
        "Config": _Config,
    }, orig_calc_size, cleaned_debug_calls


def _fake_ocr_factory(_ocr_dir):
    def _read(_margin):
        return SimpleNamespace(
            boxes=[[[0, 120], [50, 120], [50, 140], [0, 140]], [[0, 280], [50, 280], [50, 300], [0, 300]]],
            txts=["Flute", "Timpani"],
        )

    return _read


def test_run_fullscore_homr_success_and_cover_skip(tmp_path):
    p1 = tmp_path / "page_001.png"
    p2 = tmp_path / "page_002.png"
    Image.new("RGB", (600, 600), (255, 255, 255)).save(p1)
    Image.new("RGB", (600, 600), (255, 255, 255)).save(p2)

    mods, orig_calc, cleaned = _build_fake_homr("skip_p1")
    page_events: list[tuple[int, int]] = []
    staff_events: list[tuple[int, int, int, int]] = []

    outs = run_fullscore_homr(
        [p1, p2],
        page_cb=lambda i, t: page_events.append((i, t)),
        staff_cb=lambda pi, pt, si, st: staff_events.append((pi, pt, si, st)),
        homr_modules=mods,
        ocr_reader_factory=_fake_ocr_factory,
    )
    assert outs == [p2.with_suffix(".musicxml")]
    assert page_events == [(1, 2), (2, 2)]
    assert staff_events == [(2, 2, 1, 2), (2, 2, 2, 2)]
    assert len(cleaned) == 1
    assert mods["resize"].calc_target_image_size is orig_calc

    sc = read_page_sidecar(outs[0])
    assert sc is not None and sc.page_number == 2
    assert [l.full_name for l in sc.labels] == ["Flute", "Timpani"]
    assert sc.tempo is None

    # Tempo & Rehearsal OCR hits on header crop are stored in sidecar, and failures do not crash page
    def _ocr_with_tempo(_ocr_dir):
        def _read(crop):
            if crop.shape[1] > 400:
                return SimpleNamespace(
                    boxes=[[[180, 40], [360, 40], [360, 70], [180, 70]], [[420, 40], [452, 40], [452, 76], [420, 76]]],
                    txts=["Allegro = 120", "A"],
                )
            return SimpleNamespace(
                boxes=[[[0, 120], [50, 120], [50, 140], [0, 140]], [[0, 280], [50, 280], [50, 300], [0, 300]]],
                txts=["Flute", "Timpani"],
            )

        return _read

    mods_t, _, _ = _build_fake_homr("ok")
    outs_t = run_fullscore_homr([p2], homr_modules=mods_t, ocr_reader_factory=_ocr_with_tempo)
    sc_t = read_page_sidecar(outs_t[0])
    assert sc_t is not None and sc_t.tempo is not None
    assert sc_t.tempo.text == "Allegro" and sc_t.tempo.bpm == 120
    assert len(sc_t.positioned_tempos) == 1
    assert len(sc_t.rehearsal_marks) == 1
    assert sc_t.rehearsal_marks[0].mark.label == "A"
    assert sc_t.rehearsal_marks[0].mark.enclosure == "none"

    def _ocr_tempo_boom(_ocr_dir):
        def _read(crop):
            if crop.shape[1] > 400:
                raise RuntimeError("ocr header glitch")
            return SimpleNamespace(boxes=[], txts=[])

        return _read

    outs_boom_t = run_fullscore_homr([p2], homr_modules=mods_t, ocr_reader_factory=_ocr_tempo_boom)
    assert len(outs_boom_t) == 1
    assert read_page_sidecar(outs_boom_t[0]).tempo is None
    assert read_page_sidecar(outs_boom_t[0]).rehearsal_marks == ()


def test_run_fullscore_homr_empty_staves_partial_fail_all_fail_and_cancel(tmp_path):
    p1 = tmp_path / "page_001.png"
    p2 = tmp_path / "page_002.png"
    Image.new("RGB", (400, 400), (255, 255, 255)).save(p1)
    Image.new("RGB", (400, 400), (255, 255, 255)).save(p2)

    # Empty staves on p1 skips cleanly, p2 succeeds
    mods_empty, _, _ = _build_fake_homr("empty_staves")
    assert run_fullscore_homr([p1, p2], homr_modules=mods_empty, ocr_reader_factory=_fake_ocr_factory) == [
        p2.with_suffix(".musicxml")
    ]

    # Unexpected failure on p1 logs warning and returns p2
    mods_boom, _, _ = _build_fake_homr("boom_p1")
    assert run_fullscore_homr([p1, p2], homr_modules=mods_boom, ocr_reader_factory=_fake_ocr_factory) == [
        p2.with_suffix(".musicxml")
    ]

    # All non-music -> OmrError
    mods_all_empty, _, _ = _build_fake_homr("all_empty")
    with pytest.raises(OmrError, match="No music staves found"):
        run_fullscore_homr([p1], homr_modules=mods_all_empty, ocr_reader_factory=_fake_ocr_factory)

    # All fatal error -> OmrError with failure details
    mods_all_boom, _, _ = _build_fake_homr("all_boom")
    with pytest.raises(OmrError, match="fatal segnet error"):
        run_fullscore_homr([p1], homr_modules=mods_all_boom, ocr_reader_factory=_fake_ocr_factory)

    # Immediate cancel + mid-staff cancel
    mods_ok, orig_calc, _ = _build_fake_homr("ok")
    with pytest.raises(CancelledError):
        run_fullscore_homr([p1], is_cancelled=lambda: True, homr_modules=mods_ok, ocr_reader_factory=_fake_ocr_factory)

    calls = {"n": 0}

    def cancel_after_one():
        calls["n"] += 1
        return calls["n"] >= 4

    with pytest.raises(CancelledError):
        run_fullscore_homr([p1], is_cancelled=cancel_after_one, homr_modules=mods_ok, ocr_reader_factory=_fake_ocr_factory)
    assert mods_ok["resize"].calc_target_image_size is orig_calc


def test_run_fullscore_homr_barline_detection_error_and_disabled_chunking(tmp_path, monkeypatch, caplog):
    import pdf2musescore.fullscore.runner as runner_mod

    p1 = tmp_path / "page_001.png"
    Image.new("RGB", (400, 400), (255, 255, 255)).save(p1)

    mods, _, _ = _build_fake_homr("ok")
    monkeypatch.setattr(
        runner_mod,
        "detect_system_barlines_with_skew",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("barline probe boom")),
    )
    with caplog.at_level("WARNING"):
        outs = run_fullscore_homr([p1], homr_modules=mods, ocr_reader_factory=_fake_ocr_factory)
    assert len(outs) == 1
    assert "System barline detection failed" in caplog.text

    outs_disabled = run_fullscore_homr(
        [p1],
        enable_staff_chunking=False,
        homr_modules=mods,
        ocr_reader_factory=_fake_ocr_factory,
    )
    assert len(outs_disabled) == 1
