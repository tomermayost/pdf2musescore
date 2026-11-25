from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pdf2musescore.engines.homr_engine import CancelledError
from pdf2musescore.fullscore.homr_patch import (
    DEFAULT_TARGET_WIDTH,
    MAX_TARGET_HEIGHT,
    calc_fullscore_image_size,
    fullscore_homr_context,
)


def test_calc_fullscore_image_size_scaling_and_clamp(caplog):
    assert calc_fullscore_image_size(3840, 5400) == (3840, 5400)
    assert calc_fullscore_image_size(0, 100) == (1, 100)
    assert calc_fullscore_image_size(1920, 2700, target_width=3840) == (3840, 5400)

    # Abnormally tall page gets clamped to MAX_TARGET_HEIGHT proportionally
    with caplog.at_level("WARNING"):
        w, h = calc_fullscore_image_size(1000, 4000, target_width=3840, max_height=MAX_TARGET_HEIGHT)
    assert h == MAX_TARGET_HEIGHT
    assert w == round(1000 * (MAX_TARGET_HEIGHT / 4000))
    assert "Clamped full-score target height" in caplog.text


def _fake_homr_modules():
    orig_calc_size = lambda w, h: (1920, round(h * 1920 / w))
    orig_calc_region = lambda staff, regions: np.array([10, 50, 200, 40])  # inverted y!
    orig_parse = lambda debug, idx, staff, img, regions, cfg, page_to_input_image=None: [f"sym-{idx}"]
    orig_extract = lambda pre, p, step_size=320, use_cache=False, use_gpu_inference=False: (step_size, p)
    orig_ellipses = lambda nh, min_size=(4, 4), **kw: min_size
    orig_clahe = lambda img: ("clahe", img)

    resize_mod = SimpleNamespace(calc_target_image_size=orig_calc_size)
    staff_mod = SimpleNamespace(
        _calculate_region=orig_calc_region,
        parse_staff_image=orig_parse,
        identity=lambda pt: pt,
    )
    main_mod = SimpleNamespace(
        extract=orig_extract,
        create_bounding_ellipses=orig_ellipses,
        color_adjust=SimpleNamespace(apply_clahe=orig_clahe),
    )
    return resize_mod, staff_mod, main_mod, {
        "calc_size": orig_calc_size,
        "calc_region": orig_calc_region,
        "parse": orig_parse,
        "extract": orig_extract,
        "ellipses": orig_ellipses,
        "clahe": orig_clahe,
    }


def test_fullscore_homr_context_patches_and_restores_on_exit(caplog):
    resize_mod, staff_mod, main_mod, orig = _fake_homr_modules()
    seen_staff_indices: list[int] = []

    with caplog.at_level("INFO"):
        with fullscore_homr_context(
            target_width=4000,
            segnet_step_size=240,
            min_notehead_size=3,
            preprocess_mode="none",
            staff_progress_cb=seen_staff_indices.append,
            homr_resize=resize_mod,
            homr_staff_parsing=staff_mod,
            homr_main=main_mod,
        ):
            assert resize_mod.calc_target_image_size(2000, 3000) == (4000, 6000)
            assert main_mod.extract("img", "p.png") == (240, "p.png")
            assert main_mod.create_bounding_ellipses("nh") == (3, 3)
            assert main_mod.color_adjust.apply_clahe("raw") == "raw"

            # Degenerate region (both inverted y and inverted x)
            bad_staff = SimpleNamespace(min_x=50, max_x=20, min_y=100, max_y=100, average_unit_size=5.0)
            staff_mod._calculate_region = staff_mod._calculate_region
            reg = staff_mod._calculate_region(bad_staff, None)
            assert reg[2] > reg[0] and reg[3] > reg[1]

            # Normal region passes through safely
            staff_mod_orig = staff_mod._calculate_region
            # Also test parse_staff_image progress callback
            assert staff_mod.parse_staff_image(None, 2, bad_staff, None, None, None) == ["sym-2"]
            assert seen_staff_indices == [2]

    assert "Degenerate staff region y=" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= 30]
    # All originals strictly restored
    assert resize_mod.calc_target_image_size is orig["calc_size"]
    assert staff_mod._calculate_region is orig["calc_region"]
    assert staff_mod.parse_staff_image is orig["parse"]
    assert main_mod.extract is orig["extract"]
    assert main_mod.create_bounding_ellipses is orig["ellipses"]
    assert main_mod.color_adjust.apply_clahe is orig["clahe"]


def test_fullscore_homr_context_degenerate_x_and_normal_region():
    resize_mod, staff_mod, main_mod, orig = _fake_homr_modules()
    staff_mod._calculate_region = lambda staff, regions: np.array([200, 100, 100, 160])

    with fullscore_homr_context(
        target_width=DEFAULT_TARGET_WIDTH,
        homr_resize=resize_mod,
        homr_staff_parsing=staff_mod,
        homr_main=main_mod,
    ):
        staff = SimpleNamespace(min_x=100, max_x=300, min_y=110, max_y=150, average_unit_size=6.0)
        reg = staff_mod._calculate_region(staff, None)
        assert reg[0] < reg[2]
        assert reg[1] == 100 and reg[3] == 160

    # CLAHE untouched when preprocess_mode != "none"
    assert main_mod.color_adjust.apply_clahe is orig["clahe"]


def test_fullscore_homr_context_staff_error_skips_but_cancel_propagates(caplog):
    resize_mod, staff_mod, main_mod, orig = _fake_homr_modules()

    def flaky_parse(debug, idx, staff, img, regions, cfg, page_to_input_image=None):
        if idx == 1:
            raise RuntimeError("cv2 resize empty")
        return ["ok"]

    staff_mod.parse_staff_image = flaky_parse
    cancelled = {"flag": False}

    with caplog.at_level("ERROR"):
        with pytest.raises(CancelledError):
            with fullscore_homr_context(
                is_cancelled=lambda: cancelled["flag"],
                homr_resize=resize_mod,
                homr_staff_parsing=staff_mod,
                homr_main=main_mod,
            ):
                assert staff_mod.parse_staff_image(None, 1, None, None, None, None) == []
                assert staff_mod.parse_staff_image(None, 2, None, None, None, None) == ["ok"]
                cancelled["flag"] = True
                staff_mod.parse_staff_image(None, 3, None, None, None, None)

    assert "parse_staff_image failed on staff 1" in caplog.text
    # Restored even after CancelledError
    assert resize_mod.calc_target_image_size is orig["calc_size"]
    assert staff_mod.parse_staff_image is flaky_parse
