"""Tests for Phase 1 release readiness, licensing alignment, CI provenance, and media generation."""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "generate_media.py"


def _load_generate_media():
    spec = importlib.util.spec_from_file_location("generate_media", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compose_before_after_creates_valid_image(tmp_path: Path) -> None:
    gm = _load_generate_media()

    before = Image.new("RGB", (600, 800), (255, 255, 255))
    draw = ImageDraw.Draw(before)
    for y in range(120, 420, 20):
        draw.line([(80, y), (520, y)], fill=(15, 23, 42), width=3)

    # Blank white image also exercises the no-ink bounding-box branch cleanly
    after = Image.new("RGB", (600, 800), (255, 255, 255))

    out_path = tmp_path / "before-after.png"
    res = gm.compose_before_after(before, after, out_path, width=1400)
    assert res == out_path
    assert out_path.is_file()

    with Image.open(out_path) as rendered:
        assert rendered.mode == "RGB"
        assert rendered.width == 1400
        assert rendered.height >= 400


def test_generate_social_preview_dimensions_and_missing_inputs(tmp_path: Path) -> None:
    gm = _load_generate_media()

    icon_path = tmp_path / "icon.png"
    shot_path = tmp_path / "shot.png"
    out_path = tmp_path / "social.png"

    with pytest.raises(FileNotFoundError):
        gm.generate_social_preview(icon_path, shot_path, out_path)

    Image.new("RGBA", (128, 128), (37, 99, 235, 255)).save(icon_path, format="PNG")
    with pytest.raises(FileNotFoundError):
        gm.generate_social_preview(icon_path, shot_path, out_path)

    Image.new("RGB", (846, 913), (31, 43, 62)).save(shot_path, format="PNG")
    res = gm.generate_social_preview(icon_path, shot_path, out_path)
    assert res == out_path
    assert out_path.is_file()

    with Image.open(out_path) as banner:
        assert banner.size == (1280, 640)
        assert banner.mode == "RGB"


def test_render_pdf_first_page_and_cli_social_preview(tmp_path: Path) -> None:
    gm = _load_generate_media()

    with pytest.raises(FileNotFoundError):
        gm.render_pdf_first_page(tmp_path / "missing.pdf", tmp_path / "out.png")

    fixture_pdf = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.pdf"
    rendered_pdf = gm.render_pdf_first_page(fixture_pdf, tmp_path / "bach_p1.png", dpi=100)
    assert rendered_pdf.is_file()
    with Image.open(rendered_pdf) as img:
        assert img.width > 100 and img.height > 100

    social_out = tmp_path / "cli_social.png"
    rc = gm.main([
        "--social-preview",
        "--log-level", "DEBUG",
        "--social-out", str(social_out),
    ])
    assert rc == 0
    assert social_out.is_file()
    with Image.open(social_out) as banner:
        assert banner.size == (1280, 640)


def test_render_mscz_first_page_validation_and_mocked_cli(tmp_path: Path, monkeypatch) -> None:
    gm = _load_generate_media()
    fake_mscz = tmp_path / "score.mscz"
    fake_exe = tmp_path / "MuseScore4.exe"

    with pytest.raises(FileNotFoundError):
        gm.render_mscz_first_page(fake_mscz, tmp_path / "out", fake_exe)

    fake_mscz.write_bytes(b"zip")
    with pytest.raises(FileNotFoundError):
        gm.render_mscz_first_page(fake_mscz, tmp_path / "out", fake_exe)

    fake_exe.write_bytes(b"exe")
    out_dir = tmp_path / "mscz_pages"

    def fake_run_ok(cmd, capture_output, text, timeout):
        (out_dir / "after-1.png").write_bytes(b"png1")
        (out_dir / "after-2.png").write_bytes(b"png2")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gm.subprocess, "run", fake_run_ok)
    first = gm.render_mscz_first_page(fake_mscz, out_dir, fake_exe)
    assert first.name == "after-1.png"

    monkeypatch.setattr(
        gm.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="MuseScore CLI failed"):
        gm.render_mscz_first_page(fake_mscz, out_dir, fake_exe)


def test_license_and_pyproject_metadata_aligned_agplv3() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3, 19 November 2007" in license_text
    assert not license_text.lstrip().startswith("MIT License")
    assert len(license_text) >= 30_000

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    proj = data["project"]
    assert proj["license"]["text"] == "AGPL-3.0-or-later"
    assert proj["authors"] == [{"name": "Tomer Mayost"}]
    assert "musescore" in proj["keywords"]
    assert any("AGPLv3+" in c for c in proj["classifiers"])
    assert proj["urls"]["Homepage"] == "https://tomermayost.github.io/pdf2musescore/"
    for key in ("Repository", "Issues", "Releases"):
        assert proj["urls"][key].startswith("https://github.com/tomermayost/pdf2musescore")


def test_readme_assets_and_trust_statements() -> None:
    gm = _load_generate_media()
    assert gm.DEFAULT_MSCZ == ROOT / "tests" / "fixtures" / "bach_cello_suite_2.mscz"
    assert gm.DEFAULT_MSCZ.is_file()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Free code signing is provided by [SignPath.io]" not in readme
    assert "AGPL-3.0-or-later" in readme
    assert "SHA256SUMS.txt" in readme
    assert "gh attestation verify" in readme
    assert not re.search(r"\]\([^)]+\.pdf\)", readme, re.IGNORECASE)

    asset_refs = re.findall(r"docs/assets/[A-Za-z0-9_.-]+\.png", readme)
    assert {"docs/assets/ui-light.png", "docs/assets/ui-dark.png", "docs/assets/before-after-bach.png"}.issubset(
        set(asset_refs)
    )
    for rel in asset_refs:
        img_path = ROOT / rel
        assert img_path.is_file(), f"Missing README asset: {rel}"
        with Image.open(img_path) as img:
            assert img.width > 200 and img.height > 200

    social_path = ROOT / "docs" / "assets" / "social-preview.png"
    assert social_path.is_file()
    with Image.open(social_path) as social:
        assert social.size == (1280, 640)


def test_release_workflow_and_issue_templates() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "id-token: write" in workflow
    assert "attestations: write" in workflow
    assert "SHA256SUMS.txt" in workflow
    assert "actions/attest-build-provenance@v2" in workflow

    tpl_dir = ROOT / ".github" / "ISSUE_TEMPLATE"
    bug = (tpl_dir / "bug_report.yml").read_text(encoding="utf-8")
    for token in ("name:", "description:", "title:", "labels:", "body:", "Open log folder"):
        assert token in bug

    feat = (tpl_dir / "feature_request.yml").read_text(encoding="utf-8")
    for token in ("name:", "description:", "title:", "labels:", "body:"):
        assert token in feat

    cfg = (tpl_dir / "config.yml").read_text(encoding="utf-8")
    assert "blank_issues_enabled: true" in cfg
    assert "https://github.com/tomermayost/pdf2musescore/discussions" in cfg
