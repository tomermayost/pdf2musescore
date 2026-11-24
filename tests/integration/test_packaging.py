"""Integration test: build the standalone executable and verify it starts."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "tmp" / "dist" / "pdf2musescore" / "pdf2musescore.exe"
SETUP_EXE = ROOT / "tmp" / "dist" / "pdf2musescore-setup.exe"

pytestmark = pytest.mark.integration


def _needs_rebuild() -> bool:
    if not EXE.is_file():
        return True
    exe_mtime = EXE.stat().st_mtime
    watched = [ROOT / "packaging" / "pdf2musescore.spec", *(ROOT / "src" / "pdf2musescore").rglob("*.py")]
    return any(p.stat().st_mtime > exe_mtime for p in watched)


def test_build_and_smoke_run():
    if _needs_rebuild():
        pyinstaller = ROOT / ".venv" / "Scripts" / "pyinstaller.exe"
        res = subprocess.run(
            [
                str(pyinstaller),
                str(ROOT / "packaging" / "pdf2musescore.spec"),
                "--distpath", str(ROOT / "tmp" / "dist"),
                "--workpath", str(ROOT / "tmp" / "build"),
                "--noconfirm",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert res.returncode == 0, res.stderr

    assert EXE.is_file()
    internal = EXE.parent / "_internal"
    assert list(internal.glob("homr-*.dist-info")), "homr dist-info metadata must be bundled for version display"
    assert list(internal.glob("rapidocr-*.dist-info")), "rapidocr dist-info metadata must be bundled for cache versioning"
    assert (internal / "pdf2musescore" / "assets" / "icon.ico").is_file()
    assert (internal / "pdf2musescore" / "assets" / "icon.png").is_file()
    internal_homr = internal / "homr"
    assert not list(internal_homr.rglob("*.onnx")), "homr .onnx models should not be bundled in the installer"
    internal_ocr = internal / "rapidocr"
    assert not list(internal_ocr.rglob("*.onnx")), "rapidocr .onnx models should be downloaded into user cache, not bundled"
    res = subprocess.run([str(EXE), "--version"], timeout=30)
    assert res.returncode == 0
    res_self = subprocess.run([str(EXE), "--self-test"], timeout=60)
    assert res_self.returncode == 0

    for candidate_models in (ROOT / "tmp" / "verify_models", ROOT / "tmp" / "models"):
        if candidate_models.is_dir() and list(candidate_models.rglob(".pdf2musescore-models.json")):
            env = dict(os.environ)
            env["PDF2MSCZ_MODELS_DIR"] = str(candidate_models)
            env["PDF2MSCZ_LOG_DIR"] = str(ROOT / "tmp" / "packaging_smoke_logs")
            res_dl = subprocess.run(
                [str(EXE), "--download-models", "--no-ui"],
                env=env,
                timeout=30,
            )
            assert res_dl.returncode == 0
            break

    if SETUP_EXE.is_file():
        assert 1_000_000 < SETUP_EXE.stat().st_size < 135_000_000
