"""Integration test: build the standalone executable and verify it starts."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "tmp" / "dist" / "pdf2musescore" / "pdf2musescore.exe"
SETUP_EXE = ROOT / "tmp" / "dist" / "pdf2musescore-setup.exe"

pytestmark = pytest.mark.integration


def test_build_and_smoke_run():
    if not EXE.is_file():
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
    internal_homr = EXE.parent / "_internal" / "homr"
    assert not list(internal_homr.rglob("*.onnx")), "homr .onnx models should be downloaded on first run, not bundled"
    res = subprocess.run([str(EXE), "--version"], timeout=30)
    assert res.returncode == 0
    res_self = subprocess.run([str(EXE), "--self-test"], timeout=60)
    assert res_self.returncode == 0
    if SETUP_EXE.is_file():
        assert 1_000_000 < SETUP_EXE.stat().st_size < 135_000_000
