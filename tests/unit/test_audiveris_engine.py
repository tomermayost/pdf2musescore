import subprocess

import pytest

from pdf2musescore.engines.audiveris import AudiverisEngine
from pdf2musescore.engines.base import OmrError


@pytest.fixture
def pdf(tmp_path):
    p = tmp_path / "in.pdf"
    p.write_bytes(b"%PDF-1.4")
    return p


def _runner(code=0, make=None, out="line1\nline2", err="warn"):
    def run(cmd, **kw):
        if make:
            make.parent.mkdir(parents=True, exist_ok=True)
            make.write_text("x")
        return subprocess.CompletedProcess(cmd, code, out, err)
    return run


def test_command(tmp_path):
    cmd = AudiverisEngine("A.exe").build_command(tmp_path / "a.pdf", tmp_path / "o")
    assert cmd[1:5] == ["-batch", "-export", "-output", str(tmp_path / "o")]
    assert cmd[-1].endswith("a.pdf")


def test_success(pdf, tmp_path):
    out = tmp_path / "out"
    eng = AudiverisEngine("A.exe", runner=_runner(make=out / "in" / "in.mxl"))
    assert [p.name for p in eng.recognize(pdf, out)] == ["in.mxl"]


def test_missing_pdf(tmp_path):
    with pytest.raises(OmrError, match="not found"):
        AudiverisEngine("A.exe").recognize(tmp_path / "x.pdf", tmp_path)


def test_nonzero_exit(pdf, tmp_path):
    with pytest.raises(OmrError, match="code 2"):
        AudiverisEngine("A.exe", runner=_runner(code=2)).recognize(pdf, tmp_path / "o")


def test_no_output(pdf, tmp_path):
    with pytest.raises(OmrError, match="no MusicXML"):
        AudiverisEngine("A.exe", runner=_runner(out=None, err=None)).recognize(pdf, tmp_path / "o")


@pytest.mark.parametrize("exc", [subprocess.TimeoutExpired("a", 1), OSError("nope")])
def test_run_errors(pdf, tmp_path, exc):
    def boom(*a, **k):
        raise exc
    with pytest.raises(OmrError):
        AudiverisEngine("A.exe", runner=boom).recognize(pdf, tmp_path / "o")
