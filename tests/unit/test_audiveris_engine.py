import subprocess

import pytest

from pdf2musescore.engines.audiveris import AudiverisEngine, parse_valid_sheets
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


def test_parse_valid_sheets():
    log_text = (
        "INFO [input#1] Sheet.java:1583 | Stored /sheet#1/sheet#1.xml\n"
        "WARN [input#2] Book.java:2044 | Error processing stub java.lang.ArrayIndexOutOfBoundsException\n"
        "INFO [input#2] Sheet.java:1583 | Stored /sheet#2/sheet#2.xml\n"
        "INFO [input#3] Sheet.java:1583 | Stored /sheet#3/sheet#3.xml\n"
    )
    assert parse_valid_sheets(log_text) == [1, 3]


def test_partial_sheet_recovery(pdf, tmp_path):
    out = tmp_path / "out"
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if "-sheets" not in cmd:
            return subprocess.CompletedProcess(
                cmd, 1,
                "INFO [input#1] Sheet.java:1583 | Stored /sheet#1/sheet#1.xml\n"
                "WARN [input#2] Book.java:2044 | Error processing stub\n"
                "INFO [input#2] Sheet.java:1583 | Stored /sheet#2/sheet#2.xml\n"
                "INFO [input#3] Sheet.java:1583 | Stored /sheet#3/sheet#3.xml\n",
                "",
            )
        assert cmd[cmd.index("-sheets") + 1] == "1,3"
        mxl = out / "in.mxl"
        mxl.parent.mkdir(parents=True, exist_ok=True)
        mxl.write_text("x")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    eng = AudiverisEngine("A.exe", runner=run)
    assert [p.name for p in eng.recognize(pdf, out)] == ["in.mxl"]
    assert len(calls) == 2
