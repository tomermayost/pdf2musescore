import subprocess

import pytest

from pdf2musescore.exporters.musescore import ExportError, MuseScoreExporter


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "s.musicxml"
    p.write_text("<x/>")
    return p


def test_export_ok(src, tmp_path):
    dst = tmp_path / "o" / "s.mscz"

    def run(cmd, **k):
        assert cmd[1] == "-o"
        dst.write_text("z")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")
    assert MuseScoreExporter("M.exe", runner=run).export(src, dst) == dst


def test_missing_source(tmp_path):
    with pytest.raises(ExportError, match="not found"):
        MuseScoreExporter("M.exe").export(tmp_path / "no.xml", tmp_path / "o.mscz")


def test_failure_code(src, tmp_path):
    run = lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, None, None)
    with pytest.raises(ExportError, match="code 1"):
        MuseScoreExporter("M.exe", runner=run).export(src, tmp_path / "o.mscz")


@pytest.mark.parametrize("exc", [subprocess.TimeoutExpired("m", 1), OSError("x")])
def test_run_errors(src, tmp_path, exc):
    def boom(*a, **k):
        raise exc
    with pytest.raises(ExportError):
        MuseScoreExporter("M.exe", runner=boom).export(src, tmp_path / "o.mscz")
