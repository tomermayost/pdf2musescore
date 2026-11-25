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
        MuseScoreExporter("M.exe", runner=run, musescore_log_dir=tmp_path).export(src, tmp_path / "o.mscz")


def test_failure_extracts_musescore_log_error(src, tmp_path):
    log_dir = tmp_path / "ms_logs"
    log_dir.mkdir()
    stale = log_dir / "MuseScore_260101_000000.log"
    stale.write_text("2026-01-01 | ERROR | main | err: stale log\n", encoding="utf-8")
    import os
    os.utime(stale, (1000.0, 1000.0))

    def run_with_log(cmd, **k):
        fresh = log_dir / "MuseScore_261125_222251.log"
        fresh.write_text(
            "2026-11-25 | ERROR | ConverterController::convertFile | failed load notation, "
            "err: [2009] <b>Incomplete measure</b>: measure 46, staff 18.\n"
            "2026-11-25 | ERROR | ConsoleApp | failed convert, error: [1320]\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 1320, "out line", "err line")

    exp = MuseScoreExporter("M.exe", runner=run_with_log, musescore_log_dir=log_dir)
    with pytest.raises(ExportError, match=r"code 1320.*\[2009\] Incomplete measure: measure 46, staff 18"):
        exp.export(src, tmp_path / "o.mscz")


@pytest.mark.parametrize("exc", [subprocess.TimeoutExpired("m", 1), OSError("x")])
def test_run_errors(src, tmp_path, exc):
    def boom(*a, **k):
        raise exc
    with pytest.raises(ExportError):
        MuseScoreExporter("M.exe", runner=boom).export(src, tmp_path / "o.mscz")
