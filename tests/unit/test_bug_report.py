"""Unit tests for privacy-sanitized bug reporting, fingerprinting, crash detection, ZIPs, and Worker submission."""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.parse
import zipfile
from pathlib import Path

import pytest

from pdf2musescore.bug_report import (
    BugReportError,
    BugReportResult,
    build_github_issue_url,
    collect_bug_report,
    compute_error_fingerprint,
    create_diagnostic_zip,
    detect_previous_crash,
    extract_musescore_error,
    read_log_tail,
    redact_text,
    submit_bug_report,
)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    def read(self, n: int = -1) -> bytes:
        return self._stream.read(n)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        return None


def test_redact_text_strips_user_paths_and_personal_filenames() -> None:
    pdf_path = Path(r"C:\Users\Alice Smith\Desktop\Private Divorce Agreement.pdf")
    out_path = Path(r"C:\Users\Alice Smith\Documents\Private Divorce Agreement.mscz")

    raw = (
        r"2026-11-25 INFO Converting C:\Users\Alice Smith\Desktop\Private Divorce Agreement.pdf "
        r"(size=123 bytes) -> C:\Users\Alice Smith\Documents\Private Divorce Agreement.mscz "
        r"(work dir C:\Users\Alice Smith\AppData\Local\pdf2musescore\work\job-1)" + "\n"
        r"Selected source file: C:\Users\Bob\Music\My Secret Sonata.pdf (4 pages)" + "\n"
        "Dropped file: /Users/charlie/Desktop/confidential_etude.pdf\n"
        "Crash inside /home/david/work/score.pdf and C:/Users/Eve/OneDrive/secret.pdf\n"
        "Standalone reference: unreleased_symphony.pdf"
    )
    redacted = redact_text(raw, extra_paths=[pdf_path, out_path], extra_names=["Alice Smith"])

    assert "Alice Smith" not in redacted
    assert "Bob" not in redacted
    assert "charlie" not in redacted
    assert "david" not in redacted
    assert "Eve" not in redacted
    assert "Private Divorce Agreement" not in redacted
    assert "My Secret Sonata" not in redacted
    assert "confidential_etude.pdf" not in redacted
    assert "unreleased_symphony.pdf" not in redacted
    assert "<USER>" in redacted
    assert "<PDF_FILE>" in redacted
    assert "pdf2musescore" in redacted


def test_compute_error_fingerprint_deterministic_and_distinct() -> None:
    tb_user_a = (
        "Traceback (most recent call last):\n"
        '  File "C:\\Users\\Alice\\AppData\\Local\\Temp\\_MEI111\\pdf2musescore\\pipeline.py", line 267, in convert\n'
        "    self.exporter_factory(musescore).export(xml, staged_mscz)\n"
        '  File "C:\\Users\\Alice\\AppData\\Local\\Temp\\_MEI111\\pdf2musescore\\exporters\\musescore.py", line 99, in export\n'
        "    raise ExportError(...)\n"
    )
    tb_user_b = (
        "Traceback (most recent call last):\n"
        '  File "D:\\Dev\\Bob\\src\\pdf2musescore\\pipeline.py", line 268, in convert\n'
        "    self.exporter_factory(musescore).export(xml, staged_mscz)\n"
        '  File "D:\\Dev\\Bob\\src\\pdf2musescore\\exporters\\musescore.py", line 101, in export\n'
        "    raise ExportError(...)\n"
    )

    fp_a = compute_error_fingerprint(
        "ExportError",
        "Reading page 2 of 15 with AI...",
        traceback_text=tb_user_a,
        error_message="MuseScore export failed (code 1320): err: [2009] Incomplete measure",
    )
    fp_b = compute_error_fingerprint(
        "ExportError",
        "Reading page 9 of 20 with AI...",
        traceback_text=tb_user_b,
        error_message="MuseScore export failed (code 1320): err: [2009] Incomplete measure",
    )
    assert len(fp_a) == 16
    assert fp_a == fp_b

    # Different error type, stage, or MuseScore code yields a distinct fingerprint
    fp_diff_type = compute_error_fingerprint("OmrError", "Reading page 2 of 15 with AI...", traceback_text=tb_user_a)
    fp_diff_code = compute_error_fingerprint(
        "ExportError",
        "Reading page 2 of 15 with AI...",
        traceback_text=tb_user_a,
        error_message="MuseScore export failed (code 1320): err: [2015] Bad tag",
    )
    assert fp_diff_type != fp_a
    assert fp_diff_code != fp_a

    # No-traceback fallback ignores hex addresses and digits
    fp_no_tb_1 = compute_error_fingerprint("RuntimeError", "Stage 1", error_message="Segfault at 0x00001234 line 10")
    fp_no_tb_2 = compute_error_fingerprint("RuntimeError", "Stage 2", error_message="Segfault at 0xDEADBEEF line 99")
    assert fp_no_tb_1 == fp_no_tb_2


def test_collect_bug_report_tails_logs_and_extracts_musescore_error(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    app_log = log_dir / "pdf2musescore.log"
    app_lines = [f"2026-11-25 INFO line {i} C:\\Users\\Tomer\\Secret\\Song.pdf" for i in range(120)]
    app_log.write_text("\n".join(app_lines), encoding="utf-8")

    crash_log = log_dir / "crash.log"
    crash_log.write_text(
        "=== 2026-11-25T21:00:00 pdf2musescore 0.4.7 pid=9999 ===\n"
        "Windows fatal exception: access violation\n",
        encoding="utf-8",
    )

    staged_xml = tmp_path / "score.musicxml"
    staged_xml.write_text("<score-partwise/>", encoding="utf-8")

    src_pdf = Path(r"C:\Users\Tomer\Music\My Concerto.pdf")
    dst_mscz = Path(r"C:\Users\Tomer\Music\My Concerto.mscz")
    detail = (
        "ExportError: MuseScore export failed (code 1320): "
        "failed load notation, err: [2009] Incomplete measure: measure 46, staff 18."
    )
    assert extract_musescore_error(detail) == "failed load notation, err: [2009] Incomplete measure: measure 46, staff 18."
    assert extract_musescore_error("plain error") == ""

    report = collect_bug_report(
        user_message="MuseScore could not save the file.",
        error_detail=detail,
        error_type="ExportError",
        stage="Saving MuseScore file...",
        mode="Full Score",
        log_dir=log_dir,
        source_path=src_pdf,
        output_path=dst_mscz,
        staged_musicxml=staged_xml,
        user_note=r"Happened on C:\Users\Tomer\Music\My Concerto.pdf page 4",
        tail_lines=80,
        env_info={
            "app": "0.4.7",
            "homr": "0.7.1",
            "os": "Windows-11-10.0.26100",
            "memory": "RAM available 8.0 GB of 16.0 GB (50% in use)",
            "edition": "standalone",
            "python": "3.12.10",
            "onnxruntime_providers": "CPUExecutionProvider",
        },
    )

    assert report.mode == "Full Score"
    assert "[2009] Incomplete measure" in report.musescore_error
    assert report.staged_musicxml == staged_xml
    assert len(report.log_tail.splitlines()) == 80
    assert "line 40" in report.log_tail and "line 39" not in report.log_tail
    assert "Tomer" not in report.log_tail
    assert "My Concerto" not in report.user_note
    assert "access violation" in report.crash_tail

    md = report.to_markdown()
    assert "pdf2musescore `0.4.7`" in md
    assert "homr `0.7.1`" in md
    assert "Full Score" in md
    assert f"fp:{report.fingerprint}" in md
    assert "Tomer" not in md

    payload = report.to_payload()
    assert payload["fingerprint"] == report.fingerprint
    assert payload["app_version"] == "0.4.7"
    assert payload["mode"] == "Full Score"


def test_create_diagnostic_zip_never_includes_pdf_and_optional_musicxml(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "pdf2musescore.log").write_text(
        "INFO Converting C:\\Users\\Alice\\Desktop\\Private.pdf\n", encoding="utf-8"
    )
    (log_dir / "crash.log").write_text("Fatal error in C:\\Users\\Alice\\foo.py\n", encoding="utf-8")

    staged_xml = tmp_path / "work" / "score.musicxml"
    staged_xml.parent.mkdir()
    staged_xml.write_text("<score-partwise><part id='P1'/></score-partwise>", encoding="utf-8")

    src_pdf = tmp_path / "Private.pdf"
    src_pdf.write_bytes(b"%PDF-1.4")

    report = collect_bug_report(
        user_message="Export failed",
        error_detail="ExportError: boom",
        error_type="ExportError",
        stage="Saving MuseScore file...",
        log_dir=log_dir,
        source_path=src_pdf,
        staged_musicxml=staged_xml,
    )

    zip_no_xml = tmp_path / "out" / "diag_no_xml.zip"
    create_diagnostic_zip(
        zip_no_xml,
        report,
        log_dir=log_dir,
        include_musicxml=False,
        extra_paths=[src_pdf],
    )
    with zipfile.ZipFile(zip_no_xml) as zf:
        names = set(zf.namelist())
        assert names == {"bug_report.md", "pdf2musescore.log", "crash.log"}
        assert all(not n.lower().endswith(".pdf") for n in names)
        log_content = zf.read("pdf2musescore.log").decode("utf-8")
        assert "Alice" not in log_content
        assert "Private.pdf" not in log_content

    zip_with_xml = tmp_path / "out" / "diag_with_xml.zip"
    create_diagnostic_zip(
        zip_with_xml,
        report,
        log_dir=log_dir,
        include_musicxml=True,
        extra_paths=[src_pdf],
    )
    with zipfile.ZipFile(zip_with_xml) as zf:
        names = set(zf.namelist())
        assert names == {"bug_report.md", "pdf2musescore.log", "crash.log", "score.musicxml"}
        assert "<score-partwise>" in zf.read("score.musicxml").decode("utf-8")

    # Failure when dest_zip points at a directory raises BugReportError cleanly without leaving .tmp
    dir_target = tmp_path / "existing_dir"
    dir_target.mkdir()
    with pytest.raises(BugReportError, match="Could not create diagnostic ZIP"):
        create_diagnostic_zip(dir_target, report)
    assert not list(tmp_path.glob("*.zip.tmp"))


def test_build_github_issue_url_respects_length_limit(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    huge_lines = [f"2026-11-25 INFO {'X' * 200} line {i}" for i in range(80)]
    (log_dir / "pdf2musescore.log").write_text("\n".join(huge_lines), encoding="utf-8")

    report = collect_bug_report(
        user_message="OMR failed",
        error_detail="OmrError: could not read",
        error_type="OmrError",
        stage="Reading the music with AI...",
        log_dir=log_dir,
    )
    url = build_github_issue_url(report, max_url_chars=6500)
    assert url.startswith("https://github.com/tomermayost/pdf2musescore/issues/new?")
    assert len(url) <= 6500
    parsed = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert parsed["labels"] == ["bug"]
    assert report.fingerprint in parsed["body"][0]

    # Even with ultra-tight max_url_chars, fallback keeps URL bounded
    tiny_url = build_github_issue_url(report, max_url_chars=500)
    assert "issues/new?" in tiny_url
    assert len(tiny_url) < 4000


def test_submit_bug_report_success_duplicate_and_errors() -> None:
    report = collect_bug_report(user_message="Export failed", error_type="ExportError", stage="Export")

    with pytest.raises(BugReportError, match="not configured"):
        submit_bug_report(report, "")
    with pytest.raises(BugReportError, match="not configured"):
        submit_bug_report(report, "ftp://bad-scheme.example")

    captured: dict[str, object] = {}

    def ok_opener(req, timeout=10.0):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            json.dumps(
                {
                    "ok": True,
                    "duplicate": True,
                    "issue_number": 42,
                    "issue_url": "https://github.com/tomermayost/pdf2musescore/issues/42",
                }
            ).encode("utf-8")
        )

    res = submit_bug_report(report, "https://bugs.example.workers.dev/report", opener=ok_opener)
    assert res == BugReportResult(
        issue_url="https://github.com/tomermayost/pdf2musescore/issues/42",
        issue_number=42,
        duplicate=True,
        message="",
    )
    assert captured["method"] == "POST"
    assert captured["payload"]["fingerprint"] == report.fingerprint

    # HTTP 429 rate-limit response surfaces server message
    def rate_limit_opener(req, timeout=10.0):
        raise urllib.error.HTTPError(
            req.full_url,
            429,
            "Too Many Requests",
            {},
            io.BytesIO(json.dumps({"error": "Too many reports, try again later"}).encode("utf-8")),
        )

    with pytest.raises(BugReportError, match="HTTP 429.*Too many reports"):
        submit_bug_report(report, "https://bugs.example.workers.dev/report", opener=rate_limit_opener)

    # Network error / malformed JSON / missing issue_url raise BugReportError
    with pytest.raises(BugReportError, match="Could not send"):
        submit_bug_report(
            report,
            "https://bugs.example.workers.dev/report",
            opener=lambda _r, timeout=10.0: (_ for _ in ()).throw(TimeoutError("timed out")),
        )
    with pytest.raises(BugReportError, match="Invalid JSON"):
        submit_bug_report(
            report,
            "https://bugs.example.workers.dev/report",
            opener=lambda _r, timeout=10.0: _FakeResponse(b"not json"),
        )
    with pytest.raises(BugReportError, match="did not include issue_url"):
        submit_bug_report(
            report,
            "https://bugs.example.workers.dev/report",
            opener=lambda _r, timeout=10.0: _FakeResponse(b"{}"),
        )


def test_detect_previous_crash_handles_clean_active_dismissed_and_crashed(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Missing crash.log -> None
    assert detect_previous_crash(log_dir) is None
    assert read_log_tail(log_dir / "missing.log") == ""

    crash_log = log_dir / "crash.log"
    my_pid = os.getpid()

    # Header-only clean runs -> None
    crash_log.write_text(
        f"\n=== 2026-11-25T20:00:00 pdf2musescore 0.4.7 pid=11111 ===\n"
        f"\n=== 2026-11-25T21:00:00 pdf2musescore 0.4.7 pid={my_pid} ===\n",
        encoding="utf-8",
    )
    assert detect_previous_crash(log_dir) is None

    # Prior session crash followed by current clean startup header -> detects prior crash, redacts path
    crash_log.write_text(
        "\n=== 2026-11-25T20:00:00 pdf2musescore 0.4.7 pid=11111 ===\n"
        "Windows fatal exception: access violation\n"
        '  File "C:\\Users\\Alice\\src\\pdf2musescore\\pipeline.py", line 254 in convert\n'
        f"\n=== 2026-11-25T21:00:00 pdf2musescore 0.4.7 pid={my_pid} ===\n",
        encoding="utf-8",
    )
    crash = detect_previous_crash(log_dir)
    assert crash is not None
    assert crash.timestamp == "2026-11-25T20:00:00"
    assert "access violation" in crash.excerpt
    assert "Alice" not in crash.excerpt
    assert "<USER>" in crash.excerpt

    # Dismissed signature suppresses repeat banner
    assert detect_previous_crash(log_dir, dismissed_signature=crash.signature) is None
