"""Privacy-sanitized bug report collection, deterministic error fingerprinting, diagnostic ZIPs, and submission."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import APP_NAME, __version__
from .diagnostics import collect_environment_info

log = logging.getLogger(__name__)

GITHUB_REPO_URL = "https://github.com/tomermayost/pdf2musescore"
DEFAULT_LOG_TAIL_LINES = 80
MAX_CRASH_TAIL_LINES = 80
MAX_GITHUB_ISSUE_URL_CHARS = 6500

_WIN_USER_BS_RE = re.compile(r"(?i)[A-Za-z]:\\Users\\[^\\/:*?\"<>|\r\n]+")
_WIN_USER_FS_RE = re.compile(r"(?i)[A-Za-z]:/Users/[^\\/:*?\"<>|\r\n]+")
_POSIX_USER_RE = re.compile(r"/(?:Users|home)/[^/\s:\"<>|]+")
_CONVERTING_LINE_RE = re.compile(r"(Converting\s+).+?(\s+\(size=)")
_CONVERT_ARROW_RE = re.compile(r"(\)\s*->\s*).+?(\s+\(work dir\s+)")
_SELECTED_SOURCE_RE = re.compile(r"(Selected source file:\s+).+?(\s+\(\d+\s+page)")
_DROPPED_FILE_RE = re.compile(r"(Dropped file:\s+)[^\r\n]+")
_PATH_PDF_RE = re.compile(r"(?i)(?:[A-Za-z]:|<USER>)[\\/][^\r\n()|\"<>]+?\.pdf\b")
_BARE_PDF_RE = re.compile(r"(?i)[^\s\\/:\"*?<>|]+\.pdf\b")
_USER_MSCZ_RE = re.compile(r"(?i)<USER>[\\/][^\r\n()|\"<>]+?\.mscz\b")
_TB_FRAME_RE = re.compile(r'^\s*File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+([^\r\n]+)$', re.MULTILINE)
_FH_FRAME_RE = re.compile(r'^\s*File\s+"([^"]+)",\s+line\s+(\d+)\s+in\s+([^\r\n]+)$', re.MULTILINE)
_MS_EXPORT_DETAIL_RE = re.compile(r"MuseScore export failed\s*\(code\s*-?\d+\)\s*:\s*(.+)$", re.IGNORECASE)
_MS_ERR_CODE_RE = re.compile(r"\[(\d{3,5})\]")
_CRASH_HEADER_RE = re.compile(r"^===\s*(\S+)\s+pdf2musescore\s+\S+\s+pid=(\d+)\s*===\s*$")


class BugReportError(RuntimeError):
    """Raised when submitting or assembling a bug report fails."""


@dataclass(frozen=True)
class CrashInfo:
    """Represents a detected crash session from a prior run in ``crash.log``."""

    signature: str
    timestamp: str
    excerpt: str
    crash_log_path: Path


@dataclass(frozen=True)
class BugReportResult:
    """Result returned by the bug-report Worker endpoint."""

    issue_url: str
    issue_number: Optional[int] = None
    duplicate: bool = False
    message: str = ""


@dataclass(frozen=True)
class BugReport:
    """Privacy-sanitized diagnostic report ready for preview, Worker POST, GitHub URL, or ZIP export."""

    title: str
    error_type: str
    user_message: str
    error_detail: str
    stage: str
    mode: str
    fingerprint: str
    environment: dict[str, str]
    musescore_error: str = ""
    user_note: str = ""
    traceback: str = ""
    log_tail: str = ""
    crash_tail: str = ""
    staged_musicxml: Optional[Path] = None

    def with_user_note(self, note: str, *, extra_paths: Sequence[Path | str] = ()) -> "BugReport":
        """Return a copy with ``user_note`` attached and sanitized."""
        cleaned = redact_text(note or "", extra_paths=extra_paths).strip()
        return replace(self, user_note=cleaned)

    def to_markdown(self, *, include_logs: bool = True, max_log_chars: Optional[int] = None) -> str:
        """Format the report as clean Markdown suitable for preview, clipboard, or GitHub Issues."""
        env = self.environment
        app_ver = env.get("app", __version__)
        homr_ver = env.get("homr", "unknown")
        os_ver = env.get("os", "unknown")
        ram_info = env.get("memory", "unknown")
        edition = env.get("edition", "unknown")
        py_ver = env.get("python", "unknown")
        ort_prov = env.get("onnxruntime_providers", "unknown")

        lines: list[str] = [
            "### Summary",
            f"- **App:** pdf2musescore `{app_ver}` (homr `{homr_ver}`, edition `{edition}`)",
            f"- **OS:** {os_ver}",
            f"- **RAM:** {ram_info}",
            f"- **Python / ORT:** {py_ver} / {ort_prov}",
            f"- **Mode:** {self.mode}",
            f"- **Stage:** {self.stage or 'Unknown'}",
            f"- **Error Type:** `{self.error_type}`",
            f"- **Fingerprint:** `fp:{self.fingerprint}`",
            "",
            "### Error",
            self.user_message or "(No summary)",
        ]
        if self.error_detail:
            lines.extend(["", f"**Detail:** `{self.error_detail}`"])
        if self.musescore_error:
            lines.extend(["", f"**MuseScore 4 Log ERROR:** `{self.musescore_error}`"])

        if self.user_note:
            lines.extend(["", "### User Note", self.user_note])

        if self.traceback:
            lines.extend(["", "### Stack Trace", "```text", self.traceback.strip(), "```"])

        if include_logs:
            log_text = (self.log_tail or "").strip()
            crash_text = (self.crash_tail or "").strip()
            if max_log_chars is not None and max_log_chars >= 0:
                log_text = _tail_chars(log_text, max_log_chars)
                crash_text = _tail_chars(crash_text, max(0, max_log_chars // 2))
            if log_text:
                lines.extend(["", "### Recent Log (pdf2musescore.log)", "```text", log_text, "```"])
            if crash_text:
                lines.extend(["", "### Crash Log (crash.log)", "```text", crash_text, "```"])

        lines.extend(["", f"<!-- fp:{self.fingerprint} -->"])
        return "\n".join(lines).strip() + "\n"

    def to_payload(self) -> dict[str, Any]:
        """Build the JSON payload expected by the Cloudflare Worker endpoint."""
        env = self.environment
        return {
            "fingerprint": self.fingerprint,
            "title": self.title,
            "error_type": self.error_type,
            "stage": self.stage,
            "mode": self.mode,
            "app_version": env.get("app", __version__),
            "homr_version": env.get("homr", "unknown"),
            "os": env.get("os", "unknown"),
            "ram": env.get("memory", "unknown"),
            "edition": env.get("edition", "unknown"),
            "error_message": self.error_detail or self.user_message,
            "musescore_error": self.musescore_error,
            "user_note": self.user_note,
            "body": self.to_markdown(),
        }


def _tail_chars(text: str, limit: int) -> str:
    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text
    kept = text[-limit:]
    if "\n" in kept:
        kept = kept.split("\n", 1)[1]
    return "... (trimmed earlier lines)\n" + kept


def _placeholder_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "<PDF_FILE>"
    if ext == ".mscz":
        return "<OUTPUT_FILE>"
    return "<INPUT_FILE>"


def redact_text(
    text: str,
    *,
    extra_paths: Sequence[Path | str] = (),
    extra_names: Sequence[str] = (),
) -> str:
    """Scrub user directories and personal PDF/score filenames from ``text``."""
    if not text:
        return ""
    result = str(text)

    explicit_paths: list[tuple[str, str]] = []
    for raw in extra_paths:
        if raw is None:
            continue
        p = Path(raw)
        placeholder = _placeholder_for_path(p)
        for variant in {str(raw), str(p), p.as_posix(), os.path.normpath(str(p))}:
            if variant:
                explicit_paths.append((variant, placeholder))

    # Replace longest full paths first before scrubbing user home directories.
    explicit_paths.sort(key=lambda item: len(item[0]), reverse=True)
    for needle, repl in explicit_paths:
        result = re.sub(re.escape(needle), repl, result, flags=re.IGNORECASE)

    # Explicit home/profile paths first (handles custom non-C:\Users profiles too).
    home_candidates = {str(Path.home()), Path.home().as_posix()}
    for env_key in ("USERPROFILE", "HOME"):
        val = os.environ.get(env_key, "").strip()
        if val:
            home_candidates.add(val)
            home_candidates.add(Path(val).as_posix())
    for home_str in sorted((h for h in home_candidates if len(h) >= 3), key=len, reverse=True):
        result = re.sub(re.escape(home_str), "<USER>", result, flags=re.IGNORECASE)

    result = _WIN_USER_BS_RE.sub("<USER>", result)
    result = _WIN_USER_FS_RE.sub("<USER>", result)
    result = _POSIX_USER_RE.sub("<USER>", result)

    # Structured log lines containing arbitrary user paths (e.g. from older runs in pdf2musescore.log).
    result = _CONVERTING_LINE_RE.sub(r"\1<INPUT_FILE>\2", result)
    result = _CONVERT_ARROW_RE.sub(r"\1<OUTPUT_FILE>\2", result)
    result = _SELECTED_SOURCE_RE.sub(r"\1<INPUT_FILE>\2", result)
    result = _DROPPED_FILE_RE.sub(r"\1<INPUT_FILE>", result)

    explicit_names: list[tuple[str, str]] = []
    for raw in extra_paths:
        if raw is not None:
            p = Path(raw)
            if p.name:
                explicit_names.append((p.name, _placeholder_for_path(p)))
    for name in extra_names:
        cleaned_name = str(name or "").strip()
        if cleaned_name:
            explicit_names.append((cleaned_name, "<REDACTED>"))
    explicit_names.sort(key=lambda item: len(item[0]), reverse=True)
    for needle, repl in explicit_names:
        result = re.sub(re.escape(needle), repl, result, flags=re.IGNORECASE)

    result = _PATH_PDF_RE.sub("<PDF_FILE>", result)
    result = _BARE_PDF_RE.sub("<PDF_FILE>", result)
    result = _USER_MSCZ_RE.sub("<OUTPUT_FILE>", result)
    return result


def extract_musescore_error(detail: str) -> str:
    """Extract the MuseScore 4 log ERROR snippet from an ``ExportError`` message, if present."""
    text = (detail or "").strip()
    if not text:
        return ""
    match = _MS_EXPORT_DETAIL_RE.search(text)
    if match:
        return match.group(1).strip()
    return ""


def _normalize_frame_file(raw_path: str) -> str:
    norm = raw_path.replace("\\", "/")
    for marker in ("/pdf2musescore/", "/homr/"):
        idx = norm.lower().rfind(marker)
        if idx != -1:
            return norm[idx + 1 :]
    return Path(norm).name or norm


def compute_error_fingerprint(
    error_type: str,
    stage: str,
    traceback_text: str = "",
    error_message: str = "",
) -> str:
    """Compute a deterministic 16-hex-char fingerprint for deduplicating identical failures."""
    norm_type = (error_type or "Error").strip()
    norm_stage = re.sub(r"\b\d+\b", "<N>", (stage or "Unknown").strip())

    frames = _TB_FRAME_RE.findall(traceback_text or "")
    if not frames:
        frames = _FH_FRAME_RE.findall(traceback_text or "")

    if frames:
        sig_parts = [f"{_normalize_frame_file(f)}:{fn.strip()}" for f, _ln, fn in frames[-5:]]
        ms_codes = _MS_ERR_CODE_RE.findall(error_message or "")
        if ms_codes:
            sig_parts.append(f"ms:{ms_codes[0]}")
        signature = "|".join(sig_parts)
    else:
        clean_msg = redact_text(error_message or "")
        clean_msg = re.sub(r"0x[0-9a-fA-F]+", "<ADDR>", clean_msg)
        clean_msg = re.sub(r"\b\d+\b", "<N>", clean_msg)
        signature = clean_msg.strip()[:240]

    raw = f"{norm_type}|{norm_stage}|{signature}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def read_log_tail(
    log_path: Path,
    *,
    max_lines: int = DEFAULT_LOG_TAIL_LINES,
    extra_paths: Sequence[Path | str] = (),
) -> str:
    """Read and redact the last ``max_lines`` lines of ``log_path`` (empty string if missing)."""
    path = Path(log_path)
    if max_lines <= 0 or not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        log.warning("Could not read log file %s: %s", path, exc)
        return ""
    tail = "\n".join(lines[-max_lines:])
    return redact_text(tail, extra_paths=extra_paths)


def detect_previous_crash(
    log_dir: Path,
    *,
    dismissed_signature: str = "",
    current_pid: Optional[int] = None,
) -> Optional[CrashInfo]:
    """Detect whether ``crash.log`` contains an unacknowledged crash from a previous process."""
    crash_path = Path(log_dir) / "crash.log"
    if not crash_path.is_file():
        return None
    try:
        text = crash_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("Could not inspect %s: %s", crash_path, exc)
        return None

    blocks: list[tuple[str, Optional[int], list[str]]] = []
    cur_ts = ""
    cur_pid: Optional[int] = None
    cur_lines: list[str] = []

    for line in text.splitlines():
        m = _CRASH_HEADER_RE.match(line.strip())
        if m:
            if cur_ts or cur_lines:
                blocks.append((cur_ts, cur_pid, cur_lines))
            cur_ts = m.group(1)
            cur_pid = int(m.group(2))
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_ts or cur_lines:
        blocks.append((cur_ts, cur_pid, cur_lines))

    my_pid = os.getpid() if current_pid is None else int(current_pid)
    for ts, pid, body_lines in reversed(blocks):
        if pid is not None and pid == my_pid:
            continue
        non_empty = [ln for ln in body_lines if ln.strip()]
        if not non_empty:
            continue
        raw_block = f"{ts}|{pid}|" + "\n".join(non_empty)
        sig = hashlib.sha256(raw_block.encode("utf-8")).hexdigest()[:16]
        if dismissed_signature and sig == dismissed_signature:
            log.debug("Ignoring previously dismissed crash signature %s", sig)
            return None
        excerpt = redact_text("\n".join(non_empty[-MAX_CRASH_TAIL_LINES:]))
        log.info("Detected previous crash in %s (timestamp=%s, sig=%s)", crash_path, ts or "unknown", sig)
        return CrashInfo(
            signature=sig,
            timestamp=ts,
            excerpt=excerpt,
            crash_log_path=crash_path,
        )
    return None


def collect_bug_report(
    *,
    user_message: str,
    error_detail: str = "",
    error_type: str = "ConversionError",
    stage: str = "",
    mode: str = "Single Part",
    musescore_error: str = "",
    traceback_text: str = "",
    log_dir: Optional[Path] = None,
    source_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    staged_musicxml: Optional[Path] = None,
    user_note: str = "",
    tail_lines: int = DEFAULT_LOG_TAIL_LINES,
    env_info: Optional[dict[str, str]] = None,
) -> BugReport:
    """Assemble and sanitize a complete :class:`BugReport`."""
    if env_info is None:
        try:
            env = collect_environment_info()
        except Exception as exc:
            log.warning("Failed to collect environment info for bug report: %s", exc)
            env = {"app": __version__}
    else:
        env = dict(env_info)

    extra_paths = [p for p in (source_path, output_path) if p is not None]
    clean_user_msg = redact_text(user_message or "", extra_paths=extra_paths).strip()
    clean_detail = redact_text(error_detail or "", extra_paths=extra_paths).strip()
    raw_ms_err = musescore_error or extract_musescore_error(error_detail)
    clean_ms_err = redact_text(raw_ms_err or "", extra_paths=extra_paths).strip()
    clean_tb = redact_text(traceback_text or "", extra_paths=extra_paths).strip()
    clean_note = redact_text(user_note or "", extra_paths=extra_paths).strip()
    clean_stage = redact_text(stage or "", extra_paths=extra_paths).strip()

    log_tail = ""
    crash_tail = ""
    if log_dir is not None:
        ldir = Path(log_dir)
        log_tail = read_log_tail(ldir / f"{APP_NAME}.log", max_lines=tail_lines, extra_paths=extra_paths)
        crash_tail = read_log_tail(ldir / "crash.log", max_lines=tail_lines, extra_paths=extra_paths)

    fingerprint = compute_error_fingerprint(
        error_type=error_type,
        stage=clean_stage,
        traceback_text=clean_tb,
        error_message=clean_detail or clean_user_msg,
    )
    short_stage = (clean_stage or "Conversion").rstrip(".")
    if len(short_stage) > 55:
        short_stage = short_stage[:52] + "..."
    title = f"[Auto-Report] {error_type} at {short_stage} ({fingerprint[:8]})"

    resolved_xml = (
        Path(staged_musicxml)
        if staged_musicxml is not None and Path(staged_musicxml).is_file()
        else None
    )

    log.debug("Collected bug report (type=%s, stage=%r, fp=%s)", error_type, clean_stage, fingerprint)
    return BugReport(
        title=title,
        error_type=error_type,
        user_message=clean_user_msg,
        error_detail=clean_detail,
        stage=clean_stage,
        mode=mode,
        fingerprint=fingerprint,
        environment=env,
        musescore_error=clean_ms_err,
        user_note=clean_note,
        traceback=clean_tb,
        log_tail=log_tail,
        crash_tail=crash_tail,
        staged_musicxml=resolved_xml,
    )


def create_diagnostic_zip(
    dest_zip: Path,
    report: BugReport,
    *,
    log_dir: Optional[Path] = None,
    include_musicxml: bool = False,
    extra_paths: Sequence[Path | str] = (),
) -> Path:
    """Write an atomic diagnostic ``.zip`` with sanitized logs (and optional ``score.musicxml``).

    Never attaches the user's original PDF or image input.
    """
    target = Path(dest_zip)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=".diag.", suffix=".zip.tmp", dir=str(target.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)

        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("bug_report.md", report.to_markdown())

            if log_dir is not None:
                ldir = Path(log_dir)
                app_log = ldir / f"{APP_NAME}.log"
                if app_log.is_file():
                    raw_log = app_log.read_text(encoding="utf-8", errors="replace")
                    zf.writestr(f"{APP_NAME}.log", redact_text(raw_log, extra_paths=extra_paths))
                elif report.log_tail:
                    zf.writestr(f"{APP_NAME}.log", report.log_tail)

                crash_log = ldir / "crash.log"
                if crash_log.is_file():
                    raw_crash = crash_log.read_text(encoding="utf-8", errors="replace")
                    zf.writestr("crash.log", redact_text(raw_crash, extra_paths=extra_paths))
            else:
                if report.log_tail:
                    zf.writestr(f"{APP_NAME}.log", report.log_tail)
                if report.crash_tail:
                    zf.writestr("crash.log", report.crash_tail)

            if (
                include_musicxml
                and report.staged_musicxml is not None
                and report.staged_musicxml.is_file()
            ):
                zf.write(report.staged_musicxml, arcname="score.musicxml")

        os.replace(tmp_path, target)
        log.info("Wrote diagnostic archive %s (include_musicxml=%s)", target, include_musicxml)
        return target
    except Exception as exc:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise BugReportError(f"Could not create diagnostic ZIP: {exc}") from exc


def build_github_issue_url(
    report: BugReport,
    *,
    repo_url: str = GITHUB_REPO_URL,
    max_url_chars: int = MAX_GITHUB_ISSUE_URL_CHARS,
) -> str:
    """Build a pre-filled GitHub new-issue URL bounded to ``max_url_chars``."""
    base = f"{repo_url.rstrip('/')}/issues/new"
    for log_limit in (3500, 1800, 800, 0):
        include_logs = log_limit > 0
        body = report.to_markdown(include_logs=include_logs, max_log_chars=log_limit)
        url = f"{base}?{urllib.parse.urlencode({'title': report.title, 'body': body, 'labels': 'bug'})}"
        if len(url) <= max_url_chars:
            return url

    # Extreme fallback: trim body cleanly so browsers never reject the URL length.
    short_body = report.to_markdown(include_logs=False)[:1800]
    return f"{base}?{urllib.parse.urlencode({'title': report.title[:120], 'body': short_body, 'labels': 'bug'})}"


def submit_bug_report(
    report: BugReport,
    worker_url: str,
    *,
    timeout: float = 10.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> BugReportResult:
    """POST ``report`` as JSON to the configured Cloudflare Worker endpoint."""
    target = (worker_url or "").strip()
    parsed = urllib.parse.urlsplit(target)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise BugReportError("Bug report server URL is not configured or invalid.")

    body_bytes = json.dumps(report.to_payload(), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        target,
        data=body_bytes,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": f"pdf2musescore/{report.environment.get('app', __version__)}",
        },
    )

    log.info("Submitting bug report fp=%s to %s", report.fingerprint, target)
    try:
        with opener(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
            err_json = json.loads(err_body)
            detail = str(err_json.get("error") or err_json.get("message") or err_body)
        except Exception:
            detail = str(exc.reason or exc)
        log.error("Bug report HTTP error (%s): %s", exc.code, detail)
        raise BugReportError(f"Server returned HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        log.error("Bug report submission failed: %s", exc)
        raise BugReportError(f"Could not send bug report: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BugReportError(f"Invalid JSON response from bug report server: {exc}") from exc

    if not isinstance(data, dict):
        raise BugReportError("Unexpected response format from bug report server.")

    issue_url = str(data.get("issue_url") or data.get("url") or "").strip()
    if not issue_url:
        err = str(data.get("error") or "Server response did not include issue_url.")
        raise BugReportError(err)

    raw_num = data.get("issue_number")
    issue_number = int(raw_num) if isinstance(raw_num, (int, float)) or (isinstance(raw_num, str) and raw_num.isdigit()) else None
    duplicate = bool(data.get("duplicate", False))
    message = str(data.get("message") or "").strip()
    log.info("Bug report submitted: issue_url=%s duplicate=%s", issue_url, duplicate)
    return BugReportResult(
        issue_url=issue_url,
        issue_number=issue_number,
        duplicate=duplicate,
        message=message,
    )
