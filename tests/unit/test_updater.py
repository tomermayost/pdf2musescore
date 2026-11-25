import io
import json
from pathlib import Path

import pytest

from pdf2musescore.updater import (
    UpdateError,
    UpdateInfo,
    check_for_update,
    download_installer,
    is_newer_version,
    parse_version,
)


class FakeResponse:
    def __init__(self, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self._stream = io.BytesIO(payload)
        self.headers = headers or {}

    def read(self, n: int = -1) -> bytes:
        return self._stream.read(n)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        return None


def test_parse_and_compare_versions():
    assert parse_version("v0.3.0") == (0, 3, 0)
    assert parse_version("0.3.10-beta") == (0, 3, 10)
    assert parse_version("invalid") == (0,)
    assert is_newer_version("v0.3.1", "0.3.0") is True
    assert is_newer_version("v0.3.10", "0.3.2") is True
    assert is_newer_version("v0.3.0", "0.3.0") is False
    assert is_newer_version("v0.3", "0.3.0") is False
    assert is_newer_version("v0.2.9", "0.3.0") is False


def test_check_for_update_variants(caplog):
    def make_opener(payload_obj):
        raw = json.dumps(payload_obj).encode("utf-8")
        return lambda req, timeout=5.0: FakeResponse(raw)

    # Exact pdf2musescore-setup.exe match
    info = check_for_update(
        "0.3.0",
        opener=make_opener({
            "tag_name": "v0.4.0",
            "html_url": "https://example.com/rel",
            "body": "Release notes",
            "assets": [
                {"name": "other.zip", "browser_download_url": "https://example.com/other.zip"},
                {"name": "pdf2musescore-setup.exe", "browser_download_url": "https://example.com/setup.exe"},
            ],
        }),
    )
    assert info == UpdateInfo(
        version="0.4.0",
        download_url="https://example.com/setup.exe",
        release_url="https://example.com/rel",
        notes="Release notes",
    )

    # Versioned setup installer name preferred over other .exe assets
    info_ver = check_for_update(
        "0.4.7",
        opener=make_opener({
            "tag_name": "v0.4.8",
            "html_url": "https://example.com/rel48",
            "assets": [
                {"name": "helper.exe", "browser_download_url": "https://example.com/helper.exe"},
                {
                    "name": "pdf2musescore-0.4.8-setup-x64.exe",
                    "browser_download_url": "https://example.com/v048-setup.exe",
                },
            ],
        }),
    )
    assert info_ver is not None and info_ver.download_url == "https://example.com/v048-setup.exe"

    # Generic *setup*.exe preferred over unrelated .exe assets
    info_setup = check_for_update(
        "0.4.7",
        opener=make_opener({
            "tag_name": "v0.5.0",
            "html_url": "https://example.com/rel50",
            "assets": [
                {"name": "helper.exe", "browser_download_url": "https://example.com/helper.exe"},
                {"name": "pdf2musescore-v0.5.0-setup.exe", "browser_download_url": "https://example.com/setup50.exe"},
            ],
        }),
    )
    assert info_setup is not None and info_setup.download_url == "https://example.com/setup50.exe"

    # Fallback to any .exe asset
    info2 = check_for_update(
        "0.3.0",
        opener=make_opener({
            "tag_name": "v0.4.0",
            "html_url": "https://example.com/rel",
            "assets": [{"name": "installer-win64.exe", "browser_download_url": "https://example.com/w64.exe"}],
        }),
    )
    assert info2.download_url == "https://example.com/w64.exe"

    # Fallback to release page URL when no .exe asset exists
    info3 = check_for_update(
        "0.3.0",
        opener=make_opener({
            "tag_name": "v0.4.0",
            "html_url": "https://example.com/rel",
            "assets": [],
        }),
    )
    assert info3.download_url == "https://example.com/rel"

    # Same / older version returns None
    assert check_for_update("0.3.0", opener=make_opener({"tag_name": "v0.3.0"})) is None
    assert check_for_update("0.3.0", opener=make_opener({"tag_name": ""})) is None

    # Offline / network failure logs warning and returns None
    def offline(req, timeout=5.0):
        raise OSError("network unreachable")

    with caplog.at_level("WARNING"):
        assert check_for_update("0.3.0", opener=offline) is None
    assert "continuing offline" in caplog.text


def test_download_installer(tmp_path):
    info = UpdateInfo("0.4.0", "https://example.com/setup.exe", "https://example.com/rel")
    steps = []
    out = download_installer(
        info,
        tmp_path / "updates",
        progress=lambda p, m: steps.append(p),
        opener=lambda req, timeout=60.0: FakeResponse(b"MZ" * 100, {"Content-Length": "200"}),
    )
    assert out == tmp_path / "updates" / "pdf2musescore-setup-0.4.0.exe"
    assert out.read_bytes() == b"MZ" * 100
    assert steps and steps[-1] <= 100

    # Without Content-Length still works cleanly
    out2 = download_installer(
        info,
        tmp_path / "updates2",
        progress=lambda p, m: None,
        opener=lambda req, timeout=60.0: FakeResponse(b"MZ" * 50),
    )
    assert out2.is_file()

    # Empty response raises UpdateError and leaves no .part file
    with pytest.raises(UpdateError, match="empty"):
        download_installer(info, tmp_path / "empty", opener=lambda req, timeout=60.0: FakeResponse(b""))
    assert not list((tmp_path / "empty").glob("*.part"))

    # Network error cleans .part and raises UpdateError
    def fail(req, timeout=60.0):
        raise TimeoutError("timed out")

    with pytest.raises(UpdateError, match="Failed to download"):
        download_installer(info, tmp_path / "fail", opener=fail)
    assert not list((tmp_path / "fail").glob("*.part"))
