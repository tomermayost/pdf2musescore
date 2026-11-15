import subprocess
from pathlib import Path

import pytest

from pdf2musescore import deps
from pdf2musescore.deps import (AUDIVERIS, MUSESCORE, InstallError, ToolNotFoundError,
                                ensure_tool, find_tool, install_from_github, install_with_winget)


def _cp(code=0, out="", err=""):
    return subprocess.CompletedProcess([], code, out, err)


def _make_exe(base: Path, folder: str, exe: str, sub: str = "") -> Path:
    d = base / folder / sub if sub else base / folder
    d.mkdir(parents=True)
    p = d / exe
    p.write_text("x")
    return p


def test_find_configured(tmp_path):
    exe = tmp_path / "A.exe"
    exe.write_text("x")
    assert find_tool(AUDIVERIS, exe, search_dirs=[]) == exe


def test_find_bad_configured_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)
    exe = _make_exe(tmp_path, "Audiveris", "Audiveris.exe", "bin")
    assert find_tool(AUDIVERIS, tmp_path / "nope.exe", search_dirs=[tmp_path]) == exe


def test_find_in_install_dir(tmp_path):
    exe = _make_exe(tmp_path, "MuseScore 4", "MuseScore4.exe", "bin")
    assert find_tool(MUSESCORE, search_dirs=[tmp_path]) == exe


def test_find_on_path(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: r"C:\x\Audiveris.exe")
    assert find_tool(AUDIVERIS, search_dirs=[]) == Path(r"C:\x\Audiveris.exe")


def test_not_found(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)
    assert find_tool(AUDIVERIS, search_dirs=[]) is None


def test_program_dirs(monkeypatch):
    monkeypatch.setenv("ProgramFiles", r"C:\PF")
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert deps._program_dirs() == [Path(r"C:\PF")]


def test_winget_success_and_failure():
    calls = []
    assert install_with_winget(AUDIVERIS, lambda c, **k: calls.append(c) or _cp(0))
    assert "audiveris.org.Audiveris" in calls[0]
    assert not install_with_winget(AUDIVERIS, lambda c, **k: _cp(1, err="boom"))


def test_winget_missing():
    def raiser(*a, **k):
        raise FileNotFoundError
    assert not install_with_winget(AUDIVERIS, raiser)


class _Resp:
    def __init__(self, payload):
        import io, json
        self._b = io.BytesIO(json.dumps(payload).encode())

    def read(self, *a):
        return self._b.read(*a)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_github_install_msi():
    payload = {"assets": [{"name": "src.zip", "browser_download_url": "u0"},
                          {"name": "Audiveris-5.11-windows-x86_64.msi", "browser_download_url": "http://h/a.msi"}]}
    calls, downloads = [], []
    ok = install_from_github(AUDIVERIS, runner=lambda c, **k: calls.append(c) or _cp(0),
                             opener=lambda u, timeout: _Resp(payload),
                             download=lambda u, t: downloads.append(u))
    assert ok and downloads == ["http://h/a.msi"] and calls[0][0] == "msiexec"


def test_github_install_exe_fails():
    payload = {"assets": [{"name": "setup-win.exe", "browser_download_url": "http://h/setup-win.exe"}]}
    ok = install_from_github(AUDIVERIS, runner=lambda c, **k: _cp(3),
                             opener=lambda u, timeout: _Resp(payload), download=lambda u, t: None)
    assert not ok


def test_github_no_asset_or_no_repo():
    assert not install_from_github(AUDIVERIS, opener=lambda u, timeout: _Resp({"assets": []}),
                                   download=lambda u, t: None)
    assert not install_from_github(MUSESCORE)


def test_ensure_found(tmp_path):
    exe = _make_exe(tmp_path, "Audiveris", "Audiveris.exe")
    assert ensure_tool(AUDIVERIS, search_dirs=[tmp_path]) == exe


def test_ensure_declined(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)
    with pytest.raises(ToolNotFoundError):
        ensure_tool(AUDIVERIS, search_dirs=[])


def test_ensure_installs(tmp_path, monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)

    def runner(cmd, **k):
        _make_exe(tmp_path, "Audiveris", "Audiveris.exe")
        return _cp(0)
    assert ensure_tool(AUDIVERIS, confirm=lambda m: True, runner=runner,
                       search_dirs=[tmp_path]).name == "Audiveris.exe"


def test_ensure_install_fails(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)
    monkeypatch.setattr(deps, "install_from_github", lambda s, r: False)
    with pytest.raises(InstallError):
        ensure_tool(AUDIVERIS, confirm=lambda m: True, runner=lambda c, **k: _cp(1), search_dirs=[])


def test_ensure_installed_but_missing(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)
    with pytest.raises(InstallError):
        ensure_tool(AUDIVERIS, confirm=lambda m: True, runner=lambda c, **k: _cp(0), search_dirs=[])
