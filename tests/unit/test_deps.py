import subprocess
from pathlib import Path

import pytest

from pdf2musescore import deps
from pdf2musescore.deps import (MUSESCORE, InstallError, ToolNotFoundError,
                                ensure_tool, find_tool, install_with_winget)


def _cp(code=0, out="", err=""):
    return subprocess.CompletedProcess([], code, out, err)


def _make_exe(base: Path, folder: str, exe: str, sub: str = "") -> Path:
    d = base / folder / sub if sub else base / folder
    d.mkdir(parents=True)
    p = d / exe
    p.write_text("x")
    return p


def test_find_configured(tmp_path):
    exe = tmp_path / "M.exe"
    exe.write_text("x")
    assert find_tool(MUSESCORE, exe, search_dirs=[]) == exe


def test_find_bad_configured_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)
    exe = _make_exe(tmp_path, "MuseScore 4", "MuseScore4.exe", "bin")
    assert find_tool(MUSESCORE, tmp_path / "nope.exe", search_dirs=[tmp_path]) == exe


def test_find_in_install_dir(tmp_path):
    exe = _make_exe(tmp_path, "MuseScore 4", "MuseScore4.exe", "bin")
    assert find_tool(MUSESCORE, search_dirs=[tmp_path]) == exe


def test_find_on_path(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: r"C:\x\MuseScore4.exe")
    assert find_tool(MUSESCORE, search_dirs=[]) == Path(r"C:\x\MuseScore4.exe")


def test_not_found(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)
    assert find_tool(MUSESCORE, search_dirs=[]) is None


def test_program_dirs(monkeypatch):
    monkeypatch.setenv("ProgramFiles", r"C:\PF")
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert deps._program_dirs() == [Path(r"C:\PF")]


def test_winget_success_and_failure():
    calls = []
    assert install_with_winget(MUSESCORE, lambda c, **k: calls.append(c) or _cp(0))
    assert "Musescore.Musescore" in calls[0]
    assert not install_with_winget(MUSESCORE, lambda c, **k: _cp(1, err="boom"))


def test_winget_missing():
    def raiser(*a, **k):
        raise FileNotFoundError
    assert not install_with_winget(MUSESCORE, raiser)


def test_ensure_found(tmp_path):
    exe = _make_exe(tmp_path, "MuseScore 4", "MuseScore4.exe")
    assert ensure_tool(MUSESCORE, search_dirs=[tmp_path]) == exe


def test_ensure_declined(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)
    with pytest.raises(ToolNotFoundError):
        ensure_tool(MUSESCORE, search_dirs=[])


def test_ensure_installs(tmp_path, monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)

    def runner(cmd, **k):
        _make_exe(tmp_path, "MuseScore 4", "MuseScore4.exe")
        return _cp(0)
    assert ensure_tool(MUSESCORE, confirm=lambda m: True, runner=runner,
                       search_dirs=[tmp_path]).name == "MuseScore4.exe"


def test_ensure_install_fails(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)
    with pytest.raises(InstallError):
        ensure_tool(MUSESCORE, confirm=lambda m: True, runner=lambda c, **k: _cp(1), search_dirs=[])


def test_ensure_installed_but_missing(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)
    with pytest.raises(InstallError):
        ensure_tool(MUSESCORE, confirm=lambda m: True, runner=lambda c, **k: _cp(0), search_dirs=[])
