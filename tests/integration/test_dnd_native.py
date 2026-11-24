"""Subprocess integration tests exercising real Win32 WM_DROPFILES drag-and-drop."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PDF = ROOT / "tests" / "fixtures" / "bach_cello_suite_2.pdf"

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only native WM_DROPFILES test")


def _run_child(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_native_drop_handler_survives_multiple_real_drops(tmp_path: Path) -> None:
    sample = tmp_path / "score.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    script = f"""
import faulthandler, sys, tkinter as tk
from pathlib import Path
faulthandler.enable()
sys.path.insert(0, {str(ROOT / "src")!r})
sys.path.insert(0, {str(ROOT)!r})
from pdf2musescore.win_dnd import install_file_drop_handler
from tests.helpers.win_drop import post_fake_drop

root = tk.Tk()
delivered = []
def on_files(paths):
    delivered.append([str(p) for p in paths])
    print(f"DROP_OK:{{len(delivered)}}", flush=True)

handle = install_file_drop_handler(root, on_files)
assert handle is not None, "drop handler failed to install"

target = {str(sample)!r}
def send_drop(n=[0]):
    n[0] += 1
    post_fake_drop(handle.hwnd, [target])
    if n[0] < 5:
        root.after(200, send_drop)
    else:
        root.after(400, lambda: (handle.uninstall(), root.destroy()))

root.after(150, send_drop)
root.mainloop()
assert len(delivered) == 5, f"expected 5 drops, got {{len(delivered)}}"
print("ALL_DROPS_OK", flush=True)
"""
    res = _run_child(script)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "Fatal Python error" not in res.stderr
    assert res.stdout.count("DROP_OK:") == 5
    assert "ALL_DROPS_OK" in res.stdout


def test_full_app_survives_real_wm_dropfiles_png_and_pdf(tmp_path: Path) -> None:
    png_path = tmp_path / "sheet.png"
    Image.new("RGB", (120, 160), (255, 255, 255)).save(png_path, format="PNG")
    log_dir = tmp_path / "logs"
    work_dir = tmp_path / "work"
    models_dir = tmp_path / "models"
    prefs_path = tmp_path / "prefs.json"

    script = f"""
import faulthandler, sys, tkinter as tk
from pathlib import Path
faulthandler.enable()
sys.path.insert(0, {str(ROOT / "src")!r})
sys.path.insert(0, {str(ROOT)!r})
from pdf2musescore.config import AppConfig
from pdf2musescore.gui import App
from tests.helpers.win_drop import post_fake_drop

root = tk.Tk()
cfg = AppConfig(
    log_dir=Path({str(log_dir)!r}),
    work_dir=Path({str(work_dir)!r}),
    models_dir=Path({str(models_dir)!r}),
    check_updates=False,
)
errors = []
app = App(
    root,
    cfg,
    musescore_available=lambda: True,
    prefs_path=Path({str(prefs_path)!r}),
    enable_dnd=True,
    prefetch_models=False,
    show_error=lambda t, m: errors.append((t, m)),
)
assert app._drop_handle is not None, "App did not install drop handler"
hwnd = app._drop_handle.hwnd

def drop_png():
    post_fake_drop(hwnd, [{str(png_path)!r}])

def check_png_and_drop_pdf():
    print(f"AFTER_PNG:{{app.pdf}}|{{app.total_pages}}", flush=True)
    post_fake_drop(hwnd, [{str(FIXTURE_PDF)!r}])

def finish():
    print(f"AFTER_PDF:{{app.pdf}}|{{app.total_pages}}", flush=True)
    print(f"ERRORS:{{errors}}", flush=True)
    app.destroy()
    root.destroy()

root.after(150, drop_png)
root.after(550, check_png_and_drop_pdf)
root.after(1050, finish)
root.mainloop()
"""
    res = _run_child(script)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "Fatal Python error" not in res.stderr
    assert f"AFTER_PNG:{png_path}|1" in res.stdout
    assert f"AFTER_PDF:{FIXTURE_PDF}|4" in res.stdout
    assert "ERRORS:[]" in res.stdout
