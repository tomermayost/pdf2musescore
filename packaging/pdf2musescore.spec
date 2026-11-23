# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parent

datas = copy_metadata("homr") + copy_metadata("rapidocr") + [
    (str(ROOT / "src" / "pdf2musescore" / "assets"), "pdf2musescore/assets"),
]
binaries = []
hiddenimports = ["pdf2musescore.gui", "PIL.ImageTk", "PIL._tkinter_finder"]

for pkg in ("homr", "rapidocr", "pypdfium2", "pypdfium2_raw"):
    pkg_excludes = ["**/*.onnx", "**/*.pth", "**/*.zip"] if pkg in ("homr", "rapidocr") else []
    datas += collect_data_files(pkg, excludes=pkg_excludes)
    binaries += collect_dynamic_libs(pkg)
    hiddenimports += collect_submodules(pkg)

binaries += collect_dynamic_libs("onnxruntime")
hiddenimports += collect_submodules("onnxruntime")

# Bundle the latest VC++ 14.40+ redistributable DLLs from System32 so they override
# Python 3.10's older 14.29 msvcp140.dll (required by onnxruntime 1.23+ on Windows).
sys32 = Path(r"C:\Windows\System32")
vc_dll_names = {
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
}

a = Analysis(
    [str(ROOT / "src" / "pdf2musescore" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "torch", "torchvision"],
    noarchive=False,
)

# Strip any downloaded homr/rapidocr model weights (.onnx / .pth / .zip) from datas so
# the installer stays lightweight and downloads/updates models into the user cache.
a.datas = [
    d for d in a.datas
    if not (
        d[0].replace("/", "\\").lower().startswith(("homr\\", "rapidocr\\"))
        and d[0].lower().endswith((".onnx", ".pth", ".zip"))
    )
]

# Strip unused OpenCV video/FFmpeg DLL (~30 MB) — we only process still images.
a.binaries = [
    b for b in a.binaries
    if "opencv_videoio_ffmpeg" not in b[0].lower()
    and b[0].lower() not in vc_dll_names
]

for dll_name in sorted(vc_dll_names):
    dll_path = sys32 / dll_name
    if dll_path.is_file():
        a.binaries.append((dll_name, str(dll_path), "BINARY"))

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pdf2musescore",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=True,
    icon=str(ROOT / "src" / "pdf2musescore" / "assets" / "icon.ico"),
    version=str(ROOT / "packaging" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="pdf2musescore",
)
