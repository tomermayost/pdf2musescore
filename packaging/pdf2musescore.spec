# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).resolve().parent

datas = []
binaries = []
hiddenimports = ["pdf2musescore.gui"]

for pkg in ("homr", "musicxml", "rapidocr_onnxruntime", "pypdfium2", "pypdfium2_raw"):
    datas += collect_data_files(pkg, excludes=["*.onnx", "*.pth", "*.zip"])
    binaries += collect_dynamic_libs(pkg)
    hiddenimports += collect_submodules(pkg)

binaries += collect_dynamic_libs("onnxruntime")
hiddenimports += collect_submodules("onnxruntime")

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
