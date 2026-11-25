"""Unit tests for MSIX Store mode detection, version-scoped AI model cache, and packaging assets."""

from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from pdf2musescore import __version__, deps
from pdf2musescore.config import AppConfig, is_msix_package, load_config
from pdf2musescore.deps import MUSESCORE, ToolNotFoundError, ensure_tool
from pdf2musescore.engines.homr_engine import (
    HomrEngine,
    configure_model_cache,
    get_rapidocr_version,
    prune_stale_version_dirs,
)
from pdf2musescore.gui import App
from pdf2musescore.install_detect import MSIX_STORE_URI, MSIX_STORE_WEB_URL
from pdf2musescore.updater import UpdateInfo

ROOT = Path(__file__).resolve().parents[2]


def _load_generate_icons():
    spec = importlib.util.spec_from_file_location("generate_icons", ROOT / "scripts" / "generate_icons.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_is_msix_package_env_overrides() -> None:
    assert is_msix_package(environ={"PDF2MSCZ_STORE_MODE": "1"}) is True
    assert is_msix_package(environ={"PDF2MSCZ_STORE_MODE": "true"}) is True
    assert is_msix_package(environ={"PDF2MSCZ_STORE_MODE": "0"}) is False
    assert is_msix_package(environ={"PDF2MSCZ_STORE_MODE": "false"}) is False


def test_is_msix_package_windows_api_and_fallbacks(monkeypatch, caplog) -> None:
    # Invalid override logs warning and falls through to OS API
    with caplog.at_level("WARNING"):
        fake_kernel = SimpleNamespace(GetCurrentPackageFullName=lambda *a: 15700)
        monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=fake_kernel), raising=False)
        assert is_msix_package(environ={"PDF2MSCZ_STORE_MODE": "maybe"}) is False
    assert "Invalid PDF2MSCZ_STORE_MODE" in caplog.text

    for rc, expected in ((0, True), (122, True), (15700, False)):
        fake_kernel = SimpleNamespace(GetCurrentPackageFullName=lambda *a, code=rc: code)
        monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=fake_kernel), raising=False)
        assert is_msix_package(environ={}) is expected

    def _boom(*_a):
        raise OSError("no package api")

    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(GetCurrentPackageFullName=_boom)), raising=False)
    assert is_msix_package(environ={}) is False

    monkeypatch.setattr(sys, "platform", "linux")
    assert is_msix_package(environ={}) is False


def test_app_config_models_dir_default_and_env_override(tmp_path: Path) -> None:
    cfg_default = AppConfig()
    assert cfg_default.models_dir == Path("tmp") / "models"

    custom_models = tmp_path / "custom_models"
    cfg = load_config(tmp_path / "missing.toml", environ={"PDF2MSCZ_MODELS_DIR": str(custom_models)})
    assert cfg.models_dir == custom_models


def test_get_rapidocr_version_and_missing(monkeypatch) -> None:
    assert get_rapidocr_version() != "unknown"

    def _missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    assert get_rapidocr_version() == "unknown"


def test_configure_model_cache_redirects_and_triggers_redownload_on_upgrade(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    fake_seg_cfg = SimpleNamespace(
        segnet_path_onnx="C:/readonly/homr/segmentation/segnet_308.onnx",
        segnet_path_onnx_fp16="C:/readonly/homr/segmentation/segnet_308_fp16.onnx",
    )
    fake_main = SimpleNamespace(
        segnet_path_onnx=fake_seg_cfg.segnet_path_onnx,
        segnet_path_onnx_fp16=fake_seg_cfg.segnet_path_onnx_fp16,
    )
    fake_inf = SimpleNamespace(
        segnet_path_onnx=fake_seg_cfg.segnet_path_onnx,
        segnet_path_onnx_fp16=fake_seg_cfg.segnet_path_onnx_fp16,
    )
    fake_filepaths = SimpleNamespace(
        encoder_path="C:/readonly/homr/transformer/encoder.onnx",
        decoder_path="C:/readonly/homr/transformer/decoder.onnx",
        encoder_path_fp16="C:/readonly/homr/transformer/encoder_fp16.onnx",
        decoder_path_fp16="C:/readonly/homr/transformer/decoder_fp16.onnx",
        rhythmtokenizer="C:/readonly/homr/transformer/tokenizer_rhythm.json",
    )
    fake_tr_cfg = SimpleNamespace(default_config=SimpleNamespace(filepaths=fake_filepaths))

    # First version: 0.7.1 / 3.0.0
    homr_v1, ocr_v1 = configure_model_cache(
        models_dir,
        homr_version="0.7.1",
        rapidocr_version="3.0.0",
        homr_main=fake_main,
        homr_seg_cfg=fake_seg_cfg,
        homr_seg_inf=fake_inf,
        homr_tr_cfg=fake_tr_cfg,
    )
    assert homr_v1 == models_dir / "homr" / "0.7.1"
    assert ocr_v1 == models_dir / "rapidocr" / "3.0.0"
    assert Path(fake_seg_cfg.segnet_path_onnx) == homr_v1 / "segmentation" / "segnet_308.onnx"
    assert Path(fake_main.segnet_path_onnx) == homr_v1 / "segmentation" / "segnet_308.onnx"
    assert Path(fake_inf.segnet_path_onnx_fp16) == homr_v1 / "segmentation" / "segnet_308_fp16.onnx"
    assert Path(fake_filepaths.encoder_path) == homr_v1 / "transformer" / "encoder.onnx"
    assert Path(fake_filepaths.decoder_path) == homr_v1 / "transformer" / "decoder.onnx"
    # Tokenizer JSON paths remain in the bundled readonly directory
    assert fake_filepaths.rhythmtokenizer == "C:/readonly/homr/transformer/tokenizer_rhythm.json"

    # Simulate cached weights from the old engine version
    (homr_v1 / "segmentation" / "segnet_308.onnx").write_bytes(b"old-weights")
    (ocr_v1 / "det.onnx").write_bytes(b"old-ocr")

    # Upgrade to newer engine versions -> new directory doesn't have weights (triggers re-download)
    # and obsolete old version cache is pruned automatically
    homr_v2, ocr_v2 = configure_model_cache(
        models_dir,
        homr_version="0.8.0",
        rapidocr_version="3.1.0",
        homr_main=fake_main,
        homr_seg_cfg=fake_seg_cfg,
        homr_seg_inf=fake_inf,
        homr_tr_cfg=fake_tr_cfg,
    )
    assert homr_v2 == models_dir / "homr" / "0.8.0"
    assert ocr_v2 == models_dir / "rapidocr" / "3.1.0"
    assert not Path(fake_seg_cfg.segnet_path_onnx).exists()
    assert not homr_v1.exists()
    assert not ocr_v1.exists()


def test_configure_model_cache_patches_fresh_filepaths_instances(tmp_path: Path) -> None:
    class FakeFilePaths:
        def __init__(self) -> None:
            self.encoder_path = "C:/readonly/homr/transformer/encoder.onnx"
            self.decoder_path = "C:/readonly/homr/transformer/decoder.onnx"
            self.encoder_path_fp16 = "C:/readonly/homr/transformer/encoder_fp16.onnx"
            self.decoder_path_fp16 = "C:/readonly/homr/transformer/decoder_fp16.onnx"
            self.rhythmtokenizer = "C:/readonly/homr/transformer/tokenizer_rhythm.json"

    fake_seg_cfg = SimpleNamespace(
        segnet_path_onnx="C:/readonly/homr/segmentation/segnet.onnx",
        segnet_path_onnx_fp16="C:/readonly/homr/segmentation/segnet_fp16.onnx",
    )
    fake_main = SimpleNamespace(
        segnet_path_onnx=fake_seg_cfg.segnet_path_onnx,
        segnet_path_onnx_fp16=fake_seg_cfg.segnet_path_onnx_fp16,
    )
    fake_inf = SimpleNamespace(
        segnet_path_onnx=fake_seg_cfg.segnet_path_onnx,
        segnet_path_onnx_fp16=fake_seg_cfg.segnet_path_onnx_fp16,
    )
    fake_tr_cfg = SimpleNamespace(
        FilePaths=FakeFilePaths,
        default_config=SimpleNamespace(filepaths=FakeFilePaths()),
    )

    dir1 = tmp_path / "m1"
    dir2 = tmp_path / "m2"
    configure_model_cache(
        dir1,
        homr_version="0.7.1",
        rapidocr_version="3.10.0",
        homr_main=fake_main,
        homr_seg_cfg=fake_seg_cfg,
        homr_seg_inf=fake_inf,
        homr_tr_cfg=fake_tr_cfg,
    )
    first_wrapped = FakeFilePaths.__init__
    fp1 = FakeFilePaths()
    assert Path(fp1.encoder_path) == dir1 / "homr" / "0.7.1" / "transformer" / "encoder.onnx"
    assert fp1.rhythmtokenizer == "C:/readonly/homr/transformer/tokenizer_rhythm.json"

    # Second call updates class target dir idempotently without re-wrapping __init__
    configure_model_cache(
        dir2,
        homr_version="0.7.2",
        rapidocr_version="3.10.0",
        homr_main=fake_main,
        homr_seg_cfg=fake_seg_cfg,
        homr_seg_inf=fake_inf,
        homr_tr_cfg=fake_tr_cfg,
    )
    assert FakeFilePaths.__init__ is first_wrapped
    fp2 = FakeFilePaths()
    assert Path(fp2.decoder_path_fp16) == dir2 / "homr" / "0.7.2" / "transformer" / "decoder_fp16.onnx"

    # Real homr.transformer.configs.Config() created later also honors redirected model cache
    import homr.main as real_main
    import homr.segmentation.config as real_seg_cfg
    import homr.segmentation.inference_segnet as real_seg_inf
    import homr.transformer.configs as real_tr_cfg

    orig_seg = (
        real_seg_cfg.segnet_path_onnx,
        real_seg_cfg.segnet_path_onnx_fp16,
        real_main.segnet_path_onnx,
        real_main.segnet_path_onnx_fp16,
        real_seg_inf.segnet_path_onnx,
        real_seg_inf.segnet_path_onnx_fp16,
    )
    real_default_fp = real_tr_cfg.default_config.filepaths
    orig_tr_attrs = {
        a: getattr(real_default_fp, a)
        for a in ("encoder_path", "decoder_path", "encoder_path_fp16", "decoder_path_fp16")
    }
    orig_cls_tr_dir = getattr(real_tr_cfg.FilePaths, "_pdf2musescore_tr_dir", None)
    try:
        real_models = tmp_path / "real_models"
        hdir, _ = configure_model_cache(real_models)
        fresh_cfg = real_tr_cfg.Config()
        for attr in ("encoder_path", "decoder_path", "encoder_path_fp16", "decoder_path_fp16"):
            assert Path(getattr(fresh_cfg.filepaths, attr)).parent == hdir / "transformer"
        for tok_attr in ("rhythmtokenizer", "lifttokenizer", "pitchtokenizer", "notetokenizer"):
            assert Path(getattr(fresh_cfg.filepaths, tok_attr)).is_file()
    finally:
        (
            real_seg_cfg.segnet_path_onnx,
            real_seg_cfg.segnet_path_onnx_fp16,
            real_main.segnet_path_onnx,
            real_main.segnet_path_onnx_fp16,
            real_seg_inf.segnet_path_onnx,
            real_seg_inf.segnet_path_onnx_fp16,
        ) = orig_seg
        for attr, val in orig_tr_attrs.items():
            setattr(real_default_fp, attr, val)
        real_tr_cfg.FilePaths._pdf2musescore_tr_dir = orig_cls_tr_dir


def test_prune_stale_version_dirs_handles_missing_and_error(tmp_path: Path, monkeypatch) -> None:
    assert prune_stale_version_dirs(tmp_path / "missing", "0.7.1") == []

    root = tmp_path / "homr"
    keep = root / "0.8.0"
    old = root / "0.7.1"
    keep.mkdir(parents=True)
    old.mkdir(parents=True)
    (root / "stray.txt").write_text("keep file")

    monkeypatch.setattr("pdf2musescore.engines.homr_engine.shutil.rmtree", lambda p: (_ for _ in ()).throw(OSError("locked")))
    assert prune_stale_version_dirs(root, "0.8.0") == []
    assert old.is_dir()


def test_homr_engine_models_dir_attribute() -> None:
    eng = HomrEngine(models_dir=Path("tmp/models"))
    assert eng.models_dir == Path("tmp/models")


def test_ensure_tool_disables_winget_in_store_mode(monkeypatch) -> None:
    monkeypatch.setattr(deps.shutil, "which", lambda e: None)
    confirm_called = False

    def _confirm(_msg: str) -> bool:
        nonlocal confirm_called
        confirm_called = True
        return True

    with pytest.raises(ToolNotFoundError, match="musescore.org"):
        ensure_tool(MUSESCORE, confirm=_confirm, search_dirs=[], allow_winget=False)
    assert not confirm_called


def test_gui_store_mode_update_alerts_and_opens_store(root, tmp_path: Path) -> None:
    calls: list[str] = []
    app_disabled = App(
        root,
        AppConfig(work_dir=tmp_path / "w0", log_dir=tmp_path / "l0", check_updates=False),
        musescore_available=lambda: True,
        update_checker=lambda ver: calls.append(ver) or None,
        prefs_path=tmp_path / "prefs0.json",
        store_mode=True,
        prefetch_models=False,
    )
    try:
        root.update_idletasks()
        assert app_disabled._store_mode is True
        assert app_disabled.update_thread is None
        assert calls == []
        assert app_disabled.edition_badge_lbl is not None
        assert app_disabled.edition_badge_lbl.cget("text") == "Microsoft Store"
    finally:
        app_disabled.destroy()

    prompts: list[tuple[str, str]] = []
    opened_urls: list[str] = []
    downloads: list[object] = []
    launches: list[str] = []
    exits: list[bool] = []
    url_results = [True, False]

    def fake_open_url(url: str) -> bool:
        opened_urls.append(url)
        if url_results:
            val = url_results.pop(0)
            if isinstance(val, Exception):
                raise val
            return bool(val)
        return True

    info = UpdateInfo("9.9.9", "https://example.com/pdf2musescore-setup.exe", "https://example.com/rel")
    answers = [False, True]
    app = App(
        root,
        AppConfig(work_dir=tmp_path / "w1", log_dir=tmp_path / "l1", check_updates=True),
        musescore_available=lambda: True,
        update_checker=lambda ver: calls.append(ver) or info,
        ask_yes_no=lambda t, m: (prompts.append((t, m)) or answers.pop(0)),
        installer_downloader=lambda *a, **kw: downloads.append(a) or Path("never.exe"),
        launch_installer=lambda p: launches.append(p),
        open_url=fake_open_url,
        on_exit=lambda: exits.append(True),
        prefs_path=tmp_path / "prefs1.json",
        store_mode=True,
        prefetch_models=False,
    )
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not prompts:
            app.process_queue()
            root.update()
            time.sleep(0.02)
        assert len(calls) == 1
        assert app.update_btn.winfo_manager()
        assert "v9.9.9" in app.update_btn.cget("text")
        assert "Microsoft Store" in prompts[0][0]
        assert opened_urls == []

        # Clicking Update and confirming opens Microsoft Store deep link (never downloads .exe)
        app.on_update_clicked()
        assert opened_urls == [MSIX_STORE_URI]
        assert downloads == []
        assert launches == []
        assert exits == []

        # If ms-windows-store:// URI returns False or raises, falls back to Store web URL
        app.on_open_store_page()
        assert opened_urls[-2:] == [MSIX_STORE_URI, MSIX_STORE_WEB_URL]

        def failing_open_url(url: str) -> bool:
            opened_urls.append(url)
            if url == MSIX_STORE_URI:
                raise OSError("scheme blocked")
            return True

        app.open_url = failing_open_url
        app.on_open_store_page()
        assert opened_urls[-2:] == [MSIX_STORE_URI, MSIX_STORE_WEB_URL]
    finally:
        if app.update_thread is not None and app.update_thread.is_alive():
            app.update_thread.join(timeout=2)
        app.destroy()


def test_installer_iss_downgrade_and_dual_install_guards() -> None:
    iss_text = (ROOT / "packaging" / "installer.iss").read_text(encoding="utf-8")
    assert f'#define MyAppVersion "{__version__}"' in iss_text
    for token in (
        "InitializeSetup",
        "CompareVersionStrings",
        "GetStoreInstalledVersion",
        "GetStandaloneInstalledVersion",
        "CurUninstallStepChanged",
        "TomerMayost.PDF2MuseScore_",
        "{8D4F61C0-5A2E-4E91-9A23-7B8C1F42E101}_is1",
    ):
        assert token in iss_text

    # Models directory cleanup must be guarded in [Code], never unconditional in [UninstallDelete]
    uninstall_section = iss_text.split("[UninstallDelete]", 1)[1].split("[Code]", 1)[0]
    assert r"\models" not in uninstall_section

    code_section = iss_text.split("[Code]", 1)[1]
    begins = len(re.findall(r"\bbegin\b", code_section, flags=re.IGNORECASE))
    ends = len(re.findall(r"\bend\b", code_section, flags=re.IGNORECASE))
    assert begins > 0 and begins == ends


def test_pipeline_injects_models_dir_into_engine(tmp_path: Path) -> None:
    from pdf2musescore.pipeline import Pipeline
    from tests.unit.test_musicxml import score, write_mxl

    class FakeModelDirEngine:
        def __init__(self) -> None:
            self.models_dir = None

        def recognize(self, _pdf: Path, out_dir: Path) -> list[Path]:
            out_dir.mkdir(parents=True, exist_ok=True)
            return [write_mxl(score(1), out_dir / "x.mvt1.mxl")]

    eng = FakeModelDirEngine()
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"%PDF")
    models_root = tmp_path / "models"
    pipe = Pipeline(
        AppConfig(work_dir=tmp_path / "work", models_dir=models_root),
        ensure=lambda spec, cfg, confirm: Path("ms.exe"),
        engine_factory=lambda *a, **kw: eng,
        exporter_factory=lambda exe: SimpleNamespace(export=lambda s, d: Path(d).write_text("ok") or d),
    )
    out = pipe.convert(pdf, tmp_path / "out.musicxml")
    assert out.is_file()
    assert eng.models_dir == models_root


def test_generate_msix_assets_and_manifest_and_ci(tmp_path: Path) -> None:
    gi = _load_generate_icons()
    written = gi.generate_msix_assets(tmp_path / "msix_assets")
    expected_sizes = {
        "StoreLogo.png": (50, 50),
        "Square44x44Logo.png": (44, 44),
        "Square150x150Logo.png": (150, 150),
        "LargeTile.png": (310, 310),
        "Wide310x150Logo.png": (310, 150),
        "SplashScreen.png": (620, 300),
    }
    assert set(written) == set(expected_sizes)
    for name, size in expected_sizes.items():
        with Image.open(written[name]) as im:
            assert im.size == size

    # Verify committed MSIX assets match manifest references
    manifest_tpl = ROOT / "packaging" / "AppxManifest.xml.template"
    rendered_xml = manifest_tpl.read_text(encoding="utf-8").replace("__MSIX_VERSION__", "0.4.3.0")
    root = ET.fromstring(rendered_xml)
    ns = {
        "p": "http://schemas.microsoft.com/appx/manifest/foundation/windows10",
        "uap": "http://schemas.microsoft.com/appx/manifest/uap/windows10",
        "rescap": "http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities",
    }
    ident = root.find("p:Identity", ns)
    assert ident is not None
    assert ident.get("Name") == "TomerMayost.PDF2MuseScore"
    assert ident.get("Publisher") == "CN=BFE2770B-DDFB-4A4F-A48F-4968CE5B3B82"
    assert ident.get("Version") == "0.4.3.0"
    assert ident.get("ProcessorArchitecture") == "x64"

    assert root.findtext("p:Properties/p:DisplayName", default="", namespaces=ns) == "PDF2MuseScore"
    assert root.findtext("p:Properties/p:PublisherDisplayName", default="", namespaces=ns) == "Tomer Mayost"

    tdf = root.find("p:Dependencies/p:TargetDeviceFamily", ns)
    assert tdf is not None
    assert tdf.get("MaxVersionTested") == "10.0.26100.0"

    assert root.find("p:Capabilities/rescap:Capability[@Name='runFullTrust']", ns) is not None
    app_el = root.find("p:Applications/p:Application", ns)
    assert app_el is not None
    assert app_el.get("Executable") == "pdf2musescore.exe"
    assert app_el.get("EntryPoint") == "Windows.FullTrustApplication"

    for name, size in expected_sizes.items():
        committed = ROOT / "packaging" / "msix_assets" / name
        assert committed.is_file()
        with Image.open(committed) as im:
            assert im.size == size

    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    for token in (
        "build_msix.ps1",
        "msix-store-package",
        "pdf2musescore.msix",
        "microsoft/microsoft-store-apppublisher@v1.4",
        "9N6NCBWM3BHT",
    ):
        assert token in workflow
