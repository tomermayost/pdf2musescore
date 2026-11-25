"""Application entry point."""

from __future__ import annotations

import logging
import sys

from pdf2musescore import AUTHOR_CREDIT, __version__
from pdf2musescore.config import AppConfig, ConfigError, is_msix_package, load_config
from pdf2musescore.diagnostics import install_crash_handlers, install_stdio_capture, log_environment
from pdf2musescore.engines.homr_engine import CancelledError, get_engine_version, get_rapidocr_version
from pdf2musescore.logging_setup import setup_logging
from pdf2musescore.model_store import ModelManager


def _run_download_models(config: AppConfig, log: logging.Logger, *, no_ui: bool) -> int:
    try:
        manager = ModelManager(config.models_dir)
        st = manager.status()
        if st.ready:
            log.info("AI models already present and verified (%s, %s)", st.homr_dir, st.ocr_dir)
            return 0
        if no_ui:
            last_bucket = [-1]

            def _log_progress(done_b: int, total_b: int) -> None:
                pct = min(100, int(100 * done_b / max(total_b, 1)))
                bucket = pct // 10
                if bucket != last_bucket[0]:
                    last_bucket[0] = bucket
                    log.info(
                        "Downloading AI models: %.0f / ~%.0f MB (%d%%)",
                        done_b / 1e6,
                        max(total_b, 1) / 1e6,
                        pct,
                    )

            manager.ensure(progress=_log_progress)
            return 0
        from pdf2musescore.ui_download import run_download_window

        return int(run_download_window(manager))
    except CancelledError:
        log.info("AI model download cancelled by user")
        return 2
    except Exception:
        log.exception("AI model download failed")
        return 1


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if "--version" in args:
        if sys.stdout is not None:
            print(f"pdf2musescore {__version__} (homr {get_engine_version()}) ({AUTHOR_CREDIT})")
        return 0
    if "--self-test" in args:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import cv2  # noqa: F401
        import homr.main  # noqa: F401
        import homr.segmentation.config as homr_seg_cfg
        import homr.segmentation.inference_segnet as homr_seg_inf
        import homr.transformer.configs as homr_tr_cfg
        import onnxruntime as ort  # noqa: F401
        import pypdfium2  # noqa: F401
        import rapidocr  # noqa: F401
        from PIL import ImageTk  # noqa: F401

        from pdf2musescore.engines.homr_engine import configure_model_cache
        from pdf2musescore.ui_theme import get_assets_dir

        assets = get_assets_dir()
        assert (assets / "icon.ico").is_file()
        assert (assets / "icon.png").is_file()
        assert get_engine_version() != "unknown"
        assert get_rapidocr_version() != "unknown"
        assert ort.get_available_providers()

        orig_seg_paths = (
            homr_seg_cfg.segnet_path_onnx,
            homr_seg_cfg.segnet_path_onnx_fp16,
            homr.main.segnet_path_onnx,
            homr.main.segnet_path_onnx_fp16,
            homr_seg_inf.segnet_path_onnx,
            homr_seg_inf.segnet_path_onnx_fp16,
        )
        fp_default = homr_tr_cfg.default_config.filepaths
        orig_tr_attrs = {
            a: getattr(fp_default, a)
            for a in ("encoder_path", "decoder_path", "encoder_path_fp16", "decoder_path_fp16")
        }
        orig_cls_tr_dir = getattr(homr_tr_cfg.FilePaths, "_pdf2musescore_tr_dir", None)
        try:
            with TemporaryDirectory() as td:
                hdir, _ = configure_model_cache(Path(td))
                fresh_fp = homr_tr_cfg.Config().filepaths
                for attr in ("encoder_path", "decoder_path", "encoder_path_fp16", "decoder_path_fp16"):
                    assert Path(getattr(fresh_fp, attr)).parent == hdir / "transformer"
                assert Path(homr_seg_inf.segnet_path_onnx).parent == hdir / "segmentation"
                assert Path(fresh_fp.rhythmtokenizer).is_file()
        finally:
            (
                homr_seg_cfg.segnet_path_onnx,
                homr_seg_cfg.segnet_path_onnx_fp16,
                homr.main.segnet_path_onnx,
                homr.main.segnet_path_onnx_fp16,
                homr_seg_inf.segnet_path_onnx,
                homr_seg_inf.segnet_path_onnx_fp16,
            ) = orig_seg_paths
            for attr, val in orig_tr_attrs.items():
                setattr(fp_default, attr, val)
            homr_tr_cfg.FilePaths._pdf2musescore_tr_dir = orig_cls_tr_dir
        return 0
    try:
        config = load_config()
    except ConfigError as exc:
        setup_logging("INFO")
        logging.getLogger("pdf2musescore").error("Invalid configuration: %s", exc)
        _show_fatal(f"Invalid configuration:\n{exc}")
        return 2
    log = setup_logging(config.log_level, config.log_dir, config.third_party_log_level)
    crash_handle = install_crash_handlers(config.log_dir)
    restore_stdio = install_stdio_capture()
    try:
        log.info(
            "pdf2musescore %s (homr %s) starting (python %s, frozen=%s, msix=%s)",
            __version__,
            get_engine_version(),
            sys.version.split()[0],
            getattr(sys, "frozen", False),
            is_msix_package(),
        )
        log_environment(log)
        if "--download-models" in args:
            return _run_download_models(config, log, no_ui=("--no-ui" in args))
        from pdf2musescore.gui import run_gui
        try:
            run_gui(config)
        except Exception as exc:
            log.exception("Fatal error")
            _show_fatal(str(exc))
            return 1
        return 0
    finally:
        restore_stdio()
        crash_handle.uninstall()


def _show_fatal(message: str) -> None:  # pragma: no cover - UI popup
    try:
        from tkinter import messagebox
        messagebox.showerror("PDF to MuseScore", message)
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover
    code = main()
    import os
    os._exit(code)
