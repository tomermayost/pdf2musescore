"""Shared Raw-OMR caching fixtures for integration tests.

Caches ONLY the heavy neural-network OMR output (``homr`` SegNet + TrOmr raw XML and
staff bounding geometry) under ``tmp/integration/_raw_omr_cache/``.
Everything downstream — PDF rendering, RapidOCR (instruments, tempos, expressions,
rehearsal marks), OpenCV barlines, MusicXML sanitization/alignment/merge, and
MuseScore 4 CLI export — runs LIVE on every test execution.

Set ``PDF2MUSESCORE_NO_OMR_CACHE=1`` to bypass cache and run raw neural-net OMR live.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest

from pdf2musescore.engines import homr_engine

ROOT = Path(__file__).resolve().parents[2]
RAW_OMR_CACHE_ROOT = ROOT / "tmp" / "integration" / "_raw_omr_cache"
LOCAL_MODELS = Path(r"C:\Users\tmayo\AppData\Local\pdf2musescore\models")


@pytest.fixture(scope="session")
def shared_rapidocr_reader():
    """Load RapidOCR ONNX models once per integration test session."""
    from rapidocr import RapidOCR

    models_dir = LOCAL_MODELS if LOCAL_MODELS.is_dir() else (ROOT / "tmp" / "models")
    _, ocr_ver_dir = homr_engine.configure_model_cache(models_dir)
    reader = RapidOCR(params={"Global.model_root_dir": str(ocr_ver_dir)})
    return reader, ocr_ver_dir


def _std_page_cache_key(
    img_path: Path,
    *,
    segnet_step_size: int,
    min_notehead_size: int,
    preprocess_mode: str,
) -> str:
    h = hashlib.sha256()
    h.update(img_path.read_bytes())
    h.update(f"\0homr:{importlib.metadata.version('homr')}".encode("utf-8"))
    h.update(
        f"\0step={segnet_step_size}:mn={min_notehead_size}:pre={preprocess_mode}".encode("utf-8")
    )
    return h.hexdigest()[:20]


def _serialize_multi_staffs(
    multi_staffs: Any,
    to_input_image: Optional[Callable[[tuple[float, float]], tuple[float, float]]],
) -> dict[str, Any]:
    map_pt = to_input_image if to_input_image is not None else (lambda pt: (float(pt[0]), float(pt[1])))
    systems: list[list[dict[str, float]]] = []
    for ms in multi_staffs or ():
        st_list: list[dict[str, float]] = []
        for st in getattr(ms, "staffs", ()):
            x0, y0 = map_pt((float(st.min_x), float(st.min_y)))
            x1, y1 = map_pt((float(st.max_x), float(st.max_y)))
            st_list.append(
                {
                    "min_x": float(x0),
                    "min_y": float(y0),
                    "max_x": float(x1),
                    "max_y": float(y1),
                }
            )
        systems.append(st_list)
    return {"systems": systems}


def _deserialize_multi_staffs(payload: dict[str, Any]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(staffs=[SimpleNamespace(**st) for st in sys_staves])
        for sys_staves in payload.get("systems", [])
    ]


@pytest.fixture(autouse=True)
def _cached_standard_raw_omr(monkeypatch, shared_rapidocr_reader):
    """Cache Standard Mode raw homr neural-net output while running all OCR & MusicXML/MuseScore stages live."""
    real_default_run = homr_engine._default_run_homr

    def _run_with_raw_cache(
        image_paths,
        page_cb=None,
        is_cancelled=lambda: False,
        segnet_step_size=320,
        min_notehead_size=4,
        preprocess_mode="auto",
        models_dir=None,
    ):
        no_cache = os.environ.get("PDF2MUSESCORE_NO_OMR_CACHE", "").strip().lower() in ("1", "true", "yes")
        if no_cache:
            return real_default_run(
                image_paths,
                page_cb=page_cb,
                is_cancelled=is_cancelled,
                segnet_step_size=segnet_step_size,
                min_notehead_size=min_notehead_size,
                preprocess_mode=preprocess_mode,
                models_dir=models_dir,
            )

        ocr_reader, ocr_ver_dir = shared_rapidocr_reader
        total = len(image_paths)
        entries: list[tuple[int, Path, Path]] = []
        misses: list[tuple[int, Path, Path]] = []

        for idx, raw_p in enumerate(image_paths, start=1):
            img = Path(raw_p)
            key = _std_page_cache_key(
                img,
                segnet_step_size=int(segnet_step_size),
                min_notehead_size=int(min_notehead_size),
                preprocess_mode=str(preprocess_mode),
            )
            cdir = RAW_OMR_CACHE_ROOT / "std" / key
            entries.append((idx, img, cdir))
            if not ((cdir / "raw.musicxml").is_file() and (cdir / "geometry.json").is_file()):
                misses.append((idx, img, cdir))

        if misses:
            import homr.main as hmain
            import homr.title_detection as htitle
            import onnxruntime as ort
            from homr.main import ProcessingConfig, XmlGeneratorArguments, download_weights, process_image
            from pdf2musescore.model_store import patched_homr_download_timeout

            ort.set_default_logger_severity(3)
            if models_dir is not None:
                homr_engine.configure_model_cache(models_dir)
            with patched_homr_download_timeout():
                download_weights(segnet_use_gpu=False, transformer_use_gpu=False, coreml_encoder=False)
            htitle._reader = ocr_reader

            cfg = ProcessingConfig(
                enable_debug=False,
                enable_cache=False,
                write_staff_positions=False,
                read_staff_positions=False,
                selected_staff=-1,
                transformer_use_gpu=False,
                segnet_use_gpu=False,
                coreml_encoder=False,
                title_detection=True,
            )
            xargs = XmlGeneratorArguments(large_page=False, metronome=None, tempo=None)

            orig_extract = hmain.extract
            orig_ellipses = hmain.create_bounding_ellipses
            orig_clahe = hmain.color_adjust.apply_clahe
            orig_detect = hmain.detect_staffs_in_image
            captured: dict[str, Any] = {}

            def _capturing_detect(*args, **kwargs):
                res = orig_detect(*args, **kwargs)
                captured["multi_staffs"] = res[0]
                captured["to_input_image"] = res[5] if len(res) > 5 else None
                return res

            hmain.detect_staffs_in_image = _capturing_detect
            hmain.extract = lambda pre, p, step_size=320, use_cache=False, use_gpu_inference=False: orig_extract(
                pre, p, step_size=int(segnet_step_size), use_cache=use_cache, use_gpu_inference=use_gpu_inference
            )
            hmain.create_bounding_ellipses = lambda nm, min_size=(4, 4), **kw: orig_ellipses(
                nm, min_size=(int(min_notehead_size), int(min_notehead_size)), **kw
            )
            if (preprocess_mode or "auto").strip().lower() == "none":
                hmain.color_adjust.apply_clahe = lambda img: img

            try:
                for _idx, img, cdir in misses:
                    if is_cancelled():
                        raise homr_engine.CancelledError("Conversion aborted by user.")
                    captured.clear()
                    process_image(str(img), cfg, xargs)
                    xml_path = img.with_suffix(".musicxml")
                    if xml_path.is_file():
                        cdir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(xml_path, cdir / "raw.musicxml")
                        geom = _serialize_multi_staffs(
                            captured.get("multi_staffs", ()),
                            captured.get("to_input_image"),
                        )
                        (cdir / "geometry.json").write_text(
                            json.dumps(geom, indent=2) + "\n", encoding="utf-8"
                        )
            finally:
                hmain.detect_staffs_in_image = orig_detect
                hmain.extract = orig_extract
                hmain.create_bounding_ellipses = orig_ellipses
                hmain.color_adjust.apply_clahe = orig_clahe

        outputs: list[Path] = []
        for idx, img, cdir in entries:
            if is_cancelled():
                raise homr_engine.CancelledError("Conversion aborted by user.")
            if page_cb is not None:
                page_cb(idx, total)
            raw_xml = cdir / "raw.musicxml"
            geom_json = cdir / "geometry.json"
            if not (raw_xml.is_file() and geom_json.is_file()):
                continue
            dst_xml = img.with_suffix(".musicxml")
            shutil.copy2(raw_xml, dst_xml)
            multi_staffs = _deserialize_multi_staffs(json.loads(geom_json.read_text(encoding="utf-8")))
            homr_engine.annotate_standard_page_instruments(
                img,
                dst_xml,
                multi_staffs,
                None,
                ocr_reader=ocr_reader,
                ocr_dir=ocr_ver_dir,
            )
            outputs.append(dst_xml)

        if not outputs:
            raise homr_engine.OmrError(f"homr could not read any of the {total} page(s)")
        return outputs

    orig_recognize = homr_engine.HomrEngine.recognize

    def _patched_recognize(self, pdf, out_dir):
        if not self.full_score and self._run is real_default_run:
            self._run = _run_with_raw_cache
        return orig_recognize(self, pdf, out_dir)

    monkeypatch.setattr(homr_engine, "_default_run_homr", _run_with_raw_cache)
    monkeypatch.setattr(homr_engine.HomrEngine, "recognize", _patched_recognize)
