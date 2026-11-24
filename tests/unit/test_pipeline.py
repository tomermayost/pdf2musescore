from pathlib import Path

import pytest

from pdf2musescore.config import AppConfig
from pdf2musescore.deps import InstallError, ToolNotFoundError
from pdf2musescore.engines.base import OmrEngine, OmrError
from pdf2musescore.engines.homr_engine import CancelledError
from pdf2musescore.exporters.musescore import ExportError
from pdf2musescore.pipeline import ConversionError, Pipeline

from .test_musicxml import score, write_mxl


class FakeEngine(OmrEngine):
    def __init__(self, n=2, exc=None, cb=None, pages_spec=None):
        self.n, self.exc, self.cb, self.pages_spec = n, exc, cb, pages_spec

    def recognize(self, pdf, out_dir):
        if self.exc:
            raise self.exc
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.cb:
            self.cb(1, max(self.n, 1))
        return [write_mxl(score(2), out_dir / f"x.mvt{i}.mxl") for i in range(1, self.n + 1)]


class FakeExporter:
    def __init__(self, exe, exc=None):
        self.exc = exc

    def export(self, src, dst):
        if self.exc:
            raise self.exc
        Path(dst).write_text("mscz")
        return dst


@pytest.fixture
def pdf(tmp_path):
    p = tmp_path / "song.pdf"
    p.write_bytes(b"%PDF")
    return p


def make(tmp_path, level="INFO", engine=None, exporter=FakeExporter, ensure=None, steps=None, is_cancelled=lambda: False):
    seen = []
    p = Pipeline(
        AppConfig(work_dir=tmp_path / "work", log_level=level),
        progress=lambda pct, m: (steps if steps is not None else []).append(pct),
        is_cancelled=is_cancelled,
        ensure=ensure or (lambda spec, cfg, confirm: seen.append(spec.name) or Path(spec.name + ".exe")),
        engine_factory=lambda dpi, cb, pages_spec=None, is_cancelled=None: (
            engine if engine is not None else FakeEngine(cb=cb, pages_spec=pages_spec)
        ),
        exporter_factory=exporter,
    )
    return p, seen


def test_convert_mscz(tmp_path, pdf, caplog):
    steps: list[int] = []
    messages: list[str] = []
    p, seen = make(tmp_path, steps=steps)
    p.progress = lambda pct, m: (steps.append(pct), messages.append(m))
    with caplog.at_level("INFO", logger="pdf2musescore.pipeline"):
        out = p.convert(pdf, tmp_path / "out.mscz", pages_spec="2-3")
    assert out.read_text() == "mscz"
    assert seen == ["MuseScore"]
    assert 100 in steps
    assert not any((tmp_path / "work").iterdir())
    assert "sha256=" in caplog.text
    assert all("first run" not in m.lower() for m in messages)


def test_convert_cancelled(tmp_path, pdf):
    p, _ = make(tmp_path, is_cancelled=lambda: True)
    with pytest.raises(CancelledError):
        p.convert(pdf, tmp_path / "out.mscz")


def test_convert_musicxml_skips_musescore(tmp_path, pdf):
    import xml.etree.ElementTree as ET
    p, seen = make(tmp_path)
    out = p.convert(pdf, tmp_path / "Custom Output Title.musicxml")
    assert seen == [] and out.is_file()
    assert ET.parse(out).findtext("work/work-title") == "Custom Output Title"


def test_convert_unicode_paths(tmp_path):
    heb_dir = tmp_path / "תווים" / "מנחם"
    heb_dir.mkdir(parents=True)
    pdf = heb_dir / "באך סוויטה 2.pdf"
    pdf.write_bytes(b"%PDF")
    seen_inputs = []

    class InspectEngine(FakeEngine):
        def recognize(self, p, out_dir):
            seen_inputs.append((str(p), str(out_dir)))
            return super().recognize(p, out_dir)

    p, _ = make(tmp_path, engine=InspectEngine())
    out = p.convert(pdf, heb_dir / "תוצאה.mscz")
    assert out.is_file()
    assert seen_inputs[0][0].isascii() and seen_inputs[0][1].isascii()


def test_debug_keeps_work_dir(tmp_path, pdf):
    p, _ = make(tmp_path, level="DEBUG")
    p.convert(pdf, tmp_path / "out.mscz")
    assert any((tmp_path / "work").iterdir())


def test_bad_suffix_and_missing_pdf(tmp_path, pdf):
    p, _ = make(tmp_path)
    with pytest.raises(ConversionError, match="Unsupported"):
        p.convert(pdf, tmp_path / "out.pdf")
    with pytest.raises(ConversionError, match="could not be found"):
        p.convert(tmp_path / "nope.pdf", tmp_path / "o.mscz")


def _raiser(exc):
    def f(*a, **k):
        raise exc
    return f


@pytest.mark.parametrize("kwargs,match,detail_type", [
    (dict(ensure=_raiser(ToolNotFoundError("x"))), "not installed", "ToolNotFoundError"),
    (dict(ensure=_raiser(InstallError("x"))), "Installing", "InstallError"),
    (dict(engine=FakeEngine(exc=OmrError("x"))), "could not be read", "OmrError"),
    (dict(engine=FakeEngine(exc=OmrError("Page 5 is out of range"))), "out of range", "OmrError"),
    (dict(engine=FakeEngine(n=0)), "combined", "MergeError"),
    (dict(exporter=lambda exe: FakeExporter(exe, ExportError("x"))), "MuseScore could not", "ExportError"),
    (dict(engine=FakeEngine(exc=RuntimeError("unexpected boom"))), "Something unexpected", "RuntimeError"),
])
def test_errors_mapped(tmp_path, pdf, kwargs, match, detail_type, caplog):
    p, _ = make(tmp_path, **kwargs)
    with caplog.at_level("ERROR", logger="pdf2musescore.pipeline"):
        with pytest.raises(ConversionError, match=match) as ei:
            p.convert(pdf, tmp_path / "o.mscz")
    assert ei.value.user_message
    assert detail_type in ei.value.detail
    assert ei.value.stage
    assert "Traceback" in caplog.text
    # Failed conversion preserves its work folder for diagnosis
    assert any(d.name.startswith("job-") for d in (tmp_path / "work").iterdir())


def test_prune_old_jobs_keeps_newest_three(tmp_path, pdf):
    import os
    from pdf2musescore.pipeline import _prune_old_jobs

    work_root = tmp_path / "work"
    work_root.mkdir()
    dirs = []
    for i in range(5):
        d = work_root / f"job-2026010{i}-000000-1"
        d.mkdir()
        (d / "page_001.png").write_bytes(b"png")
        os.utime(d, (1000 + i * 100, 1000 + i * 100))
        dirs.append(d)

    removed = _prune_old_jobs(work_root, keep=3)
    assert len(removed) == 2
    remaining = sorted(p.name for p in work_root.iterdir())
    assert remaining == [d.name for d in dirs[-3:]]

    # Successful conversion also runs prune and cleans up its own temp work dir
    (work_root / "job-20260109-000000-1").mkdir()
    os.utime(work_root / "job-20260109-000000-1", (2000, 2000))
    p, _ = make(tmp_path)
    p.convert(pdf, tmp_path / "ok.mscz")
    assert len(list(work_root.iterdir())) == 3


def test_default_engine_branch_and_options_error(tmp_path, pdf, monkeypatch):
    from pdf2musescore.options import ConversionOptions, OptionsError
    import pdf2musescore.pipeline as pl_mod

    p = Pipeline(
        AppConfig(work_dir=tmp_path / "work"),
        ensure=lambda spec, cfg, confirm: Path("ms.exe"),
        exporter_factory=FakeExporter,
    )
    monkeypatch.setattr(
        pl_mod,
        "_default_engine_factory",
        lambda dpi, cb, pages_spec=None, is_cancelled=None, options=None: FakeEngine(cb=cb, pages_spec=pages_spec),
    )
    p.engine_factory = pl_mod._default_engine_factory
    opts = ConversionOptions(render_dpi=350, preprocess_mode="faded", transpose_semitones=2)
    out = p.convert(pdf, tmp_path / "out.mscz", options=opts)
    assert out.is_file()

    # OptionsError raised inside convert is mapped to ConversionError
    p_err, _ = make(tmp_path, engine=FakeEngine(exc=OptionsError("bad option value")))
    with pytest.raises(ConversionError, match="bad option value"):
        p_err.convert(pdf, tmp_path / "o.mscz")


def test_default_factories():
    from pdf2musescore.options import ConversionOptions
    p = Pipeline(AppConfig())
    eng = p.engine_factory(
        300, None, "1-2", lambda: False,
        ConversionOptions(render_dpi=400, preprocess_mode="faded", deskew=True, segnet_step_size=240, min_notehead_size=3),
    )
    assert eng.dpi == 400 and eng.pages_spec == "1-2" and eng.preprocess_mode == "faded" and eng.deskew is True
    assert p.confirm("x") is False
    p.progress(1, "x")
