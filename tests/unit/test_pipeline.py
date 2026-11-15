from pathlib import Path

import pytest

from pdf2musescore.config import AppConfig
from pdf2musescore.deps import InstallError, ToolNotFoundError
from pdf2musescore.engines.base import OmrEngine, OmrError
from pdf2musescore.exporters.musescore import ExportError
from pdf2musescore.pipeline import ConversionError, Pipeline

from .test_musicxml import score, write_mxl


class FakeEngine(OmrEngine):
    def __init__(self, n=2, exc=None, cb=None):
        self.n, self.exc, self.cb = n, exc, cb

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


def make(tmp_path, level="INFO", engine=None, exporter=FakeExporter, ensure=None, steps=None):
    seen = []
    p = Pipeline(
        AppConfig(work_dir=tmp_path / "work", log_level=level),
        progress=lambda pct, m: (steps if steps is not None else []).append(pct),
        ensure=ensure or (lambda spec, cfg, confirm: seen.append(spec.name) or Path(spec.name + ".exe")),
        engine_factory=lambda dpi, cb: engine if engine is not None else FakeEngine(cb=cb),
        exporter_factory=exporter,
    )
    return p, seen


def test_convert_mscz(tmp_path, pdf):
    steps = []
    p, seen = make(tmp_path, steps=steps)
    out = p.convert(pdf, tmp_path / "out.mscz")
    assert out.read_text() == "mscz"
    assert seen == ["MuseScore"]
    assert 100 in steps
    assert not any((tmp_path / "work").iterdir())


def test_convert_musicxml_skips_musescore(tmp_path, pdf):
    p, seen = make(tmp_path)
    out = p.convert(pdf, tmp_path / "out.musicxml")
    assert seen == [] and out.is_file()


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


@pytest.mark.parametrize("kwargs,match", [
    (dict(ensure=_raiser(ToolNotFoundError("x"))), "not installed"),
    (dict(ensure=_raiser(InstallError("x"))), "Installing"),
    (dict(engine=FakeEngine(exc=OmrError("x"))), "could not be read"),
    (dict(engine=FakeEngine(n=0)), "combined"),
    (dict(exporter=lambda exe: FakeExporter(exe, ExportError("x"))), "MuseScore could not"),
])
def test_errors_mapped(tmp_path, pdf, kwargs, match):
    p, _ = make(tmp_path, **kwargs)
    with pytest.raises(ConversionError, match=match) as ei:
        p.convert(pdf, tmp_path / "o.mscz")
    assert ei.value.user_message


def test_default_factories():
    p = Pipeline(AppConfig())
    eng = p.engine_factory(300, None)
    assert eng.dpi == 300
    assert p.confirm("x") is False
    p.progress(1, "x")
