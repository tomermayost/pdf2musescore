import pytest

from pdf2musescore.options import (
    CLEF_CHOICES,
    INSTRUMENT_PRESETS,
    PARAMETER_DOCS,
    PREPROCESS_MODES,
    ConversionOptions,
    OptionsError,
    format_parameter_guide,
)


def test_default_options():
    opts = ConversionOptions()
    assert opts.render_dpi == 300
    assert opts.preprocess_mode == "auto"
    assert opts.deskew is False
    assert opts.segnet_step_size == 320
    assert opts.min_notehead_size == 4
    assert opts.transpose_semitones == 0
    assert opts.clef_override == "auto"
    assert opts.instrument_name == "auto"
    assert opts.title_override is None


def test_options_normalization():
    opts = ConversionOptions(
        render_dpi=400,
        preprocess_mode="  FADED ",
        deskew=True,
        segnet_step_size=240,
        min_notehead_size=3,
        transpose_semitones=2,
        clef_override=" BASS ",
        instrument_name="  Trombone ",
        title_override="  My Sonata  ",
    )
    assert opts.preprocess_mode == "faded"
    assert opts.clef_override == "bass"
    assert opts.instrument_name == "Trombone"
    assert opts.title_override == "My Sonata"

    empty = ConversionOptions(instrument_name="  ", title_override="   ")
    assert empty.instrument_name == "auto"
    assert empty.title_override is None


@pytest.mark.parametrize("kwargs,match", [
    (dict(render_dpi=50), "Render DPI"),
    (dict(render_dpi=700), "Render DPI"),
    (dict(preprocess_mode="magic"), "preprocess_mode"),
    (dict(segnet_step_size=40), "SegNet Tile Step"),
    (dict(segnet_step_size=400), "SegNet Tile Step"),
    (dict(min_notehead_size=1), "Min Notehead Size"),
    (dict(min_notehead_size=20), "Min Notehead Size"),
    (dict(transpose_semitones=-30), "Transpose semitones"),
    (dict(transpose_semitones=30), "Transpose semitones"),
    (dict(clef_override="baritone"), "Clef override"),
])
def test_options_validation_errors(kwargs, match):
    with pytest.raises(OptionsError, match=match):
        ConversionOptions(**kwargs)


def test_parameter_guide_and_constants():
    guide = format_parameter_guide()
    assert "2. PDF Rendering & AI Engine" in guide
    assert "3. Score Post-Processing" in guide
    for doc in PARAMETER_DOCS:
        assert doc.label in guide
        assert doc.examples in guide
    assert "faded" in PREPROCESS_MODES
    assert "tenor" in CLEF_CHOICES
    assert "Trombone" in INSTRUMENT_PRESETS
