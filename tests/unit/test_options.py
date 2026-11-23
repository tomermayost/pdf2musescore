import pytest

from pdf2musescore.options import (
    CLEF_CHOICES,
    INSTRUMENT_PRESETS,
    PARAMETER_DOCS,
    PREPROCESS_MODES,
    SCAN_PRESETS,
    ConversionOptions,
    OptionsError,
    detect_scan_preset,
    format_parameter_guide,
    get_parameter_tooltip,
    get_scan_preset,
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
    assert opts.keep_line_breaks is False


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
        keep_line_breaks=True,
    )
    assert opts.preprocess_mode == "faded"
    assert opts.clef_override == "bass"
    assert opts.instrument_name == "Trombone"
    assert opts.keep_line_breaks is True

    empty = ConversionOptions(instrument_name="  ")
    assert empty.instrument_name == "auto"


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
        tip = get_parameter_tooltip(doc.key)
        assert doc.label in tip and doc.examples in tip
    with pytest.raises(KeyError):
        get_parameter_tooltip("nonexistent")
    assert "faded" in PREPROCESS_MODES
    assert "tenor" in CLEF_CHOICES
    assert "Trombone" in INSTRUMENT_PRESETS


def test_scan_presets_and_detection():
    keys = [p.key for p in SCAN_PRESETS]
    assert keys == ["clean", "faded", "photo"]

    for preset in SCAN_PRESETS:
        assert get_scan_preset(preset.key) == preset
        opts = ConversionOptions(
            render_dpi=preset.render_dpi,
            preprocess_mode=preset.preprocess_mode,
            deskew=preset.deskew,
            segnet_step_size=preset.segnet_step_size,
            min_notehead_size=preset.min_notehead_size,
            transpose_semitones=2,  # score post-processing should not break engine preset detection
        )
        assert detect_scan_preset(opts) == preset.key

    custom = ConversionOptions(render_dpi=420)
    assert detect_scan_preset(custom) is None

    with pytest.raises(KeyError):
        get_scan_preset("unknown")
