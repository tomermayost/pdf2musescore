import json
from pathlib import Path

from pdf2musescore import preferences as pref_mod
from pdf2musescore.preferences import (
    UserPreferences,
    default_preferences_path,
    load_preferences,
    save_preferences,
)


def test_default_preferences_path_dev_and_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(pref_mod, "is_frozen", lambda: False)
    assert default_preferences_path() == Path("tmp") / "preferences.json"

    monkeypatch.setattr(pref_mod, "is_frozen", lambda: True)
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert default_preferences_path() == tmp_path / "pdf2musescore" / "preferences.json"


def test_load_missing_returns_defaults_with_clamped_dpi(tmp_path):
    missing = tmp_path / "nope.json"
    prefs = load_preferences(missing, default_dpi=350)
    assert prefs == UserPreferences(render_dpi=350)
    assert load_preferences(missing, default_dpi=999).render_dpi == 300
    assert load_preferences(missing, default_dpi="bad").render_dpi == 300  # type: ignore[arg-type]


def test_save_and_load_roundtrip(tmp_path):
    p = tmp_path / "nested" / "preferences.json"
    original = UserPreferences(
        theme="light",
        advanced_expanded=True,
        log_expanded=False,
        render_dpi=400,
        preprocess_mode="faded",
        deskew=True,
        segnet_step_size=240,
        min_notehead_size=3,
        transpose_semitones=-2,
        clef_override="bass",
        instrument_name="Cello",
        keep_line_breaks=True,
        last_dir="C:/Scores/Bach",
    )
    assert save_preferences(original, p) is True
    assert p.is_file()
    # Temp files should be cleaned up after atomic replace
    assert list(p.parent.glob("*.tmp")) == []

    loaded = load_preferences(p)
    assert loaded == original


def test_corrupt_and_out_of_range_preferences_fall_back_safely(tmp_path, caplog):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert load_preferences(corrupt) == UserPreferences()
    assert "Could not read preferences" in caplog.text

    non_dict = tmp_path / "list.json"
    non_dict.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_preferences(non_dict) == UserPreferences()

    bad_fields = tmp_path / "bad_fields.json"
    bad_fields.write_text(
        json.dumps(
            {
                "theme": "neon",
                "advanced_expanded": "yes",
                "log_expanded": False,
                "render_dpi": 999,
                "preprocess_mode": "unknown",
                "deskew": True,
                "segnet_step_size": True,
                "min_notehead_size": "abc",
                "transpose_semitones": 99,
                "clef_override": "subcontrabass",
                "instrument_name": 123,
                "keep_line_breaks": True,
                "last_dir": 456,
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        sanitized = load_preferences(bad_fields, default_dpi=300)

    assert sanitized == UserPreferences(
        theme="dark",
        advanced_expanded=False,
        log_expanded=False,
        render_dpi=300,
        preprocess_mode="auto",
        deskew=True,
        segnet_step_size=320,
        min_notehead_size=4,
        transpose_semitones=0,
        clef_override="auto",
        instrument_name="auto",
        keep_line_breaks=True,
        last_dir="",
    )


def test_save_preferences_handles_io_failure(tmp_path, caplog):
    # Making parent an existing regular file causes mkdir/save to raise OSError
    blocker = tmp_path / "file_not_dir"
    blocker.write_text("x", encoding="utf-8")
    with caplog.at_level("WARNING"):
        ok = save_preferences(UserPreferences(), blocker / "preferences.json")
    assert ok is False
    assert "Failed to save user preferences" in caplog.text
