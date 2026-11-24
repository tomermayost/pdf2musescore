from pathlib import Path

import pytest

from pdf2musescore.config import AppConfig, ConfigError, default_config_path, load_config


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(tmp_path / "missing.toml", environ={})
    assert cfg == AppConfig()
    assert cfg.log_level == "INFO"
    assert cfg.third_party_log_level == "INFO"
    assert cfg.check_updates is True


def test_file_values_loaded(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text(
        'log_level = "debug"\n'
        'third_party_log_level = "warning"\n'
        'musescore_path = "C:/M/MuseScore4.exe"\n'
        'render_dpi = 250\n'
        'check_updates = false\n'
    )
    cfg = load_config(f, environ={})
    assert cfg.log_level == "DEBUG"
    assert cfg.third_party_log_level == "WARNING"
    assert cfg.musescore_path == Path("C:/M/MuseScore4.exe")
    assert cfg.render_dpi == 250
    assert cfg.check_updates is False


def test_env_overrides_file(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('log_level = "DEBUG"\ncheck_updates = false\n')
    cfg = load_config(
        f,
        environ={
            "PDF2MSCZ_LOG_LEVEL": "error",
            "PDF2MSCZ_THIRD_PARTY_LOG_LEVEL": "debug",
            "PDF2MSCZ_CHECK_UPDATES": "1",
            "OTHER": "x",
        },
    )
    assert cfg.log_level == "ERROR"
    assert cfg.third_party_log_level == "DEBUG"
    assert cfg.check_updates is True
    assert load_config(f, environ={"PDF2MSCZ_CHECK_UPDATES": "off"}).check_updates is False


def test_unknown_key_warns(tmp_path, caplog):
    f = tmp_path / "c.toml"
    f.write_text('bogus = 1\n')
    with caplog.at_level("WARNING"):
        load_config(f, environ={})
    assert "bogus" in caplog.text


def test_empty_env_value_becomes_none(tmp_path):
    cfg = load_config(tmp_path / "x.toml", environ={"PDF2MSCZ_MUSESCORE_PATH": ""})
    assert cfg.musescore_path is None


@pytest.mark.parametrize("content", ['log_level = "LOUD"\n', 'third_party_log_level = "LOUD"\n',
                                     "render_dpi = 50\n",
                                     'render_dpi = "abc"\n', 'check_updates = "maybe"\n',
                                     "not = valid = toml\n"])
def test_invalid_values_raise(tmp_path, content):
    f = tmp_path / "c.toml"
    f.write_text(content)
    with pytest.raises(ConfigError):
        load_config(f, environ={})


def test_default_config_path_uses_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert default_config_path() == tmp_path / "pdf2musescore" / "config.toml"


def test_load_config_uses_default_path(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    p = tmp_path / "pdf2musescore"
    p.mkdir()
    (p / "config.toml").write_text('log_level = "WARNING"\n')
    assert load_config(environ={}).log_level == "WARNING"
