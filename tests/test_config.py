import pytest

from localflow.config import Config, config_dir


def test_defaults():
    cfg = Config()
    assert cfg.model == "base"
    assert cfg.mode == "hold"
    assert cfg.backend == "auto"
    assert cfg.language is None


def test_config_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALFLOW_CONFIG_DIR", str(tmp_path / "cfg"))
    assert config_dir() == tmp_path / "cfg"


def test_save_load_roundtrip(tmp_path):
    cfg = Config()
    cfg.model = "small"
    cfg.hotkey = "f9"
    cfg.mode = "toggle"
    cfg.spoken_punctuation = True
    cfg.max_seconds = 60.0
    path = cfg.save(tmp_path / "config.toml")
    loaded = Config.load(path)
    assert loaded.model == "small"
    assert loaded.hotkey == "f9"
    assert loaded.mode == "toggle"
    assert loaded.spoken_punctuation is True
    assert loaded.max_seconds == 60.0
    # None language stays None (rendered as a comment)
    assert loaded.language is None


def test_load_missing_file_gives_defaults(tmp_path):
    cfg = Config.load(tmp_path / "nope.toml")
    assert cfg.model == "base"


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('modle = "base"\n')
    with pytest.raises(ValueError, match="modle"):
        Config.load(p)


def test_invalid_mode_rejected(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('mode = "sideways"\n')
    with pytest.raises(ValueError, match="mode"):
        Config.load(p)
