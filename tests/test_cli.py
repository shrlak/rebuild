import json

from localflow.cli import main
from localflow.config import Config


class TestDictCommands:
    def test_add_list_remove(self, capsys):
        assert main(["dict", "add", "wispr", "Wispr"]) == 0
        assert main(["dict", "list"]) == 0
        listing = json.loads(capsys.readouterr().out.split("added:")[-1].split("\n", 1)[-1])
        assert listing["replacements"] == {"wispr": "Wispr"}
        assert main(["dict", "remove", "wispr"]) == 0
        assert main(["dict", "remove", "wispr"]) == 1

    def test_vocab(self, capsys):
        assert main(["dict", "vocab", "CTranslate2", "Anthropic"]) == 0
        out = capsys.readouterr().out
        assert "CTranslate2" in out and "Anthropic" in out


class TestConfigCommands:
    def test_init_show_path(self, capsys):
        assert main(["config", "init"]) == 0
        wrote = capsys.readouterr().out
        assert "config.toml" in wrote
        assert main(["config", "show"]) == 0
        assert "model = 'base'" in capsys.readouterr().out
        assert main(["config", "path"]) == 0
        assert "config.toml" in capsys.readouterr().out

    def test_loads_saved_config(self):
        cfg = Config()
        cfg.model = "tiny"
        cfg.save()
        assert Config.load().model == "tiny"


class TestHistoryCommands:
    def test_empty_history(self, capsys):
        assert main(["history"]) == 0
        assert "no dictations" in capsys.readouterr().out
        assert main(["stats"]) == 0
        assert "dictations: 0" in capsys.readouterr().out

    def test_last_empty(self, capsys):
        assert main(["last"]) == 1

    def test_history_and_last_after_record(self, capsys):
        from localflow.history import History

        History(Config().history_path).record("hello world", "hello world", "en", 1.0, 0.2)
        assert main(["history"]) == 0
        assert "hello world" in capsys.readouterr().out
        assert main(["last"]) == 0
        assert capsys.readouterr().out.strip() == "hello world"
        assert main(["stats"]) == 0
        assert "total words: 2" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "localflow" in capsys.readouterr().out
