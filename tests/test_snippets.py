from localflow.dictionary import Dictionary


def make_dict(tmp_path) -> Dictionary:
    d = Dictionary(tmp_path / "dictionary.json")
    d.add_snippet("sign off", "Best,\nSpencer")
    d.add_snippet("my address", "123 Main St, Springfield")
    return d


def test_snippet_expands_verbatim(tmp_path):
    d = make_dict(tmp_path)
    assert d.apply_snippets("Sign off") == "Best,\nSpencer"


def test_snippet_case_insensitive_and_inline(tmp_path):
    d = make_dict(tmp_path)
    out = d.apply_snippets("Send it to my address please.")
    assert out == "Send it to 123 Main St, Springfield please."


def test_trailing_period_swallowed_at_end(tmp_path):
    d = make_dict(tmp_path)
    assert d.apply_snippets("Thanks for the help. Sign off.") == (
        "Thanks for the help. Best,\nSpencer"
    )


def test_no_partial_word_match(tmp_path):
    d = Dictionary(tmp_path / "d.json")
    d.add_snippet("sig", "SIGNATURE")
    assert d.apply_snippets("design signal") == "design signal"


def test_snippet_persistence(tmp_path):
    d = make_dict(tmp_path)
    d.save()
    d2 = Dictionary(d.path)
    assert d2.snippets["sign off"] == "Best,\nSpencer"
    assert d2.apply_snippets("sign off") == "Best,\nSpencer"


def test_remove_snippet(tmp_path):
    d = make_dict(tmp_path)
    assert d.remove_snippet("sign off") is True
    assert d.remove_snippet("sign off") is False
    assert d.apply_snippets("sign off") == "sign off"


def test_triggers_join_hotwords(tmp_path):
    d = make_dict(tmp_path)
    assert "sign off" in d.hotwords()


def test_cli_snippet_commands(tmp_path, monkeypatch, capsys):
    from localflow.cli import main

    assert main(["snippet", "add", "sign off", r"Best,\nSpencer"]) == 0
    assert main(["snippet", "list"]) == 0
    out = capsys.readouterr().out
    assert "sign off" in out
    assert main(["snippet", "remove", "sign off"]) == 0
    assert main(["snippet", "remove", "sign off"]) == 1
