from pathlib import Path

from localflow.dictionary import Dictionary


def make_dict(tmp_path) -> Dictionary:
    return Dictionary(tmp_path / "dictionary.json")


class TestReplacements:
    def test_case_insensitive_whole_word(self, tmp_path):
        d = make_dict(tmp_path)
        d.add_replacement("wispr", "Wispr")
        assert d.apply("i love wispr and WISPR") == "i love Wispr and Wispr"

    def test_no_partial_word_match(self, tmp_path):
        d = make_dict(tmp_path)
        d.add_replacement("cat", "CAT")
        assert d.apply("catalog of cats and a cat") == "catalog of cats and a CAT"

    def test_multiword_and_longest_first(self, tmp_path):
        d = make_dict(tmp_path)
        d.add_replacement("local", "LOCAL")
        d.add_replacement("local flow", "LocalFlow")
        assert d.apply("use local flow locally") == "use LocalFlow locally"
        assert d.apply("a local shop") == "a LOCAL shop"

    def test_remove(self, tmp_path):
        d = make_dict(tmp_path)
        d.add_replacement("foo", "bar")
        assert d.remove_replacement("foo") is True
        assert d.remove_replacement("foo") is False
        assert d.apply("foo") == "foo"

    def test_persistence(self, tmp_path):
        d = make_dict(tmp_path)
        d.add_replacement("gpt", "GPT")
        d.add_vocabulary("CTranslate2")
        d.save()
        d2 = Dictionary(d.path)
        assert d2.apply("gpt") == "GPT"
        assert d2.vocabulary == ["CTranslate2"]

    def test_missing_file_is_empty(self, tmp_path):
        d = Dictionary(Path(tmp_path) / "nope.json")
        assert d.apply("anything") == "anything"
        assert d.hotwords() is None


class TestVocabulary:
    def test_dedupe_case_insensitive(self, tmp_path):
        d = make_dict(tmp_path)
        d.add_vocabulary("Kubernetes")
        d.add_vocabulary("kubernetes")
        assert d.vocabulary == ["Kubernetes"]

    def test_hotwords_includes_replacement_targets(self, tmp_path):
        d = make_dict(tmp_path)
        d.add_vocabulary("Anthropic")
        d.add_replacement("cloud", "Claude")
        hw = d.hotwords()
        assert "Anthropic" in hw and "Claude" in hw

    def test_remove_vocabulary(self, tmp_path):
        d = make_dict(tmp_path)
        d.add_vocabulary("Zig")
        assert d.remove_vocabulary("zig") is True
        assert d.remove_vocabulary("zig") is False
