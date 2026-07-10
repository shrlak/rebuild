from localflow.cleanup import (
    apply_backtrack,
    apply_spoken_newlines,
    apply_spoken_punctuation,
    clean,
    extract_press_enter,
    normalize,
    remove_fillers,
)


class TestRemoveFillers:
    def test_basic_fillers(self):
        assert clean("um so I think uh we should go") == "So I think we should go"

    def test_filler_with_comma(self):
        assert clean("Um, hello there") == "Hello there"

    def test_filler_mid_sentence(self):
        assert clean("I was, uh, thinking about it.") == "I was, thinking about it."

    def test_does_not_eat_real_words(self):
        # "umbrella" contains "um"; "usher" contains "uh"
        assert remove_fillers("my umbrella and the usher") == "my umbrella and the usher"

    def test_hmm_variants(self):
        assert clean("hmm let me see") == "Let me see"

    def test_case_insensitive(self):
        assert clean("UM. UH. hello") == "Hello"

    def test_apostrophe_words_untouched(self):
        assert remove_fillers("er no wait") == "no wait"
        assert "er" not in remove_fillers("er ").split()


class TestSpokenCommands:
    def test_new_line(self):
        assert clean("first item new line second item") == "First item\nSecond item"

    def test_new_paragraph(self):
        out = clean("intro new paragraph body")
        assert out == "Intro\n\nBody"

    def test_newline_swallows_adjacent_punctuation(self):
        assert apply_spoken_newlines("Hello, new line. World") == "Hello\nWorld"

    def test_spoken_punctuation_opt_in(self):
        raw = "hello comma world period"
        assert clean(raw, spoken_punctuation=False) == "Hello comma world period"
        assert clean(raw, spoken_punctuation=True) == "Hello, world."

    def test_question_mark(self):
        assert (
            apply_spoken_punctuation("are you there question mark")
            == "are you there?"
        )

    def test_bullet(self):
        out = clean("shopping new bullet eggs new bullet milk", spoken_punctuation=True)
        assert out == "Shopping\n- Eggs\n- Milk"


class TestBacktrack:
    def test_scratch_that(self):
        raw = "I'll bring cookies, scratch that, I'll bring brownies."
        assert clean(raw) == "I'll bring brownies."

    def test_strike_that(self):
        assert clean("send it Monday strike that send it Friday") == "Send it Friday"

    def test_does_not_cross_sentences(self):
        raw = "The meeting is at noon. Bring slides, scratch that, bring handouts."
        assert clean(raw) == "The meeting is at noon. Bring handouts."

    def test_multiple_corrections_keep_last(self):
        raw = "call Bob scratch that call Sue scratch that call Ann"
        assert clean(raw) == "Call Ann"

    def test_disabled(self):
        raw = "cookies scratch that brownies"
        assert "scratch that" in clean(raw, backtrack=False)

    def test_trigger_at_start(self):
        assert apply_backtrack("scratch that hello") == "hello"


class TestPressEnter:
    def test_trailing_press_enter_stripped(self):
        text, send = extract_press_enter("Sounds good, press enter.")
        assert text == "Sounds good"
        assert send is True

    def test_mid_sentence_press_enter_kept(self):
        text, send = extract_press_enter("press enter to continue the setup")
        assert send is False
        assert "press enter" in text

    def test_no_command(self):
        text, send = extract_press_enter("just some text")
        assert (text, send) == ("just some text", False)


class TestNormalize:
    def test_collapses_spaces(self):
        assert normalize("a    b\t c") == "A b c"

    def test_space_before_punct(self):
        assert normalize("hello , world .") == "Hello, world."

    def test_capitalizes_sentences(self):
        assert normalize("one. two! three? four") == "One. Two! Three? Four"

    def test_capitalize_after_newline(self):
        assert normalize("hello\nworld") == "Hello\nWorld"

    def test_no_capitalize_flag(self):
        assert normalize("hello. world", capitalize=False) == "hello. world"

    def test_leading_orphan_punctuation_removed(self):
        assert normalize(", hello") == "Hello"

    def test_blank_line_runs_capped(self):
        assert normalize("a\n\n\n\nb") == "A\n\nB"


class TestFullPipeline:
    def test_realistic_utterance(self):
        raw = "um, hey can you, uh, send me the report new line thanks"
        assert clean(raw) == "Hey can you, send me the report\nThanks"

    def test_empty_after_cleanup(self):
        assert clean("um uh hmm") == ""

    def test_plain_text_passthrough(self):
        assert clean("This is already clean text.") == "This is already clean text."
