"""End-to-end tests using real synthesized speech and the real ASR model.

espeak-ng generates a WAV, faster-whisper transcribes it, and the full
pipeline (dictionary -> cleanup -> inject -> history) runs on the result.
Marked slow: the first run downloads the "base" model (~75 MB).
"""

import subprocess
import sys
from pathlib import Path

import pytest

from localflow.config import Config

from .conftest import synthesize


def word_overlap(expected: str, actual: str) -> float:
    expected_words = set(expected.lower().replace(",", "").replace(".", "").split())
    actual_words = set(actual.lower().replace(",", "").replace(".", "").split())
    if not expected_words:
        return 0.0
    return len(expected_words & actual_words) / len(expected_words)


GGML_TINY = Path.home() / ".cache" / "localflow" / "models" / "ggml-tiny.bin"


def pick_engine() -> tuple[str, str]:
    """Prefer faster-whisper; fall back to a local GGML file for whispercpp."""
    from localflow.asr import create_transcriber

    try:
        t = create_transcriber(engine="faster-whisper", model="base")
        t.load()
        return "faster-whisper", "base"
    except Exception:
        if GGML_TINY.exists():
            return "whispercpp", str(GGML_TINY)
        pytest.skip("no ASR model available (Hugging Face unreachable and no "
                    f"GGML model at {GGML_TINY})")


@pytest.fixture(scope="session")
def engine_model() -> tuple[str, str]:
    return pick_engine()


@pytest.fixture(scope="session")
def transcriber(engine_model):
    from localflow.asr import create_transcriber

    engine, model = engine_model
    t = create_transcriber(engine=engine, model=model)
    t.load()
    return t


@pytest.mark.slow
def test_asr_transcribes_synthesized_speech(tmp_path, transcriber):
    wav = tmp_path / "speech.wav"
    sentence = "Hello world. This is a test of local dictation on this machine."
    synthesize(sentence, wav)
    result = transcriber.transcribe(str(wav), language="en")
    assert result.text, "expected non-empty transcript"
    assert word_overlap(sentence, result.text) >= 0.5, result.text
    assert result.language == "en"
    assert result.audio_seconds > 1.0


@pytest.mark.slow
def test_full_pipeline_with_history(tmp_path, transcriber, engine_model, capsys):
    from localflow.app import LocalFlowApp

    wav = tmp_path / "speech.wav"
    synthesize("The quick brown fox jumps over the lazy dog.", wav)

    cfg = Config.load()
    cfg.engine, cfg.model = engine_model
    app = LocalFlowApp(cfg, backend="stdout")
    app.transcriber = transcriber  # reuse the session model
    entry = app.transcribe_file(wav)

    assert entry is not None
    printed = capsys.readouterr().out
    assert entry.text in printed
    assert "fox" in entry.text.lower()
    assert entry.words > 4
    assert entry.wpm > 0
    # history actually persisted
    from localflow.history import History

    stored = History(cfg.history_path).entries()
    assert len(stored) == 1
    assert stored[0].text == entry.text


@pytest.mark.slow
def test_dictionary_replacement_applies_in_pipeline(tmp_path, transcriber, engine_model):
    from localflow.app import LocalFlowApp

    wav = tmp_path / "speech.wav"
    synthesize("I really enjoy using whisper for dictation.", wav)

    cfg = Config.load()
    cfg.engine, cfg.model = engine_model
    app = LocalFlowApp(cfg, backend="stdout")
    app.transcriber = transcriber
    app.dictionary.add_replacement("whisper", "Whisper™")
    entry = app.transcribe_file(wav)
    assert entry is not None
    if "whisper" in entry.raw_text.lower():
        assert "Whisper™" in entry.text


@pytest.mark.slow
def test_cli_transcribe_subprocess(tmp_path, engine_model):
    """The real CLI, as a user would run it: file in, cleaned text on stdout."""
    wav = tmp_path / "speech.wav"
    synthesize("Testing the command line interface right now.", wav)
    import os

    engine, model = engine_model
    env = dict(os.environ)
    env["LOCALFLOW_CONFIG_DIR"] = str(tmp_path / "cfg")
    proc = subprocess.run(
        [sys.executable, "-m", "localflow.cli", "transcribe", str(wav),
         "--backend", "stdout", "--engine", engine, "--model", model],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert word_overlap("testing the command line interface right now", proc.stdout) >= 0.5, (
        proc.stdout
    )
