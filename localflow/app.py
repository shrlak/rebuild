"""Wires the pipeline together: hotkey -> record -> ASR -> cleanup -> inject."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

from . import cleanup
from .asr import create_transcriber
from .audio import Recorder, duration_seconds
from .config import Config
from .dictionary import Dictionary
from .history import Entry, History
from .inject import get_injector

MIN_UTTERANCE_SECONDS = 0.25  # ignore accidental key taps
SILENCE_PEAK = 0.004  # below this absolute peak, treat the clip as silence


class LocalFlowApp:
    def __init__(self, config: Config | None = None, backend: str | None = None):
        self.config = config or Config.load()
        self.transcriber = create_transcriber(
            engine=self.config.engine,
            model=self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
            beam_size=self.config.beam_size,
        )
        self.dictionary = Dictionary(self.config.dictionary_path)
        self.history = History(
            self.config.history_path, enabled=self.config.history_enabled
        )
        self.injector = get_injector(backend or self.config.backend)

    # --- core pipeline ---

    def process_audio(self, audio: np.ndarray) -> Entry | None:
        """Transcribe -> dictionary -> cleanup -> inject -> history."""
        if duration_seconds(audio, self.config.sample_rate) < MIN_UTTERANCE_SECONDS:
            return None
        if audio.size and float(np.abs(audio).max()) < SILENCE_PEAK:
            return None  # dead air: don't let the model hallucinate on it
        result = self.transcriber.transcribe(
            audio,
            language=self.config.language,
            hotwords=self.dictionary.hotwords(),
        )
        if not result.text.strip():
            return None
        text = self.dictionary.apply(result.text)
        text = cleanup.clean(
            text,
            remove_fillers_enabled=self.config.remove_fillers,
            backtrack=self.config.backtrack,
            spoken_newlines=self.config.spoken_newlines,
            spoken_punctuation=self.config.spoken_punctuation,
            capitalize=self.config.capitalize_sentences,
        )
        send_enter = False
        if self.config.press_enter_command:
            text, send_enter = cleanup.extract_press_enter(text)
        text = self.dictionary.apply_snippets(text)
        if not text:
            return None
        self._inject_with_fallback(text)
        if send_enter:
            self.injector.press_enter()
        return self.history.record(
            text=text,
            raw_text=result.text,
            language=result.language,
            audio_seconds=result.audio_seconds,
            latency_seconds=result.latency_seconds,
        )

    def _inject_with_fallback(self, text: str) -> None:
        """Never lose a dictation: fall back to the clipboard on failure."""
        from .inject import ClipboardInjector

        try:
            self.injector.inject(text)
        except Exception as exc:
            if isinstance(self.injector, ClipboardInjector):
                raise
            try:
                ClipboardInjector().inject(text)
            except Exception:
                print(text, flush=True)  # last resort: don't swallow the text
                raise RuntimeError(
                    f"{self.injector.name} injection failed ({exc}); "
                    "text printed above"
                ) from exc
            print(
                f"warning: {self.injector.name} injection failed ({exc}); "
                "text is on the clipboard — paste it manually",
                file=sys.stderr,
            )

    def transcribe_file(self, path: str | Path) -> Entry | None:
        from .audio import load_wav

        return self.process_audio(load_wav(str(path), self.config.sample_rate))

    # --- interactive daemon ---

    def run(self) -> None:
        from .hotkey import HotkeyListener

        recorder = Recorder(
            sample_rate=self.config.sample_rate, max_seconds=self.config.max_seconds
        )
        print(f"LocalFlow ready — model={self.config.model} "
              f"backend={self.injector.name}", file=sys.stderr)
        print("Loading ASR model...", file=sys.stderr)
        self.transcriber.load()
        if self.config.mode == "hold":
            print(f"Hold [{self.config.hotkey}] and speak "
                  "(double-tap for hands-free, tap again to stop). "
                  "Ctrl+C to quit.", file=sys.stderr)
        else:
            print(f"Press [{self.config.hotkey}] to start/stop dictation. "
                  "Ctrl+C to quit.", file=sys.stderr)

        def on_activate():
            try:
                recorder.start()
                if self.config.sound_cues:
                    from .feedback import play_cue

                    play_cue("start")
                print("● recording...", file=sys.stderr)
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)

        def on_deactivate():
            audio = recorder.stop()
            if self.config.sound_cues:
                from .feedback import play_cue

                play_cue("stop")
            print(f"○ processing {duration_seconds(audio):.1f}s...",
                  file=sys.stderr)
            # Process off the hotkey-listener thread so the next dictation
            # can start while this one transcribes.
            threading.Thread(
                target=self._process_and_report, args=(audio,), daemon=True
            ).start()

        listener = HotkeyListener(
            self.config.hotkey, self.config.mode, on_activate, on_deactivate
        )
        listener.start()
        try:
            listener.join()
        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
            if recorder.recording:
                recorder.stop()
        print("bye", file=sys.stderr)

    def _process_and_report(self, audio: np.ndarray) -> None:
        try:
            entry = self.process_audio(audio)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return
        if entry is None:
            print("(nothing heard)", file=sys.stderr)
        else:
            preview = entry.text if len(entry.text) < 80 else entry.text[:77] + "..."
            print(
                f"✓ {entry.words} words in {entry.latency_seconds:.1f}s "
                f"({entry.wpm:.0f} wpm): {preview}",
                file=sys.stderr,
            )
