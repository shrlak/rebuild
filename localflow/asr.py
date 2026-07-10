"""Speech-to-text engines. Everything runs fully local.

Two interchangeable engines:

- ``faster-whisper`` (default): CTranslate2 backend, int8 CPU quantization,
  built-in Silero VAD, ``hotwords`` biasing. Models auto-download from
  Hugging Face on first use, then cached.
- ``whispercpp``: pywhispercpp bindings over whisper.cpp. Accepts GGML model
  *names* (auto-download) or a *path* to a ggml-*.bin file — handy where
  Hugging Face is unreachable or a GGML model is already on disk.

Both expose ``transcribe(audio, language, hotwords) -> TranscriptionResult``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ENGINES = ("faster-whisper", "whispercpp")


@dataclass
class TranscriptionResult:
    text: str
    language: str
    audio_seconds: float
    latency_seconds: float


def _as_audio(audio: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(audio, (str, Path)):
        from .audio import load_wav

        return load_wav(audio)
    return audio


class FasterWhisperTranscriber:
    engine = "faster-whisper"

    def __init__(
        self,
        model: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        except Exception:
            # Hub unreachable (offline/proxy): retry against the local cache.
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True,
            )

    def transcribe(
        self,
        audio: np.ndarray | str | Path,
        language: str | None = None,
        hotwords: str | None = None,
    ) -> TranscriptionResult:
        self.load()
        audio = _as_audio(audio)
        started = time.monotonic()
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            vad_filter=True,
            hotwords=hotwords,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            audio_seconds=float(info.duration),
            latency_seconds=time.monotonic() - started,
        )


class WhisperCppTranscriber:
    engine = "whispercpp"

    def __init__(self, model: str = "tiny", n_threads: int | None = None):
        self.model_name = model
        self.n_threads = n_threads
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from pywhispercpp.model import Model

        kwargs = {"print_progress": False, "redirect_whispercpp_logs_to": None}
        if self.n_threads:
            kwargs["n_threads"] = self.n_threads
        self._model = Model(self.model_name, **kwargs)

    def transcribe(
        self,
        audio: np.ndarray | str | Path,
        language: str | None = None,
        hotwords: str | None = None,
    ) -> TranscriptionResult:
        self.load()
        audio = _as_audio(audio)
        started = time.monotonic()
        kwargs = {"language": language or "auto"}
        if hotwords:
            kwargs["initial_prompt"] = hotwords
        segments = self._model.transcribe(audio, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return TranscriptionResult(
            text=text,
            language=language or "auto",
            audio_seconds=audio.size / 16000.0,
            latency_seconds=time.monotonic() - started,
        )


# Back-compat alias: the original single-engine API.
Transcriber = FasterWhisperTranscriber


def create_transcriber(
    engine: str = "faster-whisper",
    model: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
):
    if engine == "faster-whisper":
        return FasterWhisperTranscriber(
            model=model, device=device, compute_type=compute_type, beam_size=beam_size
        )
    if engine == "whispercpp":
        return WhisperCppTranscriber(model=model)
    raise ValueError(f"unknown ASR engine {engine!r}; choose from {ENGINES}")
