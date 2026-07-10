"""Audio capture and WAV file I/O.

The Recorder wraps a sounddevice input stream and accumulates 16 kHz mono
float32 frames between start() and stop(). sounddevice (PortAudio) is an
optional dependency: file-based workflows and tests work without it.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import soundfile as sf


class Recorder:
    def __init__(self, sample_rate: int = 16000, max_seconds: float = 120.0):
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()
        self.recording = False

    def start(self) -> None:
        if self.recording:
            return
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "Microphone capture needs the 'sounddevice' package and a "
                "working audio device (pip install 'localflow[desktop]')."
            ) from exc
        with self._lock:
            self._chunks = []
        self._max_frames = int(self.max_seconds * self.sample_rate)
        self._frames = 0

        def callback(indata, frames, time_info, status):
            with self._lock:
                if self._frames < self._max_frames:
                    self._chunks.append(indata[:, 0].copy())
                    self._frames += frames

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        self.recording = True

    def stop(self) -> np.ndarray:
        if not self.recording:
            return np.zeros(0, dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.recording = False
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._chunks)


def load_wav(path: str | Path, target_rate: int = 16000) -> np.ndarray:
    """Load any WAV/FLAC/OGG as mono float32 at target_rate."""
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != target_rate:
        mono = resample(mono, rate, target_rate)
    return mono


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    sf.write(str(path), audio, sample_rate)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resample. Fine for speech-to-ASR purposes."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.size / src_rate
    n_out = int(round(duration * dst_rate))
    src_t = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    dst_t = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(dst_t, src_t, audio).astype(np.float32)


def duration_seconds(audio: np.ndarray, sample_rate: int = 16000) -> float:
    return float(audio.size) / float(sample_rate)
