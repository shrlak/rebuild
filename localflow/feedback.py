"""Audio cues: a short rising blip when recording starts, falling when it
stops — the audible equivalent of Wispr's Flow Bar state change.

Cues are fire-and-forget and never raise: on machines without an audio
output device (headless, CI) they silently do nothing.
"""

from __future__ import annotations

import numpy as np

_RATE = 44100


def _tone(freq: float, seconds: float = 0.07, volume: float = 0.12) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(_RATE * seconds), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    # 5 ms fade in/out to avoid clicks
    fade = max(1, int(0.005 * _RATE))
    envelope = np.ones_like(wave)
    envelope[:fade] = np.linspace(0.0, 1.0, fade)
    envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    return (volume * wave * envelope).astype(np.float32)


def play_cue(kind: str) -> None:
    """kind: "start" (rising) or "stop" (falling). Non-blocking, never raises."""
    try:
        import sounddevice as sd

        if kind == "start":
            cue = np.concatenate([_tone(660), _tone(880)])
        else:
            cue = np.concatenate([_tone(880), _tone(660)])
        sd.play(cue, _RATE, blocking=False)
    except Exception:
        pass
