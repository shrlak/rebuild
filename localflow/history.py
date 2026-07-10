"""Dictation history + stats (words dictated, WPM, time saved vs typing)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

TYPING_WPM_BASELINE = 40.0  # average typing speed used for "time saved"


@dataclass
class Entry:
    ts: float
    text: str
    raw_text: str
    language: str
    audio_seconds: float
    latency_seconds: float
    words: int
    wpm: float


class History:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled

    def record(
        self,
        text: str,
        raw_text: str,
        language: str,
        audio_seconds: float,
        latency_seconds: float,
    ) -> Entry:
        words = len(text.split())
        minutes = audio_seconds / 60.0
        entry = Entry(
            ts=time.time(),
            text=text,
            raw_text=raw_text,
            language=language,
            audio_seconds=round(audio_seconds, 3),
            latency_seconds=round(latency_seconds, 3),
            words=words,
            wpm=round(words / minutes, 1) if minutes > 0 else 0.0,
        )
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return entry

    def entries(self) -> list[Entry]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(Entry(**json.loads(line)))
        return out

    def stats(self) -> dict:
        entries = self.entries()
        total_words = sum(e.words for e in entries)
        total_audio = sum(e.audio_seconds for e in entries)
        speaking_minutes = total_audio / 60.0
        typing_minutes = total_words / TYPING_WPM_BASELINE
        return {
            "dictations": len(entries),
            "total_words": total_words,
            "total_audio_seconds": round(total_audio, 1),
            "average_wpm": round(total_words / speaking_minutes, 1)
            if speaking_minutes > 0
            else 0.0,
            "estimated_minutes_saved": round(
                max(typing_minutes - speaking_minutes, 0.0), 1
            ),
        }
