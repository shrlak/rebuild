"""Personal dictionary: custom vocabulary + replacements.

Two mechanisms, mirroring Wispr Flow's dictionary:

- ``vocabulary``: words/names the ASR should be biased toward. Fed to
  faster-whisper as ``hotwords`` so "CTranslate2" doesn't come out as
  "see translate too".
- ``replacements``: exact post-ASR substitutions applied case-insensitively
  on word boundaries, e.g. "wispr" -> "Wispr", "local flow" -> "LocalFlow".
- ``snippets``: voice-triggered canned text — say the trigger phrase, get the
  block, e.g. "sign off" -> "Best,\nSpencer". Applied after cleanup so the
  block's own formatting/capitalization is preserved verbatim.

Stored as JSON so it's trivially editable by hand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


class Dictionary:
    def __init__(self, path: Path):
        self.path = path
        self.replacements: dict[str, str] = {}
        self.vocabulary: list[str] = []
        self.snippets: dict[str, str] = {}
        self._pattern: re.Pattern | None = None
        self.load()

    def load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.replacements = dict(data.get("replacements", {}))
            self.vocabulary = list(data.get("vocabulary", []))
            self.snippets = dict(data.get("snippets", {}))
        self._compile()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "replacements": self.replacements,
                    "vocabulary": self.vocabulary,
                    "snippets": self.snippets,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )

    def _compile(self) -> None:
        if not self.replacements:
            self._pattern = None
            return
        # longest keys first so "local flow" wins over "local"
        keys = sorted(self.replacements, key=len, reverse=True)
        self._pattern = re.compile(
            r"(?<![\w'-])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w'-])",
            re.IGNORECASE,
        )

    # --- editing ---

    def add_replacement(self, spoken: str, written: str) -> None:
        self.replacements[spoken.strip().lower()] = written.strip()
        self._compile()

    def remove_replacement(self, spoken: str) -> bool:
        removed = self.replacements.pop(spoken.strip().lower(), None) is not None
        self._compile()
        return removed

    def add_snippet(self, trigger: str, text: str) -> None:
        self.snippets[trigger.strip().lower()] = text

    def remove_snippet(self, trigger: str) -> bool:
        return self.snippets.pop(trigger.strip().lower(), None) is not None

    def add_vocabulary(self, word: str) -> None:
        word = word.strip()
        if word and word.lower() not in (w.lower() for w in self.vocabulary):
            self.vocabulary.append(word)

    def remove_vocabulary(self, word: str) -> bool:
        before = len(self.vocabulary)
        self.vocabulary = [w for w in self.vocabulary if w.lower() != word.strip().lower()]
        return len(self.vocabulary) != before

    # --- application ---

    def apply(self, text: str) -> str:
        if not self._pattern:
            return text
        return self._pattern.sub(
            lambda m: self.replacements[m.group(0).lower()], text
        )

    def apply_snippets(self, text: str) -> str:
        """Expand snippet triggers. Runs on *cleaned* text so blocks are
        inserted verbatim; a trigger at the end of the text also swallows
        the sentence period the ASR put after it."""
        for trigger, block in sorted(self.snippets.items(), key=lambda i: -len(i[0])):
            pattern = re.compile(
                r"(?<![\w'-])" + re.escape(trigger) + r"(?![\w'-])[.!?]?\s*$|"
                r"(?<![\w'-])" + re.escape(trigger) + r"(?![\w'-])",
                re.IGNORECASE,
            )
            text = pattern.sub(lambda m: block, text)
        return text

    def hotwords(self) -> str | None:
        """Bias string for the ASR (vocabulary + replacement targets)."""
        words = list(self.vocabulary)
        words += [v for v in self.replacements.values() if v not in words]
        words += [t for t in self.snippets if t not in words]
        return " ".join(words) if words else None
