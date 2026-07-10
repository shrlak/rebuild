"""Text cleanup pipeline: what Wispr Flow calls "auto-edits".

Pure functions only — deterministic and unit-testable. Order of operations
in `clean()`:

1. strip filler words (um, uh, ...)
2. apply personal-dictionary replacements (done by caller via dictionary.py)
3. spoken commands ("new line", optionally "period" etc.)
4. whitespace/punctuation normalization + sentence capitalization
"""

from __future__ import annotations

import re

# Conservative filler list: things that are ~never intentional dictation.
# ("like"/"you know" are too risky to strip without context.)
FILLERS = (
    "um", "umm", "ums", "uh", "uhh", "uhm", "er", "erm", "err",
    "ah", "ahh", "hmm", "hm", "mhm", "mm-hmm", "mmm",
)

_FILLER_RE = re.compile(
    r"(?<![\w'-])(" + "|".join(re.escape(f) for f in FILLERS) + r")(?![\w'-])[,.!?]?\s*",
    re.IGNORECASE,
)

# Spoken layout commands (always safe to honor for dictation).
_NEWLINE_RE = re.compile(r"[,.]?\s*\b(new\s+paragraph)\b[,.]?\s*", re.IGNORECASE)
_LINEBREAK_RE = re.compile(r"[,.]?\s*\b(new\s*line)\b[,.]?\s*", re.IGNORECASE)

# Backtrack: "bring cookies, scratch that, bring brownies" -> "bring brownies".
# Greedy prefix match within the sentence keeps only what follows the last
# trigger, matching how people restate the corrected clause in full.
_BACKTRACK_RE = re.compile(
    r"[^.!?\n]*[,.]?\s*\b(?:scratch|strike)\s+that\b[,.]?\s*",
    re.IGNORECASE,
)

# Trailing "press enter": stripped from the text; the caller sends the key.
_PRESS_ENTER_RE = re.compile(r"[,.!?\s]*\bpress\s+enter\b[.!?\s]*$", re.IGNORECASE)

# Spoken punctuation (opt-in: "period" etc. can be legitimate words).
_SPOKEN_PUNCT = [
    (re.compile(r"\s*\bfull\s+stop\b", re.IGNORECASE), "."),
    (re.compile(r"\s*\bperiod\b", re.IGNORECASE), "."),
    (re.compile(r"\s*\bcomma\b", re.IGNORECASE), ","),
    (re.compile(r"\s*\bquestion\s+mark\b", re.IGNORECASE), "?"),
    (re.compile(r"\s*\bexclamation\s+(?:mark|point)\b", re.IGNORECASE), "!"),
    (re.compile(r"\s*\bsemicolon\b", re.IGNORECASE), ";"),
    (re.compile(r"\s*\bcolon\b", re.IGNORECASE), ":"),
    (re.compile(r"\s*\bellipsis\b", re.IGNORECASE), "..."),
    (re.compile(r"\s*\bnew\s+bullet\b[,.]?\s*", re.IGNORECASE), "\n- "),
]


def remove_fillers(text: str) -> str:
    return _FILLER_RE.sub("", text)


def apply_backtrack(text: str) -> str:
    return _BACKTRACK_RE.sub("", text)


def extract_press_enter(text: str) -> tuple[str, bool]:
    """Strip a trailing spoken "press enter"; report whether it was there."""
    stripped = _PRESS_ENTER_RE.sub("", text)
    return stripped, stripped != text


def apply_spoken_newlines(text: str) -> str:
    text = _NEWLINE_RE.sub("\n\n", text)
    text = _LINEBREAK_RE.sub("\n", text)
    return text


def apply_spoken_punctuation(text: str) -> str:
    for pattern, mark in _SPOKEN_PUNCT:
        text = pattern.sub(mark, text)
    return text


def normalize(text: str, capitalize: bool = True) -> str:
    """Fix spacing artifacts left by removals, then capitalize sentences."""
    # collapse runs of spaces/tabs (not newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # no space before closing punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # collapse duplicate commas/periods introduced by removals ("um, ," cases)
    text = re.sub(r"([,;:])\s*([,;:])", r"\1", text)
    text = re.sub(r"(?<!\.)\.\s*\.(?!\.)", ".", text)
    # space after sentence punctuation if a word follows
    text = re.sub(r"([,.;:!?])(?=[A-Za-z])", r"\1 ", text)
    # trim spaces around newlines, cap blank runs at one blank line
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # leading orphan punctuation (filler removed at sentence start)
    text = re.sub(r"^[ ,.;:]+", "", text)
    text = re.sub(r"\n[ ,.;:]+", "\n", text)
    text = text.strip()
    if capitalize and text:
        text = _capitalize_sentences(text)
    return text


def _capitalize_sentences(text: str) -> str:
    out = list(text)
    cap_next = True
    for i, ch in enumerate(out):
        if cap_next and ch.isalpha():
            out[i] = ch.upper()
            cap_next = False
        elif ch in ".!?\n":
            cap_next = True
        elif not ch.isspace() and ch not in "\"'([{-":
            cap_next = False
    return "".join(out)


def clean(
    text: str,
    *,
    remove_fillers_enabled: bool = True,
    backtrack: bool = True,
    spoken_newlines: bool = True,
    spoken_punctuation: bool = False,
    capitalize: bool = True,
) -> str:
    """Run the full cleanup pipeline over a raw ASR transcript."""
    if remove_fillers_enabled:
        text = remove_fillers(text)
    if backtrack:
        text = apply_backtrack(text)
    if spoken_newlines:
        text = apply_spoken_newlines(text)
    if spoken_punctuation:
        text = apply_spoken_punctuation(text)
    return normalize(text, capitalize=capitalize)
