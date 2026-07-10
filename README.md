# LocalFlow

**Local push-to-talk dictation.** Hold a hotkey, speak, release — cleaned-up
text is pasted into whatever app has focus. A rebuild of
[Wispr Flow](https://wisprflow.ai)'s core loop where **everything runs on your
machine**: on-device Whisper ASR, rule-based cleanup, no audio or text ever
leaves your computer. No account, no subscription, works offline.

```
hold hotkey ──► mic ──► Whisper (local) ──► cleanup ──► pasted at your cursor
                                              │
              fillers removed · "scratch that" backtrack · "new line" ·
              personal dictionary · sentence capitalization · "press enter"
```

## Quick start

```bash
pip install -e ".[desktop]"      # or: pip install -e ".[desktop,whispercpp]"
localflow run                    # hold ctrl+alt+space and speak
```

The first run downloads the Whisper `base` model (~75 MB) from Hugging Face,
after which everything is offline.

Try it without a microphone:

```bash
localflow transcribe recording.wav          # full pipeline, prints the result
localflow transcribe recording.wav --raw    # raw ASR output, no cleanup
```

## What it does

- **Push-to-talk**: hold `ctrl+alt+space` (configurable, e.g. `f9`), speak,
  release. **Double-tap** the hotkey for hands-free mode (recording stays on;
  tap once more to stop), or set `mode = "toggle"` for tap-to-start/stop.
  Audio cues blip when recording starts/stops (`sound_cues = false` to mute).
- **Local ASR**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (CTranslate2, int8 on CPU, Silero VAD) by default; or
  [pywhispercpp](https://github.com/absadiki/pywhispercpp) (`engine =
  "whispercpp"`) which also accepts a path to any `ggml-*.bin` model file.
- **Auto-edits** (all rule-based, deterministic, unit-tested):
  - filler removal: "um, I think uh we should" → "I think we should"
  - backtrack: "the demo is Tuesday, scratch that, the demo is Thursday" →
    "The demo is Thursday"
  - "new line" / "new paragraph" → real line breaks; optional spoken
    punctuation ("comma", "period", "new bullet", ...)
  - sentence capitalization + whitespace/punctuation normalization
  - end with "press enter" to strip those words and press the Enter key —
    great for chat and LLM prompts
- **Personal dictionary**: `localflow dict add "wispr" "Wispr"` fixes
  spellings after ASR; `localflow dict vocab CTranslate2 Anthropic` biases
  the recognizer toward your jargon (fed as hotwords/initial prompt).
- **Snippets**: `localflow snippet add "sign off" "Best,\nSpencer"` — say
  "sign off" and the block is inserted verbatim, formatting intact.
- **Insertion backends**: `paste` (clipboard + Cmd/Ctrl+V with clipboard
  save/restore — the default, same mechanism Wispr uses), `type` (simulated
  keystrokes), `wtype` (Wayland-native, auto-selected on pure-Wayland
  sessions), `clipboard` (copy only), `stdout` (headless/piping). If
  injection fails the text always lands on the clipboard — a dictation is
  never lost. `localflow last --copy` re-copies the most recent one.
- **Command mode** (optional, off by default): select text, hold the command
  hotkey (`ctrl+alt+c`), and speak an instruction — "make this more formal",
  "translate to French", "turn into bullet points" — and the selection is
  replaced with the result. Needs an LLM: set `llm_backend = "openai-compat"`
  to use a local Ollama/llama.cpp server (fully on-machine), or
  `"anthropic"` for the Claude API. `localflow rewrite "instruction" --text
  "..."` (or pipe stdin) runs the same thing from the terminal.
- **History & stats**: `localflow history`, `localflow stats` (words, WPM,
  estimated time saved). Stored as plain JSONL in your config dir; disable
  with `history_enabled = false`.
- **100+ languages**: `language` unset = auto-detect per utterance, or pin
  e.g. `language = "de"`.

## Configuration

```bash
localflow config init    # writes ~/.config/localflow/config.toml
localflow config show
```

```toml
# ~/.config/localflow/config.toml
model = "base"              # tiny/base/small/medium/large-v3 (+.en variants)
engine = "faster-whisper"   # or "whispercpp" (model may be a ggml file path)
hotkey = "ctrl+alt+space"   # e.g. "f9", "cmd+shift+d"
mode = "hold"               # "hold" = push-to-talk, "toggle" = tap on/off
backend = "auto"            # auto/paste/type/clipboard/stdout
spoken_punctuation = false  # say "comma"/"period" to punctuate
```

Model size guide (CPU, int8): `tiny` is fastest and fine for clear speech;
`base` is the sweet spot; `small` if accuracy matters more than ~2-3× slower
transcription. On Apple Silicon or a GPU, `small`/`medium` are comfortable.

## Platform setup

**macOS** — grant two permissions on first run: **Microphone** and
**Accessibility** (System Settings → Privacy & Security) — the latter is
required for the global hotkey listener and the synthetic paste keystroke.
The terminal/app you launch `localflow` from is what needs the grant.

**Windows** — no special permissions; run from any terminal.

**Linux (X11)** — needs `xclip` or `xsel` for the paste backend
(`sudo apt install xclip`) and PortAudio for the mic
(`sudo apt install libportaudio2`). **Wayland** — text insertion works via
the `wtype` backend (`sudo apt install wtype`; auto-selected on pure-Wayland
sessions), but pynput's global hotkey listener still needs XWayland — a
compositor-native hotkey path (`evdev`) is on the v2 list.

**Headless / servers** — skip the `desktop` extra; `localflow transcribe`
and the `stdout`/`clipboard` backends work without a GUI.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[desktop,whispercpp,dev]"
.venv/bin/pytest -m "not slow"   # fast unit tests (<1s)
.venv/bin/pytest                 # + end-to-end: espeak-ng speech → real model
```

The e2e tests synthesize speech with `espeak-ng` and run it through a real
Whisper model (faster-whisper `base`, or a local GGML tiny model at
`~/.cache/localflow/models/ggml-tiny.bin` when Hugging Face is unreachable).

See [PLAN.md](PLAN.md) for the architecture, design decisions, research on
how Wispr Flow works, and the v2 roadmap (streaming partials, command mode,
tray icon, Wayland).
