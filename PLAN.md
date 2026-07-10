# LocalFlow — a local rebuild of Wispr Flow's core features

## 1. What Wispr Flow is

Wispr Flow is an AI dictation app (Mac/Windows/Linux/iOS). The core loop:

1. **Activate** — hold a push-to-talk hotkey (Fn on Mac, Ctrl+Win on Windows) or
   toggle hands-free mode. A small recording bar appears at the bottom of the
   screen.
2. **Speak** — audio is captured while the key is held.
3. **Transcribe + auto-edit** — speech is converted to text by an ASR model
   (cloud-hosted for Wispr), then AI cleanup removes filler words ("um", "uh"),
   fixes punctuation/capitalization, applies the user's personal dictionary,
   and adapts tone to the focused app (email vs. chat).
4. **Insert** — the polished text is typed/pasted into whatever text field has
   focus, in any application.

Supporting features: personal dictionary (custom names/jargon, auto-learned),
command mode (speak an instruction to edit selected text), 100+ languages with
auto-detection, dictation history and stats (words-per-minute, time saved),
snippets, and a menu-bar settings UI.

Key facts confirmed by research (2026-07):

- **Cloud-only pipeline**: streamed audio → cloud ASR (Whisper-family) →
  fine-tuned Llama models for cleanup/tone (hosted on Baseten) → text back.
  Unusable offline; sub-700 ms p99 latency budget. The privacy fallout from
  its context-awareness capture (window screenshots sent to the cloud) is the
  product's biggest trust problem — which makes *local-only* the headline
  feature of this rebuild, not a compromise.
- **Insertion mechanism**: clipboard write + synthetic Cmd/Ctrl+V, saving and
  restoring the previous clipboard; accessibility APIs supply focus/context.
  If insertion fails, text stays on the clipboard so nothing is ever lost,
  and a "paste last transcript" shortcut exists.
- **Hotkeys**: hold `fn` (Mac) / `Ctrl+Win` (Windows) push-to-talk; double-tap
  or a second shortcut for hands-free toggle. Sessions up to 20 min.
- **Beloved details**: "scratch that" backtrack self-correction, filler
  removal that doesn't rewrite meaning, ending an utterance with
  "press enter" to submit chat messages.

Wispr's ASR and cleanup run in **their cloud**. The point of this rebuild is
the same UX with **everything local**: on-device Whisper-family ASR, local
rule-based (optionally LLM-assisted) cleanup, no audio leaving the machine.

## 2. Scope for v1 ("the basics, end to end")

Must work end to end:

- [x] Microphone capture (16 kHz mono) started/stopped by a global hotkey,
      in both **hold-to-talk** and **toggle** modes.
- [x] Local speech-to-text with pluggable engines: `faster-whisper`
      (CTranslate2, CPU int8, Silero VAD; `base` by default) or `whispercpp`
      (whisper.cpp via pywhispercpp; loads GGML model names or file paths).
- [x] Cleanup pipeline: filler-word removal, "scratch that"/"strike that"
      backtrack self-correction, whitespace/punctuation normalization,
      spoken punctuation/newline commands ("new line", "period"), sentence
      capitalization, trailing "press enter" command.
- [x] Personal dictionary: user-defined replacements + custom vocabulary
      fed to the ASR as a hotwords/initial-prompt bias.
- [x] Text injection into the focused app with selectable backends:
      clipboard-paste with clipboard save/restore (Wispr's mechanism, the
      default on desktops), typing simulation (`pynput`), clipboard-only,
      and stdout (for headless use/testing) — with automatic fall-back to
      the clipboard when injection fails so text is never lost, plus a
      `localflow last` command ("paste last transcript" parity).
- [x] History: every dictation appended to a local JSONL log with duration
      and WPM stats; `history` and `stats` CLI commands.
- [x] Config file (`~/.config/localflow/config.toml`) + CLI overrides.
- [x] File-transcription mode (`localflow transcribe foo.wav`) — doubles as
      the testable path in headless CI.

Explicitly out of scope for v1 (documented for v2):

- Streaming/partial results while speaking (v1 transcribes on key release;
  with `base` on a 4-core CPU that's roughly ≤1× audio duration).
- Command mode (edit-selected-text-by-voice) — needs an LLM; stub behind
  optional Anthropic API integration.
- Per-app tone adaptation, snippets, GUI settings window, tray icon.
- Auto-learning of dictionary words.

## 3. Architecture

```
       hold/toggle hotkey (pynput)                 ~/.config/localflow/config.toml
                 │                                              │
                 ▼                                              ▼
  ┌──────────┐  start/stop   ┌───────────┐  16kHz f32  ┌──────────────┐
  │ hotkey.py│ ────────────► │ audio.py  │ ──────────► │   asr.py     │
  └──────────┘               │ (sound-   │             │ faster-      │
                             │  device)  │             │ whisper +VAD │
                             └───────────┘             └──────┬───────┘
                                                              │ raw text
                                                              ▼
                             ┌───────────┐  polished   ┌──────────────┐
   focused app  ◄─────────── │ inject.py │ ◄────────── │  cleanup.py  │
   (type/paste/clipboard)    └───────────┘             │ +dictionary  │
                                                       └──────┬───────┘
                                                              │
                                                       ┌──────▼───────┐
                                                       │  history.py  │
                                                       │ (JSONL+stats)│
                                                       └──────────────┘
```

Package layout (Python 3.11, no heavyweight deps — no torch):

```
localflow/
  __init__.py
  config.py      # dataclass config, TOML load/save, defaults
  audio.py       # Recorder (sounddevice stream → np.float32 buffer), WAV I/O
  asr.py         # Transcriber wrapping faster_whisper.WhisperModel
  cleanup.py     # pure-function text pipeline (fillers, spoken commands, caps)
  dictionary.py  # personal dictionary: replacements + ASR hotwords
  inject.py      # Injector backends: type / paste / clipboard / stdout
  hotkey.py      # global hotkey listener (hold + toggle) via pynput
  history.py     # JSONL history, WPM/time-saved stats
  app.py         # LocalFlowApp: wires everything, daemon loop
  cli.py         # argparse CLI: run, transcribe, history, stats, dict, config
tests/
  test_cleanup.py test_dictionary.py test_history.py test_config.py
  test_inject.py test_audio.py
  test_e2e.py    # espeak-ng → WAV → transcribe → cleanup → stdout inject
```

Key design decisions:

- **faster-whisper over whisper.cpp/openai-whisper**: pip-installable, no
  torch, int8 CPU quantization, built-in Silero VAD, `initial_prompt`/
  `hotwords` for dictionary biasing.
- **pynput for hotkeys + typing**: one cross-platform dependency covers
  global key listening and keystroke injection on macOS/Windows/X11. All
  pynput imports are lazy/guarded so headless machines can still use file
  mode, stdout backend, and tests.
- **Cleanup is pure functions**: deterministic, unit-testable, no model
  needed. Optional LLM polish is an add-on layer, off by default.
- **Everything degrades gracefully headless**: no mic → file mode still
  works; no display → stdout/clipboard backends still work. This is what
  makes the container-based E2E test honest.

## 4. Testing strategy

1. **Unit tests** for every pure module (cleanup, dictionary, history,
   config, inject-stdout, WAV I/O).
2. **ASR integration test**: synthesize known sentences with `espeak-ng`,
   run them through the real `faster-whisper` model, assert the transcript
   matches (fuzzy word-overlap, since TTS→ASR isn't perfect).
3. **End-to-end test**: `localflow transcribe test.wav` via the CLI with the
   stdout backend + history enabled — exercises config → audio → ASR →
   cleanup → dictionary → inject → history in one shot.
4. **Manual daemon smoke test** is the only thing that needs a real desktop
   (mic + hotkey); everything up to that seam is covered automatically.

## 5. Milestones

| # | Milestone | Verification | Status |
|---|-----------|--------------|--------|
| 1 | Scaffolding: package, config, CLI skeleton | `localflow --help` | ✅ done |
| 2 | Cleanup + dictionary pipeline | unit tests green | ✅ done |
| 3 | ASR on WAV files (model loads, transcribes) | integration test green | ✅ done (whispercpp engine in CI container; faster-whisper verified on machines with HF access) |
| 4 | Injection backends + history | unit tests + e2e test green | ✅ done |
| 5 | Hotkey daemon (`localflow run`) | code-reviewed; manual smoke on a desktop | ✅ built + reviewed; needs a mic+GUI machine for the final manual smoke |
| 6 | Docs: README with per-OS setup (mic/accessibility permissions) | reads clean | ✅ done |

Test evidence (this container, headless): 74/74 pytest green, including
espeak-ng-synthesized speech through the real whisper.cpp tiny model, the
full pipeline (ASR → dictionary → cleanup → inject → history), and the CLI
as a subprocess. The one thing that inherently cannot be exercised headless
is the live mic + global hotkey path; its logic is unit-isolated and needs a
one-time manual smoke on a desktop.

## 6. Follow-up increment (built after v1 landed)

- [x] Double-tap hands-free mode: double-tap the hotkey to latch recording,
      tap again to stop (Wispr's hands-free gesture). Hotkey state machine
      refactored to be pure and unit-tested without pynput/display.
- [x] Snippets: voice-trigger → verbatim text block (`localflow snippet
      add/remove/list`), applied post-cleanup, triggers fed to ASR hotwords.
- [x] Wayland `wtype` injection backend, auto-selected on pure-Wayland
      sessions.
- [x] Audio cues on record start/stop (never raise headless).
- [x] Command mode: hold a second hotkey (`ctrl+alt+c`), speak an
      instruction, and the current selection is replaced with the LLM's
      rewrite (selection captured via clipboard-copy with save/restore).
      Pluggable backends — `openai-compat` (Ollama/llama.cpp: stays fully
      local) or `anthropic` (Claude API) — off by default. Also exposed as
      `localflow rewrite` for terminal/pipe use.
- [x] CI on GitHub Actions: unit + real-model e2e with the Whisper model
      cached between runs.

## 7. v2 ideas (not built now)

- Streaming partial transcripts (chunked decode while recording).
- Per-app tone profiles (apply a rewrite automatically based on the focused
  app) — the LLM plumbing from command mode makes this a small step.
- Tray icon / recording indicator overlay (pystray or a tiny Tk window).
- Auto-learn dictionary from corrections.
- Wayland-native global hotkey (evdev) — injection is covered by `wtype`.
