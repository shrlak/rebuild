"""Deliver finished text into the focused application.

Backends, mirroring how dictation apps insert text:

- ``type``: simulate keystrokes with pynput (works on macOS, Windows, X11).
- ``paste``: put text on the clipboard and send Cmd/Ctrl+V, then restore the
  previous clipboard. Fastest for long text.
- ``clipboard``: copy only; the user pastes manually.
- ``stdout``: print to stdout — for piping, headless machines, and tests.

``auto`` picks: type when a GUI session is detectable, else stdout.
"""

from __future__ import annotations

import sys
import time


class Injector:
    name = "base"

    def inject(self, text: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def press_enter(self) -> None:
        """Send the Enter key (spoken "press enter"). No-op by default."""


def _send_enter_key() -> None:
    from pynput.keyboard import Controller, Key

    keyboard = Controller()
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)


class StdoutInjector(Injector):
    name = "stdout"

    def inject(self, text: str) -> None:
        print(text, flush=True)


class ClipboardInjector(Injector):
    name = "clipboard"

    def inject(self, text: str) -> None:
        import pyperclip

        pyperclip.copy(text)


class TypeInjector(Injector):
    name = "type"

    def inject(self, text: str) -> None:
        from pynput.keyboard import Controller

        Controller().type(text)

    def press_enter(self) -> None:
        _send_enter_key()


class PasteInjector(Injector):
    name = "paste"

    def inject(self, text: str) -> None:
        import pyperclip
        from pynput.keyboard import Controller, Key

        keyboard = Controller()
        try:
            previous = pyperclip.paste()
        except Exception:
            previous = None
        pyperclip.copy(text)
        modifier = Key.cmd if sys.platform == "darwin" else Key.ctrl
        time.sleep(0.05)  # let the clipboard settle before the paste chord
        with keyboard.pressed(modifier):
            keyboard.press("v")
            keyboard.release("v")
        if previous is not None:
            time.sleep(0.3)  # target app must read the clipboard first
            pyperclip.copy(previous)

    def press_enter(self) -> None:
        _send_enter_key()


def gui_available() -> bool:
    if sys.platform in ("darwin", "win32"):
        return True
    import os

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def get_injector(backend: str = "auto") -> Injector:
    # Paste is what Wispr Flow uses on desktop: instant for long text and
    # immune to keyboard-layout quirks. Set backend="type" if your platform
    # lacks clipboard tooling (Linux needs xclip/xsel or wl-clipboard).
    if backend == "auto":
        backend = "paste" if gui_available() else "stdout"
    backends = {
        "stdout": StdoutInjector,
        "clipboard": ClipboardInjector,
        "type": TypeInjector,
        "paste": PasteInjector,
    }
    if backend not in backends:
        raise ValueError(f"unknown injection backend {backend!r}")
    return backends[backend]()
