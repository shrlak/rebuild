"""Global hotkey listening for push-to-talk / toggle activation.

Uses pynput's low-level listener so we can implement *hold* semantics
(record while all hotkey keys are down, stop when any is released), which
pynput's GlobalHotKeys API doesn't support.

Hotkey strings are '+'-separated, e.g. "ctrl+alt+space", "f9", "cmd+shift+d".
"""

from __future__ import annotations

import threading
from typing import Callable


def parse_hotkey(spec: str):
    """Return the set of pynput key objects for a hotkey spec string."""
    from pynput.keyboard import Key, KeyCode

    aliases = {
        "ctrl": Key.ctrl, "control": Key.ctrl,
        "alt": Key.alt, "option": Key.alt, "opt": Key.alt,
        "shift": Key.shift,
        "cmd": Key.cmd, "command": Key.cmd, "super": Key.cmd, "win": Key.cmd,
        "space": Key.space, "tab": Key.tab, "enter": Key.enter,
        "esc": Key.esc, "escape": Key.esc,
        "caps_lock": Key.caps_lock, "capslock": Key.caps_lock,
        "home": Key.home, "end": Key.end,
        "insert": Key.insert, "pause": Key.pause,
        "scroll_lock": Key.scroll_lock, "menu": Key.menu,
    }
    for i in range(1, 21):
        aliases[f"f{i}"] = getattr(Key, f"f{i}")

    keys = set()
    for part in spec.lower().split("+"):
        part = part.strip()
        if not part:
            continue
        if part in aliases:
            keys.add(aliases[part])
        elif len(part) == 1:
            keys.add(KeyCode.from_char(part))
        else:
            raise ValueError(f"unknown key {part!r} in hotkey {spec!r}")
    if not keys:
        raise ValueError(f"empty hotkey spec {spec!r}")
    return keys


def _canonical(listener, key):
    """Normalize left/right modifier variants (ctrl_l -> ctrl, etc.)."""
    from pynput.keyboard import Key

    key = listener.canonical(key)
    mods = {
        Key.ctrl_l: Key.ctrl, Key.ctrl_r: Key.ctrl,
        Key.alt_l: Key.alt, Key.alt_r: Key.alt, Key.alt_gr: Key.alt,
        Key.shift_l: Key.shift, Key.shift_r: Key.shift,
        Key.cmd_l: Key.cmd, Key.cmd_r: Key.cmd,
    }
    return mods.get(key, key)


class HotkeyListener:
    """Fires on_activate/on_deactivate around the hotkey.

    mode="hold":   activate when the full combo goes down, deactivate when
                   any key of it is released (push-to-talk).
    mode="toggle": each full combo press flips active on/off.
    """

    def __init__(
        self,
        hotkey: str,
        mode: str,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ):
        if mode not in ("hold", "toggle"):
            raise ValueError(f"mode must be 'hold' or 'toggle', got {mode!r}")
        self.combo = parse_hotkey(hotkey)
        self.mode = mode
        self.on_activate = on_activate
        self.on_deactivate = on_deactivate
        self.active = False
        self._pressed = set()
        self._combo_down = False
        self._lock = threading.Lock()
        self._listener = None

    # The two callbacks below run on pynput's listener thread.

    def _handle_press(self, key) -> None:
        key = _canonical(self._listener, key)
        with self._lock:
            self._pressed.add(key)
            if self._combo_down or not self.combo.issubset(self._pressed):
                return
            self._combo_down = True
            if self.mode == "hold":
                self.active = True
                fire = self.on_activate
            else:
                self.active = not self.active
                fire = self.on_activate if self.active else self.on_deactivate
        fire()

    def _handle_release(self, key) -> None:
        key = _canonical(self._listener, key)
        with self._lock:
            self._pressed.discard(key)
            combo_was_down = self._combo_down
            if key in self.combo:
                self._combo_down = False
            fire = None
            if (
                self.mode == "hold"
                and combo_was_down
                and key in self.combo
                and self.active
            ):
                self.active = False
                fire = self.on_deactivate
        if fire:
            fire()

    def start(self) -> None:
        from pynput import keyboard

        self._listener = keyboard.Listener(
            on_press=self._handle_press, on_release=self._handle_release
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def join(self) -> None:
        if self._listener:
            self._listener.join()
