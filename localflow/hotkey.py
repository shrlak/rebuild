"""Global hotkey listening for push-to-talk / toggle activation.

Uses pynput's low-level listener so we can implement *hold* semantics
(record while all hotkey keys are down, stop when any is released), which
pynput's GlobalHotKeys API doesn't support.

Hold mode also supports Wispr Flow's hands-free gesture: **double-tap** the
hotkey to lock recording on (release does nothing), then tap once more to
stop. The first quick tap of a double-tap yields a sub-quarter-second clip
that the app discards as an accidental tap.

Hotkey strings are '+'-separated, e.g. "ctrl+alt+space", "f9", "cmd+shift+d".

The key-state machine (`_on_key_down`/`_on_key_up`) is pure and driven with
already-normalized key objects, so it is unit-testable without pynput or a
display; the pynput listener is only wired up in `start()`.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

DOUBLE_TAP_SECONDS = 0.4


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

    mode="hold":   activate while the full combo is down (push-to-talk);
                   double-tap locks hands-free, one more tap unlocks.
    mode="toggle": each full combo press flips active on/off.
    """

    def __init__(
        self,
        hotkey: str | set,
        mode: str,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
        double_tap_seconds: float = DOUBLE_TAP_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        if mode not in ("hold", "toggle"):
            raise ValueError(f"mode must be 'hold' or 'toggle', got {mode!r}")
        self.combo = hotkey if isinstance(hotkey, set) else parse_hotkey(hotkey)
        self.mode = mode
        self.on_activate = on_activate
        self.on_deactivate = on_deactivate
        self.double_tap_seconds = double_tap_seconds
        self.active = False
        self.locked = False  # hands-free latch (hold mode only)
        self._clock = clock
        self._pressed = set()
        self._combo_down = False
        self._last_combo_press = float("-inf")
        self._lock = threading.Lock()
        self._listener = None

    # --- pure state machine (keys must already be canonical) ---

    def _on_key_down(self, key) -> None:
        with self._lock:
            self._pressed.add(key)
            if self._combo_down or not self.combo.issubset(self._pressed):
                return
            self._combo_down = True
            fire = None
            if self.mode == "toggle":
                self.active = not self.active
                fire = self.on_activate if self.active else self.on_deactivate
            elif self.locked:
                # hands-free is on: this tap turns it off
                self.locked = False
                self.active = False
                self._last_combo_press = float("-inf")
                fire = self.on_deactivate
            else:
                now = self._clock()
                if now - self._last_combo_press < self.double_tap_seconds:
                    self.locked = True  # second tap: latch hands-free
                self._last_combo_press = now
                if not self.active:
                    self.active = True
                    fire = self.on_activate
        if fire:
            fire()

    def _on_key_up(self, key) -> None:
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
                and not self.locked
            ):
                self.active = False
                fire = self.on_deactivate
        if fire:
            fire()

    # --- pynput wiring (runs on the listener thread) ---

    def _handle_press(self, key) -> None:
        self._on_key_down(_canonical(self._listener, key))

    def _handle_release(self, key) -> None:
        self._on_key_up(_canonical(self._listener, key))

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
