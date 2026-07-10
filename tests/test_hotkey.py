"""Unit tests for the hotkey state machine.

The listener's key logic is driven directly with plain strings as "keys",
so no pynput, keyboard, or display is required.
"""

from localflow.hotkey import HotkeyListener


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


def make(mode="hold", combo=None):
    events = []
    clock = FakeClock()
    listener = HotkeyListener(
        hotkey=combo or {"ctrl", "space"},
        mode=mode,
        on_activate=lambda: events.append("on"),
        on_deactivate=lambda: events.append("off"),
        clock=clock,
    )
    return listener, events, clock


class TestHoldMode:
    def test_push_to_talk(self):
        listener, events, _ = make()
        listener._on_key_down("ctrl")
        assert events == []
        listener._on_key_down("space")
        assert events == ["on"]
        listener._on_key_up("space")
        assert events == ["on", "off"]

    def test_release_of_any_combo_key_deactivates(self):
        listener, events, _ = make()
        listener._on_key_down("ctrl")
        listener._on_key_down("space")
        listener._on_key_up("ctrl")
        assert events == ["on", "off"]

    def test_non_combo_keys_ignored(self):
        listener, events, _ = make()
        listener._on_key_down("a")
        listener._on_key_up("a")
        listener._on_key_down("ctrl")
        listener._on_key_up("ctrl")
        assert events == []

    def test_extra_keys_do_not_break_combo(self):
        listener, events, _ = make()
        listener._on_key_down("ctrl")
        listener._on_key_down("a")  # typing while holding ctrl
        listener._on_key_down("space")
        assert events == ["on"]

    def test_repeated_key_events_do_not_refire(self):
        # OS auto-repeat sends key-down again while held
        listener, events, _ = make()
        listener._on_key_down("ctrl")
        listener._on_key_down("space")
        listener._on_key_down("space")
        listener._on_key_down("space")
        assert events == ["on"]

    def test_two_slow_taps_are_two_dictations(self):
        listener, events, clock = make()
        for _ in range(2):
            listener._on_key_down("ctrl")
            listener._on_key_down("space")
            clock.advance(0.1)
            listener._on_key_up("space")
            listener._on_key_up("ctrl")
            clock.advance(1.0)
        assert events == ["on", "off", "on", "off"]


class TestDoubleTapHandsFree:
    def tap(self, listener, clock, hold=0.05):
        listener._on_key_down("ctrl")
        listener._on_key_down("space")
        clock.advance(hold)
        listener._on_key_up("space")
        listener._on_key_up("ctrl")

    def test_double_tap_locks_recording(self):
        listener, events, clock = make()
        self.tap(listener, clock)          # first tap: on, off (short clip)
        clock.advance(0.2)                 # within double-tap window
        listener._on_key_down("ctrl")
        listener._on_key_down("space")     # second tap: on + lock
        listener._on_key_up("space")
        listener._on_key_up("ctrl")        # release ignored: locked
        assert events == ["on", "off", "on"]
        assert listener.locked and listener.active

    def test_tap_while_locked_stops(self):
        listener, events, clock = make()
        self.tap(listener, clock)
        clock.advance(0.2)
        self.tap(listener, clock)          # locks
        clock.advance(2.0)                 # speak for a while, hands-free
        self.tap(listener, clock)          # tap again: unlock + stop
        assert events == ["on", "off", "on", "off"]
        assert not listener.locked and not listener.active

    def test_slow_second_tap_does_not_lock(self):
        listener, events, clock = make()
        self.tap(listener, clock)
        clock.advance(1.0)                 # too slow for a double-tap
        self.tap(listener, clock)
        assert not listener.locked
        assert events == ["on", "off", "on", "off"]

    def test_hold_after_unlock_is_normal_ptt(self):
        listener, events, clock = make()
        self.tap(listener, clock)
        clock.advance(0.2)
        self.tap(listener, clock)          # locked
        clock.advance(2.0)
        self.tap(listener, clock)          # unlocked
        clock.advance(2.0)
        listener._on_key_down("ctrl")
        listener._on_key_down("space")
        clock.advance(1.0)
        listener._on_key_up("space")
        assert events == ["on", "off", "on", "off", "on", "off"]


class TestToggleMode:
    def test_toggle(self):
        listener, events, _ = make(mode="toggle")
        listener._on_key_down("ctrl")
        listener._on_key_down("space")
        assert events == ["on"]
        listener._on_key_up("space")
        listener._on_key_up("ctrl")
        assert events == ["on"]  # release does nothing in toggle mode
        listener._on_key_down("ctrl")
        listener._on_key_down("space")
        assert events == ["on", "off"]


def test_parse_hotkey_requires_pynput():
    import pytest

    pytest.importorskip("pynput", exc_type=ImportError)
    from localflow.hotkey import parse_hotkey

    keys = parse_hotkey("ctrl+alt+space")
    assert len(keys) == 3
    with pytest.raises(ValueError):
        parse_hotkey("ctrl+bogus_key")
    with pytest.raises(ValueError):
        parse_hotkey("")
