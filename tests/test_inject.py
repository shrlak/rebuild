import pytest

from localflow.inject import StdoutInjector, get_injector, gui_available


def test_stdout_injector(capsys):
    StdoutInjector().inject("hello there")
    assert capsys.readouterr().out == "hello there\n"


def test_get_injector_by_name():
    assert get_injector("stdout").name == "stdout"
    assert get_injector("clipboard").name == "clipboard"
    assert get_injector("type").name == "type"
    assert get_injector("paste").name == "paste"


def test_get_injector_unknown():
    with pytest.raises(ValueError):
        get_injector("carrier-pigeon")


def test_auto_falls_back_to_stdout_headless(monkeypatch):
    import sys

    if sys.platform in ("darwin", "win32"):
        pytest.skip("headless fallback is linux-only behavior")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert not gui_available()
    assert get_injector("auto").name == "stdout"


def test_auto_picks_paste_with_display(monkeypatch):
    import sys

    if sys.platform in ("darwin", "win32"):
        pytest.skip("DISPLAY check is linux-only behavior")
    monkeypatch.setenv("DISPLAY", ":0")
    assert get_injector("auto").name == "paste"


def test_auto_picks_wtype_on_pure_wayland(monkeypatch):
    import sys

    if sys.platform in ("darwin", "win32"):
        pytest.skip("Wayland check is linux-only behavior")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr("localflow.inject.shutil.which", lambda name: "/usr/bin/wtype")
    assert get_injector("auto").name == "wtype"


def test_wtype_injector_invokes_wtype(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "localflow.inject.subprocess.run",
        lambda cmd, check: calls.append(cmd),
    )
    inj = get_injector("wtype")
    inj.inject("hello there")
    inj.press_enter()
    assert calls == [["wtype", "--", "hello there"], ["wtype", "-k", "Return"]]


def test_play_cue_never_raises_headless():
    from localflow.feedback import play_cue

    play_cue("start")
    play_cue("stop")
