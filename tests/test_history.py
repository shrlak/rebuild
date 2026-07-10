from localflow.history import History


def test_record_and_read(tmp_path):
    h = History(tmp_path / "history.jsonl")
    e = h.record(
        text="hello world out there",
        raw_text="um hello world out there",
        language="en",
        audio_seconds=3.0,
        latency_seconds=0.8,
    )
    assert e.words == 4
    assert e.wpm == 80.0  # 4 words / 0.05 min
    entries = h.entries()
    assert len(entries) == 1
    assert entries[0].text == "hello world out there"
    assert entries[0].raw_text.startswith("um")


def test_disabled_history_returns_entry_but_writes_nothing(tmp_path):
    h = History(tmp_path / "history.jsonl", enabled=False)
    e = h.record("hi", "hi", "en", 1.0, 0.1)
    assert e.words == 1
    assert not h.path.exists()
    assert h.entries() == []


def test_stats(tmp_path):
    h = History(tmp_path / "history.jsonl")
    h.record("one two three four", "raw", "en", 2.0, 0.5)
    h.record("five six", "raw", "en", 1.0, 0.4)
    s = h.stats()
    assert s["dictations"] == 2
    assert s["total_words"] == 6
    assert s["total_audio_seconds"] == 3.0
    assert s["average_wpm"] == 120.0
    assert s["estimated_minutes_saved"] > 0


def test_empty_stats(tmp_path):
    s = History(tmp_path / "none.jsonl").stats()
    assert s["dictations"] == 0
    assert s["average_wpm"] == 0.0
