import numpy as np

from localflow.audio import duration_seconds, load_wav, resample, save_wav


def make_tone(seconds=1.0, rate=16000, hz=440.0):
    t = np.linspace(0, seconds, int(seconds * rate), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_save_load_roundtrip(tmp_path):
    audio = make_tone()
    path = tmp_path / "tone.wav"
    save_wav(path, audio, 16000)
    loaded = load_wav(path)
    assert loaded.dtype == np.float32
    assert abs(loaded.size - audio.size) < 4
    assert np.max(np.abs(loaded[:1000] - audio[:1000])) < 0.01


def test_load_resamples_to_16k(tmp_path):
    audio = make_tone(seconds=1.0, rate=44100)
    path = tmp_path / "hi.wav"
    save_wav(path, audio, 44100)
    loaded = load_wav(path, target_rate=16000)
    assert abs(loaded.size - 16000) < 10


def test_load_stereo_downmixes(tmp_path):
    import soundfile as sf

    mono = make_tone()
    stereo = np.stack([mono, mono], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, 16000)
    loaded = load_wav(path)
    assert loaded.ndim == 1


def test_resample_identity():
    audio = make_tone()
    assert resample(audio, 16000, 16000) is audio


def test_resample_empty():
    empty = np.zeros(0, dtype=np.float32)
    assert resample(empty, 44100, 16000).size == 0


def test_duration():
    assert duration_seconds(np.zeros(16000, dtype=np.float32)) == 1.0
