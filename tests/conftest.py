import shutil
import subprocess

import pytest


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Keep every test away from the real ~/.config/localflow."""
    monkeypatch.setenv("LOCALFLOW_CONFIG_DIR", str(tmp_path / "localflow-config"))
    return tmp_path


def synthesize(text: str, path, voice: str = "en-us", wpm: int = 150) -> None:
    """Generate a speech WAV with espeak-ng for ASR testing."""
    espeak = shutil.which("espeak-ng")
    if espeak is None:
        pytest.skip("espeak-ng not installed")
    subprocess.run(
        [espeak, "-v", voice, "-s", str(wpm), "-g", "6", "-w", str(path), text],
        check=True,
        capture_output=True,
    )
