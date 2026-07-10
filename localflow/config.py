"""Configuration: dataclass defaults + TOML file at ~/.config/localflow/."""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def config_dir() -> Path:
    base = os.environ.get("LOCALFLOW_CONFIG_DIR")
    if base:
        return Path(base)
    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(xdg) / "localflow"


@dataclass
class Config:
    # --- ASR ---
    engine: str = "faster-whisper"  # "faster-whisper" or "whispercpp"
    model: str = "base"            # tiny/base/small/medium/large-v3, *.en variant,
                                   # or (whispercpp) a path to a ggml-*.bin file
    language: str | None = None    # None = auto-detect
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    # --- activation ---
    hotkey: str = "ctrl+alt+space"
    mode: str = "hold"             # "hold" (push-to-talk) or "toggle"
    # --- output ---
    backend: str = "auto"          # auto/type/paste/clipboard/stdout
    # --- cleanup ---
    remove_fillers: bool = True
    backtrack: bool = True         # "scratch that" / "strike that" self-correction
    spoken_newlines: bool = True   # "new line" / "new paragraph" -> line breaks
    spoken_punctuation: bool = False  # "period", "comma", ... -> marks
    capitalize_sentences: bool = True
    press_enter_command: bool = True  # trailing "press enter" sends the Enter key
    # --- audio ---
    sample_rate: int = 16000
    max_seconds: float = 120.0     # safety cap per utterance
    # --- storage ---
    history_enabled: bool = True

    @property
    def dictionary_path(self) -> Path:
        return config_dir() / "dictionary.json"

    @property
    def history_path(self) -> Path:
        return config_dir() / "history.jsonl"

    @classmethod
    def config_path(cls) -> Path:
        return config_dir() / "config.toml"

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or cls.config_path()
        cfg = cls()
        if path.exists():
            data = tomllib.loads(path.read_text())
            valid = {f.name for f in dataclasses.fields(cls)}
            for key, value in data.items():
                if key not in valid:
                    raise ValueError(f"Unknown config key {key!r} in {path}")
                setattr(cfg, key, value)
        if cfg.mode not in ("hold", "toggle"):
            raise ValueError(f"mode must be 'hold' or 'toggle', got {cfg.mode!r}")
        if cfg.backend not in ("auto", "type", "paste", "clipboard", "stdout"):
            raise ValueError(f"unknown backend {cfg.backend!r}")
        if cfg.engine not in ("faster-whisper", "whispercpp"):
            raise ValueError(f"unknown engine {cfg.engine!r}")
        return cfg

    def save(self, path: Path | None = None) -> Path:
        path = path or self.config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# LocalFlow configuration\n"]
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if value is None:
                lines.append(f"# {f.name} = ...  (unset: auto)")
                continue
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, (int, float)):
                rendered = str(value)
            else:
                rendered = '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
            lines.append(f"{f.name} = {rendered}")
        path.write_text("\n".join(lines) + "\n")
        return path
