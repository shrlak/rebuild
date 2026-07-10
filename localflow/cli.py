"""Command-line interface for LocalFlow."""

from __future__ import annotations

import argparse
import datetime
import json
import sys

from . import __version__
from .config import Config
from .dictionary import Dictionary
from .history import History


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="localflow",
        description="Local push-to-talk dictation with on-device Whisper ASR.",
    )
    p.add_argument("--version", action="version", version=f"localflow {__version__}")
    p.add_argument("--config", help="path to config.toml (default: user config dir)")
    sub = p.add_subparsers(dest="command")

    run = sub.add_parser("run", help="start the dictation daemon (hotkey mode)")
    run.add_argument("--backend", choices=["auto", "type", "paste", "clipboard", "stdout"])
    run.add_argument("--model", help="whisper model size override (tiny/base/small/...)")
    run.add_argument("--engine", choices=["faster-whisper", "whispercpp"])
    run.add_argument("--hotkey", help="hotkey override, e.g. 'ctrl+alt+space' or 'f9'")
    run.add_argument("--mode", choices=["hold", "toggle"])

    tr = sub.add_parser("transcribe", help="transcribe an audio file through the full pipeline")
    tr.add_argument("file", help="audio file (wav/flac/ogg)")
    tr.add_argument("--backend", default="stdout",
                    choices=["auto", "type", "paste", "clipboard", "stdout"])
    tr.add_argument("--model", help="whisper model size override")
    tr.add_argument("--engine", choices=["faster-whisper", "whispercpp"])
    tr.add_argument("--raw", action="store_true", help="print raw ASR text, skip cleanup")
    tr.add_argument("--json", action="store_true", help="print full result entry as JSON")

    rw = sub.add_parser("rewrite", help="command mode: apply an instruction to text via the configured LLM")
    rw.add_argument("instruction", help="e.g. 'make this more formal'")
    rw.add_argument("--text", help="text to transform (default: read from stdin if piped)")

    hist = sub.add_parser("history", help="show recent dictations")
    hist.add_argument("-n", type=int, default=10, help="number of entries (default 10)")

    last = sub.add_parser("last", help="print the most recent dictation")
    last.add_argument("--copy", action="store_true", help="also copy it to the clipboard")

    sub.add_parser("stats", help="show dictation stats")

    d = sub.add_parser("dict", help="manage the personal dictionary")
    dsub = d.add_subparsers(dest="dict_command", required=True)
    dadd = dsub.add_parser("add", help="add replacement: spoken form -> written form")
    dadd.add_argument("spoken")
    dadd.add_argument("written")
    drm = dsub.add_parser("remove", help="remove a replacement")
    drm.add_argument("spoken")
    dvocab = dsub.add_parser("vocab", help="add vocabulary word(s) to bias the ASR")
    dvocab.add_argument("words", nargs="+")
    dsub.add_parser("list", help="show dictionary contents")

    s = sub.add_parser("snippet", help="manage voice-triggered snippets")
    ssub = s.add_subparsers(dest="snippet_command", required=True)
    sadd = ssub.add_parser("add", help="add snippet: trigger phrase -> inserted text")
    sadd.add_argument("trigger")
    sadd.add_argument("text", help=r"snippet body; \n becomes a newline")
    srm = ssub.add_parser("remove", help="remove a snippet")
    srm.add_argument("trigger")
    ssub.add_parser("list", help="show snippets")

    c = sub.add_parser("config", help="show or initialize configuration")
    csub = c.add_subparsers(dest="config_command", required=True)
    csub.add_parser("show", help="print effective config")
    csub.add_parser("init", help="write a default config.toml to edit")
    csub.add_parser("path", help="print the config file path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.command:
        build_parser().print_help()
        return 0

    from pathlib import Path

    cfg = Config.load(Path(args.config)) if args.config else Config.load()

    if args.command in ("run", "transcribe"):
        if getattr(args, "model", None):
            cfg.model = args.model
        if getattr(args, "engine", None):
            cfg.engine = args.engine
        if getattr(args, "hotkey", None):
            cfg.hotkey = args.hotkey
        if getattr(args, "mode", None):
            cfg.mode = args.mode

        from .app import LocalFlowApp

        if args.command == "run":
            LocalFlowApp(cfg, backend=args.backend).run()
            return 0

        # transcribe
        app = LocalFlowApp(cfg, backend=args.backend)
        if args.raw:
            result = app.transcriber.transcribe(
                args.file, language=cfg.language, hotwords=app.dictionary.hotwords()
            )
            print(result.text)
            return 0
        entry = app.transcribe_file(args.file)
        if entry is None:
            print("(no speech detected)", file=sys.stderr)
            return 1
        if args.json:
            from dataclasses import asdict

            print(json.dumps(asdict(entry), ensure_ascii=False, indent=2))
        return 0

    if args.command == "rewrite":
        from .llm import create_backend

        backend = create_backend(
            cfg.llm_backend, model=cfg.llm_model, base_url=cfg.llm_base_url
        )
        if backend is None:
            print(
                "command mode is disabled — set llm_backend to 'openai-compat' "
                "(local, e.g. Ollama) or 'anthropic' in the config",
                file=sys.stderr,
            )
            return 1
        text = args.text
        if text is None and not sys.stdin.isatty():
            text = sys.stdin.read()
        print(backend.rewrite(args.instruction, text or None))
        return 0

    if args.command == "history":
        entries = History(cfg.history_path).entries()
        if not entries:
            print("no dictations yet")
            return 0
        for e in entries[-args.n:]:
            when = datetime.datetime.fromtimestamp(e.ts).strftime("%Y-%m-%d %H:%M")
            print(f"[{when}] ({e.words}w, {e.wpm:.0f}wpm) {e.text}")
        return 0

    if args.command == "last":
        entries = History(cfg.history_path).entries()
        if not entries:
            print("no dictations yet", file=sys.stderr)
            return 1
        text = entries[-1].text
        print(text)
        if args.copy:
            from .inject import ClipboardInjector

            ClipboardInjector().inject(text)
            print("(copied to clipboard)", file=sys.stderr)
        return 0

    if args.command == "stats":
        for key, value in History(cfg.history_path).stats().items():
            print(f"{key.replace('_', ' ')}: {value}")
        return 0

    if args.command == "dict":
        d = Dictionary(cfg.dictionary_path)
        if args.dict_command == "add":
            d.add_replacement(args.spoken, args.written)
            d.save()
            print(f"added: {args.spoken!r} -> {args.written!r}")
        elif args.dict_command == "remove":
            if d.remove_replacement(args.spoken):
                d.save()
                print(f"removed {args.spoken!r}")
            else:
                print(f"not found: {args.spoken!r}", file=sys.stderr)
                return 1
        elif args.dict_command == "vocab":
            for word in args.words:
                d.add_vocabulary(word)
            d.save()
            print(f"vocabulary: {', '.join(d.vocabulary)}")
        elif args.dict_command == "list":
            print(json.dumps(
                {"replacements": d.replacements, "vocabulary": d.vocabulary},
                indent=2, ensure_ascii=False,
            ))
        return 0

    if args.command == "snippet":
        d = Dictionary(cfg.dictionary_path)
        if args.snippet_command == "add":
            d.add_snippet(args.trigger, args.text.replace("\\n", "\n"))
            d.save()
            print(f"added snippet {args.trigger!r}")
        elif args.snippet_command == "remove":
            if d.remove_snippet(args.trigger):
                d.save()
                print(f"removed snippet {args.trigger!r}")
            else:
                print(f"not found: {args.trigger!r}", file=sys.stderr)
                return 1
        elif args.snippet_command == "list":
            print(json.dumps(d.snippets, indent=2, ensure_ascii=False))
        return 0

    if args.command == "config":
        if args.config_command == "show":
            import dataclasses

            for f in dataclasses.fields(cfg):
                print(f"{f.name} = {getattr(cfg, f.name)!r}")
        elif args.config_command == "init":
            path = cfg.save()
            print(f"wrote {path}")
        elif args.config_command == "path":
            print(Config.config_path())
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
