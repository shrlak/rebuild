"""Command-mode backend tests. No network: servers and SDKs are faked."""

import io
import json
import urllib.request

import pytest

from localflow.llm import (
    DEFAULT_MODELS,
    SYSTEM_PROMPT,
    AnthropicBackend,
    OpenAICompatBackend,
    build_user_message,
    create_backend,
)


class TestFactory:
    def test_none_disables(self):
        assert create_backend("none") is None

    def test_openai_compat(self):
        backend = create_backend("openai-compat", base_url="http://localhost:8080/v1")
        assert backend.name == "openai-compat"
        assert backend.base_url == "http://localhost:8080/v1"
        assert backend.model == DEFAULT_MODELS["openai-compat"]

    def test_anthropic_default_model(self):
        backend = create_backend("anthropic")
        assert backend.model == "claude-opus-4-8"

    def test_model_override(self):
        assert create_backend("openai-compat", model="qwen2.5").model == "qwen2.5"

    def test_unknown_backend(self):
        with pytest.raises(ValueError):
            create_backend("skynet")


class TestPromptShape:
    def test_with_text(self):
        msg = build_user_message("make it formal", "hey what's up")
        assert "Instruction: make it formal" in msg
        assert "hey what's up" in msg

    def test_without_text(self):
        msg = build_user_message("write a haiku about rain", None)
        assert "No text is selected" in msg

    def test_system_prompt_demands_bare_output(self):
        assert "ONLY the resulting text" in SYSTEM_PROMPT


class TestOpenAICompat:
    def test_rewrite_round_trip(self, monkeypatch):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data)
            reply = {"choices": [{"message": {"content": "  Good day to you.  "}}]}

            class Response(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return Response(json.dumps(reply).encode())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        backend = OpenAICompatBackend(base_url="http://localhost:11434/v1")
        out = backend.rewrite("make it formal", "hey")
        assert out == "Good day to you."
        assert captured["url"] == "http://localhost:11434/v1/chat/completions"
        assert captured["body"]["model"] == DEFAULT_MODELS["openai-compat"]
        roles = [m["role"] for m in captured["body"]["messages"]]
        assert roles == ["system", "user"]
        assert "make it formal" in captured["body"]["messages"][1]["content"]


class TestAnthropic:
    def test_rewrite_uses_sdk(self, monkeypatch):
        anthropic = pytest.importorskip("anthropic")
        captured = {}

        class FakeBlock:
            type = "text"
            text = "Bonjour le monde"

        class FakeMessages:
            def create(self, **kwargs):
                captured.update(kwargs)

                class Response:
                    content = [FakeBlock()]

                return Response()

        class FakeClient:
            messages = FakeMessages()

        monkeypatch.setattr(anthropic, "Anthropic", lambda: FakeClient())
        backend = AnthropicBackend()
        out = backend.rewrite("translate to french", "hello world")
        assert out == "Bonjour le monde"
        assert captured["model"] == "claude-opus-4-8"
        assert captured["system"] == SYSTEM_PROMPT
        assert captured["output_config"] == {"effort": "low"}
        assert "hello world" in captured["messages"][0]["content"]


class TestAppIntegration:
    def test_run_command_disabled_raises(self, monkeypatch):
        from localflow.app import LocalFlowApp
        from localflow.config import Config

        cfg = Config.load()
        app = LocalFlowApp(cfg, backend="stdout")
        assert app.command_backend is None
        with pytest.raises(RuntimeError, match="command mode is disabled"):
            app.run_command("do something")

    def test_run_command_with_backend(self):
        from localflow.app import LocalFlowApp
        from localflow.config import Config

        cfg = Config.load()
        app = LocalFlowApp(cfg, backend="stdout")

        class FakeBackend:
            def rewrite(self, instruction, text=None):
                return f"[{instruction}] {text}"

        app.command_backend = FakeBackend()
        assert app.run_command("shout", "hi") == "[shout] hi"


class TestCliRewrite:
    def test_disabled_by_default(self, capsys):
        from localflow.cli import main

        assert main(["rewrite", "make it formal", "--text", "hey"]) == 1
        assert "disabled" in capsys.readouterr().err

    def test_with_openai_compat(self, monkeypatch, tmp_path, capsys):
        from localflow.cli import main
        from localflow.config import Config

        cfg = Config()
        cfg.llm_backend = "openai-compat"
        cfg.save()

        def fake_urlopen(request, timeout):
            reply = {"choices": [{"message": {"content": "FORMAL HEY"}}]}

            class Response(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return Response(json.dumps(reply).encode())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        assert main(["rewrite", "make it formal", "--text", "hey"]) == 0
        assert capsys.readouterr().out.strip() == "FORMAL HEY"
