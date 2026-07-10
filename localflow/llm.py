"""Command mode LLM backends: apply a spoken instruction to text.

Wispr Flow's command mode ("make this more professional", "translate to
French") requires a language model. LocalFlow keeps this optional and
pluggable — off by default to preserve the local-only story:

- ``openai-compat``: any OpenAI-compatible chat-completions server. Point it
  at Ollama or llama.cpp on localhost and command mode stays fully local.
  Uses only the standard library (urllib), no extra dependency.
- ``anthropic``: the Claude API via the official SDK (``pip install
  anthropic``); needs ANTHROPIC_API_KEY. Not local — opt-in for people who
  want the best rewrite quality.

Backends implement ``rewrite(instruction, text) -> str``.
"""

from __future__ import annotations

import json
import urllib.request

SYSTEM_PROMPT = (
    "You are the text-editing engine inside a voice dictation app. The user "
    "spoke an instruction, and may have selected text for it to act on. "
    "Apply the instruction and return ONLY the resulting text - no "
    "explanations, no surrounding quotes, no markdown code fences. If no "
    "text is provided, generate exactly the text the instruction asks for."
)

DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai-compat": "llama3.2",
}


def build_user_message(instruction: str, text: str | None) -> str:
    if text:
        return f"Instruction: {instruction}\n\nText:\n{text}"
    return f"Instruction: {instruction}\n\n(No text is selected.)"


class OpenAICompatBackend:
    """Talks to any /v1/chat/completions server (Ollama, llama.cpp, ...)."""

    name = "openai-compat"

    def __init__(self, base_url: str = "http://localhost:11434/v1",
                 model: str | None = None, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model or DEFAULT_MODELS["openai-compat"]
        self.timeout = timeout

    def rewrite(self, instruction: str, text: str | None = None) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(instruction, text)},
            ],
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read())
        return data["choices"][0]["message"]["content"].strip()


class AnthropicBackend:
    """Claude API via the official SDK. Requires ANTHROPIC_API_KEY."""

    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_MODELS["anthropic"]
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "the anthropic backend needs the SDK: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def rewrite(self, instruction: str, text: str | None = None) -> str:
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=4096,
            output_config={"effort": "low"},  # fast turnaround for edits
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": build_user_message(instruction, text)}
            ],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()


def create_backend(backend: str, model: str | None = None,
                   base_url: str = "http://localhost:11434/v1"):
    """Return a command-mode backend, or None when disabled."""
    if backend == "none":
        return None
    if backend == "openai-compat":
        return OpenAICompatBackend(base_url=base_url, model=model)
    if backend == "anthropic":
        return AnthropicBackend(model=model)
    raise ValueError(
        f"unknown llm backend {backend!r}; choose none, openai-compat, or anthropic"
    )
