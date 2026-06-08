"""Backend-resolution and SSE-parsing tests (no server required)."""

import pytest

from llmfit.backends import parse_sse_line, resolve
from llmfit.ollama import OllamaError


def test_resolve_ollama_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    kind, label, url = resolve(None, None)
    assert kind == "ollama"
    assert "11434" in url


def test_resolve_openai_aliases():
    for name, port in [("llamacpp", "8080"), ("mlx", "8080"),
                       ("lmstudio", "1234"), ("vllm", "8000")]:
        kind, label, url = resolve(name, None)
        assert kind == name
        assert port in url


def test_resolve_explicit_url_wins():
    _, _, url = resolve("mlx", "http://gpu.box:9999/v1")
    assert url == "http://gpu.box:9999/v1"


def test_resolve_unknown_raises():
    with pytest.raises(OllamaError):
        resolve("nope", None)


def test_parse_sse_content():
    line = 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
    assert parse_sse_line(line) == "Hello"


def test_parse_sse_done_sentinel():
    assert parse_sse_line("data: [DONE]") == "[DONE]"


def test_parse_sse_ignores_role_only_and_blanks():
    assert parse_sse_line('data: {"choices":[{"delta":{"role":"assistant"}}]}') is None
    assert parse_sse_line("") is None
    assert parse_sse_line(": keep-alive comment") is None
    assert parse_sse_line("data: not-json") is None
