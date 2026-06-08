"""Benchmark backends beyond Ollama.

llmfit's hardware sizing (``scan`` / ``can``) is runtime-agnostic and needs no
server at all. This module adds *measured* benchmarking for any
OpenAI-compatible local server — **llama.cpp** (``llama-server``), **MLX**
(``mlx_lm.server``), **LM Studio**, **vLLM**, etc. — by streaming a completion
and timing tokens as they arrive.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .ollama import BenchResult, OllamaError

# OpenAI-compatible servers and their usual base URLs.
OPENAI_COMPAT_ALIASES = {
    "llamacpp": ("llama.cpp", "http://localhost:8080/v1"),
    "llama.cpp": ("llama.cpp", "http://localhost:8080/v1"),
    "mlx": ("MLX", "http://localhost:8080/v1"),
    "lmstudio": ("LM Studio", "http://localhost:1234/v1"),
    "vllm": ("vLLM", "http://localhost:8000/v1"),
    "openai": ("OpenAI-compatible", "http://localhost:8080/v1"),
}

BACKEND_CHOICES = ["ollama", "llamacpp", "mlx", "lmstudio", "vllm", "openai"]


@dataclass
class OpenAICompatBench:
    base_url: str = "http://localhost:8080/v1"
    label: str = "OpenAI-compatible"
    api_key: str = "-"
    timeout: float = 300.0

    def is_up(self) -> bool:
        try:
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            urllib.request.urlopen(req, timeout=5).close()
            return True
        except urllib.error.URLError:
            return False

    def models(self) -> list[str]:
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            body = json.loads(urllib.request.urlopen(req, timeout=5).read())
        except urllib.error.URLError as exc:
            raise OllamaError(f"Could not reach {self.label} at {self.base_url}.") from exc
        return [m.get("id", "?") for m in body.get("data", [])]

    def benchmark(self, model: str, *, num_tokens: int = 128) -> BenchResult:
        """Stream a completion and time prefill (time-to-first-token) vs decode."""
        payload = {
            "model": model,
            "messages": [{"role": "user",
                          "content": "Write a detailed paragraph about the "
                                     "history of computing."}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": num_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        start = time.perf_counter()
        first_token_at: float | None = None
        last_token_at = start
        count = 0
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    piece = parse_sse_line(raw.decode("utf-8", "ignore"))
                    if piece is None:
                        continue
                    if piece == "[DONE]":
                        break
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    count += 1
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            raise OllamaError(f"{self.label} returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Could not reach {self.label} at {self.base_url}."
            ) from exc

        if first_token_at is None or count == 0:
            raise OllamaError("server streamed no tokens.")
        ttft = first_token_at - start
        decode_span = max(last_token_at - first_token_at, 1e-9)
        # tokens after the first are produced during the decode span.
        decode_tps = (count - 1) / decode_span if count > 1 else 0.0
        return BenchResult(
            model=model,
            eval_tokens=count,
            eval_tps=decode_tps,
            prompt_tokens=0,
            prompt_tps=0.0,
            load_seconds=ttft,  # time-to-first-token (prefill + any load)
        )


def parse_sse_line(line: str) -> str | None:
    """Extract one generated-token delta from an SSE line, or signal completion.

    Returns the delta text, the literal ``"[DONE]"`` sentinel, or ``None`` for
    lines with no generated token (blank lines, role-only deltas, comments).
    For throughput, a reasoning-model's ``reasoning`` tokens count too — they
    are real decoded tokens, so we measure them.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return "[DONE]"
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    choices = obj.get("choices") or [{}]
    delta = choices[0].get("delta") or {}
    token = delta.get("content") or delta.get("reasoning")
    return token if token else None


def resolve(name: str | None, base_url: str | None) -> tuple[str, str, str]:
    """Return (kind, label, base_url) for a backend name. Raises for unknown."""
    name = (name or "ollama").strip().lower()
    if name == "ollama":
        return "ollama", "Ollama", os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434")
    if name in OPENAI_COMPAT_ALIASES:
        label, default_url = OPENAI_COMPAT_ALIASES[name]
        return name, label, (base_url or os.environ.get("LLMFIT_BASE_URL") or default_url)
    raise OllamaError(
        f"Unknown backend '{name}'. Choose: {', '.join(BACKEND_CHOICES)}."
    )
