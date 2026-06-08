"""Minimal, dependency-free client for a local Ollama server.

Only the standard library is used, and every request targets the local host
(``$OLLAMA_HOST`` or ``http://localhost:11434``), so llmfit never touches the
public internet.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


@dataclass
class InstalledModel:
    name: str
    size_bytes: int          # on-disk size reported by Ollama
    param_size: str          # e.g. "7.6B" (may be "")
    quant: str               # e.g. "Q4_K_M" (may be "")

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9


@dataclass
class LoadedModel:
    name: str
    size_bytes: int          # VRAM/RAM footprint while loaded
    size_vram_bytes: int     # portion on GPU
    expires_at: str

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9

    @property
    def gpu_fraction(self) -> float:
        return self.size_vram_bytes / self.size_bytes if self.size_bytes else 0.0


@dataclass
class BenchResult:
    model: str
    eval_tokens: int
    eval_tps: float          # generation tokens / second
    prompt_tokens: int
    prompt_tps: float        # prompt (prefill) tokens / second
    load_seconds: float


@dataclass
class Ollama:
    host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    timeout: float = 600.0

    def _base(self) -> str:
        h = self.host
        if not h.startswith("http"):
            h = "http://" + h
        return h.rstrip("/")

    def _get(self, path: str, timeout: float | None = None) -> dict:
        url = f"{self._base()}{path}"
        try:
            with urllib.request.urlopen(url, timeout=timeout or 5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.host}. "
                "Is it running? Start it with `ollama serve`."
            ) from exc

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base()}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.host}. "
                "Is it running? Start it with `ollama serve`."
            ) from exc

    def is_up(self) -> bool:
        try:
            self._get("/api/tags")
            return True
        except OllamaError:
            return False

    def installed(self) -> list[InstalledModel]:
        body = self._get("/api/tags")
        out: list[InstalledModel] = []
        for m in body.get("models", []):
            details = m.get("details", {}) or {}
            out.append(InstalledModel(
                name=m.get("name", "?"),
                size_bytes=int(m.get("size", 0)),
                param_size=details.get("parameter_size", "") or "",
                quant=details.get("quantization_level", "") or "",
            ))
        return sorted(out, key=lambda x: x.size_bytes, reverse=True)

    def loaded(self) -> list[LoadedModel]:
        body = self._get("/api/ps")
        out: list[LoadedModel] = []
        for m in body.get("models", []):
            out.append(LoadedModel(
                name=m.get("name", "?"),
                size_bytes=int(m.get("size", 0)),
                size_vram_bytes=int(m.get("size_vram", 0)),
                expires_at=m.get("expires_at", "") or "",
            ))
        return out

    def benchmark(self, model: str, *, num_tokens: int = 128) -> BenchResult:
        """Run a short generation and report real prefill/decode throughput."""
        body = self._post("/api/generate", {
            "model": model,
            "prompt": "Write a detailed paragraph about the history of computing.",
            "stream": False,
            "think": False,
            "options": {"num_predict": num_tokens, "temperature": 0.7},
        })
        if "error" in body:
            raise OllamaError(body["error"])

        def tps(count_key: str, dur_key: str) -> tuple[int, float]:
            count = int(body.get(count_key, 0) or 0)
            dur_ns = int(body.get(dur_key, 0) or 0)
            return count, (count / (dur_ns / 1e9)) if dur_ns else 0.0

        eval_count, eval_tps = tps("eval_count", "eval_duration")
        prompt_count, prompt_tps = tps("prompt_eval_count", "prompt_eval_duration")
        load_ns = int(body.get("load_duration", 0) or 0)
        return BenchResult(model, eval_count, eval_tps, prompt_count,
                           prompt_tps, load_ns / 1e9)
