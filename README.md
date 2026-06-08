<div align="center">

# 📏 llmfit

**Right-size local LLMs for your hardware. Find out what your machine can actually run, at what quantization, and how fast, before you download 40GB.**

[![CI](https://github.com/tylrcc/llmfit/actions/workflows/ci.yml/badge.svg)](https://github.com/tylrcc/llmfit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Works with Ollama](https://img.shields.io/badge/works%20with-Ollama-black.svg)](https://ollama.com)

</div>

---

You found a shiny new 32B model. Will it run on your laptop? At which quantization? Fast enough to use? Today you find out by downloading 20GB and watching it either swap to death or work great. **`llmfit` answers the question first.**

It reads your actual hardware (RAM, CPU, GPU / unified memory), estimates each model's memory footprint per quantization, tells you what fits and how comfortably, benchmarks **real** tokens/sec, and gives you a live `htop`-style view of what's loaded in Ollama right now.

```bash
$ llmfit can 32b
```
```text
Yes — a 32B model fits on your 19 GB budget at Q2_K or smaller.

quant   est. to run  fit
Q5_K_M      26.2 GB  ✗ won't fit
Q4_K_M      22.4 GB  ✗ won't fit
Q3_K_M      18.7 GB  ✗ won't fit
Q2_K        15.3 GB  ~ tight
```

100% local. No telemetry, no account, no network call that leaves your machine.

## Install

```bash
pip install llmfit        # or: pipx install llmfit
llmfit                    # full report for your machine
```

> Works with or without [Ollama](https://ollama.com). With Ollama running, llmfit also analyses your installed and loaded models. The hardware report and `can`/catalog suggestions work standalone.

---

## `llmfit` — the full report

Run it with no arguments to see your machine, your installed models graded for fit, and the most capable models you could comfortably add.

```text
🖥  Your machine
        chip  Apple M5 Pro
       cores  15
      memory  26 GB total  ·  10 GB free now
         gpu  Apple M5 Pro GPU  ·  unified memory
model budget  19 GB  (safe to spend on a model)

📦 Installed models
model                    est. to run  fit
glm-4.7-flash:latest         21.0 GB  ✗ won't fit
gemma4:26b                   19.8 GB  ✗ won't fit
qwen3.5:27b-q4_K_M           19.5 GB  ✗ won't fit
qwen3.5:9b                    7.3 GB  ✓ comfortable
nomic-embed-text:latest       0.3 GB  ✓ comfortable

💡 You can comfortably run (≈ Q4_K_M estimates)
model                    params  est. to run  fit
Phi 14B                     14B       9.8 GB  ✓ comfortable  reasoning-tuned
Qwen2.5 / Qwen3 14B         14B       9.8 GB  ✓ comfortable  capable mid-size
DeepSeek-R1 distill 14B     14B       9.8 GB  ✓ comfortable  reasoning
Gemma 9B                     9B       6.3 GB  ✓ comfortable  quality mid-size
```

The **model budget** is the memory llmfit considers safe to spend on a model: bounded by VRAM on a discrete NVIDIA GPU, ~72% of unified memory on Apple Silicon, or a conservative slice of RAM for CPU-only machines, always leaving headroom for the OS.

## `llmfit can <size>` — instant yes/no

```bash
$ llmfit can 70b
```
```text
No — a 70B model won't fit your 19 GB budget, even at the smallest quantization.
```

Pass a raw size (`32b`, `13b`, `7b`) or a tagged name (`qwen2.5:14b`). You get a per-quantization breakdown so you can see exactly where the line is.

## `llmfit models` — grade what you've installed

A detailed table of every installed Ollama model: parameter count, quantization, on-disk size, estimated memory to run, and whether it fits your budget. Great for spotting the model that's quietly forcing everything onto the CPU.

## `llmfit bench [model]` — real tokens/sec

Estimates are nice; measurements are better. `bench` runs a short generation and reports actual throughput.

```bash
$ llmfit bench qwen3.5:9b
```
```text
            model  qwen3.5:9b
        load time  7.50 s
 prompt (prefill)  42.1 tok/s (22 tokens)
generate (decode)  22.9 tok/s (96 tokens)
```

With no model given, it benchmarks your smallest generative model.

**Not on Ollama?** `bench` measures any OpenAI-compatible local server too, by streaming a completion and timing tokens as they arrive:

```bash
# llama.cpp:  llama-server -m model.gguf --port 8080
# MLX:        mlx_lm.server --port 8080
$ llmfit bench my-model --backend llamacpp
$ llmfit bench my-model --backend mlx
$ llmfit bench my-model --backend openai --url http://localhost:8080/v1
```
```text
              model  my-model
            backend  llama.cpp  http://localhost:8080/v1
time to first token  0.35 s
  generate (decode)  21.3 tok/s (64 tokens)
```

> The hardware report (`llmfit`, `llmfit can`) is runtime-agnostic and needs no server at all. The Ollama-specific views (`models`, `top`) read Ollama's API; `bench` works with Ollama, llama.cpp, MLX, LM Studio, and vLLM.

## `llmfit top` — htop for your local models

A live view of system memory and CPU, plus the models currently resident in Ollama: how much memory each holds, whether it's on GPU or spilled to CPU, and when it will unload.

```text
╭───────────────── Apple M5 Pro  ·  15 cores ──────────────────╮
│ RAM  ███████████████████████░   95%   19.8 / 26 GB           │
│ CPU  ██████████░░░░░░░░░░░░░░   40%                           │
╰──────────────────────────────────────────────────────────────╯
╭──────────────────── Ollama — resident models ────────────────╮
│  loaded model     size      placement      unloads in        │
│  qwen3.5:9b       14.4 GB   100% GPU              5m          │
╰──────────────────────────────────────────────────────────────╯
```

---

## How it works

- **Hardware** (`hardware.py`) is detected via `sysctl` (macOS) / `/proc` (Linux) and `nvidia-smi`, with the *budgeting* math kept pure and unit-tested.
- **Footprint estimation** (`catalog.py`) models memory as `params × bits-per-weight ÷ 8`, plus a KV-cache term that scales with context, plus overhead. Bits-per-weight values are tabulated per GGUF quantization (`Q4_K_M`, `Q8_0`, …).
- **Installed models** are graded against their *real* on-disk size from Ollama's API, so that path doesn't rely on estimates at all.
- **`bench`** reads Ollama's own `eval_count` / `eval_duration` timings for honest prefill and decode rates.

> ⚠️ The fit numbers are **engineering estimates** to guide a download decision, not exact predictions. They are clearly labelled as estimates throughout, and `llmfit bench` gives you measured ground truth. Real footprint varies with context length, batch size, and KV-cache quantization.

It's a small, readable codebase with three dependencies: `click`, `rich`, `psutil`.

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Where your Ollama server lives |

## Develop

```bash
git clone https://github.com/tylrcc/llmfit
cd llmfit
pip install -e ".[dev]"
pytest            # fully offline; no GPU or Ollama required
```

Contributions welcome, especially sharper footprint math and more hardware backends (AMD/ROCm, Intel Arc). See [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

- [ ] AMD (ROCm) and Intel Arc GPU detection
- [ ] Context-length slider: see footprint at 8K / 32K / 128K
- [ ] `llmfit pull <size>` — suggest and pull the best-fitting tag for a family
- [ ] Multi-GPU budgets

## License

[MIT](LICENSE) © tylrcc

<div align="center">
<sub>Stop downloading models that won't run. If llmfit saved you a 20GB mistake, consider leaving a ⭐.</sub>
</div>
