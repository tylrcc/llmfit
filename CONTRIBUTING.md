# Contributing to llmfit

Thanks for helping out! llmfit aims to be a small, accurate, fully-local utility
for sizing models to hardware. PRs of any size are welcome.

## Getting set up

```bash
git clone https://github.com/tylrcc/llmfit
cd llmfit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The test suite is **fully offline** — hardware probing and Ollama calls are
isolated from the pure logic, so you do not need a GPU or a running Ollama to
develop or to pass CI.

## Great first contributions

- **Sharper footprint math.** The estimates in `llmfit/catalog.py` (bits-per-weight
  per quant, KV-cache scaling, overhead factors) are deliberately simple. Better
  models, backed by measurements, are very welcome — keep them transparent and
  add cases to `tests/test_catalog.py`.
- **More hardware backends.** `llmfit/hardware.py` detects Apple Silicon and
  NVIDIA today. AMD (ROCm), Intel Arc, and better multi-GPU handling are open.
- **Catalog updates.** Keep `CATALOG` to widely-used, current open models.

## Principles

1. **Local only.** No network calls except to the local Ollama host. No telemetry.
2. **Honest estimates.** Anything approximate is labelled as an estimate in the
   UI, and `llmfit bench` gives real measured numbers.
3. **Few dependencies.** Currently `click`, `rich`, `psutil`. Please discuss
   before adding more.
4. **Readable over clever.**

## Submitting

- Run `pytest` and make sure it passes.
- Keep PRs focused and describe the user-facing change.
