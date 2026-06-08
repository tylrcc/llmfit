"""Model-footprint estimation and a small catalog of popular local models.

Estimates are intentionally approximate and clearly labelled as such in the UI;
they exist to steer a download decision, not to predict memory to the megabyte.
Use ``llmfit bench`` for real measured numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

# Effective bits-per-weight for common GGUF quantizations, including the small
# overhead from metadata and unquantized tensors. Ordered best-quality first.
QUANT_BPW: dict[str, float] = {
    "F16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.6,
    "Q5_K_M": 5.7,
    "Q4_K_M": 4.8,
    "Q4_0": 4.5,
    "Q3_K_M": 3.9,
    "Q2_K": 3.1,
}

# Quality-ordered list (best -> smallest) used when searching for a quant.
QUANT_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
DEFAULT_QUANT = "Q4_K_M"


@dataclass(frozen=True)
class CatalogModel:
    family: str
    params_b: float
    note: str = ""

    @property
    def label(self) -> str:
        size = f"{self.params_b:g}B"
        return f"{self.family} {size}"


# A curated, deliberately small set of widely-used open models, by size class.
CATALOG: list[CatalogModel] = [
    CatalogModel("Qwen2.5 / Qwen3", 0.5, "tiny, runs anywhere"),
    CatalogModel("Llama 3.2", 1.0, "edge / CPU friendly"),
    CatalogModel("Gemma", 2.0, "small assistant"),
    CatalogModel("Llama 3.2", 3.0, "fast general use"),
    CatalogModel("Phi", 3.8, "strong for its size"),
    CatalogModel("Mistral", 7.0, "classic 7B baseline"),
    CatalogModel("Qwen2.5 / Qwen3", 7.0, "great all-rounder"),
    CatalogModel("Llama 3.1", 8.0, "popular default"),
    CatalogModel("Gemma", 9.0, "quality mid-size"),
    CatalogModel("Phi", 14.0, "reasoning-tuned"),
    CatalogModel("Qwen2.5 / Qwen3", 14.0, "capable mid-size"),
    CatalogModel("DeepSeek-R1 distill", 14.0, "reasoning"),
    CatalogModel("Gemma", 27.0, "near-frontier local"),
    CatalogModel("Qwen2.5 / Qwen3", 32.0, "high quality"),
    CatalogModel("DeepSeek-R1 distill", 32.0, "strong reasoning"),
    CatalogModel("Llama 3.3", 70.0, "frontier-class, heavy"),
    CatalogModel("Qwen2.5 / Qwen3", 72.0, "frontier-class, heavy"),
]


def weights_gb(params_b: float, quant: str = DEFAULT_QUANT) -> float:
    """Approximate on-disk / weight memory for a model at a quantization."""
    bpw = QUANT_BPW.get(quant, QUANT_BPW[DEFAULT_QUANT])
    return params_b * bpw / 8.0


def kv_cache_gb(params_b: float, ctx: int = 4096) -> float:
    """Rough KV-cache size; scales with model size and context length."""
    return 0.5 * (params_b / 7.0) * (ctx / 4096.0)


def runtime_gb(params_b: float, quant: str = DEFAULT_QUANT, ctx: int = 4096) -> float:
    """Estimated total memory to *run* a model: weights + KV cache + overhead."""
    return weights_gb(params_b, quant) * 1.05 + kv_cache_gb(params_b, ctx)


# Fit classification ---------------------------------------------------------- #
COMFORTABLE = "comfortable"
TIGHT = "tight"
MAXED = "max"
WONT_FIT = "wont_fit"


def classify(footprint_gb: float, budget_gb: float) -> str:
    if budget_gb <= 0:
        return WONT_FIT
    ratio = footprint_gb / budget_gb
    if ratio <= 0.60:
        return COMFORTABLE
    if ratio <= 0.85:
        return TIGHT
    if ratio <= 1.0:
        return MAXED
    return WONT_FIT


def best_quant_for(params_b: float, budget_gb: float, ctx: int = 4096) -> str | None:
    """Highest-quality quant whose runtime footprint fits the budget, if any."""
    for quant in QUANT_ORDER:
        if runtime_gb(params_b, quant, ctx) <= budget_gb:
            return quant
    return None
