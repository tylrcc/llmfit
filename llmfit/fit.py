"""Combine detected hardware with the catalog and installed models to decide
what fits, at what quantization, and how comfortably.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import catalog
from .catalog import CatalogModel, classify, runtime_gb
from .hardware import Hardware, parse_param_size
from .ollama import InstalledModel


@dataclass
class ModelFit:
    label: str
    params_b: float | None
    footprint_gb: float
    status: str               # COMFORTABLE / TIGHT / MAXED / WONT_FIT
    quant: str
    detail: str = ""


def fit_installed(model: InstalledModel, budget_gb: float) -> ModelFit:
    """Fit using the model's real on-disk size (accurate, no guessing)."""
    params = parse_param_size(model.param_size)
    # Loaded footprint runs a bit above on-disk size (KV cache + overhead).
    footprint = model.size_gb * 1.1
    if params:
        footprint = max(footprint, runtime_gb(params, model.quant or catalog.DEFAULT_QUANT))
    return ModelFit(
        label=model.name,
        params_b=params,
        footprint_gb=round(footprint, 1),
        status=classify(footprint, budget_gb),
        quant=model.quant or "?",
        detail=f"{model.size_gb:.1f} GB on disk",
    )


def fit_catalog(model: CatalogModel, budget_gb: float,
                quant: str = catalog.DEFAULT_QUANT) -> ModelFit:
    footprint = runtime_gb(model.params_b, quant)
    return ModelFit(
        label=model.label,
        params_b=model.params_b,
        footprint_gb=round(footprint, 1),
        status=classify(footprint, budget_gb),
        quant=quant,
        detail=model.note,
    )


def recommend(budget_gb: float, limit: int = 4) -> list[ModelFit]:
    """The most capable catalog models that still run comfortably, plus the
    biggest one that fits if you push it."""
    fits = [fit_catalog(m, budget_gb) for m in catalog.CATALOG]
    comfortable = [f for f in fits if f.status == catalog.COMFORTABLE]
    comfortable.sort(key=lambda f: f.params_b or 0, reverse=True)
    picks = comfortable[:limit]

    stretch = [f for f in fits if f.status in (catalog.TIGHT, catalog.MAXED)]
    stretch.sort(key=lambda f: f.params_b or 0, reverse=True)
    if stretch:
        picks.append(stretch[0])
    return picks


@dataclass
class CanRunAnswer:
    query: str
    params_b: float | None
    budget_gb: float
    best_quant: str | None
    options: list[tuple[str, float, str]]  # (quant, footprint_gb, status)


def can_run(query: str, budget_gb: float) -> CanRunAnswer:
    """Answer 'can I run X?' for a size like '32b' or a name like 'qwen:14b'."""
    params = parse_param_size(query)
    options: list[tuple[str, float, str]] = []
    best = None
    if params:
        for quant in catalog.QUANT_ORDER:
            fp = runtime_gb(params, quant)
            status = classify(fp, budget_gb)
            options.append((quant, round(fp, 1), status))
            if best is None and status in (catalog.COMFORTABLE, catalog.TIGHT,
                                           catalog.MAXED):
                best = quant
    return CanRunAnswer(query, params, budget_gb, best, options)
