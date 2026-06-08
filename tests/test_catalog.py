"""Footprint-estimation and classification tests (pure math, no hardware)."""

import pytest

from llmfit import catalog
from llmfit.catalog import (
    best_quant_for,
    classify,
    runtime_gb,
    weights_gb,
)


def test_weights_scale_with_params_and_quant():
    # A 7B at Q4_K_M (4.8 bpw) is roughly 4.2 GB of weights.
    assert weights_gb(7, "Q4_K_M") == pytest.approx(7 * 4.8 / 8, rel=1e-6)
    # Higher quant => larger.
    assert weights_gb(7, "Q8_0") > weights_gb(7, "Q4_K_M")


def test_runtime_exceeds_raw_weights():
    # Runtime includes KV cache + overhead, so it must exceed bare weights.
    assert runtime_gb(13, "Q4_K_M") > weights_gb(13, "Q4_K_M")


def test_classify_thresholds():
    assert classify(5, 10) == catalog.COMFORTABLE     # 50%
    assert classify(8, 10) == catalog.TIGHT           # 80%
    assert classify(9.5, 10) == catalog.MAXED         # 95%
    assert classify(12, 10) == catalog.WONT_FIT       # 120%
    assert classify(5, 0) == catalog.WONT_FIT         # no budget


def test_best_quant_picks_highest_that_fits():
    # Tiny budget forces a small quant (or none).
    small = best_quant_for(70, budget_gb=24)
    assert small in (None, "Q2_K")
    # Generous budget allows a high-quality quant for a 7B.
    big = best_quant_for(7, budget_gb=48)
    assert big == "Q8_0"


def test_best_quant_returns_none_when_nothing_fits():
    assert best_quant_for(405, budget_gb=8) is None
