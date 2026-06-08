"""Fit-engine tests using synthetic models and budgets (no Ollama needed)."""

from llmfit import catalog
from llmfit.fit import can_run, fit_installed, recommend
from llmfit.ollama import InstalledModel


def test_fit_installed_uses_real_size():
    m = InstalledModel("llama3.1:8b", size_bytes=int(4.7e9),
                       param_size="8.0B", quant="Q4_K_M")
    f = fit_installed(m, budget_gb=17.0)
    assert f.params_b == 8.0
    assert f.footprint_gb >= 4.7  # at least on-disk size
    assert f.status in (catalog.COMFORTABLE, catalog.TIGHT)


def test_huge_model_wont_fit_small_budget():
    m = InstalledModel("llama3.1:70b", size_bytes=int(40e9),
                       param_size="70B", quant="Q4_K_M")
    f = fit_installed(m, budget_gb=17.0)
    assert f.status == catalog.WONT_FIT


def test_recommend_only_returns_models_that_fit():
    picks = recommend(budget_gb=17.0)
    assert picks  # something always fits a 17 GB budget
    # Comfortable picks must genuinely be comfortable.
    comfortable = [p for p in picks if p.status == catalog.COMFORTABLE]
    assert all(p.footprint_gb <= 17.0 for p in picks)
    # The most capable comfortable pick should come first.
    if len(comfortable) > 1:
        assert comfortable[0].params_b >= comfortable[1].params_b


def test_can_run_yes_for_small_model():
    ans = can_run("7b", budget_gb=17.0)
    assert ans.params_b == 7.0
    assert ans.best_quant is not None
    assert ans.options  # one row per quant


def test_can_run_no_for_oversized_model():
    ans = can_run("70b", budget_gb=8.0)
    assert ans.params_b == 70.0
    assert ans.best_quant is None


def test_can_run_rejects_garbage():
    ans = can_run("banana", budget_gb=17.0)
    assert ans.params_b is None
    assert ans.options == []
