"""Hardware budgeting and parsing tests (no real probing required)."""

from llmfit.hardware import GPU, Hardware, memory_budget_gb, parse_param_size


def _hw(**kw) -> Hardware:
    base = dict(os="Darwin", arch="arm64", chip="Apple M-test", cpu_cores=10,
                total_ram_gb=32.0, available_ram_gb=20.0, apple_silicon=True,
                gpu=None)
    base.update(kw)
    return Hardware(**base)


def test_apple_silicon_budget_is_share_of_unified_ram():
    hw = _hw(total_ram_gb=32.0, apple_silicon=True)
    assert memory_budget_gb(hw) == round(32.0 * 0.72, 1)


def test_nvidia_budget_is_bounded_by_vram():
    hw = _hw(apple_silicon=False, total_ram_gb=64.0,
             gpu=GPU("RTX 4090", 24.0, "nvidia"))
    assert memory_budget_gb(hw) == 24.0


def test_cpu_only_budget_is_conservative():
    hw = _hw(apple_silicon=False, total_ram_gb=16.0, gpu=None)
    assert memory_budget_gb(hw) == round(16.0 * 0.60, 1)


def test_parse_param_size_variants():
    assert parse_param_size("7B") == 7.0
    assert parse_param_size("70b") == 70.0
    assert parse_param_size("7.6B") == 7.6
    assert parse_param_size("500M") == 0.5
    assert parse_param_size("qwen2.5:14b") == 14.0
    assert parse_param_size("nonsense") is None
    assert parse_param_size("") is None
