import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic.cuda.panel as cuda_panel
from pandas_ta_classic.cuda import (
    PanelIndicatorSpec,
    PanelIndicatorTiming,
    benchmark_panel_indicators,
    panel_auto_engine_policy,
    panel_indicators,
    panel_indicators_cudf,
    panel_indicators_pandas,
    static_panel_engine_policy,
)


def make_panel(symbols=4, rows=60):
    rng = np.random.default_rng(2027)
    symbol_values = np.repeat([f"S{i}" for i in range(symbols)], rows)
    innovations = rng.normal(0.0002, 0.018, size=(symbols, rows))
    close = 100.0 * np.exp(np.cumsum(innovations, axis=1)).reshape(-1)
    spread = rng.uniform(0.001, 0.025, size=close.size)
    open_ = close * (1.0 + rng.normal(0.0, 0.004, size=close.size))
    return pd.DataFrame(
        {
            "symbol": symbol_values,
            "open": open_,
            "high": np.maximum(open_, close) * (1.0 + spread),
            "low": np.minimum(open_, close) * (1.0 - spread),
            "close": close,
            "volume": rng.lognormal(14.0, 0.8, size=close.size),
        }
    )


def test_panel_indicators_pandas_has_expected_default_columns():
    frame = make_panel()

    result = panel_indicators_pandas(frame)

    assert len(result) == len(frame)
    assert {
        "return",
        "log_return",
        "mom_10",
        "roc_10",
        "sma_20",
        "stdev_20",
        "variance_20",
        "zscore_20",
        "bb_lower_20",
        "bb_mid_20",
        "bb_upper_20",
        "donchian_low_20",
        "donchian_high_20",
        "stoch_20",
        "true_range",
        "atr_14",
        "willr_14",
    } == set(result.columns)
    assert result["return"].isna().groupby(frame["symbol"]).first().all()
    assert result["sma_20"].notna().any()


def test_panel_indicators_auto_matches_pandas_for_variable_length_symbols():
    parts = []
    for idx, rows in enumerate([21, 32, 47, 73]):
        part = make_panel(symbols=1, rows=rows)
        part["symbol"] = f"V{idx}"
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)

    result = panel_indicators(frame, engine="auto", benchmark_repeats=1)
    expected = panel_indicators_pandas(frame)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False, check_exact=False, rtol=1e-6, atol=1e-6)


def test_panel_indicators_accepts_string_and_spec_subset():
    frame = make_panel()
    specs = ["sma_10", PanelIndicatorSpec("roc", length=3)]

    result = panel_indicators_pandas(frame, specs)

    assert list(result.columns) == ["sma_10", "roc_3"]


def test_panel_indicators_cudf_matches_pandas_when_available():
    cudf = pytest.importorskip("cudf")
    frame = make_panel()
    try:
        gpu_frame = cudf.from_pandas(frame)
    except MemoryError as exc:
        pytest.skip(f"cuDF allocation unavailable: {exc}")

    result = panel_indicators_cudf(gpu_frame)
    expected = panel_indicators_pandas(frame)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False, check_exact=False, rtol=1e-6, atol=1e-6)


def test_panel_auto_policy_reports_every_spec_when_cudf_is_available():
    pytest.importorskip("cudf")
    frame = make_panel(symbols=3, rows=70)
    specs = [PanelIndicatorSpec("sma", length=10), PanelIndicatorSpec("stoch", length=10)]
    try:
        timings = benchmark_panel_indicators(frame, specs, repeats=1)
        policy = panel_auto_engine_policy(frame, specs, repeats=1, refresh=True)
    except MemoryError as exc:
        pytest.skip(f"cuDF allocation unavailable: {exc}")

    assert {timing.spec.key for timing in timings} == {"sma_10", "stoch_10"}
    assert set(policy) == {"sma_10", "stoch_10"}
    assert set(policy.values()) <= {"pandas", "cudf"}



def test_static_panel_policy_uses_pandas_for_small_panels():
    frame = make_panel(symbols=1, rows=100)

    policy = static_panel_engine_policy(frame, ["sma_20", "stdev_20", "return"])

    assert policy == {"sma_20": "pandas", "stdev_20": "pandas", "return": "pandas"}


def test_static_panel_policy_uses_cuda_for_large_rolling_families():
    frame = make_panel(symbols=20, rows=5001)

    policy = static_panel_engine_policy(frame, ["sma_20", "stdev_20", "return", "roc_10"])

    assert policy == {
        "sma_20": "cudf",
        "stdev_20": "cudf",
        "return": "pandas",
        "roc_10": "pandas",
    }



def test_auto_policy_uses_static_registry_without_benchmark(tmp_path, monkeypatch):
    frame = make_panel(symbols=20, rows=5002)

    def fail_benchmark(*args, **kwargs):
        raise AssertionError("benchmark should not run by default")

    monkeypatch.setattr(cuda_panel, "benchmark_panel_indicators", fail_benchmark)
    policy = cuda_panel.panel_auto_engine_policy(
        frame, ["sma_33", "return"], policy_cache_path=tmp_path / "missing.json"
    )

    assert policy == {"sma_33": "cudf", "return": "pandas"}


def test_auto_policy_can_persist_calibrated_policy(tmp_path, monkeypatch):
    frame = make_panel(symbols=2, rows=80)
    cache_path = tmp_path / "policy.json"

    def fake_benchmark(*args, **kwargs):
        return [
            PanelIndicatorTiming(
                spec=PanelIndicatorSpec("sma", length=77),
                pandas_seconds=2.0,
                cudf_seconds=1.0,
                selected_engine="cudf",
                speedup=2.0,
            )
        ]

    monkeypatch.setattr(cuda_panel, "benchmark_panel_indicators", fake_benchmark)
    calibrated = cuda_panel.panel_auto_engine_policy(
        frame,
        ["sma_77"],
        calibrate=True,
        refresh=True,
        policy_cache_path=cache_path,
    )
    assert calibrated == {"sma_77": "cudf"}

    cuda_panel._AUTO_POLICY_CACHE.clear()

    def fail_benchmark(*args, **kwargs):
        raise AssertionError("persisted policy should avoid benchmarking")

    monkeypatch.setattr(cuda_panel, "benchmark_panel_indicators", fail_benchmark)
    cached = cuda_panel.panel_auto_engine_policy(frame, ["sma_77"], policy_cache_path=cache_path)

    assert cached == {"sma_77": "cudf"}
