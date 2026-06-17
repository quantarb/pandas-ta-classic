import numpy as np
import pandas as pd
import pytest

from pandas_ta_classic.cuda import (
    auto_engine_policy,
    benchmark_indicator_engines,
    rolling_indicators,
    rolling_indicators_cudf,
    rolling_indicators_pandas,
)


def make_panel(symbols=3, rows=45):
    rng = np.random.default_rng(1337)
    symbol_values = np.repeat([f"S{i}" for i in range(symbols)], rows)
    innovations = rng.normal(0.0002, 0.018, size=(symbols, rows))
    close = 100.0 * np.exp(np.cumsum(innovations, axis=1)).reshape(-1)
    spread = rng.uniform(0.001, 0.025, size=close.size)
    open_ = close * (1.0 + rng.normal(0.0, 0.004, size=close.size))
    return pd.DataFrame(
        {
            "symbol": symbol_values,
            "high": np.maximum(open_, close) * (1.0 + spread),
            "low": np.minimum(open_, close) * (1.0 - spread),
            "close": close,
        }
    )


def test_rolling_indicators_pandas_has_expected_columns_and_shape():
    frame = make_panel()

    result = rolling_indicators_pandas(frame)

    assert len(result) == len(frame)
    assert list(result.columns) == [
        "return_1",
        "sma_20",
        "stdev_20",
        "zscore_20",
        "bb_upper_20",
        "bb_lower_20",
        "donchian_high_20",
        "donchian_low_20",
        "stoch_20",
        "atr_sma_14",
    ]
    assert result["return_1"].isna().groupby(frame["symbol"]).first().all()
    assert result["sma_20"].notna().any()
    assert result["atr_sma_14"].notna().any()


def test_rolling_indicators_auto_matches_pandas_result():
    frame = make_panel()

    result = rolling_indicators(frame, engine="auto", benchmark_repeats=1)
    expected = rolling_indicators_pandas(frame)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False, check_exact=False, rtol=1e-6, atol=1e-6)


def test_rolling_indicators_validates_engine_and_columns():
    frame = make_panel()

    with pytest.raises(ValueError, match="engine must be"):
        rolling_indicators(frame, engine="bad")
    with pytest.raises(ValueError, match="missing required columns"):
        rolling_indicators_pandas(frame.drop(columns=["high"]))


def test_rolling_indicators_cudf_matches_pandas_when_cudf_is_available():
    cudf = pytest.importorskip("cudf")
    frame = make_panel()
    try:
        gpu_frame = cudf.from_pandas(frame)
    except MemoryError as exc:
        pytest.skip(f"cuDF allocation unavailable: {exc}")

    result = rolling_indicators_cudf(gpu_frame)
    expected = rolling_indicators_pandas(frame)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_auto_engine_policy_reports_every_indicator_when_cudf_is_available():
    pytest.importorskip("cudf")
    frame = make_panel(symbols=4, rows=60)

    try:
        timings = benchmark_indicator_engines(frame, repeats=1)
        policy = auto_engine_policy(frame, repeats=1, refresh=True)
    except MemoryError as exc:
        pytest.skip(f"cuDF allocation unavailable: {exc}")

    assert {timing.indicator for timing in timings} == set(rolling_indicators_pandas(frame).columns)
    assert set(policy) == set(rolling_indicators_pandas(frame).columns)
    assert set(policy.values()) <= {"pandas", "cudf"}


def test_auto_matches_pandas_for_variable_length_symbols():
    parts = []
    for idx, rows in enumerate([15, 31, 47, 60]):
        part = make_panel(symbols=1, rows=rows).copy()
        part["symbol"] = f"V{idx}"
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)

    result = rolling_indicators(frame, engine="auto", benchmark_repeats=1)
    expected = rolling_indicators_pandas(frame)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False, check_exact=False, rtol=1e-6, atol=1e-6)
