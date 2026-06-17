"""Batched rolling OHLCV indicators with optional cuDF acceleration.

This module is intentionally panel-oriented.  It computes a compact set of
common rolling features for many symbols in one long OHLCV frame.  The public
``engine="auto"`` path benchmarks each output column for the input shape and
uses CUDA only for indicators where cuDF is faster than pandas.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd


Engine = Literal["auto", "pandas", "cudf", "cuda", "gpu"]
IndicatorEngine = Literal["pandas", "cudf"]
INDICATOR_COLUMNS = (
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
)
_AUTO_ENGINE_CACHE: dict[tuple[int, int, tuple[int, ...], tuple[str, ...]], dict[str, IndicatorEngine]] = {}


@dataclass(frozen=True)
class IndicatorTiming:
    """Per-indicator timing result for pandas and cuDF backends."""

    indicator: str
    pandas_seconds: float
    cudf_seconds: float
    selected_engine: IndicatorEngine
    speedup: float


def cuda_available() -> bool:
    """Return True when CuPy can see at least one CUDA device."""

    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def synchronize_cuda() -> None:
    """Synchronize the current CUDA stream when CuPy is installed."""

    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError("CuPy is required to synchronize CUDA streams.") from exc

    cp.cuda.get_current_stream().synchronize()


def rolling_indicators(
    frame: pd.DataFrame,
    *,
    symbol: str = "symbol",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    engine: Engine = "auto",
    as_pandas: bool = True,
    benchmark_repeats: int = 2,
) -> pd.DataFrame:
    """Compute common batched rolling indicators.

    Args:
        frame: Long OHLCV panel with one row per symbol/date observation.
        symbol: Symbol/grouping column name.
        high: High price column name.
        low: Low price column name.
        close: Close price column name.
        engine: ``"auto"``, ``"pandas"``, ``"cudf"``, ``"cuda"``, or ``"gpu"``.
        as_pandas: Convert cuDF results back to pandas. Ignored for pandas and
            mixed auto results.
        benchmark_repeats: Number of timing repeats for first-call auto
            calibration. The best timing is used.

    Returns:
        A DataFrame aligned to ``frame.index`` with return, SMA, stdev, zscore,
        Bollinger Band, Donchian, stochastic, and ATR-SMA columns.
    """

    normalized_engine = engine.lower()
    if normalized_engine not in {"auto", "pandas", "cudf", "cuda", "gpu"}:
        raise ValueError("engine must be one of: auto, pandas, cudf, cuda, gpu")

    if normalized_engine == "pandas":
        return rolling_indicators_pandas(
            frame,
            symbol=symbol,
            high=high,
            low=low,
            close=close,
        )

    if normalized_engine == "auto":
        return rolling_indicators_auto(
            frame,
            symbol=symbol,
            high=high,
            low=low,
            close=close,
            benchmark_repeats=benchmark_repeats,
        )

    return rolling_indicators_cudf(
        frame,
        symbol=symbol,
        high=high,
        low=low,
        close=close,
        as_pandas=as_pandas,
    )


def rolling_indicators_auto(
    frame: pd.DataFrame,
    *,
    symbol: str = "symbol",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    benchmark_repeats: int = 2,
) -> pd.DataFrame:
    """Compute indicators with the fastest measured backend per output column."""

    if not cuda_available():
        return rolling_indicators_pandas(frame, symbol=symbol, high=high, low=low, close=close)

    try:
        import cudf
    except ImportError:
        return rolling_indicators_pandas(frame, symbol=symbol, high=high, low=low, close=close)

    _validate_columns(frame, (symbol, high, low, close))
    try:
        gpu_frame = cudf.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
        policy = auto_engine_policy(
            frame,
            gpu_frame=gpu_frame,
            symbol=symbol,
            high=high,
            low=low,
            close=close,
            repeats=benchmark_repeats,
        )
    except (MemoryError, RuntimeError):
        return rolling_indicators_pandas(frame, symbol=symbol, high=high, low=low, close=close)
    pandas_state = None
    cudf_state = None
    columns = {}
    for indicator in INDICATOR_COLUMNS:
        selected_engine = policy[indicator]
        if selected_engine == "cudf":
            try:
                if cudf_state is None:
                    cudf_state = _make_cudf_state(gpu_frame, symbol=symbol, high=high, low=low, close=close)
                columns[indicator] = _CUDF_INDICATORS[indicator](cudf_state).to_pandas()
            except (MemoryError, RuntimeError):
                if pandas_state is None:
                    pandas_state = _make_pandas_state(frame, symbol=symbol, high=high, low=low, close=close)
                columns[indicator] = _PANDAS_INDICATORS[indicator](pandas_state)
        else:
            if pandas_state is None:
                pandas_state = _make_pandas_state(frame, symbol=symbol, high=high, low=low, close=close)
            columns[indicator] = _PANDAS_INDICATORS[indicator](pandas_state)
    return pd.DataFrame(columns, index=frame.index)


def auto_engine_policy(
    frame: pd.DataFrame,
    *,
    gpu_frame=None,
    symbol: str = "symbol",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    repeats: int = 2,
    refresh: bool = False,
) -> dict[str, IndicatorEngine]:
    """Return the cached fastest backend for each indicator on this input shape."""

    _validate_columns(frame, (symbol, high, low, close))
    cache_key = _auto_cache_key(frame, symbol=symbol, high=high, low=low, close=close)
    if not refresh and cache_key in _AUTO_ENGINE_CACHE:
        return dict(_AUTO_ENGINE_CACHE[cache_key])
    timings = benchmark_indicator_engines(
        frame,
        gpu_frame=gpu_frame,
        symbol=symbol,
        high=high,
        low=low,
        close=close,
        repeats=repeats,
    )
    policy = {timing.indicator: timing.selected_engine for timing in timings}
    _AUTO_ENGINE_CACHE[cache_key] = policy
    return dict(policy)


def benchmark_indicator_engines(
    frame: pd.DataFrame,
    *,
    gpu_frame=None,
    symbol: str = "symbol",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    repeats: int = 2,
) -> list[IndicatorTiming]:
    """Benchmark each indicator and report whether pandas or cuDF is faster."""

    try:
        import cudf
    except ImportError as exc:
        raise ImportError("cuDF is required to benchmark CUDA indicator engines.") from exc

    _validate_columns(frame, (symbol, high, low, close))
    if gpu_frame is None:
        gpu_frame = cudf.from_pandas(frame)
    timings = []
    for indicator in INDICATOR_COLUMNS:
        pandas_seconds = _time_best(
            lambda indicator=indicator: _compute_pandas_indicator(
                frame, indicator, symbol=symbol, high=high, low=low, close=close
            ),
            repeats,
        )
        cudf_seconds = _time_best(
            lambda indicator=indicator: _compute_cudf_indicator(
                gpu_frame, indicator, symbol=symbol, high=high, low=low, close=close
            ),
            repeats,
            synchronize=True,
        )
        selected_engine: IndicatorEngine = "cudf" if cudf_seconds < pandas_seconds else "pandas"
        timings.append(
            IndicatorTiming(
                indicator=indicator,
                pandas_seconds=pandas_seconds,
                cudf_seconds=cudf_seconds,
                selected_engine=selected_engine,
                speedup=pandas_seconds / cudf_seconds if cudf_seconds > 0 else float("inf"),
            )
        )
    return timings


def rolling_indicators_pandas(
    frame: pd.DataFrame,
    *,
    symbol: str = "symbol",
    high: str = "high",
    low: str = "low",
    close: str = "close",
) -> pd.DataFrame:
    """Pandas implementation for batched rolling indicators."""

    state = _make_pandas_state(frame, symbol=symbol, high=high, low=low, close=close)
    return pd.DataFrame(
        {indicator: _PANDAS_INDICATORS[indicator](state) for indicator in INDICATOR_COLUMNS},
        index=frame.index,
    )


def rolling_indicators_cudf(
    frame,
    *,
    symbol: str = "symbol",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    as_pandas: bool = True,
):
    """cuDF implementation for batched rolling indicators."""

    try:
        import cudf
    except ImportError as exc:
        raise ImportError(
            "cuDF is required for CUDA rolling indicators. Install the cuda extra "
            "or call rolling_indicators(..., engine='pandas')."
        ) from exc

    gpu_frame = cudf.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    state = _make_cudf_state(gpu_frame, symbol=symbol, high=high, low=low, close=close)
    result = cudf.DataFrame({indicator: _CUDF_INDICATORS[indicator](state) for indicator in INDICATOR_COLUMNS})
    return result.to_pandas() if as_pandas else result


def _compute_pandas_indicator(
    frame: pd.DataFrame,
    indicator: str,
    *,
    symbol: str,
    high: str,
    low: str,
    close: str,
):
    _validate_columns(frame, (symbol, high, low, close))
    grouped = frame.groupby(symbol, sort=False, observed=True)
    close_values = frame[close]
    if indicator == "return_1":
        previous_close = grouped[close].shift(1)
        return close_values / previous_close - 1.0
    if indicator == "sma_20":
        return _drop_group_level(grouped[close].rolling(20, min_periods=20).mean())
    if indicator == "stdev_20":
        return _drop_group_level(grouped[close].rolling(20, min_periods=20).std(ddof=1))
    if indicator in {"zscore_20", "bb_upper_20", "bb_lower_20"}:
        mean_20 = _drop_group_level(grouped[close].rolling(20, min_periods=20).mean())
        std_20 = _drop_group_level(grouped[close].rolling(20, min_periods=20).std(ddof=1))
        if indicator == "zscore_20":
            return (close_values - mean_20) / std_20
        if indicator == "bb_upper_20":
            return mean_20 + 2.0 * std_20
        return mean_20 - 2.0 * std_20
    if indicator == "donchian_high_20":
        return _drop_group_level(grouped[high].rolling(20, min_periods=20).max())
    if indicator == "donchian_low_20":
        return _drop_group_level(grouped[low].rolling(20, min_periods=20).min())
    if indicator == "stoch_20":
        high_20 = _drop_group_level(grouped[high].rolling(20, min_periods=20).max())
        low_20 = _drop_group_level(grouped[low].rolling(20, min_periods=20).min())
        denominator = (high_20 - low_20).replace(0.0, np.nan)
        return 100.0 * (close_values - low_20) / denominator
    if indicator == "atr_sma_14":
        previous_close = grouped[close].shift(1)
        true_range = pd.concat(
            [
                frame[high] - frame[low],
                (frame[high] - previous_close).abs(),
                (frame[low] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        temp = frame[[symbol]].copy()
        temp["true_range"] = true_range
        return _drop_group_level(
            temp.groupby(symbol, sort=False, observed=True)["true_range"]
            .rolling(14, min_periods=14)
            .mean()
        )
    raise KeyError(f"unknown indicator: {indicator}")


def _compute_cudf_indicator(frame, indicator: str, *, symbol: str, high: str, low: str, close: str):
    import cudf

    _validate_columns(frame, (symbol, high, low, close))
    grouped = frame.groupby(symbol, sort=False)
    close_values = frame[close]
    if indicator == "return_1":
        previous_close = grouped[close].shift(1)
        return close_values / previous_close - 1.0
    if indicator == "sma_20":
        return _drop_group_level(grouped[close].rolling(20, min_periods=20).mean())
    if indicator == "stdev_20":
        return _drop_group_level(grouped[close].rolling(20, min_periods=20).std(ddof=1))
    if indicator in {"zscore_20", "bb_upper_20", "bb_lower_20"}:
        mean_20 = _drop_group_level(grouped[close].rolling(20, min_periods=20).mean())
        std_20 = _drop_group_level(grouped[close].rolling(20, min_periods=20).std(ddof=1))
        if indicator == "zscore_20":
            return (close_values - mean_20) / std_20
        if indicator == "bb_upper_20":
            return mean_20 + 2.0 * std_20
        return mean_20 - 2.0 * std_20
    if indicator == "donchian_high_20":
        return _drop_group_level(grouped[high].rolling(20, min_periods=20).max())
    if indicator == "donchian_low_20":
        return _drop_group_level(grouped[low].rolling(20, min_periods=20).min())
    if indicator == "stoch_20":
        high_20 = _drop_group_level(grouped[high].rolling(20, min_periods=20).max())
        low_20 = _drop_group_level(grouped[low].rolling(20, min_periods=20).min())
        denominator = high_20 - low_20
        denominator = denominator.where(denominator != 0.0)
        return 100.0 * (close_values - low_20) / denominator
    if indicator == "atr_sma_14":
        previous_close = grouped[close].shift(1)
        true_range = cudf.concat(
            [
                frame[high] - frame[low],
                (frame[high] - previous_close).abs(),
                (frame[low] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        temp = frame[[symbol]].copy()
        temp["true_range"] = true_range
        return _drop_group_level(
            temp.groupby(symbol, sort=False)["true_range"].rolling(14, min_periods=14).mean()
        )
    raise KeyError(f"unknown indicator: {indicator}")


def _make_pandas_state(frame: pd.DataFrame, *, symbol: str, high: str, low: str, close: str) -> dict[str, object]:
    _validate_columns(frame, (symbol, high, low, close))
    grouped = frame.groupby(symbol, sort=False, observed=True)
    close_values = frame[close]
    previous_close = grouped[close].shift(1)
    high_20 = _drop_group_level(grouped[high].rolling(20, min_periods=20).max())
    low_20 = _drop_group_level(grouped[low].rolling(20, min_periods=20).min())
    mean_20 = _drop_group_level(grouped[close].rolling(20, min_periods=20).mean())
    std_20 = _drop_group_level(grouped[close].rolling(20, min_periods=20).std(ddof=1))
    true_range = pd.concat(
        [
            frame[high] - frame[low],
            (frame[high] - previous_close).abs(),
            (frame[low] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    temp = frame[[symbol]].copy()
    temp["true_range"] = true_range
    atr_14 = _drop_group_level(
        temp.groupby(symbol, sort=False, observed=True)["true_range"]
        .rolling(14, min_periods=14)
        .mean()
    )
    return {
        "close": close_values,
        "previous_close": previous_close,
        "mean_20": mean_20,
        "std_20": std_20,
        "high_20": high_20,
        "low_20": low_20,
        "atr_14": atr_14,
    }


def _make_cudf_state(frame, *, symbol: str, high: str, low: str, close: str) -> dict[str, object]:
    import cudf

    _validate_columns(frame, (symbol, high, low, close))
    grouped = frame.groupby(symbol, sort=False)
    close_values = frame[close]
    previous_close = grouped[close].shift(1)
    high_20 = _drop_group_level(grouped[high].rolling(20, min_periods=20).max())
    low_20 = _drop_group_level(grouped[low].rolling(20, min_periods=20).min())
    mean_20 = _drop_group_level(grouped[close].rolling(20, min_periods=20).mean())
    std_20 = _drop_group_level(grouped[close].rolling(20, min_periods=20).std(ddof=1))
    true_range = cudf.concat(
        [
            frame[high] - frame[low],
            (frame[high] - previous_close).abs(),
            (frame[low] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    temp = frame[[symbol]].copy()
    temp["true_range"] = true_range
    atr_14 = _drop_group_level(
        temp.groupby(symbol, sort=False)["true_range"].rolling(14, min_periods=14).mean()
    )
    return {
        "close": close_values,
        "previous_close": previous_close,
        "mean_20": mean_20,
        "std_20": std_20,
        "high_20": high_20,
        "low_20": low_20,
        "atr_14": atr_14,
    }


def _return_1(state: dict[str, object]):
    return state["close"] / state["previous_close"] - 1.0


def _sma_20(state: dict[str, object]):
    return state["mean_20"]


def _stdev_20(state: dict[str, object]):
    return state["std_20"]


def _zscore_20(state: dict[str, object]):
    return (state["close"] - state["mean_20"]) / state["std_20"]


def _bb_upper_20(state: dict[str, object]):
    return state["mean_20"] + 2.0 * state["std_20"]


def _bb_lower_20(state: dict[str, object]):
    return state["mean_20"] - 2.0 * state["std_20"]


def _donchian_high_20(state: dict[str, object]):
    return state["high_20"]


def _donchian_low_20(state: dict[str, object]):
    return state["low_20"]


def _stoch_20(state: dict[str, object]):
    denominator = state["high_20"] - state["low_20"]
    if hasattr(denominator, "replace"):
        denominator = denominator.replace(0.0, np.nan)
    else:
        denominator = denominator.where(denominator != 0.0)
    return 100.0 * (state["close"] - state["low_20"]) / denominator


def _atr_sma_14(state: dict[str, object]):
    return state["atr_14"]


_PANDAS_INDICATORS: dict[str, Callable[[dict[str, object]], object]] = {
    "return_1": _return_1,
    "sma_20": _sma_20,
    "stdev_20": _stdev_20,
    "zscore_20": _zscore_20,
    "bb_upper_20": _bb_upper_20,
    "bb_lower_20": _bb_lower_20,
    "donchian_high_20": _donchian_high_20,
    "donchian_low_20": _donchian_low_20,
    "stoch_20": _stoch_20,
    "atr_sma_14": _atr_sma_14,
}
_CUDF_INDICATORS = _PANDAS_INDICATORS


def _drop_group_level(series):
    return series.reset_index(level=0, drop=True)


def _time_best(function, repeats: int, *, synchronize: bool = False) -> float:
    best = float("inf")
    for _ in range(max(1, int(repeats))):
        gc.collect()
        started = time.perf_counter()
        function()
        if synchronize:
            synchronize_cuda()
        best = min(best, time.perf_counter() - started)
    return best


def _auto_cache_key(frame, *, symbol: str, high: str, low: str, close: str) -> tuple[int, int, tuple[int, ...], tuple[str, ...]]:
    if symbol not in frame.columns:
        return len(frame), 0, (), (symbol, high, low, close)
    lengths = frame.groupby(symbol, sort=False).size()
    length_values = [int(value) for value in lengths.tolist()]
    if not length_values:
        return len(frame), 0, (), (symbol, high, low, close)
    signature = _length_distribution_signature(length_values)
    return len(frame), len(length_values), signature, (symbol, high, low, close)


def _length_distribution_signature(lengths: list[int]) -> tuple[int, ...]:
    values = np.asarray(lengths, dtype=np.int64)
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]).round().astype(np.int64)
    unique_lengths = np.unique(values)
    if len(unique_lengths) <= 16:
        return tuple(int(value) for value in unique_lengths)
    return tuple(int(value) for value in quantiles)


def _validate_columns(frame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame missing required columns: {missing}")
