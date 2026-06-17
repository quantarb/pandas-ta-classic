"""Panel-oriented indicator engine with optional CUDA acceleration.

The single-symbol pandas-ta-classic functions are optimized for pandas Series.
This module targets the case that benefits most from CUDA: one long OHLCV panel
containing many symbols.  Variable-length symbols are handled with grouped
rolling operations, so the data does not need to be padded into a dense matrix.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd

from pandas_ta_classic.cuda.rolling import cuda_available, synchronize_cuda


Engine = Literal["auto", "pandas", "cudf", "cuda", "gpu"]
IndicatorEngine = Literal["pandas", "cudf"]


@dataclass(frozen=True)
class PanelIndicatorSpec:
    """A CUDA-capable panel indicator request."""

    kind: str
    length: int | None = None
    column: str = "close"
    scalar: float = 100.0
    std: float = 2.0
    ddof: int = 1

    @property
    def key(self) -> str:
        length = "" if self.length is None else f"_{self.length}"
        if self.kind in {"return", "log_return", "true_range"}:
            return self.kind
        return f"{self.kind}{length}"


@dataclass(frozen=True)
class PanelIndicatorTiming:
    """Timing result for one panel indicator spec."""

    spec: PanelIndicatorSpec
    pandas_seconds: float
    cudf_seconds: float
    selected_engine: IndicatorEngine
    speedup: float


DEFAULT_PANEL_SPECS: tuple[PanelIndicatorSpec, ...] = (
    PanelIndicatorSpec("return"),
    PanelIndicatorSpec("log_return"),
    PanelIndicatorSpec("mom", length=10),
    PanelIndicatorSpec("roc", length=10),
    PanelIndicatorSpec("sma", length=20),
    PanelIndicatorSpec("stdev", length=20),
    PanelIndicatorSpec("variance", length=20),
    PanelIndicatorSpec("zscore", length=20),
    PanelIndicatorSpec("bbands", length=20),
    PanelIndicatorSpec("donchian", length=20),
    PanelIndicatorSpec("stoch", length=20),
    PanelIndicatorSpec("true_range"),
    PanelIndicatorSpec("atr", length=14),
    PanelIndicatorSpec("willr", length=14),
    PanelIndicatorSpec("ao"),
    PanelIndicatorSpec("cmo", length=14),
    PanelIndicatorSpec("cmf", length=20),
    PanelIndicatorSpec("hvol", length=20),
    PanelIndicatorSpec("obv"),
    PanelIndicatorSpec("pvt"),
    PanelIndicatorSpec("pvi", length=1),
    PanelIndicatorSpec("nvi", length=1),
)

_AUTO_POLICY_CACHE: dict[tuple[int, int, tuple[int, ...], tuple[tuple[str, int | None, str, float, float, int], ...]], dict[str, IndicatorEngine]] = {}
_CUDA_FRIENDLY_KINDS = {
    "sma",
    "stdev",
    "variance",
    "zscore",
    "bbands",
    "donchian",
    "stoch",
    "true_range",
    "atr",
    "willr",
    "ao",
    "cmo",
    "cmf",
    "hvol",
    "obv",
    "pvt",
}
_MIN_CUDA_ROWS = 100_000
_MIN_CUDA_SYMBOLS = 16
_CUDA_KIND_THRESHOLDS = {
    "obv": (1_000_000, 256),
    "pvt": (1_000_000, 256),
}
_POLICY_CACHE_VERSION = 1


def panel_indicators(
    frame: pd.DataFrame,
    specs: Iterable[PanelIndicatorSpec | str] | None = None,
    *,
    symbol: str = "symbol",
    open_: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
    engine: Engine = "auto",
    as_pandas: bool = True,
    benchmark_repeats: int = 2,
    calibrate: bool = False,
    refresh: bool = False,
    policy_cache_path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Compute CUDA-friendly indicators for a long multi-symbol OHLCV panel.

    ``engine="auto"`` uses a static policy by default and only benchmarks when
    ``calibrate=True`` or ``refresh=True`` is passed. The returned frame is
    aligned to the input row order and supports symbols with different lengths.
    """

    normalized_engine = engine.lower()
    if normalized_engine not in {"auto", "pandas", "cudf", "cuda", "gpu"}:
        raise ValueError("engine must be one of: auto, pandas, cudf, cuda, gpu")

    normalized_specs = _normalize_specs(specs)
    if normalized_engine == "pandas":
        return panel_indicators_pandas(
            frame,
            normalized_specs,
            symbol=symbol,
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
    if normalized_engine == "auto":
        return panel_indicators_auto(
            frame,
            normalized_specs,
            symbol=symbol,
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            benchmark_repeats=benchmark_repeats,
            calibrate=calibrate,
            refresh=refresh,
            policy_cache_path=policy_cache_path,
        )
    return panel_indicators_cudf(
        frame,
        normalized_specs,
        symbol=symbol,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        as_pandas=as_pandas,
    )


def panel_indicators_auto(
    frame: pd.DataFrame,
    specs: Iterable[PanelIndicatorSpec | str] | None = None,
    *,
    symbol: str = "symbol",
    open_: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
    benchmark_repeats: int = 2,
    calibrate: bool = False,
    refresh: bool = False,
    policy_cache_path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Compute panel indicators with static, cached, or calibrated backend policy."""

    normalized_specs = _normalize_specs(specs)
    if not cuda_available():
        return panel_indicators_pandas(
            frame, normalized_specs, symbol=symbol, open_=open_, high=high, low=low, close=close, volume=volume
        )

    try:
        import cudf
    except ImportError:
        return panel_indicators_pandas(
            frame, normalized_specs, symbol=symbol, open_=open_, high=high, low=low, close=close, volume=volume
        )

    _validate_columns(frame, (symbol, high, low, close))
    try:
        gpu_frame = cudf.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
        policy = panel_auto_engine_policy(
            frame,
            normalized_specs,
            gpu_frame=gpu_frame,
            symbol=symbol,
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            repeats=benchmark_repeats,
            calibrate=calibrate,
            refresh=refresh,
            policy_cache_path=policy_cache_path,
        )
    except (MemoryError, RuntimeError):
        return panel_indicators_pandas(
            frame, normalized_specs, symbol=symbol, open_=open_, high=high, low=low, close=close, volume=volume
        )

    columns = {}
    pandas_state = None
    cudf_state = None
    for spec in normalized_specs:
        if policy[spec.key] == "cudf":
            try:
                if cudf_state is None:
                    cudf_state = _make_state(gpu_frame, symbol=symbol, open_=open_, high=high, low=low, close=close, volume=volume)
                value = _compute_spec(cudf_state, spec)
                _assign_spec_columns(columns, spec, value, to_pandas=True)
                continue
            except (MemoryError, RuntimeError):
                pass
        if pandas_state is None:
            pandas_state = _make_state(frame, symbol=symbol, open_=open_, high=high, low=low, close=close, volume=volume)
        _assign_spec_columns(columns, spec, _compute_spec(pandas_state, spec), to_pandas=False)
    return pd.DataFrame(columns, index=frame.index)


def panel_auto_engine_policy(
    frame: pd.DataFrame,
    specs: Iterable[PanelIndicatorSpec | str] | None = None,
    *,
    gpu_frame=None,
    symbol: str = "symbol",
    open_: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
    repeats: int = 2,
    calibrate: bool = False,
    refresh: bool = False,
    policy_cache_path: str | os.PathLike[str] | None = None,
) -> dict[str, IndicatorEngine]:
    """Return the backend policy for each spec on this panel shape.

    By default this uses an in-memory cache, then a persisted calibration cache,
    then the static registry/heuristic. It only benchmarks when ``calibrate=True``
    or ``refresh=True`` is passed.
    """

    normalized_specs = _normalize_specs(specs)
    cache_key = _auto_cache_key(frame, normalized_specs, symbol=symbol)
    if not refresh and cache_key in _AUTO_POLICY_CACHE:
        return dict(_AUTO_POLICY_CACHE[cache_key])

    persistent_key = _persistent_cache_key(cache_key)
    if not refresh:
        persisted = _read_persistent_policy(persistent_key, policy_cache_path)
        if persisted is not None:
            _AUTO_POLICY_CACHE[cache_key] = persisted
            return dict(persisted)

    if not calibrate and not refresh:
        policy = static_panel_engine_policy(frame, normalized_specs, symbol=symbol)
        _AUTO_POLICY_CACHE[cache_key] = policy
        return dict(policy)

    timings = benchmark_panel_indicators(
        frame,
        normalized_specs,
        gpu_frame=gpu_frame,
        symbol=symbol,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        repeats=repeats,
    )
    policy = {timing.spec.key: timing.selected_engine for timing in timings}
    _AUTO_POLICY_CACHE[cache_key] = policy
    _write_persistent_policy(persistent_key, policy, policy_cache_path)
    return dict(policy)


def static_panel_engine_policy(
    frame: pd.DataFrame,
    specs: Iterable[PanelIndicatorSpec | str] | None = None,
    *,
    symbol: str = "symbol",
) -> dict[str, IndicatorEngine]:
    """Return the built-in no-benchmark CUDA policy for a panel shape."""

    normalized_specs = _normalize_specs(specs)
    symbol_count = _symbol_count(frame, symbol=symbol)
    policy: dict[str, IndicatorEngine] = {}
    for spec in normalized_specs:
        min_rows, min_symbols = _CUDA_KIND_THRESHOLDS.get(spec.kind, (_MIN_CUDA_ROWS, _MIN_CUDA_SYMBOLS))
        use_cuda = len(frame) >= min_rows and symbol_count >= min_symbols
        policy[spec.key] = "cudf" if use_cuda and spec.kind in _CUDA_FRIENDLY_KINDS else "pandas"
    return policy


def benchmark_panel_indicators(
    frame: pd.DataFrame,
    specs: Iterable[PanelIndicatorSpec | str] | None = None,
    *,
    gpu_frame=None,
    symbol: str = "symbol",
    open_: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
    repeats: int = 2,
) -> list[PanelIndicatorTiming]:
    """Benchmark pandas versus cuDF for each CUDA-capable panel indicator."""

    try:
        import cudf
    except ImportError as exc:
        raise ImportError("cuDF is required to benchmark CUDA panel indicators.") from exc

    normalized_specs = _normalize_specs(specs)
    _validate_columns(frame, (symbol, high, low, close))
    if gpu_frame is None:
        gpu_frame = cudf.from_pandas(frame)

    timings = []
    for spec in normalized_specs:
        pandas_seconds = _time_best(
            lambda spec=spec: panel_indicators_pandas(
                frame, (spec,), symbol=symbol, open_=open_, high=high, low=low, close=close, volume=volume
            ),
            repeats,
        )
        cudf_seconds = _time_best(
            lambda spec=spec: panel_indicators_cudf(
                gpu_frame, (spec,), symbol=symbol, open_=open_, high=high, low=low, close=close, volume=volume, as_pandas=False
            ),
            repeats,
            synchronize=True,
        )
        selected_engine: IndicatorEngine = "cudf" if cudf_seconds < pandas_seconds else "pandas"
        timings.append(
            PanelIndicatorTiming(
                spec=spec,
                pandas_seconds=pandas_seconds,
                cudf_seconds=cudf_seconds,
                selected_engine=selected_engine,
                speedup=pandas_seconds / cudf_seconds if cudf_seconds > 0 else float("inf"),
            )
        )
    return timings


def panel_indicators_pandas(
    frame: pd.DataFrame,
    specs: Iterable[PanelIndicatorSpec | str] | None = None,
    *,
    symbol: str = "symbol",
    open_: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
) -> pd.DataFrame:
    """Pandas implementation for CUDA-capable panel indicators."""

    state = _make_state(frame, symbol=symbol, open_=open_, high=high, low=low, close=close, volume=volume)
    columns = {}
    for spec in _normalize_specs(specs):
        _assign_spec_columns(columns, spec, _compute_spec(state, spec), to_pandas=False)
    return pd.DataFrame(columns, index=frame.index)


def panel_indicators_cudf(
    frame,
    specs: Iterable[PanelIndicatorSpec | str] | None = None,
    *,
    symbol: str = "symbol",
    open_: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
    as_pandas: bool = True,
):
    """cuDF implementation for CUDA-capable panel indicators."""

    try:
        import cudf
    except ImportError as exc:
        raise ImportError("cuDF is required for CUDA panel indicators.") from exc

    gpu_frame = cudf.from_pandas(frame) if isinstance(frame, pd.DataFrame) else frame
    state = _make_state(gpu_frame, symbol=symbol, open_=open_, high=high, low=low, close=close, volume=volume)
    columns = {}
    for spec in _normalize_specs(specs):
        _assign_spec_columns(columns, spec, _compute_spec(state, spec), to_pandas=False)
    result = cudf.DataFrame(columns)
    return result.to_pandas() if as_pandas else result


def _make_state(frame, *, symbol: str, open_: str, high: str, low: str, close: str, volume: str) -> dict[str, object]:
    _validate_columns(frame, (symbol, high, low, close))
    grouped = frame.groupby(symbol, sort=False, observed=True) if isinstance(frame, pd.DataFrame) else frame.groupby(symbol, sort=False)
    return {
        "frame": frame,
        "grouped": grouped,
        "symbol": symbol,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _compute_spec(state: dict[str, object], spec: PanelIndicatorSpec):
    kind = spec.kind
    if kind == "return":
        return _return(state)
    if kind == "log_return":
        return _log_return(state)
    if kind == "mom":
        return _mom(state, spec.length)
    if kind == "roc":
        return _roc(state, spec.length, spec.scalar)
    if kind == "sma":
        return _rolling(state, spec.column, spec.length, "mean")
    if kind == "stdev":
        return _rolling(state, spec.column, spec.length, "std", ddof=spec.ddof)
    if kind == "variance":
        return _rolling(state, spec.column, spec.length, "var", ddof=spec.ddof)
    if kind == "zscore":
        mean = _rolling(state, spec.column, spec.length, "mean")
        std = _rolling(state, spec.column, spec.length, "std", ddof=spec.ddof)
        return (state["frame"][spec.column] - mean) / std
    if kind == "bbands":
        mean = _rolling(state, spec.column, spec.length, "mean")
        std = _rolling(state, spec.column, spec.length, "std", ddof=spec.ddof)
        return {
            f"bb_lower_{spec.length}": mean - spec.std * std,
            f"bb_mid_{spec.length}": mean,
            f"bb_upper_{spec.length}": mean + spec.std * std,
        }
    if kind == "donchian":
        return {
            f"donchian_low_{spec.length}": _rolling(state, state["low"], spec.length, "min"),
            f"donchian_high_{spec.length}": _rolling(state, state["high"], spec.length, "max"),
        }
    if kind == "stoch":
        high_n = _rolling(state, state["high"], spec.length, "max")
        low_n = _rolling(state, state["low"], spec.length, "min")
        denominator = high_n - low_n
        denominator = denominator.replace(0.0, np.nan) if hasattr(denominator, "replace") else denominator.where(denominator != 0.0)
        return spec.scalar * (state["frame"][state["close"]] - low_n) / denominator
    if kind == "true_range":
        return _true_range(state)
    if kind == "atr":
        tr = _true_range(state)
        return _rolling_series(state, tr, "true_range", spec.length, "mean")
    if kind == "willr":
        high_n = _rolling(state, state["high"], spec.length, "max")
        low_n = _rolling(state, state["low"], spec.length, "min")
        denominator = high_n - low_n
        denominator = denominator.replace(0.0, np.nan) if hasattr(denominator, "replace") else denominator.where(denominator != 0.0)
        return -spec.scalar * (high_n - state["frame"][state["close"]]) / denominator
    if kind == "ao":
        median_price = 0.5 * (state["frame"][state["high"]] + state["frame"][state["low"]])
        fast = _rolling_series(state, median_price, "median_price", 5, "mean")
        slow = _rolling_series(state, median_price, "median_price", 34, "mean")
        return fast - slow
    if kind == "cmo":
        diff = state["frame"][state["close"]] - state["grouped"][state["close"]].shift(1)
        positive = diff.clip(lower=0)
        negative = diff.clip(upper=0).abs()
        pos_sum = _rolling_series(state, positive, "positive", spec.length, "sum")
        neg_sum = _rolling_series(state, negative, "negative", spec.length, "sum")
        return spec.scalar * (pos_sum - neg_sum) / (pos_sum + neg_sum)
    if kind == "cmf":
        frame = state["frame"]
        high = frame[state["high"]]
        low = frame[state["low"]]
        close = frame[state["close"]]
        volume = frame[state["volume"]]
        denominator = high - low
        denominator = denominator.replace(0.0, np.nan) if hasattr(denominator, "replace") else denominator.where(denominator != 0.0)
        ad = (2.0 * close - (high + low)) * volume / denominator
        ad_sum = _rolling_series(state, ad, "ad", spec.length, "sum")
        vol_sum = _rolling(state, state["volume"], spec.length, "sum")
        return ad_sum / vol_sum
    if kind == "hvol":
        log_return = _log_return(state)
        return 100.0 * _rolling_series(state, log_return, "log_return", spec.length, "std", ddof=spec.ddof) * np.sqrt(252.0)
    if kind == "obv":
        sign = _signed_change(state, state["close"], initial=1.0)
        return _grouped_cumsum(state, sign * state["frame"][state["volume"]], "obv_volume")
    if kind == "pvt":
        roc = _roc(state, 1 if spec.length is None else spec.length, spec.scalar)
        return _grouped_cumsum(state, roc * state["frame"][state["volume"]], "pvt_value")
    if kind == "pvi":
        return _volume_index(state, 1 if spec.length is None else spec.length, positive=True)
    if kind == "nvi":
        return _volume_index(state, 1 if spec.length is None else spec.length, positive=False)
    raise KeyError(f"unsupported CUDA panel indicator: {kind}")


def _return(state):
    close = state["close"]
    previous = state["grouped"][close].shift(1)
    return state["frame"][close] / previous - 1.0


def _log_return(state):
    close = state["close"]
    previous = state["grouped"][close].shift(1)
    return np.log(state["frame"][close] / previous)


def _mom(state, length: int | None):
    close = state["close"]
    shifted = state["grouped"][close].shift(_require_length(length))
    return state["frame"][close] - shifted


def _roc(state, length: int | None, scalar: float):
    close = state["close"]
    shifted = state["grouped"][close].shift(_require_length(length))
    return scalar * (state["frame"][close] / shifted - 1.0)


def _signed_change(state, column: str, *, initial: float | None = None):
    series = state["frame"][column]
    previous = state["grouped"][column].shift(1)
    diff = series - previous
    sign = diff * 0.0
    sign = sign.where(diff <= 0, 1.0)
    sign = sign.where(diff >= 0, -1.0)
    if initial is not None:
        sign = sign.where(previous.notna(), initial)
    return sign


def _grouped_cumsum(state, series, name: str):
    frame = state["frame"]
    symbol = state["symbol"]
    temp = frame[[symbol]].copy()
    temp[name] = series
    grouped = temp.groupby(symbol, sort=False, observed=True) if isinstance(temp, pd.DataFrame) else temp.groupby(symbol, sort=False)
    return grouped[name].cumsum()


def _volume_index(state, length: int, *, positive: bool):
    volume_sign = _signed_change(state, state["volume"], initial=1.0)
    roc = _roc(state, length, 100.0)
    event = volume_sign.abs() * roc
    event = event.where(volume_sign > 0) if positive else event.where(volume_sign < 0)
    event = event.fillna(0.0)
    previous_volume = state["grouped"][state["volume"]].shift(1)
    event = event.where(previous_volume.notna(), 1000.0)
    return _grouped_cumsum(state, event, "volume_index")


def _true_range(state):
    frame = state["frame"]
    high = state["high"]
    low = state["low"]
    close = state["close"]
    previous_close = state["grouped"][close].shift(1)
    try:
        import cudf

        if isinstance(frame, cudf.DataFrame):
            return cudf.concat(
                [
                    frame[high] - frame[low],
                    (frame[high] - previous_close).abs(),
                    (frame[low] - previous_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
    except ImportError:
        pass
    return pd.concat(
        [
            frame[high] - frame[low],
            (frame[high] - previous_close).abs(),
            (frame[low] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _rolling(state, column: str, length: int | None, method: str, **kwargs):
    return _drop_group_level(
        getattr(state["grouped"][column].rolling(_require_length(length), min_periods=_require_length(length)), method)(**kwargs)
    )


def _rolling_series(state, series, name: str, length: int | None, method: str, **kwargs):
    frame = state["frame"]
    symbol = state["symbol"]
    temp = frame[[symbol]].copy()
    temp[name] = series
    grouped = temp.groupby(symbol, sort=False, observed=True) if isinstance(temp, pd.DataFrame) else temp.groupby(symbol, sort=False)
    return _drop_group_level(getattr(grouped[name].rolling(_require_length(length), min_periods=_require_length(length)), method)(**kwargs))


def _assign_spec_columns(columns: dict[str, object], spec: PanelIndicatorSpec, value, *, to_pandas: bool) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            columns[key] = item.to_pandas() if to_pandas and hasattr(item, "to_pandas") else item
        return
    item = value.to_pandas() if to_pandas and hasattr(value, "to_pandas") else value
    columns[spec.key] = item


def _normalize_specs(specs: Iterable[PanelIndicatorSpec | str] | None) -> tuple[PanelIndicatorSpec, ...]:
    if specs is None:
        return DEFAULT_PANEL_SPECS
    normalized = []
    for spec in specs:
        if isinstance(spec, PanelIndicatorSpec):
            normalized.append(spec)
        elif isinstance(spec, str):
            normalized.append(_spec_from_string(spec))
        else:
            raise TypeError("specs must contain PanelIndicatorSpec or str values")
    return tuple(normalized)


def _spec_from_string(value: str) -> PanelIndicatorSpec:
    value = value.lower()
    if "_" in value:
        name, maybe_length = value.rsplit("_", 1)
        if maybe_length.isdigit():
            return PanelIndicatorSpec(name, length=int(maybe_length))
    return PanelIndicatorSpec(value)


def _drop_group_level(series):
    return series.reset_index(level=0, drop=True)


def _require_length(length: int | None) -> int:
    if length is None or length <= 0:
        raise ValueError("indicator spec requires a positive length")
    return int(length)


def _time_best(function: Callable[[], object], repeats: int, *, synchronize: bool = False) -> float:
    best = float("inf")
    for _ in range(max(1, int(repeats))):
        gc.collect()
        started = time.perf_counter()
        function()
        if synchronize:
            synchronize_cuda()
        best = min(best, time.perf_counter() - started)
    return best


def _auto_cache_key(frame, specs: tuple[PanelIndicatorSpec, ...], *, symbol: str):
    if symbol not in frame.columns:
        return len(frame), 0, (), _spec_cache_key(specs)
    lengths = frame.groupby(symbol, sort=False).size()
    values = [int(value) for value in lengths.tolist()]
    return len(frame), len(values), _length_distribution_signature(values), _spec_cache_key(specs)


def _spec_cache_key(specs: tuple[PanelIndicatorSpec, ...]):
    return tuple((spec.kind, spec.length, spec.column, spec.scalar, spec.std, spec.ddof) for spec in specs)


def _length_distribution_signature(lengths: list[int]) -> tuple[int, ...]:
    if not lengths:
        return ()
    values = np.asarray(lengths, dtype=np.int64)
    unique = np.unique(values)
    if len(unique) <= 16:
        return tuple(int(value) for value in unique)
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]).round().astype(np.int64)
    return tuple(int(value) for value in quantiles)


def _symbol_count(frame, *, symbol: str) -> int:
    if symbol not in frame.columns:
        return 0
    return int(frame[symbol].nunique())


def _default_policy_cache_path() -> Path:
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root) / "pandas_ta_classic" / "cuda_panel_policy.json"
    return Path.home() / ".cache" / "pandas_ta_classic" / "cuda_panel_policy.json"


def _resolve_policy_cache_path(path: str | os.PathLike[str] | None) -> Path:
    return Path(path).expanduser() if path is not None else _default_policy_cache_path()


def _persistent_cache_key(cache_key) -> str:
    payload = {
        "version": _POLICY_CACHE_VERSION,
        "rows": cache_key[0],
        "symbols": cache_key[1],
        "length_signature": list(cache_key[2]),
        "specs": [list(item) for item in cache_key[3]],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _read_persistent_policy(
    persistent_key: str,
    path: str | os.PathLike[str] | None,
) -> dict[str, IndicatorEngine] | None:
    cache_path = _resolve_policy_cache_path(path)
    try:
        data = json.loads(cache_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    policy = data.get("policies", {}).get(persistent_key)
    if not isinstance(policy, dict):
        return None
    normalized = {str(key): value for key, value in policy.items() if value in {"pandas", "cudf"}}
    return normalized or None


def _write_persistent_policy(
    persistent_key: str,
    policy: dict[str, IndicatorEngine],
    path: str | os.PathLike[str] | None,
) -> None:
    cache_path = _resolve_policy_cache_path(path)
    try:
        data = json.loads(cache_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        data = {"version": _POLICY_CACHE_VERSION, "policies": {}}
    if not isinstance(data, dict):
        data = {"version": _POLICY_CACHE_VERSION, "policies": {}}
    data["version"] = _POLICY_CACHE_VERSION
    policies = data.setdefault("policies", {})
    if not isinstance(policies, dict):
        data["policies"] = policies = {}
    policies[persistent_key] = policy
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    except OSError:
        return


def _validate_columns(frame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame missing required columns: {missing}")
