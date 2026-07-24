"""Optional VectorTA CUDA backend for recursive panel indicators.

VectorTA must be built with CUDA support (``maturin develop --release
--features python,cuda``). The PyPI wheel is CPU-only and will not expose
``*_cuda_many_series_one_param_dev`` entry points.

Warmup semantics follow VectorTA, not ``pandas_ta`` exactly. For example,
VectorTA RSI starts one bar later than ``pandas_ta`` RSI, and VectorTA EMA
does not leave the same leading ``NaN`` region as the SMA-seeded pandas
implementation.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from pandas_ta_classic.cuda.panel import (
    PanelIndicatorSpec,
    _normalize_specs,
    _require_length,
    _validate_columns,
    panel_indicators_pandas,
)


_VECTOR_TA_KINDS = frozenset({"ema", "rsi"})
_CUDA_MANY_SERIES = {
    "ema": "ema_cuda_many_series_one_param_dev",
    "rsi": "rsi_cuda_many_series_one_param_dev",
}


def vectorta_available() -> bool:
    """Return True when VectorTA CUDA many-series entry points are importable."""

    try:
        import vector_ta
    except ImportError:
        return False
    return all(hasattr(vector_ta, name) for name in _CUDA_MANY_SERIES.values())


def panel_indicators_vectorta(
    frame: pd.DataFrame,
    specs: Iterable[PanelIndicatorSpec | str] | None = None,
    *,
    symbol: str = "symbol",
    close: str = "close",
    device_id: int = 0,
    fallback_to_cpu: bool = True,
) -> pd.DataFrame:
    """Compute RSI/EMA for a multi-symbol panel with VectorTA CUDA kernels.

    Variable-length symbols are padded into a dense time-major ``[T, N]`` matrix
    with ``NaN`` tail rows, computed on GPU, then scattered back to the input
    row order.
    """

    normalized_specs = _normalize_specs(specs)
    unsupported = [spec.kind for spec in normalized_specs if spec.kind not in _VECTOR_TA_KINDS]
    if unsupported:
        raise ValueError(f"VectorTA panel backend only supports {_VECTOR_TA_KINDS}: {unsupported}")

    _validate_columns(frame, (symbol, close))
    if not normalized_specs:
        return pd.DataFrame(index=frame.index)

    if not vectorta_available():
        if not fallback_to_cpu:
            raise ImportError(
                "VectorTA CUDA is unavailable. Build VectorTA with "
                "'maturin develop --release --features python,cuda'."
            )
        return panel_indicators_vectorta_cpu(
            frame,
            normalized_specs,
            symbol=symbol,
            close=close,
        )

    layout = _panel_to_time_major(frame, symbol=symbol, column=close)
    columns: dict[str, object] = {}
    for spec in normalized_specs:
        values = _compute_vectorta_spec(layout, spec, device_id=device_id)
        columns[spec.key] = _scatter_time_major(layout, values)
    return pd.DataFrame(columns, index=frame.index)


def panel_indicators_vectorta_cpu(
    frame: pd.DataFrame,
    specs: Iterable[PanelIndicatorSpec | str] | None = None,
    *,
    symbol: str = "symbol",
    close: str = "close",
) -> pd.DataFrame:
    """VectorTA CPU fallback for RSI/EMA on variable-length panels."""

    try:
        import vector_ta
    except ImportError as exc:
        return panel_indicators_pandas(
            frame,
            specs,
            symbol=symbol,
            close=close,
        )

    normalized_specs = _normalize_specs(specs)
    _validate_columns(frame, (symbol, close))
    columns: dict[str, object] = {}
    for spec in normalized_specs:
        if spec.kind not in _VECTOR_TA_KINDS:
            raise ValueError(f"VectorTA CPU fallback only supports {_VECTOR_TA_KINDS}")
        length = _require_length(spec.length)
        series = pd.Series(np.nan, index=frame.index, dtype=float)
        for _, part in frame.groupby(symbol, sort=False, observed=True):
            values = part[spec.column if spec.column in part.columns else close].to_numpy(dtype=np.float64)
            if spec.kind == "rsi":
                computed = vector_ta.rsi(values, period=length)
            else:
                computed = vector_ta.ema(values, period=length)
            series.loc[part.index] = np.asarray(computed, dtype=float)
        columns[spec.key] = series
    return pd.DataFrame(columns, index=frame.index)


def _compute_vectorta_spec(layout: dict[str, object], spec: PanelIndicatorSpec, *, device_id: int):
    import vector_ta

    length = _require_length(spec.length)
    cuda_name = _CUDA_MANY_SERIES[spec.kind]
    cuda_fn = getattr(vector_ta, cuda_name)
    device_values = cuda_fn(layout["data"], length, device_id)
    return _device_array_to_numpy(device_values)


def _panel_to_time_major(
    frame: pd.DataFrame,
    *,
    symbol: str,
    column: str,
) -> dict[str, object]:
    grouped = list(frame.groupby(symbol, sort=False, observed=True))
    symbols = [name for name, _ in grouped]
    lengths = [len(part) for _, part in grouped]
    max_rows = max(lengths)
    num_symbols = len(symbols)
    data = np.full((max_rows, num_symbols), np.nan, dtype=np.float32)
    row_slices: list[tuple[int, int]] = []
    for column_index, (_, part) in enumerate(grouped):
        values = part[column].to_numpy(dtype=np.float32, copy=False)
        row_count = len(values)
        data[:row_count, column_index] = values
        row_slices.append((column_index, row_count))
    return {
        "data": data,
        "symbols": symbols,
        "row_slices": row_slices,
        "index_groups": [part.index for _, part in grouped],
    }


def _scatter_time_major(layout: dict[str, object], values: np.ndarray) -> pd.Series:
    pieces = []
    for (column_index, row_count), index in zip(layout["row_slices"], layout["index_groups"], strict=True):
        pieces.append(pd.Series(values[:row_count, column_index], index=index, dtype=float))
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


def _device_array_to_numpy(device_array) -> np.ndarray:
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError("CuPy is required to read VectorTA CUDA device buffers.") from exc
    return cp.asnumpy(cp.asarray(device_array))