"""Apples-to-apples CUDA benchmark: pandas-ta-classic cuDF panel vs VectorTA CUDA.

Uses the same OHLCV panel, the same indicator parameters, and reports:
- pandas CPU baseline
- cuDF panel compute-only and transfer-inclusive
- VectorTA CUDA compute-only and transfer-inclusive

VectorTA timings include panel packing (pandas -> padded time-major f32) and
device readback (CuPy __cuda_array_interface__), because that is the real
end-to-end cost of using VectorTA from pandas today.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from examples.benchmark_cuda_panel import make_ohlcv_panel
from pandas_ta_classic.cuda import (
    panel_indicators_pandas,
    panel_indicators_vectorta_cpu,
    synchronize_cuda,
    vectorta_available,
)
from pandas_ta_classic.cuda.panel import PanelIndicatorSpec, panel_indicators_cudf


@dataclass(frozen=True)
class TimingRow:
    indicator: str
    pandas_s: float
    cudf_compute_s: float
    cudf_total_s: float
    vectorta_compute_s: float | None
    vectorta_total_s: float | None
    notes: str = ""


def time_best(label: str, function, repeats: int, *, sync: bool = False) -> float:
    best = float("inf")
    for _ in range(max(1, repeats)):
        gc.collect()
        started = time.perf_counter()
        function()
        if sync:
            synchronize_cuda()
        best = min(best, time.perf_counter() - started)
    return best


def panel_layout(frame: pd.DataFrame) -> dict[str, object]:
    grouped = list(frame.groupby("symbol", sort=False, observed=True))
    symbols = [name for name, _ in grouped]
    lengths = [len(part) for _, part in grouped]
    max_rows = max(lengths)
    num_symbols = len(symbols)
    layout = {
        "symbols": symbols,
        "lengths": lengths,
        "max_rows": max_rows,
        "num_symbols": num_symbols,
        "close": np.full((max_rows, num_symbols), np.nan, dtype=np.float32),
        "high": np.full((max_rows, num_symbols), np.nan, dtype=np.float32),
        "low": np.full((max_rows, num_symbols), np.nan, dtype=np.float32),
        "open": np.full((max_rows, num_symbols), np.nan, dtype=np.float32),
    }
    for column_index, (_, part) in enumerate(grouped):
        row_count = len(part)
        for name in ("close", "high", "low", "open"):
            layout[name][:row_count, column_index] = part[name].to_numpy(dtype=np.float32, copy=False)
    for name in ("close", "high", "low", "open"):
        matrix = layout[name]
        layout[f"{name}_flat"] = matrix.reshape(-1)
    return layout


def readback_f32(device_array) -> np.ndarray:
    import cupy as cp

    return cp.asnumpy(cp.asarray(device_array))


def run_vectorta_cuda(packed: dict[str, object], spec: PanelIndicatorSpec):
    import vector_ta

    kind = spec.kind
    length = int(spec.length)
    rows = packed["max_rows"]
    cols = packed["num_symbols"]
    if kind == "sma":
        return vector_ta.sma_cuda_many_series_one_param_dev(packed["close"], length, 0)
    if kind == "rsi":
        return vector_ta.rsi_cuda_many_series_one_param_dev(packed["close"], length, 0)
    if kind == "ema":
        return vector_ta.ema_cuda_many_series_one_param_dev(packed["close"], length, 0)
    if kind == "atr":
        return vector_ta.atr_cuda_many_series_one_param_dev(
            packed["high_flat"], packed["low_flat"], packed["close_flat"], cols, rows, length, 0
        )
    if kind == "willr":
        return vector_ta.willr_cuda_many_series_one_param_dev(
            packed["high_flat"], packed["low_flat"], packed["close_flat"], cols, rows, length, 0
        )
    if kind == "aroon":
        out = vector_ta.aroon_cuda_many_series_one_param_dev(packed["high"], packed["low"], length, 0)
        return out[0] if isinstance(out, tuple) else out
    raise KeyError(kind)


def benchmark_vectorta_indicator(frame: pd.DataFrame, spec: PanelIndicatorSpec, repeats: int) -> tuple[float, float]:
    packed = panel_layout(frame)

    def compute_only():
        run_vectorta_cuda(packed, spec)
        synchronize_cuda()

    def end_to_end():
        current = panel_layout(frame)
        out = run_vectorta_cuda(current, spec)
        synchronize_cuda()
        readback_f32(out)

    compute_seconds = time_best("compute", compute_only, repeats, sync=False)
    total_seconds = time_best("total", end_to_end, repeats, sync=False)
    return compute_seconds, total_seconds


def main() -> None:
    repeats = 2
    symbols = 800
    rows = 2500
    frame = make_ohlcv_panel(symbols, rows, variable_lengths=True)
    specs = [
        PanelIndicatorSpec("sma", length=20),
        PanelIndicatorSpec("atr", length=14),
        PanelIndicatorSpec("willr", length=14),
        PanelIndicatorSpec("aroon", length=14),
        PanelIndicatorSpec("rsi", length=14),
        PanelIndicatorSpec("ema", length=20),
    ]

    print("Apples-to-apples CUDA benchmark")
    print(f"panel: {frame['symbol'].nunique():,} symbols, {len(frame):,} rows, repeats={repeats}")
    try:
        import cupy as cp

        print(f"device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    except Exception:
        pass

    import cudf

    gpu_frame = cudf.from_pandas(frame)
    synchronize_cuda()
    h2d_seconds = time_best("h2d", lambda: cudf.from_pandas(frame), repeats, sync=True)

    vectorta_ready = vectorta_available()

    rows_out: list[TimingRow] = []
    for spec in specs:
        key = spec.key
        if spec.kind in {"rsi", "ema"}:
            pandas_seconds = time_best(
                "pandas",
                lambda spec=spec: panel_indicators_vectorta_cpu(frame, (spec,)),
                repeats,
            )
        else:
            pandas_seconds = time_best(
                "pandas",
                lambda spec=spec: panel_indicators_pandas(frame, (spec,)),
                repeats,
            )

        def one_cudf(spec=spec):
            if spec.kind in {"rsi", "ema"}:
                return
            panel_indicators_cudf(gpu_frame, (spec,), as_pandas=False)
            synchronize_cuda()

        cudf_compute = (
            float("nan")
            if spec.kind in {"rsi", "ema"}
            else time_best("cudf", one_cudf, repeats, sync=False)
        )
        if spec.kind in {"rsi", "ema"}:
            cudf_total = float("nan")
        else:
            cudf_total = h2d_seconds + cudf_compute + time_best(
                "d2h",
                lambda spec=spec: panel_indicators_cudf(gpu_frame, (spec,), as_pandas=True),
                1,
                sync=True,
            )

        notes = "cuDF panel has no CUDA path" if spec.kind in {"rsi", "ema"} else ""
        vectorta_compute = None
        vectorta_total = None
        if vectorta_ready:
            try:
                vectorta_compute, vectorta_total = benchmark_vectorta_indicator(frame, spec, repeats)
            except Exception as exc:
                notes = f"{notes} VectorTA failed: {type(exc).__name__}: {exc}".strip()
        elif not notes:
            notes = "VectorTA CUDA unavailable"

        rows_out.append(
            TimingRow(
                indicator=key,
                pandas_s=pandas_seconds,
                cudf_compute_s=cudf_compute,
                cudf_total_s=cudf_total,
                vectorta_compute_s=vectorta_compute,
                vectorta_total_s=vectorta_total,
                notes=notes,
            )
        )

    print("\nPer-indicator timings (seconds)")
    print(
        f"{'indicator':<14} {'pandas':>8} {'cudf':>8} {'cudf+xf':>8} "
        f"{'vta':>8} {'vta+e2e':>8}  notes"
    )
    def fmt(value: float | None) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "     n/a"
        return f"{value:8.3f}"

    for row in rows_out:
        print(
            f"{row.indicator:<14} {row.pandas_s:8.3f} {fmt(row.cudf_compute_s)} {fmt(row.cudf_total_s)} "
            f"{fmt(row.vectorta_compute_s)} {fmt(row.vectorta_total_s)}  {row.notes}"
        )

    cudf_supported = [r for r in rows_out if r.notes != "cuDF panel has no CUDA path"]
    if cudf_supported:
        cudf_compute_sum = sum(r.cudf_compute_s for r in cudf_supported)
        cudf_total_sum = sum(r.cudf_total_s for r in cudf_supported)
        pandas_sum = sum(r.pandas_s for r in cudf_supported)
        print("\nRolling overlap set (sma, atr, willr, aroon)")
        print(f"  pandas total:          {pandas_sum:.3f}s")
        print(f"  cuDF compute total:    {cudf_compute_sum:.3f}s  ({pandas_sum / cudf_compute_sum:.2f}x)")
        print(f"  cuDF transfer-inclusive total: {cudf_total_sum:.3f}s  ({pandas_sum / cudf_total_sum:.2f}x)")

    vectorta_rows = [r for r in rows_out if r.vectorta_total_s is not None]
    if vectorta_rows:
        vt_total = sum(r.vectorta_total_s for r in vectorta_rows)
        vt_compute = sum(r.vectorta_compute_s for r in vectorta_rows)
        print("\nVectorTA overlap + recursive set")
        print(f"  VectorTA compute total: {vt_compute:.3f}s")
        print(f"  VectorTA end-to-end total: {vt_total:.3f}s")


if __name__ == "__main__":
    main()