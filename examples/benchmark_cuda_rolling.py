from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pandas_ta_classic.cuda import benchmark_indicator_engines, rolling_indicators, synchronize_cuda


@dataclass(frozen=True)
class Timing:
    label: str
    seconds: float


def make_ohlcv_panel(symbols: int, rows_per_symbol: int, seed: int = 1337) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbol_values = np.repeat([f"S{i:04d}" for i in range(symbols)], rows_per_symbol)
    row_values = np.tile(np.arange(rows_per_symbol, dtype=np.int32), symbols)
    innovations = rng.normal(0.0002, 0.018, size=(symbols, rows_per_symbol))
    close = 100.0 * np.exp(np.cumsum(innovations, axis=1)).reshape(-1)
    spread = rng.uniform(0.001, 0.025, size=close.size)
    open_ = close * (1.0 + rng.normal(0.0, 0.004, size=close.size))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(14.0, 0.8, size=close.size)
    return pd.DataFrame(
        {
            "symbol": symbol_values,
            "row": row_values,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def timed(label: str, function, repeats: int) -> tuple[Timing, object]:
    best = float("inf")
    result = None
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - started
        best = min(best, elapsed)
    return Timing(label, best), result


def compare_results(cpu: pd.DataFrame, gpu: pd.DataFrame) -> tuple[float, float]:
    cpu_values = cpu.to_numpy(dtype=np.float64)
    gpu_values = gpu.to_numpy(dtype=np.float64)
    finite = np.isfinite(cpu_values) & np.isfinite(gpu_values)
    if not finite.any():
        return float("nan"), float("nan")
    absolute = np.abs(cpu_values[finite] - gpu_values[finite])
    scale = np.maximum(np.abs(cpu_values[finite]), 1e-12)
    return float(absolute.max()), float((absolute / scale).max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pandas versus CUDA batched rolling indicators.")
    parser.add_argument("--symbols", type=int, default=800)
    parser.add_argument("--rows", type=int, default=2500, help="Rows per symbol")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    frame = make_ohlcv_panel(args.symbols, args.rows)
    print(f"Input: {args.symbols:,} symbols x {args.rows:,} rows = {len(frame):,} rows")

    cpu_timing, cpu_result = timed(
        "pandas batched rolling indicators",
        lambda: rolling_indicators(frame, engine="pandas"),
        args.repeats,
    )
    print(f"{cpu_timing.label}: {cpu_timing.seconds:.3f}s")

    try:
        import cudf
        import cupy as cp

        print(f"CUDA device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        transfer_started = time.perf_counter()
        gpu_frame = cudf.from_pandas(frame)
        synchronize_cuda()
        host_to_device_seconds = time.perf_counter() - transfer_started

        def run_gpu():
            result = rolling_indicators(gpu_frame, engine="cudf", as_pandas=False)
            synchronize_cuda()
            return result

        _ = run_gpu()
        gpu_timing, gpu_result = timed("cuDF compute only", run_gpu, args.repeats)
        transfer_back_started = time.perf_counter()
        gpu_result_pdf = gpu_result.to_pandas()
        synchronize_cuda()
        device_to_host_seconds = time.perf_counter() - transfer_back_started
        total_gpu = host_to_device_seconds + gpu_timing.seconds + device_to_host_seconds
        max_abs, max_rel = compare_results(cpu_result, gpu_result_pdf)
        print(f"host -> device: {host_to_device_seconds:.3f}s")
        print(f"{gpu_timing.label}: {gpu_timing.seconds:.3f}s")
        print(f"device -> host: {device_to_host_seconds:.3f}s")
        print(f"cuDF transfer-inclusive: {total_gpu:.3f}s")
        print(f"speedup compute-only: {cpu_timing.seconds / gpu_timing.seconds:.2f}x")
        print(f"speedup transfer-inclusive: {cpu_timing.seconds / total_gpu:.2f}x")
        print(f"agreement: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")

        print("\nPer-indicator backend timings:")
        for item in benchmark_indicator_engines(frame, gpu_frame=gpu_frame, repeats=args.repeats):
            print(
                f"{item.indicator}: pandas={item.pandas_seconds:.6f}s "
                f"cudf={item.cudf_seconds:.6f}s speedup={item.speedup:.2f}x "
                f"selected={item.selected_engine}"
            )
    except (ImportError, RuntimeError) as exc:
        print(f"CUDA benchmark unavailable: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
