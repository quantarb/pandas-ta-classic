from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pandas_ta_classic.cuda import (
    DEFAULT_PANEL_SPECS,
    benchmark_panel_indicators,
    panel_indicators,
    panel_indicators_pandas,
    synchronize_cuda,
)


@dataclass(frozen=True)
class Timing:
    label: str
    seconds: float


def make_ohlcv_panel(
    symbols: int,
    rows_per_symbol: int,
    *,
    variable_lengths: bool = False,
    seed: int = 2027,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for idx in range(symbols):
        rows = rows_per_symbol
        if variable_lengths:
            rows = int(max(25, rows_per_symbol * rng.uniform(0.55, 1.0)))
        innovations = rng.normal(0.0002, 0.018, size=rows)
        close = 100.0 * np.exp(np.cumsum(innovations))
        spread = rng.uniform(0.001, 0.025, size=rows)
        open_ = close * (1.0 + rng.normal(0.0, 0.004, size=rows))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": f"S{idx:04d}",
                    "row": np.arange(rows, dtype=np.int32),
                    "open": open_,
                    "high": np.maximum(open_, close) * (1.0 + spread),
                    "low": np.minimum(open_, close) * (1.0 - spread),
                    "close": close,
                    "volume": rng.lognormal(14.0, 0.8, size=rows),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def timed(label: str, function, repeats: int) -> tuple[Timing, object]:
    best = float("inf")
    result = None
    for _ in range(max(1, repeats)):
        gc.collect()
        started = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - started)
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


def run_case(label: str, frame: pd.DataFrame, repeats: int) -> None:
    print(f"\n{label}: {frame['symbol'].nunique():,} symbols, {len(frame):,} rows")
    cpu_timing, cpu_result = timed(
        "pandas panel indicators",
        lambda: panel_indicators_pandas(frame, DEFAULT_PANEL_SPECS),
        repeats,
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
            result = panel_indicators(gpu_frame, DEFAULT_PANEL_SPECS, engine="cudf", as_pandas=False)
            synchronize_cuda()
            return result

        _ = run_gpu()
        gpu_timing, gpu_result = timed("cuDF compute only", run_gpu, repeats)
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
        for item in benchmark_panel_indicators(frame, DEFAULT_PANEL_SPECS, gpu_frame=gpu_frame, repeats=repeats):
            print(
                f"{item.spec.key}: pandas={item.pandas_seconds:.6f}s "
                f"cudf={item.cudf_seconds:.6f}s speedup={item.speedup:.2f}x "
                f"selected={item.selected_engine}"
            )
    except (ImportError, RuntimeError, MemoryError) as exc:
        print(f"CUDA benchmark unavailable: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CUDA-capable panel indicators.")
    parser.add_argument("--symbols", type=int, default=800)
    parser.add_argument("--rows", type=int, default=2500, help="Rows per symbol")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--variable-lengths", action="store_true")
    args = parser.parse_args()

    single = make_ohlcv_panel(1, args.rows, variable_lengths=False)
    panel = make_ohlcv_panel(args.symbols, args.rows, variable_lengths=args.variable_lengths)
    run_case("single-symbol", single, args.repeats)
    run_case("multi-symbol", panel, args.repeats)


if __name__ == "__main__":
    main()
