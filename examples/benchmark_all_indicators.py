from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import pandas_ta_classic as ta


CUDA_PANEL_IMPLEMENTED = {
    "atr",
    "bbands",
    "donchian",
    "mom",
    "roc",
    "stdev",
    "stoch",
    "true_range",
    "variance",
    "willr",
    "zscore",
    "sma",
}
CUDA_PANEL_ACCELERATED = {
    "atr",
    "bbands",
    "donchian",
    "stdev",
    "stoch",
    "true_range",
    "variance",
    "willr",
    "zscore",
    "sma",
}
CHEAP_SHIFT_ARITHMETIC = {
    "log_return",
    "mom",
    "percent_return",
    "roc",
    "rocp",
    "rocr",
    "rocr100",
}
ROLLING_WINDOW_CANDIDATES = {
    "aberration",
    "accbands",
    "ao",
    "aroon",
    "atr",
    "avolume",
    "bbands",
    "cci",
    "ce",
    "cmf",
    "cmo",
    "correl",
    "donchian",
    "entropy",
    "hvol",
    "kc",
    "kurtosis",
    "mad",
    "massi",
    "md",
    "median",
    "midpoint",
    "midprice",
    "mfi",
    "quantile",
    "rvi",
    "skew",
    "sma",
    "stdev",
    "stderr",
    "stoch",
    "stochf",
    "stochrsi",
    "tos_stdevall",
    "ui",
    "variance",
    "vhf",
    "vortex",
    "willr",
    "zscore",
}
RECURSIVE_CANDIDATES = {
    "ema",
    "fisher",
    "hwc",
    "hwma",
    "jma",
    "kama",
    "lrsi",
    "mcgd",
    "psar",
    "qqe",
    "rma",
    "rsx",
    "sarext",
    "ssf",
    "supertrend",
}
DEFAULT_EXCLUDES = {
    "above",
    "above_value",
    "below",
    "below_value",
    "cross",
    "cross_value",
    "crossany",
    "crossover",
    "lag",
    "long_run",
    "short_run",
    "td_seq",
    "ticker",
    "tsignals",
    "vp",
    "xsignals",
}


@dataclass
class IndicatorBenchmark:
    indicator: str
    status: str
    seconds: float | None
    output_type: str | None
    output_columns: int | None
    output_rows: int | None
    non_na_values: int | None
    category: str | None
    cuda_status: str
    cuda_priority: str
    error: str | None


def make_ohlcv(rows: int, seed: int = 2028) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    innovations = rng.normal(0.0002, 0.018, size=rows)
    close = 100.0 * np.exp(np.cumsum(innovations))
    spread = rng.uniform(0.001, 0.025, size=rows)
    open_ = close * (1.0 + rng.normal(0.0, 0.004, size=rows))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(14.0, 0.8, size=rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def discover_indicators(exclude: set[str]) -> list[str]:
    frame = make_ohlcv(100)
    indicators = frame.ta.indicators(as_list=True)
    return sorted(name for name in indicators if name not in exclude)


def benchmark_indicator(name: str, frame: pd.DataFrame, repeats: int, warmup: bool) -> IndicatorBenchmark:
    error = None
    best = None
    result = None
    try:
        if warmup:
            getattr(frame.copy().ta, name)()
        for _ in range(max(1, repeats)):
            gc.collect()
            work = frame.copy()
            started = time.perf_counter()
            result = getattr(work.ta, name)()
            elapsed = time.perf_counter() - started
            best = elapsed if best is None else min(best, elapsed)
        status = "ok"
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    output_type, output_columns, output_rows, non_na_values, category = summarize_output(result)
    cuda_status, cuda_priority = classify_indicator(name)
    return IndicatorBenchmark(
        indicator=name,
        status=status,
        seconds=best,
        output_type=output_type,
        output_columns=output_columns,
        output_rows=output_rows,
        non_na_values=non_na_values,
        category=category,
        cuda_status=cuda_status,
        cuda_priority=cuda_priority,
        error=error,
    )


def summarize_output(result: Any) -> tuple[str | None, int | None, int | None, int | None, str | None]:
    if result is None:
        return None, None, None, None, None
    category = getattr(result, "category", None)
    if isinstance(result, pd.Series):
        return "Series", 1, len(result), int(result.notna().sum()), category
    if isinstance(result, pd.DataFrame):
        return "DataFrame", len(result.columns), len(result), int(result.notna().sum().sum()), category
    if isinstance(result, tuple):
        columns = 0
        rows = 0
        non_na = 0
        for item in result:
            item_type, item_columns, item_rows, item_non_na, item_category = summarize_output(item)
            columns += item_columns or 0
            rows = max(rows, item_rows or 0)
            non_na += item_non_na or 0
            category = category or item_category
        return "tuple", columns, rows, non_na, category
    return type(result).__name__, None, None, None, category


def classify_indicator(name: str) -> tuple[str, str]:
    if name in CUDA_PANEL_ACCELERATED:
        return "implemented_panel_accelerated", "done"
    if name in CUDA_PANEL_IMPLEMENTED:
        return "implemented_panel_pandas_default", "low"
    if name in CHEAP_SHIFT_ARITHMETIC:
        return "pandas_likely_faster", "low"
    if name in RECURSIVE_CANDIDATES:
        return "not_implemented_recursive_candidate", "high"
    if name in ROLLING_WINDOW_CANDIDATES:
        return "not_implemented_rolling_candidate", "high"
    if name.startswith("cdl_") or name == "ha":
        return "not_implemented_candle_or_pattern", "medium"
    return "not_classified", "unknown"


def write_csv(path: Path, rows: list[IndicatorBenchmark]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, rows: list[IndicatorBenchmark]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))


def write_markdown(path: Path, rows: list[IndicatorBenchmark], *, rows_count: int, repeats: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    successful = [row for row in rows if row.status == "ok" and row.seconds is not None]
    failed = [row for row in rows if row.status != "ok"]
    slowest = sorted(successful, key=lambda row: row.seconds or 0.0, reverse=True)[:30]
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row.cuda_status] = by_status.get(row.cuda_status, 0) + 1

    lines = [
        "# Full Indicator Benchmark",
        "",
        f"- Rows: {rows_count}",
        f"- Repeats: {repeats}",
        f"- Indicators attempted: {len(rows)}",
        f"- Successful: {len(successful)}",
        f"- Failed: {len(failed)}",
        "",
        "## CUDA Classification Counts",
        "",
        "| CUDA status | Count |",
        "|---|---:|",
    ]
    for status, count in sorted(by_status.items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(
        [
            "",
            "## Slowest Successful Indicators",
            "",
            "| Indicator | Seconds | Output | Columns | CUDA status | Priority |",
            "|---|---:|---|---:|---|---|",
        ]
    )
    for row in slowest:
        lines.append(
            f"| `{row.indicator}` | {row.seconds:.6f} | {row.output_type or ''} | "
            f"{row.output_columns or 0} | `{row.cuda_status}` | `{row.cuda_priority}` |"
        )
    if failed:
        lines.extend(["", "## Failed Indicators", "", "| Indicator | Error |", "|---|---|"])
        for row in failed:
            error = (row.error or "").replace("|", "\\|")
            lines.append(f"| `{row.indicator}` | {error} |")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark every pandas-ta-classic indicator through df.ta.")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of indicators for smoke testing")
    parser.add_argument("--include-fail-prone", action="store_true", help="Also try signal/helper indicators")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output-dir", default="benchmark_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclude = set() if args.include_fail_prone else DEFAULT_EXCLUDES
    indicators = discover_indicators(exclude)
    if args.limit is not None:
        indicators = indicators[: args.limit]
    frame = make_ohlcv(args.rows)
    rows = []
    for index, name in enumerate(indicators, start=1):
        result = benchmark_indicator(name, frame, args.repeats, warmup=not args.no_warmup)
        rows.append(result)
        seconds = "error" if result.seconds is None else f"{result.seconds:.6f}s"
        print(f"[{index:03d}/{len(indicators):03d}] {name}: {result.status} {seconds} {result.cuda_status}")

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "all_indicators.csv", rows)
    write_json(output_dir / "all_indicators.json", rows)
    write_markdown(output_dir / "all_indicators.md", rows, rows_count=args.rows, repeats=args.repeats)
    print(f"\nWrote reports to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
