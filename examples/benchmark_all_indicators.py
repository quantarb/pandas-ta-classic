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
    "ao",
    "aroon",
    "atr",
    "bbands",
    "cmf",
    "cmo",
    "donchian",
    "hvol",
    "maxindex",
    "minindex",
    "minmaxindex",
    "mom",
    "nvi",
    "obv",
    "pvi",
    "pvt",
    "roc",
    "sma",
    "stdev",
    "stoch",
    "true_range",
    "variance",
    "willr",
    "wma",
    "zscore",
}
CUDA_PANEL_ACCELERATED = {
    "ao",
    "aroon",
    "atr",
    "bbands",
    "cmf",
    "cmo",
    "donchian",
    "hvol",
    "maxindex",
    "minindex",
    "minmaxindex",
    "obv",
    "pvt",
    "sma",
    "stdev",
    "stoch",
    "true_range",
    "variance",
    "willr",
    "wma",
    "zscore",
}
KNOWN_MULTI_SYMBOL_CUDA_FASTER = [
    ("sma_20", 0.097314, 0.028157, 3.46),
    ("stdev_20", 0.109964, 0.026522, 4.15),
    ("variance_20", 0.107695, 0.026422, 4.08),
    ("zscore_20", 0.178124, 0.058276, 3.06),
    ("bbands_20", 0.183413, 0.063285, 2.90),
    ("donchian_20", 0.191756, 0.056011, 3.42),
    ("stoch_20", 0.194958, 0.066594, 2.93),
    ("true_range", 0.110752, 0.048955, 2.26),
    ("atr_14", 0.226672, 0.076771, 2.95),
    ("willr_14", 0.195842, 0.066453, 2.95),
    ("ao", 0.237657, 0.065949, 3.60),
    ("cmo_14", 0.288722, 0.091176, 3.17),
    ("cmf_20", 0.222484, 0.064756, 3.44),
    ("hvol_20", 0.157391, 0.064591, 2.44),
    ("obv", 0.076254, 0.064259, 1.19),
    ("pvt", 0.072061, 0.061851, 1.17),
]
KNOWN_MULTI_SYMBOL_CUDA_SLOWER = [
    ("return", 0.024997, 0.027727, 0.90),
    ("log_return", 0.027402, 0.030672, 0.89),
    ("mom_10", 0.024407, 0.028189, 0.87),
    ("roc_10", 0.025015, 0.031136, 0.80),
    ("pvi_1", 0.090129, 0.123777, 0.73),
    ("nvi_1", 0.089718, 0.125182, 0.72),
]
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
    "hvol",
    "entropy",
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
    "maxindex",
    "minindex",
    "minmaxindex",
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
COMPOSITE_ROLLING_CANDIDATES = {
    "adx", "adxr", "amat", "brar", "chop", "cksp", "coppock", "dm",
    "dx", "hma", "hilo", "ichimoku", "inertia", "kdj", "kst",
    "macd", "macdext", "macdfix", "minus_dm", "natr", "pgo", "plus_dm", "pmax", "ppo", "rsi", "rvgi",
    "smi", "squeeze", "squeeze_pro", "stc", "tsi", "uo", "vfi",
    "vidya", "vwmacd",
}
WEIGHTED_ROLLING_CANDIDATES = {
    "alma", "fwma", "linreg", "linregangle", "linregintercept",
    "linregslope", "pwma", "sinwma", "swma", "t3", "tema",
    "trima", "tsf", "wma", "zlma",
}
ELEMENTWISE_OR_PRICE_TRANSFORMS = {
    "acos", "ad", "add", "asin", "atan", "avgprice", "beta",
    "bias", "bop", "ceil", "cfo", "cg", "cos", "cosh", "cpr",
    "cti", "cvi", "decay", "decreasing", "dema", "div", "dpo",
    "drawdown", "dsp", "edecay", "efi", "emv", "eom", "er",
    "eri", "exp", "floor", "fosc", "hl2", "hlc3", "increasing",
    "ln", "log10", "ma", "marketfi", "medprice", "minmax", "mmar", "npabs",
    "mult", "npround", "ohlc4", "pdist", "po", "psl", "pvol",
    "pvr", "qstick", "rainbow", "rolling_max", "rolling_min",
    "rolling_sum", "rocp", "rocr", "rocr100", "sin", "sinh", "slope", "sqrt", "sub", "tan",
    "tanh", "thermo", "todeg", "torad", "trix", "trixh", "trunc",
    "ttm_trend", "typprice", "vosc", "vwma", "wcp",
}
HILBERT_OR_CYCLE_CANDIDATES = {
    "ebsw", "ht_dcperiod", "ht_dcphase", "ht_phasor", "ht_sine",
    "ht_trendline", "ht_trendmode", "mama", "msw",
}
CUMULATIVE_VOLUME_CANDIDATES = {
    "adosc", "aobv", "apo", "kvo", "mavp", "nvi", "obv", "pvi",
    "pvo", "pvt", "vwap", "wad",
}
SIGNAL_HELPERS = {
    "above", "above_value", "below", "below_value", "cross",
    "cross_value", "crossany", "crossover", "lag", "long_run",
    "short_run", "td_seq", "tsignals", "vp", "xsignals",
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
    if name in COMPOSITE_ROLLING_CANDIDATES:
        return "not_implemented_composite_rolling_candidate", "medium"
    if name in WEIGHTED_ROLLING_CANDIDATES:
        return "not_implemented_weighted_rolling_candidate", "medium"
    if name in HILBERT_OR_CYCLE_CANDIDATES:
        return "not_implemented_cycle_or_hilbert_candidate", "medium"
    if name in CUMULATIVE_VOLUME_CANDIDATES:
        return "not_implemented_cumulative_volume_candidate", "medium"
    if name in SIGNAL_HELPERS:
        return "signal_or_helper_not_cuda_target", "low"
    if name in ELEMENTWISE_OR_PRICE_TRANSFORMS:
        return "pandas_likely_faster_or_low_value_cuda", "low"
    if name.startswith("cdl_") or name == "ha":
        return "not_implemented_candle_or_pattern", "medium"
    return "needs_manual_inspection", "unknown"


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
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row.cuda_status] = by_status.get(row.cuda_status, 0) + 1

    implemented = sorted(
        [row for row in successful if row.cuda_status.startswith("implemented_panel")],
        key=lambda row: row.indicator,
    )
    candidate_statuses = {
        "not_implemented_rolling_candidate",
        "not_implemented_composite_rolling_candidate",
        "not_implemented_weighted_rolling_candidate",
        "not_implemented_recursive_candidate",
        "not_implemented_cycle_or_hilbert_candidate",
        "not_implemented_cumulative_volume_candidate",
        "not_implemented_candle_or_pattern",
    }
    candidates = sorted(
        [row for row in successful if row.cuda_status in candidate_statuses],
        key=lambda row: row.seconds or 0.0,
        reverse=True,
    )
    low_value = sorted(
        [
            row
            for row in successful
            if row.cuda_status in {"pandas_likely_faster", "pandas_likely_faster_or_low_value_cuda", "signal_or_helper_not_cuda_target"}
        ],
        key=lambda row: row.seconds or 0.0,
        reverse=True,
    )

    lines = [
        "# Full Indicator Benchmark",
        "",
        "This report has two different kinds of data:",
        "",
        "- Direct pandas-vs-cuDF timings for the CUDA panel indicators already implemented.",
        "- Pandas baseline timings for every df.ta indicator, used to rank what should be ported next.",
        "",
        "It does not claim CUDA is faster for indicators that have not been CUDA-ported yet.",
        "Those rows are candidates, not speedup results.",
        "",
        f"- Rows: {rows_count}",
        f"- Repeats: {repeats}",
        f"- Indicators attempted: {len(rows)}",
        f"- Successful: {len(successful)}",
        f"- Failed: {len(failed)}",
        "",
        "## Known Faster With CUDA",
        "",
        "Measured on the DGX Spark multi-symbol variable-length panel benchmark. These are the indicators engine=auto should send to cuDF for large enough multi-symbol panels; `obv` and `pvt` use a higher static threshold because they were slower on smaller panels.",
        "",
        "| Indicator | pandas | cuDF | Speedup |",
        "|---|---:|---:|---:|",
    ]
    for indicator, pandas_seconds, cudf_seconds, speedup in KNOWN_MULTI_SYMBOL_CUDA_FASTER:
        lines.append(f"| `{indicator}` | {pandas_seconds:.6f}s | {cudf_seconds:.6f}s | {speedup:.2f}x |")

    lines.extend([
        "",
        "## Known Slower With CUDA",
        "",
        "These were tested on the same multi-symbol panel and should stay on pandas by default.",
        "",
        "| Indicator | pandas | cuDF | Speedup | Default |",
        "|---|---:|---:|---:|---|",
    ])
    for indicator, pandas_seconds, cudf_seconds, speedup in KNOWN_MULTI_SYMBOL_CUDA_SLOWER:
        lines.append(f"| `{indicator}` | {pandas_seconds:.6f}s | {cudf_seconds:.6f}s | {speedup:.2f}x | pandas |")

    lines.extend([
        "",
        "## Already Implemented In Panel Engine",
        "",
        "| Indicator | Baseline pandas seconds | Panel CUDA status |",
        "|---|---:|---|",
    ])
    for row in implemented:
        lines.append(f"| `{row.indicator}` | {row.seconds:.6f}s | `{row.cuda_status}` |")

    lines.extend([
        "",
        "## Not Yet CUDA-Ported: Next Candidates",
        "",
        "These have not been benchmarked against CUDA yet because there is no CUDA implementation for them in this fork. They are ranked by pandas runtime from this full benchmark.",
        "",
        "| Indicator | pandas seconds | Output | Columns | Candidate type | Priority |",
        "|---|---:|---|---:|---|---|",
    ])
    for row in candidates[:80]:
        lines.append(
            f"| `{row.indicator}` | {row.seconds:.6f}s | {row.output_type or chr(32)} | "
            f"{row.output_columns or 0} | `{row.cuda_status}` | `{row.cuda_priority}` |"
        )

    lines.extend([
        "",
        "## Low-Value Or Pandas-Default Indicators",
        "",
        "These are cheap elementwise/shift/helper operations or signal utilities. CUDA can still be revisited if they are fused into a larger GPU-resident pipeline, but standalone CUDA is unlikely to be the first win.",
        "",
        "| Indicator | pandas seconds | Reason |",
        "|---|---:|---|",
    ])
    for row in low_value[:80]:
        lines.append(f"| `{row.indicator}` | {row.seconds:.6f}s | `{row.cuda_status}` |")

    lines.extend(["", "## Classification Counts", "", "| Status | Count |", "|---|---:|"])
    for status, count in sorted(by_status.items()):
        lines.append(f"| `{status}` | {count} |")

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
