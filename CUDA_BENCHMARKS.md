# CUDA Benchmarks

These benchmarks were run on the local DGX Spark environment with the
`pandas_ta_classic_cuda` conda environment and CUDA/cuDF stack matched to
`optimal_trader`.

Command:

```bash
conda run -n pandas_ta_classic_cuda python examples/benchmark_cuda_panel.py --symbols 800 --rows 2500 --repeats 1 --variable-lengths
```

Input shape:

- Single-symbol case: 1 symbol, 2,500 rows
- Multi-symbol case: 800 symbols, 1,531,518 total rows
- Variable symbol lengths enabled
- CUDA device: NVIDIA GB10

## Summary

CUDA is not automatically faster for every indicator. On this machine, cuDF is
slower for small single-symbol workloads because kernel launch and transfer
overhead dominate. CUDA becomes useful for large multi-symbol panel workloads,
especially grouped rolling/window indicators.

For the large multi-symbol panel across the expanded implemented set:

| Backend | Time |
|---|---:|
| pandas full panel | 3.089s |
| cuDF compute only | 1.249s |
| cuDF transfer-inclusive | 1.457s |
| compute-only speedup | 2.47x |
| transfer-inclusive speedup | 2.12x |

Agreement versus pandas:

| Metric | Value |
|---|---:|
| max absolute difference | 5.154e-06 |
| max relative difference | 5.951e-08 |

## Multi-Symbol Results

These are the important results for the CUDA panel engine. `selected` is what
`engine="auto"` should choose for this input shape. `obv` and `pvt` are
large-panel CUDA wins, so the static policy uses a higher threshold for them
than for rolling/window indicators.

### Improved With CUDA

| Indicator | pandas | cuDF | Speedup | Selected |
|---|---:|---:|---:|---|
| `sma_20` | 0.097314s | 0.028157s | 3.46x | cuDF |
| `stdev_20` | 0.109964s | 0.026522s | 4.15x | cuDF |
| `variance_20` | 0.107695s | 0.026422s | 4.08x | cuDF |
| `zscore_20` | 0.178124s | 0.058276s | 3.06x | cuDF |
| `bbands_20` | 0.183413s | 0.063285s | 2.90x | cuDF |
| `donchian_20` | 0.191756s | 0.056011s | 3.42x | cuDF |
| `stoch_20` | 0.194958s | 0.066594s | 2.93x | cuDF |
| `true_range` | 0.110752s | 0.048955s | 2.26x | cuDF |
| `atr_14` | 0.226672s | 0.076771s | 2.95x | cuDF |
| `willr_14` | 0.195842s | 0.066453s | 2.95x | cuDF |
| `ao` | 0.237657s | 0.065949s | 3.60x | cuDF |
| `cmo_14` | 0.288722s | 0.091176s | 3.17x | cuDF |
| `cmf_20` | 0.222484s | 0.064756s | 3.44x | cuDF |
| `hvol_20` | 0.157391s | 0.064591s | 2.44x | cuDF |
| `obv` | 0.076254s | 0.064259s | 1.19x | cuDF |
| `pvt` | 0.072061s | 0.061851s | 1.17x | cuDF |

### Slower With CUDA

| Indicator | pandas | cuDF | Speedup | Selected |
|---|---:|---:|---:|---|
| `return` | 0.024997s | 0.027727s | 0.90x | pandas |
| `log_return` | 0.027402s | 0.030672s | 0.89x | pandas |
| `mom_10` | 0.024407s | 0.028189s | 0.87x | pandas |
| `roc_10` | 0.025015s | 0.031136s | 0.80x | pandas |
| `pvi_1` | 0.090129s | 0.123777s | 0.73x | pandas |
| `nvi_1` | 0.089718s | 0.125182s | 0.72x | pandas |

## Single-Symbol Results

For one symbol and 2,500 rows, every individual indicator was slower on cuDF.
The auto policy should keep these on pandas.

Full batch timing:

| Backend | Time |
|---|---:|
| pandas panel indicators | 0.275s |
| cuDF compute only | 0.105s |
| cuDF transfer-inclusive | 0.420s |
| compute-only speedup | 2.61x |
| transfer-inclusive speedup | 0.66x |

The full cuDF compute batch is fast, but transfer-inclusive runtime is slower
than pandas. The per-indicator timings also show pandas winning for each
standalone single-symbol indicator.

## Implemented Panel Indicators

The reusable panel CUDA engine currently covers these indicator families:

- Accelerated on large multi-symbol panels: `sma`, `stdev`, `variance`, `zscore`, `bbands`, `donchian`, `stoch`, `true_range`, `atr`, `willr`, `ao`, `cmo`, `cmf`, `hvol`, `obv`, `pvt`
- Implemented but pandas-default because cuDF is slower in the benchmark: `return`, `log_return`, `mom`, `roc`, `pvi`, `nvi`

The full inventory report is in `benchmark_results/all_indicators.md`. It
classifies every attempted indicator as implemented, not yet CUDA-ported next
candidate, signal/helper, or low-value pandas-default.

## Interpretation

Use CUDA by default only through `engine="auto"`. Auto mode uses the static
registry by default and only benchmarks when `calibrate=True` or `refresh=True`
is passed. Calibrated policies are cached in memory and persisted to a JSON file
so future runs can reuse them. The auto policy exists because different
indicator families have different performance profiles:

- Cheap shift/arithmetic indicators are usually faster on pandas unless the data
  is already GPU-resident and part of a larger fused workload.
- Grouped rolling/window indicators are the best current CUDA candidates.
- Cumulative volume indicators can be shape-sensitive; `obv` and `pvt` need a
  larger panel before the static policy sends them to cuDF.
- Single-symbol workloads are usually too small for cuDF to win.
- Multi-symbol panels with many rows per symbol are where CUDA starts to pay
  off, even after host/device transfer cost.

For best performance, keep the data on GPU across multiple feature-generation
steps and only convert back to pandas once.
