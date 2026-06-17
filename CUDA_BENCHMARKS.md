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
overhead dominate. CUDA becomes useful for multi-symbol panel workloads,
especially grouped rolling/window indicators.

For the large multi-symbol panel:

| Backend | Time |
|---|---:|
| pandas full panel | 1.762s |
| cuDF compute only | 0.611s |
| cuDF transfer-inclusive | 0.774s |
| compute-only speedup | 2.89x |
| transfer-inclusive speedup | 2.28x |

Agreement versus pandas:

| Metric | Value |
|---|---:|
| max absolute difference | 5.154e-06 |
| max relative difference | 5.951e-08 |

## Multi-Symbol Results

These are the important results for the CUDA panel engine. `selected` is what
`engine="auto"` should choose for this input shape.

### Improved With CUDA

| Indicator | pandas | cuDF | Speedup | Selected |
|---|---:|---:|---:|---|
| `sma_20` | 0.097455s | 0.026009s | 3.75x | cuDF |
| `stdev_20` | 0.108883s | 0.027019s | 4.03x | cuDF |
| `variance_20` | 0.108234s | 0.025048s | 4.32x | cuDF |
| `zscore_20` | 0.183030s | 0.056678s | 3.23x | cuDF |
| `bbands_20` | 0.188663s | 0.062261s | 3.03x | cuDF |
| `donchian_20` | 0.198304s | 0.052655s | 3.77x | cuDF |
| `stoch_20` | 0.200812s | 0.063345s | 3.17x | cuDF |
| `true_range` | 0.110460s | 0.047664s | 2.32x | cuDF |
| `atr_14` | 0.244874s | 0.076670s | 3.19x | cuDF |
| `willr_14` | 0.200489s | 0.062253s | 3.22x | cuDF |

### Slower With CUDA

| Indicator | pandas | cuDF | Speedup | Selected |
|---|---:|---:|---:|---|
| `return` | 0.024572s | 0.028859s | 0.85x | pandas |
| `log_return` | 0.027098s | 0.029004s | 0.93x | pandas |
| `mom_10` | 0.024203s | 0.026783s | 0.90x | pandas |
| `roc_10` | 0.024907s | 0.030394s | 0.82x | pandas |

## Single-Symbol Results

For one symbol and 2,500 rows, every individual indicator was slower on cuDF.
The auto policy should keep these on pandas.

Full batch timing:

| Backend | Time |
|---|---:|
| pandas panel indicators | 0.260s |
| cuDF compute only | 0.054s |
| cuDF transfer-inclusive | 0.351s |
| compute-only speedup | 4.83x |
| transfer-inclusive speedup | 0.74x |

The full cuDF compute batch is fast, but transfer-inclusive runtime is slower
than pandas. The per-indicator timings also show pandas winning for each
standalone single-symbol indicator:

| Indicator | pandas | cuDF | Speedup | Selected |
|---|---:|---:|---:|---|
| `return` | 0.000589s | 0.002966s | 0.20x | pandas |
| `log_return` | 0.000591s | 0.004051s | 0.15x | pandas |
| `mom_10` | 0.000520s | 0.002813s | 0.18x | pandas |
| `roc_10` | 0.000569s | 0.003107s | 0.18x | pandas |
| `sma_20` | 0.001099s | 0.003184s | 0.35x | pandas |
| `stdev_20` | 0.001064s | 0.002727s | 0.39x | pandas |
| `variance_20` | 0.001040s | 0.003149s | 0.33x | pandas |
| `zscore_20` | 0.001441s | 0.005623s | 0.26x | pandas |
| `bbands_20` | 0.001531s | 0.005848s | 0.26x | pandas |
| `donchian_20` | 0.001436s | 0.005718s | 0.25x | pandas |
| `stoch_20` | 0.001525s | 0.008488s | 0.18x | pandas |
| `true_range` | 0.001067s | 0.007296s | 0.15x | pandas |
| `atr_14` | 0.002113s | 0.009568s | 0.22x | pandas |
| `willr_14` | 0.001625s | 0.008452s | 0.19x | pandas |

## Interpretation

Use CUDA by default only through `engine="auto"`. Auto mode uses the static
registry by default and only benchmarks when `calibrate=True` or `refresh=True`
is passed. Calibrated policies are cached in memory and persisted to a JSON file
so future runs can reuse them. The auto policy exists because different
indicator families have different performance profiles:

- Cheap shift/arithmetic indicators are usually faster on pandas unless the data
  is already GPU-resident and part of a larger fused workload.
- Grouped rolling/window indicators are the best current CUDA candidates.
- Single-symbol workloads are usually too small for cuDF to win.
- Multi-symbol panels with many rows per symbol are where CUDA starts to pay
  off, even after host/device transfer cost.

For best performance, keep the data on GPU across multiple feature-generation
steps and only convert back to pandas once.
