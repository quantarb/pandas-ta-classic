# Full Indicator Benchmark

- Rows: 5000
- Repeats: 1
- Indicators attempted: 217
- Successful: 217
- Failed: 0

## CUDA Classification Counts

| CUDA status | Count |
|---|---:|
| `implemented_panel_accelerated` | 10 |
| `implemented_panel_pandas_default` | 2 |
| `not_classified` | 149 |
| `not_implemented_candle_or_pattern` | 5 |
| `not_implemented_recursive_candidate` | 15 |
| `not_implemented_rolling_candidate` | 31 |
| `pandas_likely_faster` | 5 |

## Slowest Successful Indicators

| Indicator | Seconds | Output | Columns | CUDA status | Priority |
|---|---:|---|---:|---|---|
| `cdl_pattern` | 0.150888 | DataFrame | 62 | `not_implemented_candle_or_pattern` | `medium` |
| `hilo` | 0.128683 | DataFrame | 3 | `not_classified` | `unknown` |
| `hma` | 0.118515 | Series | 1 | `not_classified` | `unknown` |
| `stc` | 0.072332 | DataFrame | 3 | `not_classified` | `unknown` |
| `ha` | 0.072114 | DataFrame | 4 | `not_implemented_candle_or_pattern` | `medium` |
| `ht_phasor` | 0.056875 | DataFrame | 2 | `not_classified` | `unknown` |
| `ht_trendmode` | 0.056805 | Series | 1 | `not_classified` | `unknown` |
| `ht_dcperiod` | 0.056789 | Series | 1 | `not_classified` | `unknown` |
| `ht_sine` | 0.056413 | DataFrame | 2 | `not_classified` | `unknown` |
| `ht_dcphase` | 0.056383 | Series | 1 | `not_classified` | `unknown` |
| `ht_trendline` | 0.056268 | Series | 1 | `not_classified` | `unknown` |
| `wma` | 0.040226 | Series | 1 | `not_classified` | `unknown` |
| `coppock` | 0.040152 | Series | 1 | `not_classified` | `unknown` |
| `jma` | 0.038952 | Series | 1 | `not_implemented_recursive_candidate` | `high` |
| `wad` | 0.034591 | Series | 1 | `not_classified` | `unknown` |
| `ebsw` | 0.028495 | Series | 1 | `not_classified` | `unknown` |
| `kama` | 0.024487 | Series | 1 | `not_implemented_recursive_candidate` | `high` |
| `mama` | 0.012025 | DataFrame | 2 | `not_classified` | `unknown` |
| `aroon` | 0.010937 | DataFrame | 3 | `not_implemented_rolling_candidate` | `high` |
| `msw` | 0.009171 | DataFrame | 2 | `not_classified` | `unknown` |
| `mavp` | 0.009041 | Series | 1 | `not_classified` | `unknown` |
| `rvgi` | 0.008157 | DataFrame | 3 | `not_classified` | `unknown` |
| `minmaxindex` | 0.008010 | DataFrame | 2 | `not_classified` | `unknown` |
| `dm` | 0.006403 | DataFrame | 2 | `not_classified` | `unknown` |
| `npround` | 0.006127 | Series | 1 | `not_classified` | `unknown` |
| `squeeze_pro` | 0.004986 | DataFrame | 6 | `not_classified` | `unknown` |
| `pmax` | 0.004909 | Series | 1 | `not_classified` | `unknown` |
| `maxindex` | 0.004067 | Series | 1 | `not_classified` | `unknown` |
| `minindex` | 0.004017 | Series | 1 | `not_classified` | `unknown` |
| `adxr` | 0.003439 | DataFrame | 3 | `not_classified` | `unknown` |
