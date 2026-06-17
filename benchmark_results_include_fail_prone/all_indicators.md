# Full Indicator Benchmark

- Rows: 5000
- Repeats: 1
- Indicators attempted: 232
- Successful: 232
- Failed: 0

## CUDA Classification Counts

| CUDA status | Count |
|---|---:|
| `implemented_panel_accelerated` | 10 |
| `implemented_panel_pandas_default` | 2 |
| `not_classified` | 164 |
| `not_implemented_candle_or_pattern` | 5 |
| `not_implemented_recursive_candidate` | 15 |
| `not_implemented_rolling_candidate` | 31 |
| `pandas_likely_faster` | 5 |

## Slowest Successful Indicators

| Indicator | Seconds | Output | Columns | CUDA status | Priority |
|---|---:|---|---:|---|---|
| `td_seq` | 1.074328 | DataFrame | 2 | `not_classified` | `unknown` |
| `cdl_pattern` | 0.152915 | DataFrame | 62 | `not_implemented_candle_or_pattern` | `medium` |
| `hilo` | 0.129854 | DataFrame | 3 | `not_classified` | `unknown` |
| `hma` | 0.120397 | Series | 1 | `not_classified` | `unknown` |
| `ha` | 0.073453 | DataFrame | 4 | `not_implemented_candle_or_pattern` | `medium` |
| `stc` | 0.073044 | DataFrame | 3 | `not_classified` | `unknown` |
| `ht_phasor` | 0.057853 | DataFrame | 2 | `not_classified` | `unknown` |
| `ht_sine` | 0.057807 | DataFrame | 2 | `not_classified` | `unknown` |
| `ht_trendmode` | 0.057709 | Series | 1 | `not_classified` | `unknown` |
| `ht_dcperiod` | 0.057702 | Series | 1 | `not_classified` | `unknown` |
| `ht_trendline` | 0.057594 | Series | 1 | `not_classified` | `unknown` |
| `ht_dcphase` | 0.057432 | Series | 1 | `not_classified` | `unknown` |
| `coppock` | 0.040505 | Series | 1 | `not_classified` | `unknown` |
| `wma` | 0.040145 | Series | 1 | `not_classified` | `unknown` |
| `jma` | 0.039808 | Series | 1 | `not_implemented_recursive_candidate` | `high` |
| `wad` | 0.035861 | Series | 1 | `not_classified` | `unknown` |
| `ebsw` | 0.029376 | Series | 1 | `not_classified` | `unknown` |
| `kama` | 0.024350 | Series | 1 | `not_implemented_recursive_candidate` | `high` |
| `mama` | 0.012144 | DataFrame | 2 | `not_classified` | `unknown` |
| `aroon` | 0.011085 | DataFrame | 3 | `not_implemented_rolling_candidate` | `high` |
| `msw` | 0.009391 | DataFrame | 2 | `not_classified` | `unknown` |
| `mavp` | 0.009107 | Series | 1 | `not_classified` | `unknown` |
| `rvgi` | 0.008251 | DataFrame | 3 | `not_classified` | `unknown` |
| `minmaxindex` | 0.008142 | DataFrame | 2 | `not_classified` | `unknown` |
| `dm` | 0.006658 | DataFrame | 2 | `not_classified` | `unknown` |
| `npround` | 0.006235 | Series | 1 | `not_classified` | `unknown` |
| `pmax` | 0.005236 | Series | 1 | `not_classified` | `unknown` |
| `squeeze_pro` | 0.004993 | DataFrame | 6 | `not_classified` | `unknown` |
| `minindex` | 0.004234 | Series | 1 | `not_classified` | `unknown` |
| `maxindex` | 0.004110 | Series | 1 | `not_classified` | `unknown` |
