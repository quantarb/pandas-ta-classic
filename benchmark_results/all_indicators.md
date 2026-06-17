# Full Indicator Benchmark

This report has two different kinds of data:

- Direct pandas-vs-cuDF timings for the CUDA panel indicators already implemented.
- Pandas baseline timings for every df.ta indicator, used to rank what should be ported next.

It does not claim CUDA is faster for indicators that have not been CUDA-ported yet.
Those rows are candidates, not speedup results.

- Rows: 5000
- Repeats: 1
- Indicators attempted: 217
- Successful: 217
- Failed: 0

## Known Faster With CUDA

Measured on the DGX Spark multi-symbol variable-length panel benchmark. These are the indicators engine=auto should send to cuDF for large enough multi-symbol panels; `obv` and `pvt` use a higher static threshold because they were slower on smaller panels.

| Indicator | pandas | cuDF | Speedup |
|---|---:|---:|---:|
| `sma_20` | 0.097314s | 0.028157s | 3.46x |
| `stdev_20` | 0.109964s | 0.026522s | 4.15x |
| `variance_20` | 0.107695s | 0.026422s | 4.08x |
| `zscore_20` | 0.178124s | 0.058276s | 3.06x |
| `bbands_20` | 0.183413s | 0.063285s | 2.90x |
| `donchian_20` | 0.191756s | 0.056011s | 3.42x |
| `stoch_20` | 0.194958s | 0.066594s | 2.93x |
| `true_range` | 0.110752s | 0.048955s | 2.26x |
| `atr_14` | 0.226672s | 0.076771s | 2.95x |
| `willr_14` | 0.195842s | 0.066453s | 2.95x |
| `ao` | 0.237657s | 0.065949s | 3.60x |
| `cmo_14` | 0.288722s | 0.091176s | 3.17x |
| `cmf_20` | 0.222484s | 0.064756s | 3.44x |
| `hvol_20` | 0.157391s | 0.064591s | 2.44x |
| `obv` | 0.076254s | 0.064259s | 1.19x |
| `pvt` | 0.072061s | 0.061851s | 1.17x |

## Known Slower With CUDA

These were tested on the same multi-symbol panel and should stay on pandas by default.

| Indicator | pandas | cuDF | Speedup | Default |
|---|---:|---:|---:|---|
| `return` | 0.024997s | 0.027727s | 0.90x | pandas |
| `log_return` | 0.027402s | 0.030672s | 0.89x | pandas |
| `mom_10` | 0.024407s | 0.028189s | 0.87x | pandas |
| `roc_10` | 0.025015s | 0.031136s | 0.80x | pandas |
| `pvi_1` | 0.090129s | 0.123777s | 0.73x | pandas |
| `nvi_1` | 0.089718s | 0.125182s | 0.72x | pandas |

## Already Implemented In Panel Engine

| Indicator | Baseline pandas seconds | Panel CUDA status |
|---|---:|---|
| `ao` | 0.000422s | `implemented_panel_accelerated` |
| `atr` | 0.001296s | `implemented_panel_accelerated` |
| `bbands` | 0.001003s | `implemented_panel_accelerated` |
| `cmf` | 0.000651s | `implemented_panel_accelerated` |
| `cmo` | 0.000982s | `implemented_panel_accelerated` |
| `donchian` | 0.000631s | `implemented_panel_accelerated` |
| `hvol` | 0.000521s | `implemented_panel_accelerated` |
| `mom` | 0.000205s | `implemented_panel_pandas_default` |
| `nvi` | 0.001227s | `implemented_panel_pandas_default` |
| `obv` | 0.000870s | `implemented_panel_accelerated` |
| `pvi` | 0.001309s | `implemented_panel_pandas_default` |
| `pvt` | 0.000517s | `implemented_panel_accelerated` |
| `roc` | 0.000419s | `implemented_panel_pandas_default` |
| `sma` | 0.000353s | `implemented_panel_accelerated` |
| `stdev` | 0.000537s | `implemented_panel_accelerated` |
| `stoch` | 0.001508s | `implemented_panel_accelerated` |
| `true_range` | 0.000992s | `implemented_panel_accelerated` |
| `variance` | 0.000452s | `implemented_panel_accelerated` |
| `willr` | 0.000537s | `implemented_panel_accelerated` |
| `zscore` | 0.000532s | `implemented_panel_accelerated` |

## Not Yet CUDA-Ported: Next Candidates

These have not been benchmarked against CUDA yet because there is no CUDA implementation for them in this fork. They are ranked by pandas runtime from this full benchmark.

| Indicator | pandas seconds | Output | Columns | Candidate type | Priority |
|---|---:|---|---:|---|---|
| `cdl_pattern` | 0.151167s | DataFrame | 62 | `not_implemented_candle_or_pattern` | `medium` |
| `hilo` | 0.129468s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `hma` | 0.118833s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `ha` | 0.073806s | DataFrame | 4 | `not_implemented_candle_or_pattern` | `medium` |
| `stc` | 0.073131s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `ht_phasor` | 0.057937s | DataFrame | 2 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `ht_trendmode` | 0.057652s | Series | 1 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `ht_dcperiod` | 0.057631s | Series | 1 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `ht_trendline` | 0.057568s | Series | 1 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `ht_sine` | 0.057421s | DataFrame | 2 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `ht_dcphase` | 0.057311s | Series | 1 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `coppock` | 0.039881s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `jma` | 0.039354s | Series | 1 | `not_implemented_recursive_candidate` | `high` |
| `wma` | 0.039350s | Series | 1 | `not_implemented_weighted_rolling_candidate` | `medium` |
| `wad` | 0.035925s | Series | 1 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `ebsw` | 0.028682s | Series | 1 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `kama` | 0.024468s | Series | 1 | `not_implemented_recursive_candidate` | `high` |
| `mama` | 0.012350s | DataFrame | 2 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `aroon` | 0.011353s | DataFrame | 3 | `not_implemented_rolling_candidate` | `high` |
| `msw` | 0.009426s | DataFrame | 2 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `mavp` | 0.009113s | Series | 1 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `minmaxindex` | 0.008445s | DataFrame | 2 | `not_implemented_rolling_candidate` | `high` |
| `rvgi` | 0.008247s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `dm` | 0.006594s | DataFrame | 2 | `not_implemented_composite_rolling_candidate` | `medium` |
| `pmax` | 0.005159s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `squeeze_pro` | 0.004775s | DataFrame | 6 | `not_implemented_composite_rolling_candidate` | `medium` |
| `minindex` | 0.004274s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `maxindex` | 0.004175s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `adxr` | 0.003694s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `adx` | 0.003557s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `macdfix` | 0.003257s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `macd` | 0.003131s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `squeeze` | 0.002927s | DataFrame | 4 | `not_implemented_composite_rolling_candidate` | `medium` |
| `swma` | 0.002833s | Series | 1 | `not_implemented_weighted_rolling_candidate` | `medium` |
| `t3` | 0.002434s | Series | 1 | `not_implemented_weighted_rolling_candidate` | `medium` |
| `dx` | 0.002304s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `vidya` | 0.002302s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `ichimoku` | 0.002223s | DataFrame | 5 | `not_implemented_composite_rolling_candidate` | `medium` |
| `quantile` | 0.002191s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `aobv` | 0.002137s | DataFrame | 7 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `qqe` | 0.002108s | DataFrame | 7 | `not_implemented_recursive_candidate` | `high` |
| `inertia` | 0.002007s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `uo` | 0.001997s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `mfi` | 0.001964s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `cksp` | 0.001918s | DataFrame | 2 | `not_implemented_composite_rolling_candidate` | `medium` |
| `kvo` | 0.001916s | DataFrame | 2 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `adosc` | 0.001881s | Series | 1 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `rvi` | 0.001879s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `pgo` | 0.001758s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `median` | 0.001733s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `aberration` | 0.001726s | DataFrame | 4 | `not_implemented_rolling_candidate` | `high` |
| `skew` | 0.001715s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `kurtosis` | 0.001696s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `chop` | 0.001681s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `kc` | 0.001675s | DataFrame | 3 | `not_implemented_rolling_candidate` | `high` |
| `stochrsi` | 0.001672s | DataFrame | 2 | `not_implemented_rolling_candidate` | `high` |
| `smi` | 0.001664s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `tema` | 0.001641s | Series | 1 | `not_implemented_weighted_rolling_candidate` | `medium` |
| `cdl_z` | 0.001588s | DataFrame | 4 | `not_implemented_candle_or_pattern` | `medium` |
| `ce` | 0.001568s | DataFrame | 2 | `not_implemented_rolling_candidate` | `high` |
| `supertrend` | 0.001548s | DataFrame | 4 | `not_implemented_recursive_candidate` | `high` |
| `minus_dm` | 0.001511s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `vortex` | 0.001507s | DataFrame | 2 | `not_implemented_rolling_candidate` | `high` |
| `plus_dm` | 0.001481s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `natr` | 0.001426s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `amat` | 0.001351s | DataFrame | 2 | `not_implemented_composite_rolling_candidate` | `medium` |
| `brar` | 0.001340s | DataFrame | 2 | `not_implemented_composite_rolling_candidate` | `medium` |
| `vwap` | 0.001323s | Series | 1 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `tsi` | 0.001292s | DataFrame | 2 | `not_implemented_composite_rolling_candidate` | `medium` |
| `vfi` | 0.001270s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `macdext` | 0.001233s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `fisher` | 0.001202s | DataFrame | 2 | `not_implemented_recursive_candidate` | `high` |
| `rsi` | 0.001200s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `kst` | 0.001189s | DataFrame | 2 | `not_implemented_composite_rolling_candidate` | `medium` |
| `stochf` | 0.001178s | DataFrame | 2 | `not_implemented_rolling_candidate` | `high` |
| `kdj` | 0.001152s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `tos_stdevall` | 0.001142s | DataFrame | 7 | `not_implemented_rolling_candidate` | `high` |
| `pvo` | 0.001095s | DataFrame | 3 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `ppo` | 0.001022s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `vwmacd` | 0.000925s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |

## Low-Value Or Pandas-Default Indicators

These are cheap elementwise/shift/helper operations or signal utilities. CUDA can still be revisited if they are fused into a larger GPU-resident pipeline, but standalone CUDA is unlikely to be the first win.

| Indicator | pandas seconds | Reason |
|---|---:|---|
| `npround` | 0.006178s | `pandas_likely_faster_or_low_value_cuda` |
| `cpr` | 0.002036s | `pandas_likely_faster_or_low_value_cuda` |
| `mmar` | 0.001568s | `pandas_likely_faster_or_low_value_cuda` |
| `trix` | 0.001389s | `pandas_likely_faster_or_low_value_cuda` |
| `trixh` | 0.001378s | `pandas_likely_faster_or_low_value_cuda` |
| `thermo` | 0.001282s | `pandas_likely_faster_or_low_value_cuda` |
| `dema` | 0.001119s | `pandas_likely_faster_or_low_value_cuda` |
| `rainbow` | 0.001111s | `pandas_likely_faster_or_low_value_cuda` |
| `ttm_trend` | 0.000945s | `pandas_likely_faster_or_low_value_cuda` |
| `edecay` | 0.000904s | `pandas_likely_faster_or_low_value_cuda` |
| `psl` | 0.000866s | `pandas_likely_faster_or_low_value_cuda` |
| `pvr` | 0.000829s | `pandas_likely_faster_or_low_value_cuda` |
| `decay` | 0.000821s | `pandas_likely_faster_or_low_value_cuda` |
| `cvi` | 0.000698s | `pandas_likely_faster_or_low_value_cuda` |
| `minmax` | 0.000688s | `pandas_likely_faster_or_low_value_cuda` |
| `eri` | 0.000617s | `pandas_likely_faster_or_low_value_cuda` |
| `po` | 0.000592s | `pandas_likely_faster_or_low_value_cuda` |
| `eom` | 0.000563s | `pandas_likely_faster_or_low_value_cuda` |
| `cfo` | 0.000539s | `pandas_likely_faster_or_low_value_cuda` |
| `pdist` | 0.000531s | `pandas_likely_faster_or_low_value_cuda` |
| `drawdown` | 0.000529s | `pandas_likely_faster_or_low_value_cuda` |
| `fosc` | 0.000517s | `pandas_likely_faster_or_low_value_cuda` |
| `dsp` | 0.000496s | `pandas_likely_faster_or_low_value_cuda` |
| `ad` | 0.000488s | `pandas_likely_faster_or_low_value_cuda` |
| `cti` | 0.000483s | `pandas_likely_faster_or_low_value_cuda` |
| `efi` | 0.000471s | `pandas_likely_faster_or_low_value_cuda` |
| `qstick` | 0.000466s | `pandas_likely_faster_or_low_value_cuda` |
| `bias` | 0.000434s | `pandas_likely_faster_or_low_value_cuda` |
| `vwma` | 0.000426s | `pandas_likely_faster_or_low_value_cuda` |
| `vosc` | 0.000415s | `pandas_likely_faster_or_low_value_cuda` |
| `dpo` | 0.000405s | `pandas_likely_faster_or_low_value_cuda` |
| `emv` | 0.000390s | `pandas_likely_faster_or_low_value_cuda` |
| `cg` | 0.000389s | `pandas_likely_faster_or_low_value_cuda` |
| `bop` | 0.000388s | `pandas_likely_faster_or_low_value_cuda` |
| `marketfi` | 0.000384s | `pandas_likely_faster_or_low_value_cuda` |
| `rocp` | 0.000366s | `pandas_likely_faster` |
| `rolling_sum` | 0.000364s | `pandas_likely_faster_or_low_value_cuda` |
| `percent_return` | 0.000361s | `pandas_likely_faster` |
| `er` | 0.000359s | `pandas_likely_faster_or_low_value_cuda` |
| `tan` | 0.000344s | `pandas_likely_faster_or_low_value_cuda` |
| `rocr100` | 0.000336s | `pandas_likely_faster` |
| `log_return` | 0.000336s | `pandas_likely_faster` |
| `rolling_max` | 0.000325s | `pandas_likely_faster_or_low_value_cuda` |
| `rocr` | 0.000320s | `pandas_likely_faster` |
| `medprice` | 0.000311s | `pandas_likely_faster_or_low_value_cuda` |
| `rolling_min` | 0.000309s | `pandas_likely_faster_or_low_value_cuda` |
| `tanh` | 0.000298s | `pandas_likely_faster_or_low_value_cuda` |
| `log10` | 0.000295s | `pandas_likely_faster_or_low_value_cuda` |
| `ohlc4` | 0.000288s | `pandas_likely_faster_or_low_value_cuda` |
| `increasing` | 0.000284s | `pandas_likely_faster_or_low_value_cuda` |
| `todeg` | 0.000284s | `pandas_likely_faster_or_low_value_cuda` |
| `ln` | 0.000280s | `pandas_likely_faster_or_low_value_cuda` |
| `sin` | 0.000279s | `pandas_likely_faster_or_low_value_cuda` |
| `sqrt` | 0.000272s | `pandas_likely_faster_or_low_value_cuda` |
| `wcp` | 0.000271s | `pandas_likely_faster_or_low_value_cuda` |
| `hlc3` | 0.000268s | `pandas_likely_faster_or_low_value_cuda` |
| `avgprice` | 0.000263s | `pandas_likely_faster_or_low_value_cuda` |
| `torad` | 0.000253s | `pandas_likely_faster_or_low_value_cuda` |
| `decreasing` | 0.000253s | `pandas_likely_faster_or_low_value_cuda` |
| `hl2` | 0.000250s | `pandas_likely_faster_or_low_value_cuda` |
| `exp` | 0.000250s | `pandas_likely_faster_or_low_value_cuda` |
| `slope` | 0.000248s | `pandas_likely_faster_or_low_value_cuda` |
| `typprice` | 0.000248s | `pandas_likely_faster_or_low_value_cuda` |
| `mult` | 0.000246s | `pandas_likely_faster_or_low_value_cuda` |
| `npabs` | 0.000245s | `pandas_likely_faster_or_low_value_cuda` |
| `add` | 0.000242s | `pandas_likely_faster_or_low_value_cuda` |
| `cos` | 0.000236s | `pandas_likely_faster_or_low_value_cuda` |
| `sinh` | 0.000236s | `pandas_likely_faster_or_low_value_cuda` |
| `floor` | 0.000234s | `pandas_likely_faster_or_low_value_cuda` |
| `asin` | 0.000221s | `pandas_likely_faster_or_low_value_cuda` |
| `acos` | 0.000220s | `pandas_likely_faster_or_low_value_cuda` |
| `pvol` | 0.000212s | `pandas_likely_faster_or_low_value_cuda` |
| `trunc` | 0.000206s | `pandas_likely_faster_or_low_value_cuda` |
| `ceil` | 0.000202s | `pandas_likely_faster_or_low_value_cuda` |
| `atan` | 0.000195s | `pandas_likely_faster_or_low_value_cuda` |
| `sub` | 0.000187s | `pandas_likely_faster_or_low_value_cuda` |
| `cosh` | 0.000184s | `pandas_likely_faster_or_low_value_cuda` |
| `div` | 0.000173s | `pandas_likely_faster_or_low_value_cuda` |
| `ma` | 0.000147s | `pandas_likely_faster_or_low_value_cuda` |
| `beta` | 0.000103s | `pandas_likely_faster_or_low_value_cuda` |

## Classification Counts

| Status | Count |
|---|---:|
| `implemented_panel_accelerated` | 16 |
| `implemented_panel_pandas_default` | 4 |
| `not_implemented_candle_or_pattern` | 5 |
| `not_implemented_composite_rolling_candidate` | 35 |
| `not_implemented_cumulative_volume_candidate` | 8 |
| `not_implemented_cycle_or_hilbert_candidate` | 9 |
| `not_implemented_recursive_candidate` | 15 |
| `not_implemented_rolling_candidate` | 30 |
| `not_implemented_weighted_rolling_candidate` | 15 |
| `pandas_likely_faster` | 5 |
| `pandas_likely_faster_or_low_value_cuda` | 75 |
