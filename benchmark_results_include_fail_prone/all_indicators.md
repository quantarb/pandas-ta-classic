# Full Indicator Benchmark

This report has two different kinds of data:

- Direct pandas-vs-cuDF timings for the CUDA panel indicators already implemented.
- Pandas baseline timings for every df.ta indicator, used to rank what should be ported next.

It does not claim CUDA is faster for indicators that have not been CUDA-ported yet.
Those rows are candidates, not speedup results.

- Rows: 5000
- Repeats: 1
- Indicators attempted: 232
- Successful: 232
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
| `ao` | 0.000509s | `implemented_panel_accelerated` |
| `atr` | 0.001282s | `implemented_panel_accelerated` |
| `bbands` | 0.000963s | `implemented_panel_accelerated` |
| `cmf` | 0.000516s | `implemented_panel_accelerated` |
| `cmo` | 0.000862s | `implemented_panel_accelerated` |
| `donchian` | 0.000707s | `implemented_panel_accelerated` |
| `hvol` | 0.000528s | `implemented_panel_accelerated` |
| `mom` | 0.000254s | `implemented_panel_pandas_default` |
| `nvi` | 0.001198s | `implemented_panel_pandas_default` |
| `obv` | 0.000843s | `implemented_panel_accelerated` |
| `pvi` | 0.001300s | `implemented_panel_pandas_default` |
| `pvt` | 0.000478s | `implemented_panel_accelerated` |
| `roc` | 0.000408s | `implemented_panel_pandas_default` |
| `sma` | 0.000264s | `implemented_panel_accelerated` |
| `stdev` | 0.000567s | `implemented_panel_accelerated` |
| `stoch` | 0.001564s | `implemented_panel_accelerated` |
| `true_range` | 0.001033s | `implemented_panel_accelerated` |
| `variance` | 0.000461s | `implemented_panel_accelerated` |
| `willr` | 0.000555s | `implemented_panel_accelerated` |
| `zscore` | 0.000534s | `implemented_panel_accelerated` |

## Not Yet CUDA-Ported: Next Candidates

These have not been benchmarked against CUDA yet because there is no CUDA implementation for them in this fork. They are ranked by pandas runtime from this full benchmark.

| Indicator | pandas seconds | Output | Columns | Candidate type | Priority |
|---|---:|---|---:|---|---|
| `cdl_pattern` | 0.153255s | DataFrame | 62 | `not_implemented_candle_or_pattern` | `medium` |
| `hilo` | 0.131199s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `hma` | 0.119323s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `ha` | 0.073563s | DataFrame | 4 | `not_implemented_candle_or_pattern` | `medium` |
| `stc` | 0.073358s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `ht_phasor` | 0.058348s | DataFrame | 2 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `ht_trendmode` | 0.058283s | Series | 1 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `ht_dcperiod` | 0.058110s | Series | 1 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `ht_sine` | 0.057842s | DataFrame | 2 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `ht_trendline` | 0.057740s | Series | 1 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `ht_dcphase` | 0.057585s | Series | 1 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `coppock` | 0.040186s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `wma` | 0.039470s | Series | 1 | `not_implemented_weighted_rolling_candidate` | `medium` |
| `jma` | 0.039365s | Series | 1 | `not_implemented_recursive_candidate` | `high` |
| `wad` | 0.034639s | Series | 1 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `ebsw` | 0.028828s | Series | 1 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `kama` | 0.024335s | Series | 1 | `not_implemented_recursive_candidate` | `high` |
| `mama` | 0.012245s | DataFrame | 2 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `aroon` | 0.011472s | DataFrame | 3 | `not_implemented_rolling_candidate` | `high` |
| `msw` | 0.009424s | DataFrame | 2 | `not_implemented_cycle_or_hilbert_candidate` | `medium` |
| `mavp` | 0.009016s | Series | 1 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `minmaxindex` | 0.008364s | DataFrame | 2 | `not_implemented_rolling_candidate` | `high` |
| `rvgi` | 0.008299s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `dm` | 0.006454s | DataFrame | 2 | `not_implemented_composite_rolling_candidate` | `medium` |
| `pmax` | 0.005260s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `squeeze_pro` | 0.004595s | DataFrame | 6 | `not_implemented_composite_rolling_candidate` | `medium` |
| `minindex` | 0.004203s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `maxindex` | 0.004203s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `adxr` | 0.003770s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `adx` | 0.003457s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `macd` | 0.003140s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `macdfix` | 0.003128s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `squeeze` | 0.002850s | DataFrame | 4 | `not_implemented_composite_rolling_candidate` | `medium` |
| `swma` | 0.002779s | Series | 1 | `not_implemented_weighted_rolling_candidate` | `medium` |
| `dx` | 0.002422s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `vidya` | 0.002328s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `ichimoku` | 0.002287s | DataFrame | 5 | `not_implemented_composite_rolling_candidate` | `medium` |
| `t3` | 0.002260s | Series | 1 | `not_implemented_weighted_rolling_candidate` | `medium` |
| `quantile` | 0.002137s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `kvo` | 0.002130s | DataFrame | 2 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `uo` | 0.002082s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `qqe` | 0.002061s | DataFrame | 7 | `not_implemented_recursive_candidate` | `high` |
| `aobv` | 0.002001s | DataFrame | 7 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `inertia` | 0.001900s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `mfi` | 0.001887s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `rvi` | 0.001860s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `adosc` | 0.001846s | Series | 1 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `kc` | 0.001828s | DataFrame | 3 | `not_implemented_rolling_candidate` | `high` |
| `pgo` | 0.001821s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `median` | 0.001818s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `stochrsi` | 0.001785s | DataFrame | 2 | `not_implemented_rolling_candidate` | `high` |
| `supertrend` | 0.001765s | DataFrame | 4 | `not_implemented_recursive_candidate` | `high` |
| `ce` | 0.001742s | DataFrame | 2 | `not_implemented_rolling_candidate` | `high` |
| `cksp` | 0.001706s | DataFrame | 2 | `not_implemented_composite_rolling_candidate` | `medium` |
| `cdl_z` | 0.001706s | DataFrame | 4 | `not_implemented_candle_or_pattern` | `medium` |
| `chop` | 0.001685s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `kurtosis` | 0.001669s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `aberration` | 0.001660s | DataFrame | 4 | `not_implemented_rolling_candidate` | `high` |
| `skew` | 0.001624s | Series | 1 | `not_implemented_rolling_candidate` | `high` |
| `smi` | 0.001523s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `vortex` | 0.001504s | DataFrame | 2 | `not_implemented_rolling_candidate` | `high` |
| `plus_dm` | 0.001503s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `tema` | 0.001476s | Series | 1 | `not_implemented_weighted_rolling_candidate` | `medium` |
| `amat` | 0.001470s | DataFrame | 2 | `not_implemented_composite_rolling_candidate` | `medium` |
| `minus_dm` | 0.001407s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `natr` | 0.001401s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `stochf` | 0.001348s | DataFrame | 2 | `not_implemented_rolling_candidate` | `high` |
| `vwap` | 0.001334s | Series | 1 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `tsi` | 0.001320s | DataFrame | 2 | `not_implemented_composite_rolling_candidate` | `medium` |
| `vfi` | 0.001317s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `kdj` | 0.001307s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `brar` | 0.001288s | DataFrame | 2 | `not_implemented_composite_rolling_candidate` | `medium` |
| `kst` | 0.001279s | DataFrame | 2 | `not_implemented_composite_rolling_candidate` | `medium` |
| `pvo` | 0.001203s | DataFrame | 3 | `not_implemented_cumulative_volume_candidate` | `medium` |
| `rsi` | 0.001201s | Series | 1 | `not_implemented_composite_rolling_candidate` | `medium` |
| `fisher` | 0.001180s | DataFrame | 2 | `not_implemented_recursive_candidate` | `high` |
| `macdext` | 0.001139s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `tos_stdevall` | 0.001115s | DataFrame | 7 | `not_implemented_rolling_candidate` | `high` |
| `ppo` | 0.001085s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |
| `vwmacd` | 0.000938s | DataFrame | 3 | `not_implemented_composite_rolling_candidate` | `medium` |

## Low-Value Or Pandas-Default Indicators

These are cheap elementwise/shift/helper operations or signal utilities. CUDA can still be revisited if they are fused into a larger GPU-resident pipeline, but standalone CUDA is unlikely to be the first win.

| Indicator | pandas seconds | Reason |
|---|---:|---|
| `td_seq` | 1.061840s | `signal_or_helper_not_cuda_target` |
| `npround` | 0.006077s | `pandas_likely_faster_or_low_value_cuda` |
| `vp` | 0.002598s | `signal_or_helper_not_cuda_target` |
| `crossany` | 0.002409s | `signal_or_helper_not_cuda_target` |
| `cpr` | 0.002150s | `pandas_likely_faster_or_low_value_cuda` |
| `mmar` | 0.001491s | `pandas_likely_faster_or_low_value_cuda` |
| `crossover` | 0.001307s | `signal_or_helper_not_cuda_target` |
| `trixh` | 0.001265s | `pandas_likely_faster_or_low_value_cuda` |
| `trix` | 0.001218s | `pandas_likely_faster_or_low_value_cuda` |
| `thermo` | 0.001141s | `pandas_likely_faster_or_low_value_cuda` |
| `dema` | 0.001141s | `pandas_likely_faster_or_low_value_cuda` |
| `rainbow` | 0.001078s | `pandas_likely_faster_or_low_value_cuda` |
| `ttm_trend` | 0.001012s | `pandas_likely_faster_or_low_value_cuda` |
| `edecay` | 0.000907s | `pandas_likely_faster_or_low_value_cuda` |
| `decay` | 0.000875s | `pandas_likely_faster_or_low_value_cuda` |
| `psl` | 0.000846s | `pandas_likely_faster_or_low_value_cuda` |
| `pvr` | 0.000784s | `pandas_likely_faster_or_low_value_cuda` |
| `eri` | 0.000735s | `pandas_likely_faster_or_low_value_cuda` |
| `cvi` | 0.000687s | `pandas_likely_faster_or_low_value_cuda` |
| `po` | 0.000660s | `pandas_likely_faster_or_low_value_cuda` |
| `eom` | 0.000635s | `pandas_likely_faster_or_low_value_cuda` |
| `minmax` | 0.000623s | `pandas_likely_faster_or_low_value_cuda` |
| `pdist` | 0.000586s | `pandas_likely_faster_or_low_value_cuda` |
| `efi` | 0.000564s | `pandas_likely_faster_or_low_value_cuda` |
| `fosc` | 0.000528s | `pandas_likely_faster_or_low_value_cuda` |
| `cti` | 0.000516s | `pandas_likely_faster_or_low_value_cuda` |
| `ad` | 0.000494s | `pandas_likely_faster_or_low_value_cuda` |
| `cfo` | 0.000493s | `pandas_likely_faster_or_low_value_cuda` |
| `drawdown` | 0.000492s | `pandas_likely_faster_or_low_value_cuda` |
| `qstick` | 0.000457s | `pandas_likely_faster_or_low_value_cuda` |
| `vwma` | 0.000445s | `pandas_likely_faster_or_low_value_cuda` |
| `dsp` | 0.000443s | `pandas_likely_faster_or_low_value_cuda` |
| `vosc` | 0.000430s | `pandas_likely_faster_or_low_value_cuda` |
| `cg` | 0.000430s | `pandas_likely_faster_or_low_value_cuda` |
| `emv` | 0.000411s | `pandas_likely_faster_or_low_value_cuda` |
| `dpo` | 0.000403s | `pandas_likely_faster_or_low_value_cuda` |
| `er` | 0.000399s | `pandas_likely_faster_or_low_value_cuda` |
| `rolling_min` | 0.000395s | `pandas_likely_faster_or_low_value_cuda` |
| `bias` | 0.000377s | `pandas_likely_faster_or_low_value_cuda` |
| `bop` | 0.000373s | `pandas_likely_faster_or_low_value_cuda` |
| `rocp` | 0.000355s | `pandas_likely_faster` |
| `rocr100` | 0.000354s | `pandas_likely_faster` |
| `percent_return` | 0.000353s | `pandas_likely_faster` |
| `ohlc4` | 0.000340s | `pandas_likely_faster_or_low_value_cuda` |
| `rolling_sum` | 0.000337s | `pandas_likely_faster_or_low_value_cuda` |
| `rolling_max` | 0.000331s | `pandas_likely_faster_or_low_value_cuda` |
| `above` | 0.000324s | `signal_or_helper_not_cuda_target` |
| `log_return` | 0.000314s | `pandas_likely_faster` |
| `mult` | 0.000312s | `pandas_likely_faster_or_low_value_cuda` |
| `decreasing` | 0.000311s | `pandas_likely_faster_or_low_value_cuda` |
| `marketfi` | 0.000308s | `pandas_likely_faster_or_low_value_cuda` |
| `cos` | 0.000299s | `pandas_likely_faster_or_low_value_cuda` |
| `tanh` | 0.000294s | `pandas_likely_faster_or_low_value_cuda` |
| `ln` | 0.000292s | `pandas_likely_faster_or_low_value_cuda` |
| `sinh` | 0.000291s | `pandas_likely_faster_or_low_value_cuda` |
| `cross` | 0.000287s | `signal_or_helper_not_cuda_target` |
| `increasing` | 0.000287s | `pandas_likely_faster_or_low_value_cuda` |
| `log10` | 0.000283s | `pandas_likely_faster_or_low_value_cuda` |
| `wcp` | 0.000281s | `pandas_likely_faster_or_low_value_cuda` |
| `tan` | 0.000278s | `pandas_likely_faster_or_low_value_cuda` |
| `sin` | 0.000270s | `pandas_likely_faster_or_low_value_cuda` |
| `rocr` | 0.000264s | `pandas_likely_faster` |
| `exp` | 0.000263s | `pandas_likely_faster_or_low_value_cuda` |
| `pvol` | 0.000261s | `pandas_likely_faster_or_low_value_cuda` |
| `slope` | 0.000256s | `pandas_likely_faster_or_low_value_cuda` |
| `typprice` | 0.000255s | `pandas_likely_faster_or_low_value_cuda` |
| `above_value` | 0.000253s | `signal_or_helper_not_cuda_target` |
| `cross_value` | 0.000251s | `signal_or_helper_not_cuda_target` |
| `atan` | 0.000251s | `pandas_likely_faster_or_low_value_cuda` |
| `hlc3` | 0.000251s | `pandas_likely_faster_or_low_value_cuda` |
| `avgprice` | 0.000249s | `pandas_likely_faster_or_low_value_cuda` |
| `asin` | 0.000249s | `pandas_likely_faster_or_low_value_cuda` |
| `below_value` | 0.000248s | `signal_or_helper_not_cuda_target` |
| `lag` | 0.000243s | `signal_or_helper_not_cuda_target` |
| `cosh` | 0.000242s | `pandas_likely_faster_or_low_value_cuda` |
| `acos` | 0.000240s | `pandas_likely_faster_or_low_value_cuda` |
| `below` | 0.000239s | `signal_or_helper_not_cuda_target` |
| `hl2` | 0.000236s | `pandas_likely_faster_or_low_value_cuda` |
| `ceil` | 0.000231s | `pandas_likely_faster_or_low_value_cuda` |
| `sub` | 0.000229s | `pandas_likely_faster_or_low_value_cuda` |

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
| `signal_or_helper_not_cuda_target` | 15 |
