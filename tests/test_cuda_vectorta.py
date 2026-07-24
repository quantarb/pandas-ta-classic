import numpy as np
import pandas as pd
import pytest

from pandas_ta_classic.cuda import (
    panel_indicators,
    panel_indicators_vectorta,
    panel_indicators_vectorta_cpu,
    vectorta_available,
)


def make_panel(symbols=4, rows=60):
    rng = np.random.default_rng(2028)
    parts = []
    for idx in range(symbols):
        row_count = rows + idx * 7
        innovations = rng.normal(0.0002, 0.018, size=row_count)
        close = 100.0 * np.exp(np.cumsum(innovations))
        parts.append(
            pd.DataFrame(
                {
                    "symbol": f"S{idx}",
                    "close": close,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _expected_vectorta_cpu(frame: pd.DataFrame, specs: list[str]) -> pd.DataFrame:
    import vector_ta

    expected_parts = []
    for _, part in frame.groupby("symbol", sort=False):
        columns = {}
        for spec in specs:
            if spec == "rsi_14":
                columns[spec] = vector_ta.rsi(part["close"].to_numpy(dtype=float), period=14)
            elif spec == "ema_20":
                columns[spec] = vector_ta.ema(part["close"].to_numpy(dtype=float), period=20)
            else:
                raise AssertionError(f"unexpected spec: {spec}")
        expected_parts.append(pd.DataFrame(columns, index=part.index))
    return pd.concat(expected_parts).sort_index()


@pytest.mark.skipif(not vectorta_available(), reason="VectorTA CUDA build unavailable")
def test_panel_indicators_vectorta_matches_vectorta_cpu():
    frame = make_panel(symbols=3, rows=80)
    specs = ["rsi_14", "ema_20"]
    result = panel_indicators_vectorta(frame, specs)
    expected = _expected_vectorta_cpu(frame, specs)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False, check_exact=False, rtol=1e-4, atol=1e-4)


def test_panel_indicators_vectorta_cpu_matches_vectorta_reference():
    pytest.importorskip("vector_ta")
    frame = make_panel(symbols=3, rows=80)
    specs = ["rsi_14", "ema_20"]
    result = panel_indicators_vectorta_cpu(frame, specs)
    expected = _expected_vectorta_cpu(frame, specs)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False, check_exact=False, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not vectorta_available(), reason="VectorTA CUDA build unavailable")
def test_panel_indicators_engine_vectorta_routes_to_backend():
    frame = make_panel(symbols=2, rows=50)
    routed = panel_indicators(frame, ["rsi_14"], engine="vectorta")
    direct = panel_indicators_vectorta(frame, ["rsi_14"])
    pd.testing.assert_frame_equal(routed, direct, check_dtype=False, check_exact=False, rtol=1e-4, atol=1e-4)