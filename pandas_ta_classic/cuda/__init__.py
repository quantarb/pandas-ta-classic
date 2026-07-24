"""Optional CUDA-accelerated technical analysis helpers."""

from .rolling import (
    IndicatorTiming,
    auto_engine_policy,
    benchmark_indicator_engines,
    cuda_available,
    rolling_indicators,
    rolling_indicators_cudf,
    rolling_indicators_pandas,
    synchronize_cuda,
)
from .vectorta import (
    panel_indicators_vectorta,
    panel_indicators_vectorta_cpu,
    vectorta_available,
)
from .panel import (
    DEFAULT_PANEL_SPECS,
    PanelIndicatorSpec,
    PanelIndicatorTiming,
    benchmark_panel_indicators,
    panel_auto_engine_policy,
    panel_indicators,
    panel_indicators_auto,
    panel_indicators_cudf,
    panel_indicators_pandas,
    static_panel_engine_policy,
)

__all__ = [
    "DEFAULT_PANEL_SPECS",
    "IndicatorTiming",
    "PanelIndicatorSpec",
    "PanelIndicatorTiming",
    "auto_engine_policy",
    "benchmark_indicator_engines",
    "benchmark_panel_indicators",
    "cuda_available",
    "panel_auto_engine_policy",
    "panel_indicators",
    "panel_indicators_auto",
    "panel_indicators_cudf",
    "panel_indicators_pandas",
    "panel_indicators_vectorta",
    "panel_indicators_vectorta_cpu",
    "static_panel_engine_policy",
    "vectorta_available",
    "rolling_indicators",
    "rolling_indicators_cudf",
    "rolling_indicators_pandas",
    "synchronize_cuda",
]
