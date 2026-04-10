from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .arima_model import ARIMAModel
from .garch_model import GARCHModel
from .linear_regression_model import LinearRegressionModel


def _safe_run_model(model: Any, series: pd.Series, periods: int) -> Dict[str, Any]:
    try:
        model.fit(series)
        prediction = model.predict(periods=periods)
        metrics = model.get_metrics()
        return {
            "available": True,
            "prediction": prediction,
            "metrics": metrics,
            "error": "",
        }
    except Exception as exc:
        return {
            "available": False,
            "prediction": {},
            "metrics": {},
            "error": str(exc),
        }


def run_advanced_models(
    returns: pd.Series,
    forecast_periods: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Run advanced model layer with unified outputs for scoring/UI."""
    clean_returns = pd.Series(returns).dropna().astype(float)

    outputs = {
        "linear_regression": _safe_run_model(
            LinearRegressionModel(),
            clean_returns,
            periods=forecast_periods,
        ),
        "arima": _safe_run_model(
            ARIMAModel(order=(1, 0, 1)),
            clean_returns,
            periods=forecast_periods,
        ),
        "garch": _safe_run_model(
            GARCHModel(p=1, q=1),
            clean_returns,
            periods=forecast_periods,
        ),
    }
    return outputs
