from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import BaseForecastModel


class LinearRegressionModel(BaseForecastModel):
    """Simple linear trend model over daily portfolio returns."""

    def __init__(self) -> None:
        self._is_fitted = False
        self._intercept = 0.0
        self._slope = 0.0
        self._r2 = 0.0
        self._n = 0

    def fit(self, data: pd.Series) -> "LinearRegressionModel":
        clean = pd.Series(data).dropna().astype(float)
        if len(clean) < 10:
            raise ValueError("LinearRegressionModel requires at least 10 observations.")

        x = np.arange(len(clean), dtype=float)
        y = clean.to_numpy(dtype=float)
        self._slope, self._intercept = np.polyfit(x, y, 1)
        y_hat = self._intercept + self._slope * x

        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        self._r2 = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - ss_res / ss_tot)
        self._n = len(clean)
        self._is_fitted = True
        return self

    def predict(self, periods: int = 1) -> Dict[str, Any]:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        periods = max(1, int(periods))
        start = self._n
        future_x = np.arange(start, start + periods, dtype=float)
        forecast = self._intercept + self._slope * future_x

        return {
            "next_return": float(forecast[0]),
            "forecast_path": forecast.tolist(),
        }

    def get_metrics(self) -> Dict[str, float]:
        if not self._is_fitted:
            return {}

        next_daily_return = self._intercept + self._slope * self._n
        return {
            "trend_slope_daily": float(self._slope),
            "expected_daily_return": float(next_daily_return),
            "expected_annual_return": float(next_daily_return * 252.0),
            "confidence": float(self._r2),
        }
