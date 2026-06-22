from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import BaseForecastModel


class LinearRegressionModel(BaseForecastModel):
    """Simple linear trend model over cumulative portfolio returns."""

    def __init__(self) -> None:
        self._is_fitted = False
        self._intercept = 0.0
        self._slope = 0.0
        self._r2 = 0.0
        self._n = 0
        self._last_cum = 1.0

    def fit(self, data: pd.Series) -> "LinearRegressionModel":
        clean = pd.Series(data).dropna().astype(float)
        if len(clean) < 10:
            raise ValueError("LinearRegressionModel requires at least 10 observations.")

        cum_returns = (1.0 + clean).cumprod()
        self._last_cum = float(cum_returns.iloc[-1])

        x = np.arange(len(cum_returns), dtype=float)
        y = cum_returns.to_numpy(dtype=float)
        self._slope, self._intercept = np.polyfit(x, y, 1)
        y_hat = self._intercept + self._slope * x

        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        self._r2 = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - ss_res / ss_tot)
        self._n = len(cum_returns)
        self._is_fitted = True
        return self

    def predict(self, periods: int = 1) -> Dict[str, Any]:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        periods = max(1, int(periods))
        start = self._n
        future_x = np.arange(start, start + periods, dtype=float)
        forecast_cum = self._intercept + self._slope * future_x

        implied_returns = []
        prev = self._last_cum
        for f in forecast_cum:
            implied_returns.append(float((f / prev) - 1.0) if prev != 0 else 0.0)
            prev = f

        return {
            "next_return": float(implied_returns[0]),
            "forecast_path": forecast_cum.tolist(),
        }

    def get_metrics(self) -> Dict[str, float]:
        if not self._is_fitted:
            return {}

        # The implied daily return on the trend line is the slope divided by the current cumulative value
        implied_daily = float(self._slope / self._last_cum) if self._last_cum != 0 else 0.0

        return {
            "trend_slope_daily": float(self._slope),
            "expected_daily_return": implied_daily,
            "expected_annual_return": float(implied_daily * 252.0),
            "confidence": float(self._r2),
        }
