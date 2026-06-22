from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseForecastModel

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:  # pragma: no cover
    ExponentialSmoothing = None  # type: ignore


class ExponentialSmoothingModel(BaseForecastModel):
    """Exponential Smoothing (Holt) model for trend forecasting."""

    def __init__(self, trend: str = "add", damped_trend: bool = True) -> None:
        self.trend = trend
        self.damped_trend = damped_trend
        self._result: Optional[Any] = None
        self._is_fitted = False
        self._last_forecast: Optional[np.ndarray] = None
        self._last_conf_int: Optional[np.ndarray] = None

    def fit(self, data: pd.Series) -> "ExponentialSmoothingModel":
        if ExponentialSmoothing is None:
            raise ImportError("statsmodels is required for ExponentialSmoothingModel.")

        clean = pd.Series(data).dropna().astype(float)
        if len(clean) < 30:
            raise ValueError("ExponentialSmoothingModel requires at least 30 observations.")

        model = ExponentialSmoothing(
            clean,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=None,
            initialization_method="estimated",
        )
        self._result = model.fit(optimized=True)
        self._is_fitted = True
        return self

    def predict(self, periods: int = 1) -> Dict[str, Any]:
        if not self._is_fitted or self._result is None:
            raise ValueError("Model must be fitted before prediction.")

        periods = max(1, int(periods))
        mean_forecast = np.asarray(self._result.forecast(periods), dtype=float)

        # ExponentialSmoothing in statsmodels doesn't natively return conf intervals from get_prediction in older versions.
        # We can simulate a basic confidence interval based on the model residuals.
        residuals = self._result.resid
        std_error = np.std(residuals)
        
        # Expanding confidence interval over time (sqrt of time)
        z_score = 1.96  # 95% CI
        ci_margins = z_score * std_error * np.sqrt(np.arange(1, periods + 1))
        
        lower_bound = mean_forecast - ci_margins
        upper_bound = mean_forecast + ci_margins
        
        conf_int = np.column_stack((lower_bound, upper_bound))

        self._last_forecast = mean_forecast
        self._last_conf_int = conf_int

        return {
            "next_return": float(mean_forecast[0]),
            "forecast_path": mean_forecast.tolist(),
            "confidence_interval": conf_int.tolist(),
        }

    def get_metrics(self) -> Dict[str, float]:
        if not self._is_fitted or self._result is None:
            return {}

        if self._last_forecast is None:
            forecast = np.asarray(self._result.forecast(1), dtype=float)
            next_return = float(forecast[0])
            spread = 0.0
        else:
            next_return = float(self._last_forecast[0])
            if self._last_conf_int is not None and self._last_conf_int.shape[1] >= 2:
                spread = float(self._last_conf_int[0, 1] - self._last_conf_int[0, 0])
            else:
                spread = 0.0

        confidence = float(max(0.0, 1.0 / (1.0 + abs(spread) * 100.0)))
        return {
            "next_period_return_forecast": next_return,
            "forecast_confidence": confidence,
            "forecast_spread": spread,
        }
