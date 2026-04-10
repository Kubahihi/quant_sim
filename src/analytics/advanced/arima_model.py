from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseForecastModel

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:  # pragma: no cover - optional dependency
    ARIMA = None  # type: ignore


class ARIMAModel(BaseForecastModel):
    """ARIMA model for short-horizon return forecasting."""

    def __init__(self, order: Tuple[int, int, int] = (1, 0, 1)) -> None:
        self.order = order
        self._result: Optional[Any] = None
        self._is_fitted = False
        self._last_forecast: Optional[np.ndarray] = None
        self._last_conf_int: Optional[np.ndarray] = None

    def fit(self, data: pd.Series) -> "ARIMAModel":
        if ARIMA is None:
            raise ImportError("statsmodels is required for ARIMAModel.")

        clean = pd.Series(data).dropna().astype(float)
        if len(clean) < 30:
            raise ValueError("ARIMAModel requires at least 30 observations.")

        model = ARIMA(clean, order=self.order)
        self._result = model.fit()
        self._is_fitted = True
        return self

    def predict(self, periods: int = 1) -> Dict[str, Any]:
        if not self._is_fitted or self._result is None:
            raise ValueError("Model must be fitted before prediction.")

        periods = max(1, int(periods))
        forecast_res = self._result.get_forecast(steps=periods)
        mean_forecast = np.asarray(forecast_res.predicted_mean, dtype=float)
        conf_int = np.asarray(forecast_res.conf_int(alpha=0.05), dtype=float)

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
            forecast = np.asarray(self._result.forecast(steps=1), dtype=float)
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
