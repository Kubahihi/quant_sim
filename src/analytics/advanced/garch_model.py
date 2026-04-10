from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseForecastModel

try:
    from arch import arch_model
except Exception:  # pragma: no cover - optional dependency
    arch_model = None  # type: ignore


class GARCHModel(BaseForecastModel):
    """GARCH(1,1) volatility model for conditional risk estimates."""

    def __init__(self, p: int = 1, q: int = 1) -> None:
        self.p = p
        self.q = q
        self._result: Optional[Any] = None
        self._is_fitted = False
        self._last_conditional_vol = 0.0

    def fit(self, data: pd.Series) -> "GARCHModel":
        if arch_model is None:
            raise ImportError("arch package is required for GARCHModel.")

        clean = pd.Series(data).dropna().astype(float)
        if len(clean) < 50:
            raise ValueError("GARCHModel requires at least 50 observations.")

        # arch expects returns in percent, not decimal.
        scaled_returns = clean * 100.0
        model = arch_model(
            scaled_returns,
            mean="Zero",
            vol="GARCH",
            p=self.p,
            q=self.q,
            rescale=False,
        )
        self._result = model.fit(disp="off")
        self._is_fitted = True
        return self

    def predict(self, periods: int = 1) -> Dict[str, Any]:
        if not self._is_fitted or self._result is None:
            raise ValueError("Model must be fitted before prediction.")

        periods = max(1, int(periods))
        forecast = self._result.forecast(horizon=periods, reindex=False)
        variance_path = forecast.variance.values[-1]
        vol_path = np.sqrt(np.maximum(variance_path, 0.0)) / 100.0

        self._last_conditional_vol = float(vol_path[0])
        return {
            "next_volatility": float(vol_path[0]),
            "volatility_path": vol_path.tolist(),
        }

    def get_metrics(self) -> Dict[str, float]:
        if not self._is_fitted:
            return {}

        return {
            "conditional_volatility": float(self._last_conditional_vol),
            "volatility_annualized": float(self._last_conditional_vol * np.sqrt(252.0)),
            "confidence": float(max(0.0, 1.0 - min(0.9, self._last_conditional_vol * 10.0))),
        }
