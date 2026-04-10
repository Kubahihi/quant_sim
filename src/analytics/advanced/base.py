from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseForecastModel(ABC):
    """Unified interface for advanced forecasting/risk models."""

    @abstractmethod
    def fit(self, data: pd.Series) -> "BaseForecastModel":
        pass

    @abstractmethod
    def predict(self, periods: int = 1) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        pass
