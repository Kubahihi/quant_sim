from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..modular import run_model_bundle


def _to_legacy_output(name: str, result: Any) -> Dict[str, Any]:
    metrics = dict(getattr(result, "metrics", {}) or {})
    prediction: Dict[str, Any] = {}

    if "expected_daily_return" in metrics:
        prediction["next_return"] = float(metrics["expected_daily_return"])
    elif "next_period_return_forecast" in metrics:
        prediction["next_return"] = float(metrics["next_period_return_forecast"])
    elif "conditional_volatility" in metrics:
        prediction["next_volatility"] = float(metrics["conditional_volatility"])

    return {
        "available": bool(getattr(result, "available", False)),
        "prediction": prediction,
        "metrics": metrics,
        "error": str(getattr(result, "error", "") or ""),
        "family": getattr(result, "family", "unknown"),
        "name": name,
    }


def run_advanced_models(
    returns: pd.Series,
    forecast_periods: int = 5,
    returns_df: pd.DataFrame | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Run modular model layer with legacy-compatible dictionary outputs."""
    clean_returns = pd.Series(returns).dropna().astype(float)
    context = {
        "forecast_periods": int(forecast_periods),
        "returns_df": returns_df if returns_df is not None else pd.DataFrame({"portfolio": clean_returns}),
    }

    outputs = run_model_bundle(clean_returns, context=context)
    return {name: _to_legacy_output(name, result) for name, result in outputs.items()}
