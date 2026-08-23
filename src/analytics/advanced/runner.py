from __future__ import annotations

from typing import Any, Dict, Mapping

import pandas as pd

from ..modular import ModelResult, run_model_bundle


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


def format_advanced_model_outputs(
    outputs: Mapping[str, ModelResult],
) -> Dict[str, Dict[str, Any]]:
    """Adapt an already calculated model bundle to the legacy UI contract."""
    return {name: _to_legacy_output(name, result) for name, result in outputs.items()}


def run_advanced_models_with_bundle(
    returns: pd.Series,
    forecast_periods: int = 5,
    returns_df: pd.DataFrame | None = None,
    model_context: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, ModelResult]]:
    """Calculate models once and return both modern and legacy representations."""
    clean_returns = pd.Series(returns).dropna().astype(float)
    context = dict(model_context or {})
    context.update(
        {
            "forecast_periods": int(forecast_periods),
            "returns_df": returns_df
            if returns_df is not None
            else pd.DataFrame({"portfolio": clean_returns}),
        }
    )
    outputs = run_model_bundle(clean_returns, context=context)
    return format_advanced_model_outputs(outputs), outputs


def run_advanced_models(
    returns: pd.Series,
    forecast_periods: int = 5,
    returns_df: pd.DataFrame | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Run modular model layer with legacy-compatible dictionary outputs."""
    legacy_outputs, _ = run_advanced_models_with_bundle(
        returns,
        forecast_periods=forecast_periods,
        returns_df=returns_df,
    )
    return legacy_outputs
