from __future__ import annotations

from typing import Dict, List

import pandas as pd


def _profile_thresholds(risk_profile: str) -> Dict[str, float]:
    profile = (risk_profile or "balanced").lower()
    if profile == "conservative":
        return {
            "max_weight": 0.28,
            "volatility": 0.16,
            "min_sharpe": 0.80,
            "max_drawdown_abs": 0.20,
            "avg_corr": 0.60,
            "min_annual_return": 0.04,
        }
    if profile == "aggressive":
        return {
            "max_weight": 0.42,
            "volatility": 0.28,
            "min_sharpe": 0.35,
            "max_drawdown_abs": 0.38,
            "avg_corr": 0.78,
            "min_annual_return": 0.08,
        }

    return {
        "max_weight": 0.35,
        "volatility": 0.22,
        "min_sharpe": 0.55,
        "max_drawdown_abs": 0.30,
        "avg_corr": 0.70,
        "min_annual_return": 0.06,
    }


def _rating_from_score(score: int) -> str:
    if score >= 80:
        return "Strong"
    if score >= 65:
        return "Good"
    if score >= 45:
        return "Moderate"
    return "Weak"


def evaluate_portfolio_score(
    metrics: Dict[str, float],
    concentration: Dict[str, float],
    avg_correlation: float,
    n_assets: int,
    risk_profile: str = "balanced",
    model_signals: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Build deterministic score and flags from portfolio metrics."""
    thresholds = _profile_thresholds(risk_profile)
    flags: List[str] = []
    breakdown: List[Dict[str, object]] = []
    score = 100.0

    annualized_return = metrics.get("annualized_return", 0.0)
    if annualized_return < thresholds["min_annual_return"]:
        penalty = min(18.0, (thresholds["min_annual_return"] - annualized_return) * 120)
        score -= penalty
        flags.append("Low expected return for the selected risk profile.")
        breakdown.append({
            "rule": "return_quality",
            "penalty": round(penalty, 2),
            "detail": f"Annualized return {annualized_return:.1%}",
        })

    max_weight = concentration.get("max_weight", 0.0)
    if max_weight > thresholds["max_weight"]:
        penalty = min(20.0, (max_weight - thresholds["max_weight"]) * 120)
        score -= penalty
        flags.append("High concentration in a single position.")
        breakdown.append({
            "rule": "concentration",
            "penalty": round(penalty, 2),
            "detail": f"Largest weight {max_weight:.1%}",
        })

    effective_holdings = concentration.get("effective_holdings", 0.0)
    min_effective = max(3.0, min(8.0, n_assets * 0.60))
    if effective_holdings < min_effective:
        penalty = min(16.0, (min_effective - effective_holdings) * 2.8)
        score -= penalty
        flags.append("Weak diversification across holdings.")
        breakdown.append({
            "rule": "diversification",
            "penalty": round(penalty, 2),
            "detail": f"Effective holdings {effective_holdings:.2f}",
        })

    volatility = metrics.get("volatility", 0.0)
    if volatility > thresholds["volatility"]:
        penalty = min(18.0, (volatility - thresholds["volatility"]) * 90)
        score -= penalty
        flags.append("High portfolio volatility.")
        breakdown.append({
            "rule": "volatility",
            "penalty": round(penalty, 2),
            "detail": f"Volatility {volatility:.1%}",
        })

    sharpe = metrics.get("sharpe_ratio", 0.0)
    if sharpe < thresholds["min_sharpe"]:
        penalty = min(20.0, (thresholds["min_sharpe"] - sharpe) * 18)
        score -= penalty
        flags.append("Low Sharpe ratio.")
        breakdown.append({
            "rule": "sharpe",
            "penalty": round(penalty, 2),
            "detail": f"Sharpe {sharpe:.2f}",
        })

    max_drawdown_abs = abs(metrics.get("max_drawdown", 0.0))
    if max_drawdown_abs > thresholds["max_drawdown_abs"]:
        penalty = min(18.0, (max_drawdown_abs - thresholds["max_drawdown_abs"]) * 70)
        score -= penalty
        flags.append("Historically deep drawdowns.")
        breakdown.append({
            "rule": "drawdown",
            "penalty": round(penalty, 2),
            "detail": f"Max drawdown {metrics.get('max_drawdown', 0.0):.1%}",
        })

    if avg_correlation > thresholds["avg_corr"]:
        penalty = min(14.0, (avg_correlation - thresholds["avg_corr"]) * 40)
        score -= penalty
        flags.append("High correlation across holdings.")
        breakdown.append({
            "rule": "correlation",
            "penalty": round(penalty, 2),
            "detail": f"Average correlation {avg_correlation:.2f}",
        })

    # Optional V0.3/V0.4 model layer integration (heuristic, non-breaking).
    signals = model_signals or {}
    lr_annual = float(signals.get("lr_expected_annual_return", 0.0))
    arima_next = float(signals.get("arima_next_return", 0.0))
    garch_ann_vol = float(signals.get("garch_annualized_volatility", 0.0))

    if lr_annual < 0:
        penalty = min(10.0, abs(lr_annual) * 60.0)
        score -= penalty
        flags.append("Linear trend model indicates negative expected annual return.")
        breakdown.append({
            "rule": "lr_trend",
            "penalty": round(penalty, 2),
            "detail": f"LR expected annual return {lr_annual:.1%}",
        })

    if arima_next < 0:
        penalty = min(8.0, abs(arima_next) * 400.0)
        score -= penalty
        flags.append("ARIMA short-term forecast is negative.")
        breakdown.append({
            "rule": "arima_forecast",
            "penalty": round(penalty, 2),
            "detail": f"ARIMA next return {arima_next:.3%}",
        })

    base_vol = float(metrics.get("volatility", 0.0))
    if base_vol > 0 and garch_ann_vol > base_vol * 1.15:
        penalty = min(10.0, (garch_ann_vol - base_vol) * 40.0)
        score -= penalty
        flags.append("GARCH indicates elevated forward volatility.")
        breakdown.append({
            "rule": "garch_volatility",
            "penalty": round(penalty, 2),
            "detail": f"GARCH annualized vol {garch_ann_vol:.1%}",
        })

    if lr_annual > 0.08 and arima_next > 0 and (base_vol <= 0 or garch_ann_vol <= base_vol):
        bonus = min(6.0, lr_annual * 20.0)
        score += bonus
        breakdown.append({
            "rule": "model_alignment_bonus",
            "penalty": round(-bonus, 2),
            "detail": "LR/ARIMA positive with stable forward volatility",
        })

    final_score = int(max(0, min(100, round(score))))
    rating = _rating_from_score(final_score)

    return {
        "score": final_score,
        "rating": rating,
        "flags": flags,
        "breakdown": breakdown,
    }


def build_deterministic_fallback_review(
    score_result: Dict[str, object],
    metrics: Dict[str, float],
) -> Dict[str, object]:
    """Return fallback review text when AI layer is unavailable."""
    flags = score_result.get("flags", [])
    rating = score_result.get("rating", "Moderate")
    score = score_result.get("score", 0)

    if flags:
        risk_text = " ".join(f"- {flag}" for flag in flags[:4])
    else:
        risk_text = "- No critical deterministic warnings."

    improvements = []
    if metrics.get("annualized_return", 0.0) < 0.06:
        improvements.append("Increase expected return quality or rebalance toward stronger assets.")
    if metrics.get("volatility", 0.0) > 0.22:
        improvements.append("Reduce volatility by adding more stable holdings.")
    if metrics.get("sharpe_ratio", 0.0) < 0.55:
        improvements.append("Improve the return-to-risk mix through reweighting.")
    if abs(metrics.get("max_drawdown", 0.0)) > 0.30:
        improvements.append("Add defensive exposure or lower concentration.")
    if not improvements:
        improvements.append("Maintain periodic rebalancing and monitor correlations.")

    return {
        "source": "deterministic_fallback",
        "summary": (
            f"The portfolio scores {score}/100 ({rating}). "
            f"Annualized return is {metrics.get('annualized_return', 0.0):.1%} "
            f"with volatility at {metrics.get('volatility', 0.0):.1%}."
        ),
        "risks": risk_text,
        "improvements": " ".join(f"- {item}" for item in improvements[:4]),
        "verdict": (
            "The portfolio is usable, but it still needs optimization."
            if score < 80
            else "The portfolio looks robust against standard market scenarios."
        ),
    }


def compute_weighted_factor_score(
    factors: Dict[str, pd.Series],
    weights: Dict[str, float],
    neutral_value: float = 0.5,
) -> pd.Series:
    """
    Compute a weighted score from named factor series.

    Each factor is expected to already be normalized into [0, 1] range.
    Missing values are filled with `neutral_value` to keep scoring robust.
    """
    if not factors:
        return pd.Series(dtype=float)

    non_empty_series = [series for series in factors.values() if isinstance(series, pd.Series)]
    if not non_empty_series:
        return pd.Series(dtype=float)

    reference_index = non_empty_series[0].index
    score = pd.Series(0.0, index=reference_index, dtype=float)

    valid_weights = {key: max(0.0, float(value)) for key, value in weights.items()}
    weight_sum = float(sum(valid_weights.values()))
    if weight_sum <= 0:
        valid_weights = {key: 1.0 for key in factors}
        weight_sum = float(len(valid_weights))

    for factor_name, factor_series in factors.items():
        if not isinstance(factor_series, pd.Series):
            continue
        aligned = factor_series.reindex(reference_index).fillna(neutral_value)
        factor_weight = valid_weights.get(factor_name, 0.0) / weight_sum
        score = score + aligned * factor_weight

    return score
