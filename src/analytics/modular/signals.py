from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .registry import SignalRegistry
from .results import ModelResult, SignalResult


def _to_model_dict(models: Dict[str, ModelResult]) -> Dict[str, ModelResult]:
    return {k: v for k, v in models.items() if isinstance(v, ModelResult)}


def _safe_signal(name: str, family: str, fn, models: Dict[str, ModelResult], context: Dict[str, object]) -> SignalResult:
    try:
        result = fn(models, context)
        if isinstance(result, SignalResult):
            return result
        return SignalResult(name=name, family=family, available=False, error="Invalid signal output type")
    except Exception as exc:
        return SignalResult(name=name, family=family, available=False, error=str(exc))


def _direction(score: float, threshold: float = 0.1) -> str:
    if score > threshold:
        return "long"
    if score < -threshold:
        return "short"
    return "neutral"


def _trend(models: Dict[str, ModelResult], _: Dict[str, object]) -> SignalResult:
    lr = models.get("linear_regression")
    val = float(lr.metrics.get("expected_annual_return", 0.0)) if lr and lr.available else 0.0
    score = float(np.tanh(val * 4.0))
    return SignalResult(name="trend", family="core", available=True, score=score, direction=_direction(score), confidence=abs(score), metrics={"score": score})


def _mean_reversion(models: Dict[str, ModelResult], _: Dict[str, object]) -> SignalResult:
    pair = models.get("cointegration_pairs")
    z = float(pair.metrics.get("spread_zscore", 0.0)) if pair and pair.available else 0.0
    score = float(np.tanh(-z / 2.0))
    return SignalResult(name="mean_reversion", family="core", available=True, score=score, direction=_direction(score), confidence=float(max(0.0, 1.0 - min(1.0, abs(z) / 4.0))), metrics={"score": score, "spread_zscore": z})


def _breakout(models: Dict[str, ModelResult], context: Dict[str, object]) -> SignalResult:
    series = pd.Series(context.get("portfolio_returns", pd.Series(dtype=float))).dropna()
    if len(series) < 15:
        raise ValueError("breakout signal requires at least 15 observations")
    recent = float(series.tail(5).mean())
    base = float(series.tail(20).mean())
    score = float(np.tanh((recent - base) * 80.0))
    return SignalResult(name="breakout", family="core", available=True, score=score, direction=_direction(score), confidence=abs(score), metrics={"score": score})


def _volatility(models: Dict[str, ModelResult], _: Dict[str, object]) -> SignalResult:
    garch = models.get("garch")
    vol_ann = float(garch.metrics.get("volatility_annualized", 0.0)) if garch and garch.available else 0.0
    score = float(np.tanh((0.16 - vol_ann) * 6.0))
    return SignalResult(name="volatility", family="risk", available=True, score=score, direction=_direction(score), confidence=float(max(0.0, min(1.0, 1.0 - vol_ann))), metrics={"score": score, "volatility_annualized": vol_ann})


def _regime(models: Dict[str, ModelResult], _: Dict[str, object]) -> SignalResult:
    regime = models.get("regime_probability")
    p_off = float(regime.metrics.get("risk_off_probability", 0.5)) if regime and regime.available else 0.5
    score = float((1.0 - p_off) * 2.0 - 1.0)
    return SignalResult(name="regime", family="risk", available=True, score=score, direction=_direction(score), confidence=float(abs(score)), metrics={"score": score, "risk_off_probability": p_off})


def _posterior_confidence(models: Dict[str, ModelResult], _: Dict[str, object]) -> SignalResult:
    confs = [
        float(models.get("bayesian_drift").confidence) if models.get("bayesian_drift") else 0.0,
        float(models.get("bayesian_regression").confidence) if models.get("bayesian_regression") else 0.0,
        float(models.get("bayesian_volatility").confidence) if models.get("bayesian_volatility") else 0.0,
    ]
    score = float(np.mean(confs) * 2.0 - 1.0)
    return SignalResult(name="posterior_confidence", family="bayesian", available=True, score=score, direction=_direction(score), confidence=float(np.mean(confs)), metrics={"score": score})


def _spread(models: Dict[str, ModelResult], _: Dict[str, object]) -> SignalResult:
    pair = models.get("cointegration_pairs")
    z = float(pair.metrics.get("spread_zscore", 0.0)) if pair and pair.available else 0.0
    score = float(np.tanh(-z))
    return SignalResult(name="spread", family="pairs", available=True, score=score, direction=_direction(score), confidence=float(max(0.0, 1.0 - min(1.0, abs(z) / 5.0))), metrics={"score": score, "spread_zscore": z})


def _disagreement(models: Dict[str, ModelResult], _: Dict[str, object]) -> SignalResult:
    bma = models.get("bayesian_model_averaging")
    dis = float(bma.metrics.get("disagreement", 0.0)) if bma and bma.available else 0.0
    score = float(np.tanh(-dis * 4.0))
    return SignalResult(name="disagreement", family="meta", available=True, score=score, direction=_direction(score), confidence=float(max(0.0, min(1.0, 1.0 - dis))), metrics={"score": score, "model_disagreement": dis})


def _composite_vote(models: Dict[str, ModelResult], context: Dict[str, object]) -> SignalResult:
    base = context.get("signal_results", {})
    scores = [
        float(base.get(key).score)
        for key in ["trend", "mean_reversion", "breakout", "regime", "volatility"]
        if key in base and getattr(base.get(key), "available", False)
    ]
    if not scores:
        raise ValueError("composite vote requires base signals")
    score = float(np.mean(scores))
    return SignalResult(name="composite_vote", family="meta", available=True, score=score, direction=_direction(score), confidence=float(np.mean([abs(s) for s in scores])), metrics={"score": score})


def _risk_on_off(models: Dict[str, ModelResult], context: Dict[str, object]) -> SignalResult:
    base = context.get("signal_results", {})
    regime = float(base.get("regime").score) if "regime" in base else 0.0
    vol = float(base.get("volatility").score) if "volatility" in base else 0.0
    score = float(np.clip((regime + vol) / 2.0, -1.0, 1.0))
    return SignalResult(name="risk_on_off", family="risk", available=True, score=score, direction=_direction(score), confidence=abs(score), metrics={"score": score})


def _black_litterman_tilt(models: Dict[str, ModelResult], _: Dict[str, object]) -> SignalResult:
    bl = models.get("black_litterman")
    if not bl or not bl.available:
        raise ValueError("black_litterman_tilt requires black_litterman model output")

    implied_alpha = float(bl.metrics.get("implied_alpha_portfolio", 0.0))
    tilt_strength = float(bl.metrics.get("tilt_strength", abs(implied_alpha)))
    score = float(np.tanh(implied_alpha * 6.0))
    confidence = float(max(0.0, min(1.0, bl.confidence or bl.metrics.get("confidence", 0.0))))
    return SignalResult(
        name="black_litterman_tilt",
        family="portfolio",
        available=True,
        score=score,
        direction=_direction(score),
        confidence=confidence,
        metrics={
            "score": score,
            "implied_alpha_portfolio": implied_alpha,
            "tilt_strength": tilt_strength,
        },
    )


def _sentiment_adjusted(models: Dict[str, ModelResult], context: Dict[str, object]) -> SignalResult:
    base = context.get("signal_results", {})
    composite = float(base.get("composite_vote").score) if "composite_vote" in base else 0.0
    news_sentiment = float(context.get("news_sentiment_score", 0.0) or 0.0)
    news_relevance = float(context.get("news_relevance_coverage", 0.0) or 0.0)

    score = float(np.clip(0.75 * composite + 0.25 * news_sentiment, -1.0, 1.0))
    confidence = float(np.clip(0.65 * abs(composite) + 0.35 * min(1.0, news_relevance + abs(news_sentiment) * 0.5), 0.0, 1.0))
    return SignalResult(
        name="sentiment_adjusted",
        family="sentiment",
        available=True,
        score=score,
        direction=_direction(score),
        confidence=confidence,
        metrics={
            "score": score,
            "composite_vote": composite,
            "news_sentiment_score": news_sentiment,
            "news_relevance_coverage": news_relevance,
        },
    )


def build_signal_registry() -> SignalRegistry:
    registry = SignalRegistry()
    registry.register("trend", "core", lambda m, c: _safe_signal("trend", "core", _trend, m, c))
    registry.register("mean_reversion", "core", lambda m, c: _safe_signal("mean_reversion", "core", _mean_reversion, m, c))
    registry.register("breakout", "core", lambda m, c: _safe_signal("breakout", "core", _breakout, m, c))
    registry.register("volatility", "risk", lambda m, c: _safe_signal("volatility", "risk", _volatility, m, c))
    registry.register("regime", "risk", lambda m, c: _safe_signal("regime", "risk", _regime, m, c))
    registry.register("posterior_confidence", "bayesian", lambda m, c: _safe_signal("posterior_confidence", "bayesian", _posterior_confidence, m, c))
    registry.register("spread", "pairs", lambda m, c: _safe_signal("spread", "pairs", _spread, m, c))
    registry.register("disagreement", "meta", lambda m, c: _safe_signal("disagreement", "meta", _disagreement, m, c))
    registry.register("black_litterman_tilt", "portfolio", lambda m, c: _safe_signal("black_litterman_tilt", "portfolio", _black_litterman_tilt, m, c))
    return registry


def run_signal_bundle(models: Dict[str, ModelResult], context: Dict[str, object] | None = None) -> Dict[str, SignalResult]:
    registry = build_signal_registry()
    run_context: Dict[str, object] = dict(context or {})
    model_dict = _to_model_dict(models)
    outputs: Dict[str, SignalResult] = {}

    for entry in registry.items():
        run_context["signal_results"] = outputs
        outputs[entry.name] = entry.runner(model_dict, run_context)

    outputs["composite_vote"] = _safe_signal("composite_vote", "meta", _composite_vote, model_dict, {**run_context, "signal_results": outputs})
    outputs["risk_on_off"] = _safe_signal("risk_on_off", "risk", _risk_on_off, model_dict, {**run_context, "signal_results": outputs})
    outputs["sentiment_adjusted"] = _safe_signal("sentiment_adjusted", "sentiment", _sentiment_adjusted, model_dict, {**run_context, "signal_results": outputs})
    return outputs
