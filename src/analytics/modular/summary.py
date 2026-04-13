from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

from .results import ModelResult, NewsResult, SignalResult, SummaryResult


def _risk_flags(models: Dict[str, ModelResult], signals: Dict[str, SignalResult]) -> list[str]:
    flags: list[str] = []
    garch = models.get("garch")
    if garch and garch.available and float(garch.metrics.get("volatility_annualized", 0.0)) > 0.28:
        flags.append("Forward volatility is elevated.")

    regime = signals.get("regime")
    if regime and regime.available and float(regime.metrics.get("risk_off_probability", 0.0)) > 0.65:
        flags.append("Regime model suggests risk-off conditions.")

    disagreement = signals.get("disagreement")
    if disagreement and disagreement.available and float(disagreement.metrics.get("model_disagreement", 0.0)) > 0.08:
        flags.append("Model disagreement is high.")

    return flags


def _expected_return_view(models: Dict[str, ModelResult]) -> float:
    candidates: List[float] = []
    for model in models.values():
        if not model.available:
            continue
        metrics = model.metrics
        if "bma_expected_annual_return" in metrics:
            candidates.append(float(metrics["bma_expected_annual_return"]))
        elif "posterior_expected_annual_return" in metrics:
            candidates.append(float(metrics["posterior_expected_annual_return"]))
        elif "posterior_annual_return" in metrics:
            candidates.append(float(metrics["posterior_annual_return"]))
        elif "expected_annual_return" in metrics:
            candidates.append(float(metrics["expected_annual_return"]))
        elif "next_period_return_forecast" in metrics:
            candidates.append(float(metrics["next_period_return_forecast"]) * 252.0)
    return float(np.mean(candidates)) if candidates else 0.0


def _expected_risk_view(models: Dict[str, ModelResult]) -> float:
    vol_candidates: List[float] = []
    for key in ("garch", "ewma", "bayesian_volatility"):
        model = models.get(key)
        if model and model.available:
            vol_candidates.append(
                float(
                    model.metrics.get(
                        "volatility_annualized",
                        model.metrics.get("posterior_volatility_annualized", 0.0),
                    )
                )
            )
    return float(np.mean(vol_candidates)) if vol_candidates else 0.0


def _regime_interpretation(regime_label: str, agreement_score: float) -> str:
    if regime_label == "risk_on":
        return f"Risk-on interpretation with agreement score {agreement_score:.2f}."
    if regime_label == "risk_off":
        return f"Risk-off interpretation with agreement score {agreement_score:.2f}."
    return f"Neutral regime with agreement score {agreement_score:.2f}."


def _drawdown_implication(backtest: Dict[str, Any] | None) -> str:
    if not backtest:
        return "Drawdown implication unavailable."
    max_dd = float(backtest.get("metrics", {}).get("max_drawdown", 0.0))
    if max_dd < -0.20:
        return "Backtest drawdown profile is deep and may require tighter risk controls."
    if max_dd < -0.10:
        return "Backtest drawdown profile is moderate."
    return "Backtest drawdown profile is contained."


def _volatility_implication(expected_risk_view: float) -> str:
    if expected_risk_view > 0.30:
        return "Expected volatility is high."
    if expected_risk_view > 0.18:
        return "Expected volatility is moderate."
    return "Expected volatility is relatively low."


def _strongest_signals(signals: Dict[str, SignalResult], top_n: int = 3) -> List[Dict[str, Any]]:
    candidates = [signal for signal in signals.values() if signal.available]
    ranked = sorted(candidates, key=lambda item: abs(float(item.score)), reverse=True)[:top_n]
    return [
        {
            "name": signal.name,
            "score": float(signal.score),
            "direction": signal.direction,
            "confidence": float(signal.confidence),
        }
        for signal in ranked
    ]


def _news_implication(news: NewsResult | None, composite_score: float) -> tuple[float, float, List[Dict[str, Any]], str]:
    if news is None or not news.available:
        return 0.0, 0.0, [], "News layer unavailable."
    score = float(news.sentiment_score)
    dispersion = float(news.sentiment_dispersion)
    top_news = [
        {
            "title": item.title,
            "source": item.source,
            "relevance": float(item.relevance_score),
            "sentiment": float(item.sentiment_score),
            "label": item.sentiment_label,
        }
        for item in sorted(news.items, key=lambda item: item.relevance_score, reverse=True)[:3]
    ]
    if not news.items:
        return score, dispersion, top_news, "No relevant news items; sentiment signal is weak."
    if score > 0.2 and abs(composite_score) < 0.1:
        return score, dispersion, top_news, "Positive news sentiment but trading signals remain weak."
    if score < -0.2 and composite_score > 0.1:
        return score, dispersion, top_news, "Negative news sentiment conflicts with positive signals; conviction is reduced."
    if score > 0.2:
        return score, dispersion, top_news, "Relevant news flow is net positive and supports risk-taking."
    if score < -0.2:
        return score, dispersion, top_news, "Relevant news flow is net negative and supports caution."
    return score, dispersion, top_news, "Relevant news flow is mixed/neutral."


def _recent_changes(
    prior_run: Dict[str, Any] | None,
    composite_score: float,
    regime_label: str,
    confidence: float,
    news_sentiment: float,
) -> List[str]:
    if not prior_run:
        return ["No prior run available for change detection."]

    previous_summary = prior_run.get("summary", {})
    changes: List[str] = []

    prev_regime = str(previous_summary.get("regime_label", ""))
    if prev_regime and prev_regime != regime_label:
        changes.append(f"Regime changed from {prev_regime} to {regime_label}.")

    prev_comp = float(previous_summary.get("composite_score", 0.0) or 0.0)
    comp_delta = composite_score - prev_comp
    changes.append(f"Composite score delta vs prior run: {comp_delta:+.3f}.")

    prev_conf = float(previous_summary.get("confidence", 0.0) or 0.0)
    conf_delta = confidence - prev_conf
    changes.append(f"Confidence delta vs prior run: {conf_delta:+.3f}.")

    prev_news = float(previous_summary.get("news_sentiment", 0.0) or 0.0)
    news_delta = news_sentiment - prev_news
    changes.append(f"News sentiment delta vs prior run: {news_delta:+.3f}.")
    return changes


def build_summary(
    models: Dict[str, ModelResult],
    signals: Dict[str, SignalResult],
    news: NewsResult | None = None,
    backtest: Dict[str, Any] | None = None,
    prior_run: Dict[str, Any] | None = None,
) -> SummaryResult:
    model_conf = [m.confidence for m in models.values() if m.available]
    signal_scores = [float(s.score) for s in signals.values() if s.available]

    composite = float(np.mean(signal_scores)) if signal_scores else 0.0
    confidence = float(np.mean(model_conf)) if model_conf else 0.0
    uncertainty = float(max(0.0, min(1.0, 1.0 - confidence)))

    agreement_score = float(max(0.0, min(1.0, 1.0 - np.std(signal_scores)))) if signal_scores else 0.0
    disagreement_score = float(max(0.0, min(1.0, 1.0 - agreement_score)))

    if composite > 0.2:
        regime_label = "risk_on"
    elif composite < -0.2:
        regime_label = "risk_off"
    else:
        regime_label = "neutral"

    expected_return = _expected_return_view(models)
    expected_risk = _expected_risk_view(models)
    strongest = _strongest_signals(signals)
    news_sentiment, news_dispersion, top_news, news_implication = _news_implication(news, composite)
    recent_changes = _recent_changes(prior_run, composite, regime_label, confidence, news_sentiment)

    highlights = [
        f"Active models: {sum(1 for item in models.values() if item.available)} / {len(models)}",
        f"Active signals: {sum(1 for item in signals.values() if item.available)} / {len(signals)}",
        f"Agreement vs disagreement: {agreement_score:.3f} / {disagreement_score:.3f}",
        f"Expected return view (annualized): {expected_return:.2%}",
        f"Expected risk view (annualized volatility): {expected_risk:.2%}",
        f"Uncertainty level: {uncertainty:.3f}",
        f"News sentiment mean/dispersion: {news_sentiment:+.3f} / {news_dispersion:.3f}",
    ]

    warnings: List[str] = []
    if abs(composite) < 0.08:
        warnings.append("Signals are weak: composite score is close to neutral.")
    if agreement_score < 0.45:
        warnings.append("Signals are unstable: agreement score is low.")
    if confidence < 0.35:
        warnings.append("Model confidence is low; treat outputs cautiously.")

    risk_flags = [*dict.fromkeys([*_risk_flags(models, signals), *warnings])]

    model_snapshot = {
        key: {
            "family": value.family,
            "available": value.available,
            "confidence": value.confidence,
            "metrics": value.metrics,
            "error": value.error,
        }
        for key, value in models.items()
    }
    signal_snapshot = {
        key: {
            "family": value.family,
            "available": value.available,
            "score": value.score,
            "direction": value.direction,
            "confidence": value.confidence,
            "metrics": value.metrics,
            "error": value.error,
        }
        for key, value in signals.items()
    }

    return SummaryResult(
        generated_at=datetime.now(timezone.utc).isoformat(),
        composite_score=composite,
        regime_label=regime_label,
        confidence=confidence,
        highlights=highlights,
        model_snapshot=model_snapshot,
        signal_snapshot=signal_snapshot,
        risk_flags=risk_flags,
        strongest_signals=strongest,
        agreement_score=agreement_score,
        disagreement_score=disagreement_score,
        uncertainty=uncertainty,
        expected_return_view=expected_return,
        expected_risk_view=expected_risk,
        regime_interpretation=_regime_interpretation(regime_label, agreement_score),
        drawdown_implication=_drawdown_implication(backtest),
        volatility_implication=_volatility_implication(expected_risk),
        recent_changes=recent_changes,
        news_sentiment=news_sentiment,
        news_sentiment_dispersion=news_dispersion,
        top_relevant_news=top_news,
        news_implication=news_implication,
        warnings=warnings,
    )
