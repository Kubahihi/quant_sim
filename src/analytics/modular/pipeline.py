from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

import pandas as pd

from .backtest import deterministic_signal_backtest
from .history import list_run_records, save_run_record
from .models import run_model_bundle
from .news import build_news_analysis
from .results import RunRecord
from .signals import run_signal_bundle
from .summary import build_summary


def run_quant_stack(
    portfolio_returns: pd.Series,
    returns_df: pd.DataFrame,
    config: Dict[str, Any],
    history_dir: str = "data/run_history",
) -> Dict[str, Any]:
    context = {
        "returns_df": returns_df,
        "portfolio_returns": portfolio_returns,
        "market_weights": config.get("weights", []),
    }

    models = run_model_bundle(portfolio_returns, context=context)
    preliminary_signals = run_signal_bundle(models, context=context)
    pre_composite = float(preliminary_signals.get("composite_vote").score if preliminary_signals.get("composite_vote") else 0.0)
    pre_risk = float(preliminary_signals.get("risk_on_off").score if preliminary_signals.get("risk_on_off") else 0.0)
    model_conf = [model.confidence for model in models.values() if model.available]

    backtest = deterministic_signal_backtest(
        portfolio_returns=portfolio_returns,
        composite_signal=pre_composite,
        risk_signal=pre_risk,
        confidence=float(pd.Series(model_conf, dtype=float).mean()) if model_conf else 0.0,
    )

    start_date = config.get("start_date")
    end_date = config.get("end_date")
    start_dt = datetime.combine(start_date, datetime.min.time()) if hasattr(start_date, "year") else datetime.utcnow()
    end_dt = datetime.combine(end_date, datetime.min.time()) if hasattr(end_date, "year") else datetime.utcnow()

    news_context = {
        "tickers": config.get("tickers", []),
        "sector_keywords": config.get("sector_keywords", []),
        "news_api_key": config.get("news_api_key", ""),
        "regime_label": "risk_on" if pre_risk > 0.2 else "risk_off" if pre_risk < -0.2 else "neutral",
        "active_models": [name for name, model in models.items() if model.available],
        "active_signals": [name for name, signal in preliminary_signals.items() if signal.available],
    }
    news = build_news_analysis(
        tickers=config.get("tickers", []),
        start_date=start_dt,
        end_date=end_dt,
        context=news_context,
        max_items=int(config.get("news_max_items", 120)),
    )
    signals = run_signal_bundle(
        models,
        context={
            **context,
            "news_sentiment_score": float(news.sentiment_score),
            "news_relevance_coverage": float(news.context.get("relevance_coverage", 0.0)),
        },
    )
    prior_run = next(iter(list_run_records(base_dir=history_dir, limit=1)), None)
    summary = build_summary(models, signals, news=news, backtest=backtest, prior_run=prior_run)

    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    backtest_metrics = {
        f"backtest_{key}": value for key, value in backtest.get("metrics", {}).items()
    }
    metrics = {
        **config.get("portfolio_metrics", {}),
        **backtest_metrics,
        "composite_signal": summary.composite_score,
        "summary_confidence": summary.confidence,
        "news_sentiment_score": float(news.sentiment_score),
        "news_sentiment_dispersion": float(news.sentiment_dispersion),
    }

    record = RunRecord.now(
        run_id=run_id,
        config=config,
        universe=config.get("tickers", []),
        date_range={
            "start": getattr(start_date, "isoformat", lambda: str(start_date))(),
            "end": getattr(end_date, "isoformat", lambda: str(end_date))(),
        },
        outputs={
            "models": {name: item.to_dict() for name, item in models.items()},
            "signals": {name: item.to_dict() for name, item in signals.items()},
            "backtest": {
                "metrics": backtest.get("metrics", {}),
                "lookahead_safe": backtest.get("lookahead_safe", True),
            },
        },
        metrics=metrics,
        summary=summary.to_dict(),
        news=news.to_dict(),
        sentiment={
            "news_sentiment_score": float(news.sentiment_score),
            "news_sentiment_dispersion": float(news.sentiment_dispersion),
            "relevance_coverage": float(news.context.get("relevance_coverage", 0.0)),
            "provider_used": str(news.context.get("provider_used", "")),
            "fetch_errors": list(news.context.get("fetch_errors", [])),
        },
    )
    history_path = save_run_record(record, base_dir=history_dir)

    return {
        "models": models,
        "signals": signals,
        "summary": summary,
        "news": news,
        "backtest": backtest,
        "run_record": record,
        "history_path": str(history_path),
    }
