from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.analytics.modular.backtest import deterministic_signal_backtest
from src.analytics.modular.history import compare_runs, load_run_record, save_run_record
from src.analytics.modular.models import run_model_bundle
from src.analytics.modular.news import (
    MissingAPIKeyError,
    NewsApiProvider,
    NewsProvider,
    RawNewsItem,
    build_news_analysis,
    build_news_rows_for_ui,
    clear_news_cache,
)
from src.analytics.modular.pipeline import run_quant_stack
from src.analytics.modular.results import RunRecord
from src.analytics.modular.signals import run_signal_bundle
from src.analytics.modular.summary import build_summary
from src.analytics.modular.results import NewsResult


class DummyNewsProvider(NewsProvider):
    provider_name = "dummy"

    def fetch(self, tickers, start_date, end_date, context=None):
        return [
            RawNewsItem(
                title=f"{tickers[0]} beats earnings in risk_on setup",
                published_at=datetime.now(timezone.utc).isoformat(),
                source="dummy",
                url="https://example.com/news/1",
                summary="Strong guidance and macro tailwinds for the selected ticker.",
            ),
            RawNewsItem(
                title=f"{tickers[0]} misses estimates amid recession risk",
                published_at=(datetime.now(timezone.utc) - timedelta(days=6)).isoformat(),
                source="dummy",
                url="https://example.com/news/2",
                summary="Weak demand and downside pressure suggest a bearish setup.",
            ),
        ]


class ContextOnlyNewsProvider(NewsProvider):
    provider_name = "context_only"

    def fetch(self, tickers, start_date, end_date, context=None):
        return [
            RawNewsItem(
                title="Market update",
                published_at=datetime.now(timezone.utc).isoformat(),
                source="context",
                url="https://example.com/news/context",
                summary="General update.",
                query_context=str(tickers[0]),
            )
        ]


class EmptyNewsProvider(NewsProvider):
    provider_name = "empty"

    def fetch(self, tickers, start_date, end_date, context=None):
        return []


class CountingNewsProvider(NewsProvider):
    provider_name = "counting"

    def __init__(self):
        self.calls = 0

    def fetch(self, tickers, start_date, end_date, context=None):
        self.calls += 1
        return [
            RawNewsItem(
                title=f"{tickers[0]} guidance update",
                published_at=datetime.now(timezone.utc).isoformat(),
                source="counting",
                url="https://example.com/news/counting",
                summary="Guidance update with earnings context.",
            )
        ]


def _sample_returns(n=180):
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0004, scale=0.01, size=n)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(data, index=idx)


def _sample_returns_df(n=180):
    base = _sample_returns(n)
    df = pd.DataFrame({
        "AAA": base,
        "BBB": base * 0.8 + 0.0001,
        "CCC": base * -0.4 + 0.0002,
    })
    return df


def test_interface_consistency_models_and_signals():
    series = _sample_returns()
    returns_df = _sample_returns_df()
    models = run_model_bundle(series, context={"returns_df": returns_df})
    signals = run_signal_bundle(
        models,
        context={"portfolio_returns": series, "returns_df": returns_df, "news_sentiment_score": 0.2},
    )

    assert "bayesian_drift" in models
    assert "linear_regression" in models
    assert "logistic_regression" in models
    assert "black_litterman" in models
    assert "composite_vote" in signals
    assert "risk_on_off" in signals
    assert "black_litterman_tilt" in signals
    assert "sentiment_adjusted" in signals

    for item in models.values():
        assert hasattr(item, "available")
        assert hasattr(item, "metrics")

    for item in signals.values():
        assert hasattr(item, "score")
        assert hasattr(item, "direction")


def test_no_lookahead_and_deterministic_backtest():
    series = _sample_returns()
    result_a = deterministic_signal_backtest(series, composite_signal=0.3, risk_signal=0.2, confidence=0.7)
    result_b = deterministic_signal_backtest(series, composite_signal=0.3, risk_signal=0.2, confidence=0.7)

    assert result_a["lookahead_safe"] is True
    assert result_b["lookahead_safe"] is True
    pd.testing.assert_series_equal(result_a["equity_curve"], result_b["equity_curve"])

    # first strategy return must be zero due to lagged position (no look-ahead)
    assert float(result_a["strategy_returns"].iloc[0]) == 0.0


def test_news_relevance_scoring_orders_items():
    now = datetime.now(timezone.utc)
    result = build_news_analysis(
        tickers=["AAA"],
        start_date=now - timedelta(days=7),
        end_date=now,
        context={
            "tickers": ["AAA"],
            "sector_keywords": ["macro", "earnings"],
            "active_models": ["bayesian_drift"],
            "active_signals": ["risk_on_off"],
            "regime_label": "risk_on",
        },
        provider=DummyNewsProvider(),
    )

    assert result.available is True
    assert len(result.items) == 2
    assert result.items[0].relevance_score >= result.items[1].relevance_score
    assert all(-1.0 <= item.sentiment_score <= 1.0 for item in result.items)
    assert result.sentiment_score == pytest.approx(
        sum(item.sentiment_score * item.relevance_score for item in result.items)
        / sum(item.relevance_score for item in result.items),
        rel=1e-6,
    )
    assert "why" not in result.items[0].summary.lower()


def test_sentiment_scoring_sign_direction():
    now = datetime.now(timezone.utc)
    result = build_news_analysis(
        tickers=["AAA"],
        start_date=now - timedelta(days=7),
        end_date=now,
        context={
            "tickers": ["AAA"],
            "sector_keywords": ["macro"],
            "active_models": [],
            "active_signals": [],
            "regime_label": "neutral",
        },
        provider=DummyNewsProvider(),
    )
    sentiments = [item.sentiment_score for item in result.items]
    assert any(score > 0 for score in sentiments)
    assert any(score < 0 for score in sentiments)


def test_no_api_key_error_is_explicit(monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    provider = NewsApiProvider()
    with pytest.raises(MissingAPIKeyError):
        provider.fetch(["AAA"], datetime.now(timezone.utc) - timedelta(days=7), datetime.now(timezone.utc), context={})


def test_empty_news_result_is_structured():
    now = datetime.now(timezone.utc)
    result = build_news_analysis(
        tickers=["AAA"],
        start_date=now - timedelta(days=7),
        end_date=now,
        context={"tickers": ["AAA"], "sector_keywords": []},
        provider=EmptyNewsProvider(),
    )
    assert result.available is True
    assert result.items == []
    assert isinstance(result.context, dict)
    assert isinstance(result.error, str)


def test_news_cache_avoids_repeat_fetch():
    clear_news_cache()
    now = datetime.now(timezone.utc)
    provider = CountingNewsProvider()

    build_news_analysis(
        tickers=["AAA"],
        start_date=now - timedelta(days=7),
        end_date=now,
        context={"tickers": ["AAA"], "sector_keywords": ["earnings"]},
        provider=provider,
    )
    build_news_analysis(
        tickers=["AAA"],
        start_date=now - timedelta(days=7),
        end_date=now,
        context={"tickers": ["AAA"], "sector_keywords": ["earnings"]},
        provider=provider,
    )
    assert provider.calls == 1


def test_news_ui_rows_builder_does_not_crash():
    now = datetime.now(timezone.utc)
    result = build_news_analysis(
        tickers=["AAA"],
        start_date=now - timedelta(days=7),
        end_date=now,
        context={"tickers": ["AAA"], "sector_keywords": ["macro"]},
        provider=DummyNewsProvider(),
    )
    rows = build_news_rows_for_ui(result)
    assert isinstance(rows, list)
    assert rows
    assert "Sentiment Color" in rows[0]


def test_relevance_uses_query_context_for_ticker_binding():
    now = datetime.now(timezone.utc)
    result = build_news_analysis(
        tickers=["AAA"],
        start_date=now - timedelta(days=7),
        end_date=now,
        context={"tickers": ["AAA"], "sector_keywords": []},
        provider=ContextOnlyNewsProvider(),
    )
    assert result.items
    assert result.items[0].relevance_score >= 0.6


def test_summary_aggregation_outputs():
    series = _sample_returns()
    returns_df = _sample_returns_df()
    models = run_model_bundle(series, context={"returns_df": returns_df})
    signals = run_signal_bundle(
        models,
        context={"portfolio_returns": series, "returns_df": returns_df, "news_sentiment_score": 0.15},
    )
    news = build_news_analysis(
        tickers=["AAA"],
        start_date=datetime.now(timezone.utc) - timedelta(days=7),
        end_date=datetime.now(timezone.utc),
        context={
            "tickers": ["AAA"],
            "sector_keywords": ["macro"],
            "active_models": ["black_litterman"],
            "active_signals": ["sentiment_adjusted"],
            "regime_label": "neutral",
        },
        provider=DummyNewsProvider(),
    )
    summary = build_summary(
        models,
        signals,
        news=news,
        backtest={"metrics": {"max_drawdown": -0.12}},
        prior_run={"summary": {"regime_label": "risk_on", "composite_score": 0.2, "confidence": 0.5, "news_sentiment": 0.1}},
    )

    assert summary.regime_label in {"risk_on", "risk_off", "neutral"}
    assert isinstance(summary.highlights, list)
    assert isinstance(summary.strongest_signals, list)
    assert isinstance(summary.recent_changes, list)
    assert isinstance(summary.news_implication, str)
    assert "model_snapshot" in summary.to_dict()


def test_persistence_round_trip(tmp_path):
    record = RunRecord.now(
        run_id="run_test_001",
        config={"foo": "bar"},
        universe=["AAA", "BBB"],
        date_range={"start": "2024-01-01", "end": "2024-12-31"},
        outputs={"models": {}},
        metrics={"sharpe": 1.2},
        summary={"regime_label": "neutral"},
        news={"items": []},
        sentiment={"news_sentiment_score": 0.0},
    )

    saved_path = save_run_record(record, base_dir=tmp_path)
    loaded = load_run_record("run_test_001", base_dir=tmp_path)

    assert saved_path.exists()
    assert loaded["run_id"] == "run_test_001"
    assert loaded["metrics"]["sharpe"] == 1.2


def test_failure_isolation_short_series():
    short = _sample_returns(n=8)
    models = run_model_bundle(short, context={"returns_df": pd.DataFrame({"AAA": short})})

    assert "linear_regression" in models
    unavailable_count = sum(1 for model in models.values() if not model.available)
    assert unavailable_count > 0


def test_black_litterman_output_consistency():
    returns_df = _sample_returns_df()
    series = returns_df.mean(axis=1)
    context = {
        "returns_df": returns_df,
        "market_weights": [0.5, 0.3, 0.2],
        "bl_views": {"AAA": 0.08, "BBB": 0.06},
    }

    run_a = run_model_bundle(series, context=context)
    run_b = run_model_bundle(series, context=context)

    bl_a = run_a["black_litterman"]
    bl_b = run_b["black_litterman"]
    assert bl_a.available is True
    assert bl_b.available is True
    assert bl_a.metrics["posterior_expected_annual_return"] == pytest.approx(bl_b.metrics["posterior_expected_annual_return"])
    assert bl_a.metrics["implied_alpha_portfolio"] == pytest.approx(bl_b.metrics["implied_alpha_portfolio"])


def test_compare_runs_metric_diff():
    left = {
        "run_id": "l",
        "metrics": {"sharpe": 1.0, "volatility": 0.2},
        "summary": {"regime_label": "neutral", "composite_score": 0.1, "confidence": 0.5, "news_sentiment": 0.0},
    }
    right = {
        "run_id": "r",
        "metrics": {"sharpe": 1.3, "volatility": 0.25},
        "summary": {"regime_label": "risk_on", "composite_score": 0.2, "confidence": 0.6, "news_sentiment": 0.1},
    }

    diff = compare_runs(left, right)
    assert diff["metric_diff"]["sharpe"] == pytest.approx(0.3)
    assert diff["metric_diff"]["volatility"] == pytest.approx(0.05)
    assert diff["summary_diff"]["composite_score"] == pytest.approx(0.1)
    assert diff["summary_diff"]["news_sentiment"] == pytest.approx(0.1)


def test_run_quant_stack_preserves_portfolio_metrics_and_namespaces_backtest(tmp_path, monkeypatch):
    returns_df = _sample_returns_df()
    portfolio_returns = returns_df.mean(axis=1)
    seen_context: dict[str, object] = {}

    monkeypatch.setattr(
        "src.analytics.modular.pipeline.build_news_analysis",
        lambda *args, **kwargs: (
            seen_context.update(kwargs.get("context", {}))
            or NewsResult(
                available=True,
                items=[],
                context={"relevance_coverage": 0.0, "provider_used": "test"},
                sentiment_score=0.0,
                sentiment_dispersion=0.0,
            )
        ),
    )

    result = run_quant_stack(
        portfolio_returns=portfolio_returns,
        returns_df=returns_df,
        config={
            "tickers": ["AAA", "BBB", "CCC"],
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "start_date": portfolio_returns.index.min().date(),
            "end_date": portfolio_returns.index.max().date(),
            "portfolio_metrics": {
                "volatility": 0.123,
                "total_return": 0.456,
                "max_drawdown": -0.111,
                "sharpe_ratio": 1.234,
            },
            "news_api_key": "news-key-from-config",
        },
        history_dir=tmp_path,
    )

    saved = load_run_record(result["run_record"].run_id, base_dir=tmp_path)
    assert saved["metrics"]["volatility"] == pytest.approx(0.123)
    assert saved["metrics"]["total_return"] == pytest.approx(0.456)
    assert saved["metrics"]["max_drawdown"] == pytest.approx(-0.111)
    assert saved["metrics"]["sharpe_ratio"] == pytest.approx(1.234)
    assert "backtest_volatility" in saved["metrics"]
    assert "backtest_total_return" in saved["metrics"]
    assert "backtest_max_drawdown" in saved["metrics"]
    assert "backtest_sharpe" in saved["metrics"]
    assert seen_context["news_api_key"] == "news-key-from-config"
