"""Synthetic, offline regression cases from the 2026-09-05 audit."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import importlib
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analytics.correlation import calculate_beta
from src.analytics.dcf import calculate_multistage_dcf, prepare_dcf_inputs
from src.analytics.modular.history import list_run_records, load_run_record, save_run_record
from src.analytics.modular.results import RunRecord
from src.analytics.returns import calculate_annualized_return, calculate_cumulative_returns
from src.analytics.risk_metrics import calculate_cvar, calculate_sharpe_ratio
from src.analytics.tail_risk import empirical_expected_shortfall
from src.auth.wharton_sessions import ABSOLUTE_SESSION_SECONDS, IDLE_SESSION_SECONDS, issue_session, session_is_current
from src.data.fetchers.yahoo_fetcher import FetchResult, YahooFetcher
from src.data.price_alignment import align_daily_prices, normalize_daily_series
from src.optimization.engine import optimize_portfolio
from src.optimization.estimators import estimate_portfolio_inputs
from src.optimization.execution import build_execution_plan, estimate_trade_costs
from src.utils.rates import annual_effective_to_arithmetic


def test_discrete_cvar_matches_fractional_tail_mass_across_metrics_and_optimizer():
    returns = pd.Series([-0.2] + [0.0] * 20)
    expected = 0.2 / (21 * 0.05)
    assert calculate_cvar(returns) == pytest.approx(expected)
    engine = optimize_portfolio(returns.to_frame("A"), objective="minimum_cvar", max_weight=1)
    assert engine["success"]
    assert engine["historical_cvar_daily"] == pytest.approx(expected)
    validation = importlib.import_module("src.analytics.model_validation")
    assert validation._point_metrics(returns.to_numpy(), 0)["cvar_95"] == pytest.approx(expected)


@pytest.mark.parametrize("count,confidence", [(21, .95), (9, .8), (100, .95), (3, .99)])
def test_expected_shortfall_matches_sorted_fractional_reference(count, confidence):
    values = np.random.default_rng(29).integers(-10, 10, size=count).astype(float)
    sorted_losses = np.sort(values)[::-1]
    mass = count * (1 - confidence)
    full = int(np.floor(mass))
    expected = (sorted_losses[:full].sum() + (mass - full) * sorted_losses[full]) / mass
    assert empirical_expected_shortfall(values, confidence) == pytest.approx(expected)
    matrix = np.vstack([values, values + 10])
    np.testing.assert_allclose(empirical_expected_shortfall(matrix, confidence, axis=1), [expected, expected + 10])


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf]])
def test_expected_shortfall_rejects_invalid_samples(bad):
    with pytest.raises(ValueError):
        empirical_expected_shortfall(bad)


def test_complete_loss_is_a_valid_simple_return():
    returns = pd.Series([.1, -1., .2])
    np.testing.assert_allclose(calculate_cumulative_returns(returns), [.1, -1., -1.])
    assert calculate_annualized_return(returns) == -1
    estimates = estimate_portfolio_inputs(returns.to_frame("A"))
    assert estimates.returns.iloc[1, 0] == -1
    validation = importlib.import_module("src.analytics.model_validation")
    assert validation._point_metrics(returns.to_numpy(), 0)["annualized_return"] == -1


def test_beta_uses_identical_overlapping_observations():
    market = pd.Series([-.5, .01, .02, .03])
    assert calculate_beta(2 * market.iloc[1:], market) == pytest.approx(2)


def test_daily_price_join_preserves_local_exchange_session_dates(monkeypatch):
    levels = 100 * np.cumprod(1 + np.tile([.01, -.009], 20))
    fixtures = [(symbol, FetchResult(pd.DataFrame({"close": levels}, index=pd.date_range("2025-03-01", periods=40, freq="B", tz=tz)), True))
                for symbol, tz in [("US", "America/New_York"), ("UK", "Europe/London")]]
    monkeypatch.setattr(YahooFetcher, "_fetch_many", lambda *args, **kwargs: fixtures)
    prices = YahooFetcher().fetch_close_prices(["US", "UK"], "2025-03-01", "2025-05-01")
    assert len(prices) == 40
    assert prices.index.tz is None
    assert prices.pct_change(fill_method=None).dropna().corr().iloc[0, 1] == pytest.approx(1)


def test_bounded_alignment_allows_holidays_but_rejects_stale_prices():
    dates = pd.date_range("2025-01-06", periods=8, freq="B")
    prices = pd.DataFrame({"A": [100, np.nan, 101, 102, 103, 104, 105, 106]}, index=dates)
    assert align_daily_prices(prices).attrs["forward_filled_observations"] == 1
    prices.iloc[2:, 0] = np.nan
    with pytest.raises(ValueError, match="Stale prices for A"):
        align_daily_prices(prices)
    sparse = pd.DataFrame({"A": [100, np.nan]}, index=pd.to_datetime(["2025-01-01", "2025-02-01"]))
    with pytest.raises(ValueError, match="Stale"):
        align_daily_prices(sparse)


def test_duplicate_local_sessions_are_not_silently_merged():
    series = pd.Series([1, 2], index=pd.to_datetime(["2025-01-01 01:00", "2025-01-01 12:00"]))
    with pytest.raises(ValueError, match="duplicate"):
        normalize_daily_series(series)


@pytest.mark.parametrize("annual_rate", [.1, .03, -.01])
def test_optimizer_and_core_sharpe_use_the_same_risk_free_units(annual_rate):
    module = importlib.import_module("src.optimization.maximum_sharpe")
    periodic = np.expm1(np.log1p(annual_rate) / 252)
    returns = pd.DataFrame({"A": periodic + np.tile([.00001, -.00001], 40)})
    result = module.optimize_maximum_sharpe(returns, risk_free_rate=annual_rate)
    assert result["success"]
    assert result["sharpe_ratio"] == pytest.approx(calculate_sharpe_ratio(returns["A"], annual_rate), abs=1e-10)
    assert result["sharpe_ratio"] == pytest.approx(0, abs=1e-10)


def test_cash_estimates_remain_arithmetic_and_solver_failure_is_explicit(monkeypatch):
    module = importlib.import_module("src.optimization.maximum_sharpe")
    daily = np.expm1(np.log1p(.03) / 252)
    returns = pd.DataFrame({"RISK": np.tile([.006, -.004], 40), "CASH": daily})
    estimates = estimate_portfolio_inputs(returns)
    assert estimates.mean_returns[1] == pytest.approx(annual_effective_to_arithmetic(.03))
    monkeypatch.setattr(module, "minimize", lambda *args, **kwargs: SimpleNamespace(success=False, message="injected failure"))
    result = module.optimize_maximum_sharpe(returns, risk_free_rate=.03)
    assert not result["success"]
    assert result["status"] == "fallback_feasible"
    assert len(result["weights"]) == 0
    np.testing.assert_allclose(result["fallback_weights"], [0, 1])


def test_execution_rechecks_minimum_order_after_fees():
    plan = build_execution_plan(["A"], [1.], prices={"A": 10}, portfolio_value=100, minimum_trade_value=100, transaction_cost_bps=10)
    assert plan["trades"] == []
    assert plan["cash"] == 100


def test_missing_adv_cannot_bypass_requested_execution_limit():
    plan = build_execution_plan(["A"], [1.], prices={"A": 10}, portfolio_value=100, maximum_adv_participation=.01)
    assert not plan["success"]
    assert not plan["liquidity_constraints_satisfied"]
    assert plan["trades"] == []
    with pytest.raises(ValueError, match="ADV"):
        estimate_trade_costs([1], ["A"], portfolio_value=100, market_impact_bps=10)


def test_execution_rechecks_concentration_after_holding_selection():
    plan = build_execution_plan(["A", "B", "C"], [.4, .35, .25], prices={"A": 1, "B": 1, "C": 1}, portfolio_value=100,
                                maximum_holdings=2, max_weight=.4)
    assert not plan["success"]
    assert not plan["mandate_constraints_satisfied"]
    assert any(not row["passed"] for row in plan["constraint_report"])


def test_execution_does_not_normalize_away_current_cash():
    plan = build_execution_plan(["A", "B"], [.5, .5], prices={"A": 1, "B": 1}, portfolio_value=100,
                                current_weights=[.4, .4], turnover_limit=.25)
    assert plan["success"]
    assert plan["turnover"] == pytest.approx(.2)


@pytest.mark.parametrize("strategy,metadata", [
    ({"max_sector_weight": .5}, {"A": {"sector": "Tech"}, "B": {"sector": "Tech"}, "C": {"sector": "Health"}}),
    ({"min_cash_weight": .25}, {"C": {"is_cash": True}}),
    ({"max_beta": .7}, {"A": {"beta": 1}, "B": {"beta": 1}, "C": {"beta": 0}}),
])
def test_lot_plan_cannot_pass_a_mandate_invalidated_by_holding_selection(strategy, metadata):
    plan = build_execution_plan(["A", "B", "C"], [.4, .35, .25], prices={"A": 1, "B": 1, "C": 1},
        portfolio_value=100, maximum_holdings=2, strategy=strategy, asset_metadata=metadata)
    assert not plan["success"]


def test_execution_obeys_holding_count_from_strategy_itself():
    plan = build_execution_plan(["A", "B", "C"], [.4, .35, .25], prices={"A": 1, "B": 1, "C": 1},
        portfolio_value=100, strategy={"max_holdings": 2})
    assert plan["success"]
    assert plan["holding_count"] == 2


def test_execution_rechecks_risk_limit_after_reducing_holdings():
    covariance = np.eye(3) * .04
    original = np.array([.4, .35, .25])
    assert np.sqrt(original @ covariance @ original) < .125
    plan = build_execution_plan(["A", "B", "C"], original, prices={"A": 1, "B": 1, "C": 1},
        portfolio_value=100, maximum_holdings=2, target_volatility=.125, annualized_covariance=covariance)
    assert not plan["success"]
    assert not plan["mandate_constraints_satisfied"]
    assert any(row["name"] == "portfolio_volatility" and not row["passed"] for row in plan["constraint_report"])


def test_execution_cannot_approve_an_unverifiable_risk_limit():
    plan = build_execution_plan(["A"], [1.], prices={"A": 1}, portfolio_value=100, target_volatility=.1)
    assert not plan["success"]
    assert "covariance" in plan["message"]


def _dcf_snapshot(financial_currency="USD", multiplier=1):
    return {"ticker": "SYNTHETIC", "info": {"currency": "USD", "financialCurrency": financial_currency,
        "freeCashflow": 100 * multiplier, "sharesOutstanding": 100., "currentPrice": 100., "marketCap": 10000.,
        "totalDebt": 20 * multiplier, "totalCash": 10 * multiplier, "beta": 1.}}


def test_dcf_requires_explicit_verified_currency_conversion():
    mixed = prepare_dcf_inputs(_dcf_snapshot("TWD", 25))
    assert not calculate_multistage_dcf(mixed, {})["available"]
    converted = prepare_dcf_inputs(_dcf_snapshot("TWD", 25), financial_to_quote_rate=1 / 25)
    reference = prepare_dcf_inputs(_dcf_snapshot())
    assert calculate_multistage_dcf(converted, {})["fair_value_per_share"] == pytest.approx(calculate_multistage_dcf(reference, {})["fair_value_per_share"])


@pytest.mark.parametrize("field", ["cash", "debt", "current_price", "shares_outstanding", "discount_rate"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_dcf_inputs_are_not_available(field, bad_value):
    result = calculate_multistage_dcf(prepare_dcf_inputs(_dcf_snapshot()), {field: bad_value})
    assert not result["available"]


def _record(run_id="test_run"):
    return RunRecord.now(run_id, config={"tickers": ["A"], "news_api_key": "FAKE-SECRET", "nested": [{"access_token": "FAKE-TOKEN", "visible": 1}]},
                         universe=["A"], date_range={}, outputs={}, metrics={}, summary={}, news={})


def test_history_redacts_credentials_on_creation_save_and_legacy_read(tmp_path):
    record = _record()
    assert record.config == {"tickers": ["A"]}
    record.config["password"] = "FAKE-PASSWORD"
    path = save_run_record(record, tmp_path)
    assert "FAKE-" not in path.read_text()
    path.write_text(json.dumps({**record.to_dict(), "config": {"news_api_key": "LEGACY-FAKE"}}))
    assert load_run_record(record.run_id, tmp_path)["config"] == {}


def test_team_history_is_durable_and_separate_from_personal_ids(tmp_path):
    @contextmanager
    def connect():
        connection = sqlite3.connect(tmp_path / "team.db")
        try:
            yield connection
        finally:
            connection.close()
    assert list_run_records(team_connection_factory=connect) == []
    save_run_record(_record(), team_connection_factory=connect)
    rows = list_run_records(team_connection_factory=connect)
    assert rows[0]["run_id"] == "test_run"
    assert "news_api_key" not in rows[0]["config"]
    assert load_run_record("test_run", team_connection_factory=connect) == rows[0]
    with pytest.raises(FileNotFoundError):
        load_run_record("missing", team_connection_factory=connect)
    with pytest.raises(ValueError, match="mutually exclusive"):
        save_run_record(_record("second"), user_id=1, team_connection_factory=connect)


def test_history_defaults_are_independent_of_working_directory_and_sort_by_timestamp(tmp_path, monkeypatch):
    history = importlib.import_module("src.analytics.modular.history")
    monkeypatch.setattr(history, "LEGACY_HISTORY_DIR", tmp_path / "history")
    monkeypatch.chdir(tmp_path)
    newer = _record("aaa")
    newer.timestamp = "2026-09-06T00:00:00+00:00"
    older = _record("zzz")
    older.timestamp = "2025-09-06T00:00:00+00:00"
    save_run_record(newer)
    save_run_record(older)
    assert list_run_records(limit=1)[0]["run_id"] == "aaa"
    assert load_run_record("aaa")["timestamp"] == newer.timestamp


def test_history_config_uses_an_allowlist_and_sanitizes_direct_constructor():
    payload = _record().to_dict()
    payload["config"] = {"tickers": ["A"], "unrecognized_credential": "FAKE-SECRET",
                         "rebalance_constraints": {"max_weight": .4, "access_token": "FAKE-TOKEN"}}
    record = RunRecord(**payload)
    assert record.config == {"tickers": ["A"], "rebalance_constraints": {"max_weight": .4}}
    record.config["unrecognized_credential"] = "FAKE-SECRET"
    assert "FAKE-" not in json.dumps(record.to_dict())


def test_local_history_sanitizer_is_scoped_repeatable_and_dry_by_default(tmp_path):
    from scripts.sanitize_run_history import sanitize_local_history
    history = tmp_path / "run_history"
    history.mkdir()
    record = {**_record().to_dict(), "config": {"news_api_key": "FAKE-SECRET", "tickers": ["A"]}}
    content = json.dumps(record)
    path = history / "run.json"
    path.write_text(content)
    with sqlite3.connect(tmp_path / "test.db") as connection:
        connection.execute("CREATE TABLE user_data (data_type TEXT, content_json TEXT)")
        connection.executemany("INSERT INTO user_data VALUES (?, ?)", [("run_history", content), ("portfolio", content)])
    report = sanitize_local_history(tmp_path)
    assert report["counts"]["records_requiring_redaction"] == 2
    assert path.read_text() == content
    assert "FAKE-" not in json.dumps(report)
    report = sanitize_local_history(tmp_path, apply=True)
    assert report["counts"]["records_redacted"] == 2
    assert json.loads(path.read_text())["config"] == {"tickers": ["A"]}
    with sqlite3.connect(tmp_path / "test.db") as connection:
        rows = dict(connection.execute("SELECT data_type, content_json FROM user_data"))
    assert "FAKE-" not in rows["run_history"]
    assert rows["portfolio"] == content
    assert sanitize_local_history(tmp_path)["counts"].get("records_requiring_redaction", 0) == 0


@pytest.mark.parametrize("missing", ["currency", "financialCurrency"])
def test_dcf_cannot_value_unidentified_currency_units(missing):
    snapshot = _dcf_snapshot()
    snapshot["info"].pop(missing)
    result = calculate_multistage_dcf(prepare_dcf_inputs(snapshot), {})
    assert not result["available"]
    assert "currencies" in result["error"]


def test_dcf_rejects_nonfinite_upside_even_with_finite_fair_value():
    result = calculate_multistage_dcf(prepare_dcf_inputs(_dcf_snapshot()), {"current_price": 1e-320})
    assert not result["available"]


def test_personal_history_is_forwarded_consistently_through_the_pipeline(monkeypatch):
    pipeline = importlib.import_module("src.analytics.modular.pipeline")
    from src.analytics.modular.results import NewsResult
    calls = []
    monkeypatch.setattr(pipeline, "list_run_records", lambda **kwargs: calls.append(("read", kwargs)) or [])
    monkeypatch.setattr(pipeline, "save_run_record", lambda record, **kwargs: calls.append(("write", kwargs)) or "db://fake")
    monkeypatch.setattr(pipeline, "build_news_analysis", lambda **kwargs: NewsResult(available=False))
    returns = pd.Series(np.tile([.01, -.009], 40))
    result = pipeline.run_quant_stack(returns, returns.to_frame("A"), {"tickers": ["A"], "news_api_key": "FAKE-KEY"}, user_id=77, precomputed_models={})
    assert [item[1]["user_id"] for item in calls] == [77, 77]
    assert "news_api_key" not in result["run_record"].config


def test_run_history_ui_keeps_dictionary_records(monkeypatch):
    from ui.pages import wharton_dash
    tables = []
    fake_st = SimpleNamespace(markdown=lambda *a, **k: None, caption=lambda *a, **k: None,
        info=lambda *a, **k: None, warning=lambda *a, **k: None,
        dataframe=lambda table, **kwargs: tables.append(table))
    record = {"run_id": "known_run", "timestamp": "2026-09-05", "universe": ["A"], "metrics": {"sharpe_ratio": 1.25}}
    monkeypatch.setattr(wharton_dash, "st", fake_st)
    monkeypatch.setattr(wharton_dash, "_load_modular_history", lambda: SimpleNamespace(list_run_records=lambda **kwargs: [record]))
    wharton_dash._render_run_history({})
    assert tables[0].iloc[0]["Run ID"] == "known_run"
    assert tables[0].iloc[0]["Tickers"] == "A"


def test_session_expires_and_is_revoked_by_password_or_role_changes():
    user = {"id": 1, "username": "test", "role": "quant", "primary_module": "quant"}
    session = issue_session(user, "hash-a", now=1000)
    assert session_is_current(session, user, "hash-a", now=1001)
    assert not session_is_current(session, user, "hash-b", now=1001)
    assert not session_is_current(session, {**user, "role": "reader"}, "hash-a", now=1001)
    assert not session_is_current(session, user, "hash-a", now=1000 + IDLE_SESSION_SECONDS)
    session["_session_last_seen"] = 1000 + ABSOLUTE_SESSION_SECONDS - 1
    assert not session_is_current(session, user, "hash-a", now=1000 + ABSOLUTE_SESSION_SECONDS)
    assert not session_is_current(user, user, "hash-a", now=1001)


def test_missing_garch_dependency_does_not_create_a_duplicate_ewma_vote(monkeypatch):
    models = importlib.import_module("src.analytics.modular.models")
    monkeypatch.setattr(models, "_load_arch_model_factory", lambda: None)
    result = models._garch_model(pd.Series(np.tile([.006, -.004], 40)), {})
    assert not result.available
    assert result.payload["alternative_model"] == "ewma"


def test_api_watchlist_reads_personal_database_history(monkeypatch):
    from src.api import handlers
    history = importlib.import_module("src.analytics.modular.history")
    calls = []
    def records(**kwargs):
        calls.append(kwargs)
        return [{"universe": [" nvda ", "NVDA", "msft"]}]
    monkeypatch.setattr(history, "list_run_records", records)
    assert handlers._latest_run_universe_tickers(77) == ["NVDA", "MSFT"]
    assert calls == [{"user_id": 77, "limit": 50}]


def test_api_overview_filters_30_calendar_days_and_sorts_before_limiting(monkeypatch):
    from src.api import handlers
    portfolio = importlib.import_module("src.portfolio_tracker.manager")
    swing = importlib.import_module("src.swing_tracker.manager")
    today = datetime.now(timezone.utc).date()
    trades = [SimpleNamespace(status="closed", exit_date=today - timedelta(days=days), ticker=f"D{days}", realized_pnl=1.) for days in [400, 31, 30, 29, 28, 27, 26, 25, 24, 23, 0]]
    trades.append(SimpleNamespace(status="invalidated", exit_date=today, ticker="INVALID", realized_pnl=-1.))
    monkeypatch.setattr(portfolio, "load_portfolio", lambda *args, **kwargs: {"positions": []})
    monkeypatch.setattr(swing, "load_trade_book", lambda **kwargs: trades)
    monkeypatch.setattr(swing, "open_trade_rows", lambda rows: [])
    monkeypatch.setattr(handlers, "_fetch_price_snapshot", lambda *args, **kwargs: {})
    response = handlers.handle_overview({"id": 77})
    assert response.success, response.error
    assert response.data["trading"]["closed_trades_30d"] == 8
    assert response.data["recent_activity"][0]["ticker"] == "D0"
    assert all(row["ticker"] != "INVALID" for row in response.data["recent_activity"])


def test_walk_forward_uses_evolving_nav_and_checks_forced_exit_adv(monkeypatch):
    module = importlib.import_module("src.optimization.walk_forward")
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    returns = pd.DataFrame({"A": np.tile([.04, .02], 30), "B": np.tile([.02, .04], 30)}, index=dates)
    actual_costs = module.estimate_trade_costs
    calls = []
    def cost(*args, **kwargs):
        calls.append(kwargs["portfolio_value"])
        return actual_costs(*args, **kwargs)
    monkeypatch.setattr(module, "estimate_trade_costs", cost)
    walk = module.run_optimization_walk_forward(returns, optimizer="minimum_variance", train_periods=20, rebalance_periods=10,
        portfolio_value=100, transaction_cost_bps=0, average_daily_dollar_volume={"A": 10000, "B": 10000}, max_weight=1)
    assert walk["success"]
    assert calls[0] == 100
    assert calls[2] > 100
    assert walk["windows"][-1]["ending_nav"] == pytest.approx(100 * (1 + walk["net_returns"]).prod())
    membership = pd.DataFrame(True, index=dates, columns=["A", "B"])
    membership.loc[dates[29]:, "A"] = False
    walk = module.run_optimization_walk_forward(returns, optimizer="minimum_variance", train_periods=20, rebalance_periods=10,
        portfolio_value=100, transaction_cost_bps=0, average_daily_dollar_volume={"A": 100, "B": 100},
        max_adv_participation=.01, max_weight=1, universe_membership=membership)
    assert any("A" in window["liquidity_failures"] for window in walk["windows"])
    for window in walk["windows"]:
        if window["liquidity_failures"]:
            assert not window["optimizer_success"]
            assert window["turnover"] == 0


def test_departure_liquidity_solver_finds_feasible_buys_from_actual_holdings():
    from src.optimization.walk_forward import run_optimization_walk_forward
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    rng = np.random.default_rng(43)
    returns = pd.DataFrame(rng.normal(0, [.02, .001, .02], (30, 3)), index=dates, columns=["A", "B", "C"])
    membership = pd.DataFrame(True, index=dates, columns=returns.columns)
    membership.loc[dates[19]:, "A"] = False
    result = run_optimization_walk_forward(returns, optimizer="minimum_variance", train_periods=20,
        rebalance_periods=10, initial_weights=[.5, .25, .25], portfolio_value=100,
        transaction_cost_bps=0, max_weight=1, turnover_limit=1.01, universe_membership=membership,
        average_daily_dollar_volume={"A": 10000, "B": 100, "C": 10000}, max_adv_participation=.05)
    assert result["success"]
    target = result["weights_history"].iloc[0]
    assert target["A"] == 0
    assert target["B"] <= .3 + 2e-5
    assert target["C"] >= .7 - 2e-5
    assert result["windows"][0]["turnover"] <= 1.01 + 2e-5


def test_walk_forward_stops_at_bankruptcy_before_missing_later_returns():
    from src.optimization.walk_forward import run_optimization_walk_forward
    dates = pd.date_range("2025-01-01", periods=40, freq="B")
    returns = pd.DataFrame(np.random.default_rng(5).normal(0, .01, (40, 2)), index=dates, columns=["A", "B"])
    returns.iloc[25] = -1
    returns.loc[dates[26]:, "A"] = np.nan
    membership = pd.DataFrame(True, index=dates, columns=returns.columns)
    result = run_optimization_walk_forward(returns, optimizer="minimum_variance", train_periods=20,
        rebalance_periods=10, universe_membership=membership, transaction_cost_bps=0)
    assert result["net_returns"].index[-1] == dates[25]
    assert result["metrics"]["total_return"] == -1
    assert result["windows"][-1]["ending_nav"] == 0
    assert result["windows"][-1]["test_end"] == dates[25]
    assert not result["evaluation_complete"]
    assert not result["success"]


def test_walk_forward_never_reopens_a_security_after_total_loss():
    from src.optimization.walk_forward import run_optimization_walk_forward
    dates = pd.date_range("2025-01-01", periods=50, freq="B")
    returns = pd.DataFrame(np.random.default_rng(15).normal(0, .01, (50, 2)), index=dates, columns=["A", "B"])
    returns.iloc[25, 0] = -1
    result = run_optimization_walk_forward(returns, optimizer="minimum_variance", train_periods=20,
        rebalance_periods=10, transaction_cost_bps=0)
    assert result["success"]
    assert (result["weights_history"].loc[dates[30]:, "A"] == 0).all()
