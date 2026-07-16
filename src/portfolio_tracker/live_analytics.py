"""Cash-aware analytics for the live Wharton competition portfolio.

The competition table is a position ledger, not a daily NAV ledger.  This
module therefore keeps two different analytical scopes explicit:

* ledger outputs are actual P/L calculated from the recorded entry, exit and
  current prices; and
* time-series outputs are a *current-weight historical proxy*.  They describe
  how today's exposures would have behaved over the supplied return history,
  not the realised competition track record.

The module is deliberately pure: callers provide prices and return series, so
it performs no network, database, UI or scenario-engine work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.benchmark import (
    calculate_active_risk_metrics,
    calculate_return_contribution,
    calculate_risk_contribution,
)
from src.analytics.correlation import calculate_correlation_matrix
from src.analytics.portfolio_metrics import (
    calculate_average_correlation,
    calculate_concentration_metrics,
    calculate_portfolio_core_metrics,
    calculate_portfolio_daily_returns,
)
from src.analytics.risk_metrics import calculate_cvar, calculate_var

from .wharton_competition import INITIAL_CAPITAL_USD, calculate_portfolio_performance


ACTUAL_LEDGER_LABEL = (
    "Actual tracked ledger P/L from recorded entry, exit, and current prices."
)
RISK_PROXY_LABEL = (
    "Current-weight historical risk proxy; not the realised competition return path."
)

_TICKER_ATTRIBUTION_COLUMNS = [
    "Ticker",
    "Sector",
    "ClientGoal",
    "OpenLots",
    "ClosedLots",
    "OpenQuantity",
    "ClosedQuantity",
    "OpenCost",
    "TotalCost",
    "CurrentValue",
    "RealizedPnL",
    "UnrealizedPnL",
    "TotalPnL",
    "ReturnContribution",
    "ContributionToTotalPnL",
    "GrossPnLImpactPct",
    "PriceSource",
]

_GROUP_ATTRIBUTION_COLUMNS = [
    "TickerCount",
    "OpenLots",
    "ClosedLots",
    "OpenCost",
    "TotalCost",
    "CurrentValue",
    "RealizedPnL",
    "UnrealizedPnL",
    "TotalPnL",
    "ReturnContribution",
    "ContributionToTotalPnL",
    "GrossPnLImpactPct",
]

_OPEN_EXPOSURE_COLUMNS = [
    "Ticker",
    "Sector",
    "ClientGoal",
    "Quantity",
    "OpenCost",
    "CurrentPrice",
    "CurrentValue",
    "UnrealizedPnL",
    "UnrealizedReturn",
    "Weight",
    "OpenLots",
    "PriceSource",
    "LivePriceAvailable",
]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _optional_positive_float(value: Any) -> float | None:
    number = _finite_float(value, default=np.nan)
    return float(number) if np.isfinite(number) and number > 0 else None


def _normalise_ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalise_positions(
    positions: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], int]:
    clean: list[dict[str, Any]] = []
    ignored = 0
    for item in positions or []:
        if not isinstance(item, Mapping):
            ignored += 1
            continue
        ticker = _normalise_ticker(item.get("ticker"))
        if not ticker:
            ignored += 1
            continue

        row = dict(item)
        row["ticker"] = ticker
        row["status"] = (
            "closed" if str(item.get("status") or "open").strip().lower() == "closed" else "open"
        )
        row["quantity"] = _finite_float(item.get("quantity"))
        row["entry_price"] = _finite_float(item.get("entry_price"))
        row["last_price"] = _optional_positive_float(item.get("last_price"))
        row["exit_price"] = _optional_positive_float(item.get("exit_price"))
        clean.append(row)
    return clean, ignored


def _normalise_live_prices(live_prices: Mapping[str, Any] | None) -> dict[str, float]:
    clean: dict[str, float] = {}
    for ticker, value in (live_prices or {}).items():
        symbol = _normalise_ticker(ticker)
        price = _optional_positive_float(value)
        if symbol and price is not None:
            clean[symbol] = price
    return clean


def _normalise_returns(asset_returns: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(asset_returns, pd.DataFrame) or asset_returns.empty:
        return pd.DataFrame()

    source = asset_returns.copy()
    normalised_columns = [_normalise_ticker(column) for column in source.columns]
    output: dict[str, pd.Series] = {}
    for symbol in dict.fromkeys(normalised_columns):
        if not symbol:
            continue
        indices = [idx for idx, name in enumerate(normalised_columns) if name == symbol]
        values = source.iloc[:, indices].apply(pd.to_numeric, errors="coerce")
        output[symbol] = values.mean(axis=1)

    frame = pd.DataFrame(output, index=source.index).replace([np.inf, -np.inf], np.nan)
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).mean()
    try:
        frame = frame.sort_index()
    except TypeError:
        pass
    return frame


def _normalise_benchmark_returns(benchmark_returns: Any) -> pd.Series:
    if benchmark_returns is None:
        return pd.Series(dtype=float, name="benchmark")
    if isinstance(benchmark_returns, pd.DataFrame):
        if benchmark_returns.shape[1] == 0:
            return pd.Series(dtype=float, name="benchmark")
        values = benchmark_returns.iloc[:, 0]
    else:
        try:
            values = pd.Series(benchmark_returns)
        except Exception:
            return pd.Series(dtype=float, name="benchmark")

    series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if series.index.has_duplicates:
        series = series.groupby(level=0).mean()
    try:
        series = series.sort_index()
    except TypeError:
        pass
    return series.dropna().astype(float).rename("benchmark")


def _payload_from_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    direct = dict(record)
    payload = direct.get("payload")
    if isinstance(payload, Mapping):
        return {**direct, **dict(payload)}
    payload_json = direct.get("payload_json")
    if isinstance(payload_json, str) and payload_json.strip():
        try:
            decoded = json.loads(payload_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, Mapping):
            return {**direct, **dict(decoded)}
    return direct


def _normalise_theses(thesis_by_ticker: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for ticker, record in (thesis_by_ticker or {}).items():
        symbol = _normalise_ticker(ticker)
        if symbol:
            output[symbol] = _payload_from_record(record)
    return output


def _thesis_labels(ticker: str, theses: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    payload = dict(theses.get(ticker, {}))
    sector = str(payload.get("sector") or "Unassigned").strip() or "Unassigned"
    goal = str(payload.get("primary_goal") or "").strip()
    if not goal:
        goals = payload.get("goals")
        if isinstance(goals, Sequence) and not isinstance(goals, (str, bytes)) and goals:
            goal = str(goals[0] or "").strip()
    return sector, goal or "Unassigned"


def _add_contribution_columns(frame: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    total_pnl = float(result["TotalPnL"].sum())
    gross_pnl = float(result["TotalPnL"].abs().sum())
    result["ReturnContribution"] = result["TotalPnL"] / initial_capital
    result["ContributionToTotalPnL"] = (
        result["TotalPnL"] / total_pnl if not np.isclose(total_pnl, 0.0) else 0.0
    )
    result["GrossPnLImpactPct"] = (
        result["TotalPnL"].abs() / gross_pnl if not np.isclose(gross_pnl, 0.0) else 0.0
    )
    return result


def _build_ledger_attribution(
    performance_rows: Sequence[Mapping[str, Any]],
    theses: Mapping[str, Mapping[str, Any]],
    initial_capital: float,
) -> pd.DataFrame:
    aggregated: dict[str, dict[str, Any]] = {}
    for item in performance_rows:
        ticker = _normalise_ticker(item.get("ticker"))
        if not ticker:
            continue
        sector, goal = _thesis_labels(ticker, theses)
        row = aggregated.setdefault(
            ticker,
            {
                "Ticker": ticker,
                "Sector": sector,
                "ClientGoal": goal,
                "OpenLots": 0,
                "ClosedLots": 0,
                "OpenQuantity": 0.0,
                "ClosedQuantity": 0.0,
                "OpenCost": 0.0,
                "TotalCost": 0.0,
                "CurrentValue": 0.0,
                "RealizedPnL": 0.0,
                "UnrealizedPnL": 0.0,
                "TotalPnL": 0.0,
                "_price_sources": set(),
            },
        )
        status = str(item.get("status") or "open").lower()
        quantity = _finite_float(item.get("quantity"))
        cost = _finite_float(item.get("cost"))
        pnl = _finite_float(item.get("pnl"))
        row["TotalCost"] += cost
        row["TotalPnL"] += pnl
        if status == "closed":
            row["ClosedLots"] += 1
            row["ClosedQuantity"] += quantity
            row["RealizedPnL"] += pnl
        else:
            row["OpenLots"] += 1
            row["OpenQuantity"] += quantity
            row["OpenCost"] += cost
            row["CurrentValue"] += _finite_float(item.get("current_value"))
            row["UnrealizedPnL"] += pnl
            source = str(item.get("price_source") or "unknown").strip()
            if source:
                row["_price_sources"].add(source)

    rows: list[dict[str, Any]] = []
    for row in aggregated.values():
        clean = dict(row)
        clean["PriceSource"] = ", ".join(sorted(clean.pop("_price_sources"))) or "n/a"
        rows.append(clean)
    if not rows:
        return pd.DataFrame(columns=_TICKER_ATTRIBUTION_COLUMNS)

    frame = _add_contribution_columns(pd.DataFrame(rows), initial_capital)
    frame["_sort"] = frame["TotalPnL"].abs()
    frame = frame.sort_values(["_sort", "Ticker"], ascending=[False, True]).drop(columns="_sort")
    return frame[_TICKER_ATTRIBUTION_COLUMNS].reset_index(drop=True)


def _group_attribution(
    ticker_attribution: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    output_columns = [group_column, *_GROUP_ATTRIBUTION_COLUMNS]
    if ticker_attribution.empty:
        return pd.DataFrame(columns=output_columns)

    numeric = [
        "OpenLots",
        "ClosedLots",
        "OpenCost",
        "TotalCost",
        "CurrentValue",
        "RealizedPnL",
        "UnrealizedPnL",
        "TotalPnL",
        "ReturnContribution",
    ]
    grouped = ticker_attribution.groupby(group_column, dropna=False)
    result = grouped[numeric].sum().reset_index()
    counts = grouped["Ticker"].nunique().rename("TickerCount").reset_index()
    result = counts.merge(result, on=group_column, how="left")

    total_pnl = float(result["TotalPnL"].sum())
    gross_pnl = float(result["TotalPnL"].abs().sum())
    result["ContributionToTotalPnL"] = (
        result["TotalPnL"] / total_pnl if not np.isclose(total_pnl, 0.0) else 0.0
    )
    result["GrossPnLImpactPct"] = (
        result["TotalPnL"].abs() / gross_pnl if not np.isclose(gross_pnl, 0.0) else 0.0
    )
    result["_sort"] = result["TotalPnL"].abs()
    result = result.sort_values(["_sort", group_column], ascending=[False, True]).drop(columns="_sort")
    return result[output_columns].reset_index(drop=True)


def _build_open_exposures(
    ticker_attribution: pd.DataFrame,
    live_prices: Mapping[str, float],
    equity: float,
) -> pd.DataFrame:
    if ticker_attribution.empty:
        return pd.DataFrame(columns=_OPEN_EXPOSURE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for item in ticker_attribution.to_dict("records"):
        if int(item.get("OpenLots") or 0) <= 0:
            continue
        quantity = _finite_float(item.get("OpenQuantity"))
        current_value = _finite_float(item.get("CurrentValue"))
        open_cost = _finite_float(item.get("OpenCost"))
        ticker = _normalise_ticker(item.get("Ticker"))
        rows.append(
            {
                "Ticker": ticker,
                "Sector": str(item.get("Sector") or "Unassigned"),
                "ClientGoal": str(item.get("ClientGoal") or "Unassigned"),
                "Quantity": quantity,
                "OpenCost": open_cost,
                "CurrentPrice": current_value / quantity if not np.isclose(quantity, 0.0) else np.nan,
                "CurrentValue": current_value,
                "UnrealizedPnL": _finite_float(item.get("UnrealizedPnL")),
                "UnrealizedReturn": (
                    _finite_float(item.get("UnrealizedPnL")) / open_cost
                    if not np.isclose(open_cost, 0.0)
                    else 0.0
                ),
                "Weight": current_value / equity if not np.isclose(equity, 0.0) else 0.0,
                "OpenLots": int(item.get("OpenLots") or 0),
                "PriceSource": str(item.get("PriceSource") or "n/a"),
                "LivePriceAvailable": ticker in live_prices,
            }
        )
    if not rows:
        return pd.DataFrame(columns=_OPEN_EXPOSURE_COLUMNS)
    frame = pd.DataFrame(rows)
    frame["_sort"] = frame["CurrentValue"].abs()
    frame = frame.sort_values(["_sort", "Ticker"], ascending=[False, True]).drop(columns="_sort")
    return frame[_OPEN_EXPOSURE_COLUMNS].reset_index(drop=True)


def _empty_risk_metrics() -> dict[str, Any]:
    return {
        "available": False,
        "scope": RISK_PROXY_LABEL,
        "partial_coverage": False,
        "observations": 0,
        "daily_return_mean": 0.0,
        "annualized_return": 0.0,
        "volatility": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "total_return": 0.0,
        "historical_var_95": 0.0,
        "historical_cvar_95": 0.0,
        "average_correlation": 0.0,
        "hhi_including_cash": 0.0,
        "effective_holdings_including_cash": 0.0,
        "max_weight_including_cash": 0.0,
        "invested_hhi": 0.0,
        "invested_effective_holdings": 0.0,
        "invested_max_weight": 0.0,
    }


def _empty_proxy_outputs(benchmark_ticker: str) -> tuple[dict[str, Any], dict[str, Any]]:
    risk = _empty_risk_metrics()
    benchmark = calculate_active_risk_metrics(
        portfolio_returns=pd.Series(dtype=float),
        benchmark_returns=pd.Series(dtype=float),
        benchmark_ticker=benchmark_ticker,
    )
    benchmark["scope"] = RISK_PROXY_LABEL
    return risk, benchmark


def build_live_competition_analytics(
    positions: Sequence[Mapping[str, Any]] | None,
    live_prices: Mapping[str, Any] | None,
    asset_returns: pd.DataFrame | None,
    benchmark_returns: pd.Series | None = None,
    benchmark_ticker: str = "SPY",
    initial_capital: float = INITIAL_CAPITAL_USD,
    risk_free_rate: float = 0.03,
    thesis_by_ticker: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build actual ledger and cash-aware current-exposure analytics.

    ``asset_returns`` must contain daily return series, not price levels.  The
    historical metrics intentionally hold today's weights constant and include
    a zero-return cash balance.  Missing holdings are represented by a zero-
    return ``UNMODELED`` bucket and explicitly disclosed in coverage warnings;
    this preserves the accounting weights without pretending the risk estimate
    is complete.
    """
    capital = _finite_float(initial_capital, default=np.nan)
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("initial_capital must be a positive finite number.")
    risk_free = _finite_float(risk_free_rate)
    benchmark_symbol = _normalise_ticker(benchmark_ticker)

    warnings: list[str] = []

    def warn(message: str) -> None:
        if message and message not in warnings:
            warnings.append(message)

    clean_positions, ignored_positions = _normalise_positions(positions)
    prices = _normalise_live_prices(live_prices)
    theses = _normalise_theses(thesis_by_ticker)
    returns = _normalise_returns(asset_returns)
    benchmark = _normalise_benchmark_returns(benchmark_returns)

    if ignored_positions:
        warn(f"Ignored {ignored_positions} invalid position row(s) without a usable ticker.")
    if not clean_positions:
        warn("No tracked positions are available; the ledger is all cash.")
    if any(row["quantity"] <= 0 or row["entry_price"] <= 0 for row in clean_positions):
        warn("At least one position has a non-positive quantity or entry price.")
    if any(row["status"] == "closed" and row.get("exit_price") is None for row in clean_positions):
        warn("At least one closed position lacks an exit price and is valued at entry price.")

    performance = calculate_portfolio_performance(
        clean_positions,
        live_prices=prices,
        initial_capital=capital,
    )
    ticker_attribution = _build_ledger_attribution(
        performance.get("positions", []),
        theses=theses,
        initial_capital=capital,
    )
    sector_attribution = _group_attribution(ticker_attribution, "Sector")
    goal_attribution = _group_attribution(ticker_attribution, "ClientGoal")

    current_equity = _finite_float(performance.get("equity"))
    current_cash = _finite_float(performance.get("cash_before_pnl")) + _finite_float(
        performance.get("realized_pnl")
    )
    open_exposures = _build_open_exposures(ticker_attribution, prices, current_equity)
    open_market_value = float(open_exposures["CurrentValue"].sum()) if not open_exposures.empty else 0.0
    accounting_gap = current_equity - (current_cash + open_market_value)
    if not np.isclose(accounting_gap, 0.0, atol=max(1e-6, abs(current_equity) * 1e-10)):
        warn(f"Portfolio accounting identity is out by ${accounting_gap:,.2f}.")
    if current_cash < -1e-6:
        warn("Current cash is negative; the tracker implies leverage or an inconsistent trade ledger.")

    if current_equity > 0:
        current_weights = pd.Series(
            {
                **{
                    str(row["Ticker"]): float(row["CurrentValue"]) / current_equity
                    for row in open_exposures.to_dict("records")
                },
                "CASH": current_cash / current_equity,
            },
            dtype=float,
            name="Weight",
        )
    else:
        current_weights = pd.Series(dtype=float, name="Weight")
        warn("Current equity is non-positive, so portfolio weights and risk proxy are unavailable.")

    open_tickers = open_exposures["Ticker"].astype(str).tolist() if not open_exposures.empty else []
    live_tickers = sorted(set(open_tickers).intersection(prices))
    manual_tickers = sorted(
        {
            str(row.get("ticker"))
            for row in performance.get("positions", [])
            if str(row.get("status") or "open") != "closed" and row.get("price_source") == "manual"
        }
    )
    fallback_tickers = sorted(
        {
            str(row.get("ticker"))
            for row in performance.get("positions", [])
            if str(row.get("status") or "open") != "closed" and row.get("price_source") == "entry fallback"
        }
    )
    if manual_tickers:
        warn("Manual prices are used for: " + ", ".join(manual_tickers) + ".")
    if fallback_tickers:
        warn("Entry-price fallbacks are used for: " + ", ".join(fallback_tickers) + ".")

    history_tickers = [
        ticker
        for ticker in open_tickers
        if ticker in returns.columns and int(returns[ticker].notna().sum()) >= 2
    ]
    missing_history_tickers = [ticker for ticker in open_tickers if ticker not in history_tickers]
    if missing_history_tickers:
        warn(
            "No usable return history for: "
            + ", ".join(missing_history_tickers)
            + "; proxy risk is understated for this exposure."
        )

    exposure_values = (
        open_exposures.set_index("Ticker")["CurrentValue"].astype(float)
        if not open_exposures.empty
        else pd.Series(dtype=float)
    )
    total_absolute_open_value = float(exposure_values.abs().sum())
    modeled_absolute_value = float(exposure_values.reindex(history_tickers).fillna(0.0).abs().sum())
    history_value_coverage = (
        modeled_absolute_value / total_absolute_open_value
        if total_absolute_open_value > 0
        else 1.0
    )

    aligned_risky_returns = pd.DataFrame()
    if history_tickers:
        aligned_risky_returns = returns[history_tickers].dropna(how="any").astype(float)
        if aligned_risky_returns.shape[0] < 2:
            warn("Open holdings do not have at least two overlapping return observations.")
            aligned_risky_returns = pd.DataFrame(columns=history_tickers)

    proxy_asset_returns = pd.DataFrame()
    proxy_weights = pd.Series(dtype=float, name="Weight")
    portfolio_returns = pd.Series(dtype=float, name="current_weight_proxy")
    risk_metrics, benchmark_metrics = _empty_proxy_outputs(benchmark_symbol)
    correlation = pd.DataFrame()
    risk_contribution = pd.DataFrame(
        columns=["Ticker", "Weight", "MarginalVolatility", "RiskContribution", "RiskBudgetPct", "Sector", "ClientGoal"]
    )
    proxy_return_contribution = pd.DataFrame(
        columns=[
            "Ticker",
            "Weight",
            "TotalContributionApprox",
            "AnnualizedContributionApprox",
            "ContributionShare",
            "MeanDailyContribution",
            "Sector",
            "ClientGoal",
        ]
    )

    risk_proxy_available = False
    partial_coverage = bool(missing_history_tickers)
    can_model_risky = bool(history_tickers and not aligned_risky_returns.empty)
    all_cash = not open_tickers

    if current_equity > 0 and (can_model_risky or all_cash):
        if can_model_risky:
            proxy_asset_returns = aligned_risky_returns.copy()
            proxy_index = aligned_risky_returns.index
        else:
            proxy_index = returns.dropna(how="all").index
            if len(proxy_index) < 2:
                proxy_index = benchmark.index
            proxy_asset_returns = pd.DataFrame(index=proxy_index)

        if len(proxy_asset_returns.index) >= 2:
            proxy_asset_returns["CASH"] = 0.0
            weight_values: dict[str, float] = {
                ticker: float(current_weights.get(ticker, 0.0)) for ticker in history_tickers
            }
            weight_values["CASH"] = float(current_weights.get("CASH", 0.0))
            unmodeled_weight = float(
                sum(float(current_weights.get(ticker, 0.0)) for ticker in missing_history_tickers)
            )
            if not np.isclose(unmodeled_weight, 0.0):
                proxy_asset_returns["UNMODELED"] = 0.0
                weight_values["UNMODELED"] = unmodeled_weight

            proxy_weights = pd.Series(weight_values, dtype=float, name="Weight")
            proxy_asset_returns = proxy_asset_returns.reindex(columns=proxy_weights.index)
            portfolio_returns = calculate_portfolio_daily_returns(
                proxy_asset_returns,
                proxy_weights.to_numpy(dtype=float),
            ).rename("current_weight_proxy")

            core = calculate_portfolio_core_metrics(portfolio_returns, risk_free_rate=risk_free)
            full_concentration = calculate_concentration_metrics(current_weights.to_numpy(dtype=float))
            invested_values = exposure_values.to_numpy(dtype=float)
            if invested_values.size and not np.isclose(np.abs(invested_values).sum(), 0.0):
                invested_weights = np.abs(invested_values) / np.abs(invested_values).sum()
                invested_concentration = calculate_concentration_metrics(invested_weights)
            else:
                invested_concentration = calculate_concentration_metrics(np.array([], dtype=float))

            correlation = calculate_correlation_matrix(aligned_risky_returns) if can_model_risky else pd.DataFrame()
            risk_metrics = {
                "available": True,
                "scope": RISK_PROXY_LABEL,
                "partial_coverage": partial_coverage,
                "observations": int(len(portfolio_returns)),
                **core,
                "historical_var_95": calculate_var(portfolio_returns, 0.95),
                "historical_cvar_95": calculate_cvar(portfolio_returns, 0.95),
                "average_correlation": calculate_average_correlation(correlation),
                "hhi_including_cash": full_concentration["hhi"],
                "effective_holdings_including_cash": full_concentration["effective_holdings"],
                "max_weight_including_cash": full_concentration["max_weight"],
                "invested_hhi": invested_concentration["hhi"],
                "invested_effective_holdings": invested_concentration["effective_holdings"],
                "invested_max_weight": invested_concentration["max_weight"],
            }
            benchmark_metrics = calculate_active_risk_metrics(
                portfolio_returns=portfolio_returns,
                benchmark_returns=benchmark,
                benchmark_ticker=benchmark_symbol,
                risk_free_rate=risk_free,
            )
            benchmark_metrics["scope"] = RISK_PROXY_LABEL

            risk_contribution = calculate_risk_contribution(
                proxy_asset_returns,
                proxy_weights.to_numpy(dtype=float),
            )
            proxy_return_contribution = calculate_return_contribution(
                proxy_asset_returns,
                proxy_weights.to_numpy(dtype=float),
            )

            def enrich(frame: pd.DataFrame) -> pd.DataFrame:
                if frame.empty:
                    return frame
                enriched = frame.copy()
                sectors: list[str] = []
                goals: list[str] = []
                for ticker in enriched["Ticker"].astype(str):
                    if ticker == "CASH":
                        sector, goal = "Liquidity", "Liquidity"
                    elif ticker == "UNMODELED":
                        sector, goal = "Unmodeled", "Unmodeled"
                    else:
                        sector, goal = _thesis_labels(ticker, theses)
                    sectors.append(sector)
                    goals.append(goal)
                enriched["Sector"] = sectors
                enriched["ClientGoal"] = goals
                return enriched

            risk_contribution = enrich(risk_contribution)
            proxy_return_contribution = enrich(proxy_return_contribution)
            risk_proxy_available = True

    if benchmark_symbol and benchmark.empty:
        warn(f"Benchmark return history for {benchmark_symbol} is unavailable.")
    elif benchmark_symbol and risk_proxy_available and not benchmark_metrics.get("benchmark_available"):
        warn(f"Benchmark {benchmark_symbol} has insufficient overlap with the risk proxy.")

    open_ticker_count = len(open_tickers)
    coverage = {
        "tracked_positions": len(clean_positions),
        "open_lots": int(sum(row["status"] != "closed" for row in clean_positions)),
        "open_tickers": open_ticker_count,
        "live_price_tickers": len(live_tickers),
        "manual_price_tickers": len(manual_tickers),
        "entry_fallback_tickers": len(fallback_tickers),
        "price_coverage_pct": len(live_tickers) / open_ticker_count if open_ticker_count else 1.0,
        "history_tickers": len(history_tickers),
        "history_coverage_pct": len(history_tickers) / open_ticker_count if open_ticker_count else 1.0,
        "history_value_coverage_pct": history_value_coverage,
        "modeled_portfolio_weight": float(
            sum(float(current_weights.get(ticker, 0.0)) for ticker in history_tickers)
        ),
        "return_observations": int(len(portfolio_returns)),
        "benchmark_observations": int(benchmark_metrics.get("benchmark_obs", 0) or 0),
        "missing_history_tickers": missing_history_tickers,
        "live_price_symbols": live_tickers,
        "manual_price_symbols": manual_tickers,
        "entry_fallback_symbols": fallback_tickers,
    }

    return {
        "available": bool(clean_positions),
        "risk_proxy_available": risk_proxy_available,
        "labels": {
            "ledger": ACTUAL_LEDGER_LABEL,
            "risk_proxy": RISK_PROXY_LABEL,
        },
        "ledger_performance": performance,
        "ledger_attribution": ticker_attribution.copy(),
        "ledger_attribution_by_ticker": ticker_attribution,
        "ledger_attribution_by_sector": sector_attribution,
        "ledger_attribution_by_goal": goal_attribution,
        "open_exposures": open_exposures,
        "current_equity": current_equity,
        "current_cash": current_cash,
        "current_weights": current_weights,
        "proxy_weights": proxy_weights,
        "proxy_asset_returns": proxy_asset_returns,
        "portfolio_returns": portfolio_returns,
        "risk_metrics": risk_metrics,
        "benchmark_metrics": benchmark_metrics,
        "correlation": correlation,
        "risk_contribution": risk_contribution,
        "proxy_return_contribution": proxy_return_contribution,
        "coverage": coverage,
        "warnings": warnings,
    }


__all__ = [
    "ACTUAL_LEDGER_LABEL",
    "RISK_PROXY_LABEL",
    "build_live_competition_analytics",
]
