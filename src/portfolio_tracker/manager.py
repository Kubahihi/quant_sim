from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re

import numpy as np
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Legacy portfolio directory (for backward compatibility)
LEGACY_PORTFOLIO_DIR = PROJECT_ROOT / "data" / "portfolios"


def _get_portfolio_dir(user_id: int | None = None) -> Path:
    """
    Get the portfolio directory for a user.
    
    If user_id is provided, returns user-specific directory.
    Otherwise, returns the legacy directory for backward compatibility.
    """
    if user_id is not None:
        user_dir = PROJECT_ROOT / "data" / "users" / str(user_id) / "portfolios"
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    return LEGACY_PORTFOLIO_DIR


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", (name or "default").strip())
    return cleaned.strip("_") or "default"


def _portfolio_path(name: str, user_id: int | None = None) -> Path:
    """Get the file path for a portfolio, scoped to user if provided."""
    portfolio_dir = _get_portfolio_dir(user_id)
    return portfolio_dir / f"{_sanitize_name(name)}.json"


def _normalize_position(raw: dict[str, Any]) -> dict[str, Any]:
    ticker = str(raw.get("ticker", "")).strip().upper()
    shares = float(raw.get("shares", 0.0) or 0.0)
    cost_basis = raw.get("cost_basis")
    target_weight = raw.get("target_weight")

    normalized = {
        "ticker": ticker,
        "shares": shares,
        "cost_basis": float(cost_basis) if cost_basis not in (None, "") else None,
        "target_weight": float(target_weight) if target_weight not in (None, "") else None,
        "notes": str(raw.get("notes", "") or "").strip(),
    }
    return normalized


def _default_portfolio(name: str = "default") -> dict[str, Any]:
    return {
        "name": _sanitize_name(name),
        "created_at": _timestamp_now(),
        "updated_at": _timestamp_now(),
        "positions": [],
    }


def list_portfolios(user_id: int | None = None) -> list[str]:
    """
    List saved portfolio names for a user.
    
    If user_id is provided, lists portfolios from user-specific directory.
    Otherwise, lists from legacy directory for backward compatibility.
    """
    portfolio_dir = _get_portfolio_dir(user_id)
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for file_path in sorted(portfolio_dir.glob("*.json")):
        names.append(file_path.stem)
    return names


def load_portfolio(name: str = "default", user_id: int | None = None) -> dict[str, Any]:
    """
    Load a portfolio JSON file or return an empty default structure.
    
    If user_id is provided, loads from user-specific directory.
    Otherwise, loads from legacy directory for backward compatibility.
    """
    path = _portfolio_path(name, user_id)
    if not path.exists():
        return _default_portfolio(name)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_portfolio(name)

    portfolio = _default_portfolio(str(payload.get("name") or name))
    portfolio["created_at"] = str(payload.get("created_at") or portfolio["created_at"])
    portfolio["updated_at"] = str(payload.get("updated_at") or portfolio["updated_at"])

    raw_positions = payload.get("positions", [])
    positions = []
    for item in raw_positions if isinstance(raw_positions, list) else []:
        if not isinstance(item, dict):
            continue
        position = _normalize_position(item)
        if position["ticker"] and position["shares"] != 0:
            positions.append(position)
    portfolio["positions"] = positions
    return portfolio


def save_portfolio(portfolio: dict[str, Any], name: str | None = None, user_id: int | None = None) -> Path:
    """
    Persist a portfolio dictionary.
    
    If user_id is provided, saves to user-specific directory.
    Otherwise, saves to legacy directory for backward compatibility.
    """
    portfolio_dir = _get_portfolio_dir(user_id)
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    portfolio_name = _sanitize_name(name or str(portfolio.get("name", "default")))
    path = _portfolio_path(portfolio_name, user_id)

    normalized = _default_portfolio(portfolio_name)
    normalized["created_at"] = str(portfolio.get("created_at") or normalized["created_at"])
    normalized["updated_at"] = _timestamp_now()

    positions: list[dict[str, Any]] = []
    for item in portfolio.get("positions", []):
        if not isinstance(item, dict):
            continue
        position = _normalize_position(item)
        if position["ticker"] and abs(position["shares"]) > 0:
            positions.append(position)
    normalized["positions"] = positions

    path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return path


def add_position(
    portfolio: dict[str, Any],
    ticker: str,
    shares: float,
    cost_basis: float | None = None,
    target_weight: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Add a new position or increase an existing one."""
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return portfolio

    updated = dict(portfolio)
    positions = [dict(item) for item in updated.get("positions", []) if isinstance(item, dict)]

    for item in positions:
        if str(item.get("ticker", "")).upper() != symbol:
            continue

        old_shares = float(item.get("shares", 0.0) or 0.0)
        new_shares = old_shares + float(shares)
        if new_shares == 0:
            positions = [row for row in positions if str(row.get("ticker", "")).upper() != symbol]
            updated["positions"] = positions
            updated["updated_at"] = _timestamp_now()
            return updated

        old_cost = item.get("cost_basis")
        if cost_basis is not None and old_cost is not None and old_shares > 0 and shares > 0:
            blended_cost = (float(old_cost) * old_shares + float(cost_basis) * float(shares)) / new_shares
            item["cost_basis"] = float(blended_cost)
        elif cost_basis is not None:
            item["cost_basis"] = float(cost_basis)

        item["shares"] = float(new_shares)
        if target_weight is not None:
            item["target_weight"] = float(target_weight)
        if notes:
            item["notes"] = str(notes)
        updated["positions"] = positions
        updated["updated_at"] = _timestamp_now()
        return updated

    positions.append(
        {
            "ticker": symbol,
            "shares": float(shares),
            "cost_basis": float(cost_basis) if cost_basis is not None else None,
            "target_weight": float(target_weight) if target_weight is not None else None,
            "notes": str(notes or "").strip(),
        }
    )
    updated["positions"] = positions
    updated["updated_at"] = _timestamp_now()
    return updated


def remove_position(portfolio: dict[str, Any], ticker: str) -> dict[str, Any]:
    """Remove a ticker from the portfolio."""
    symbol = str(ticker or "").strip().upper()
    updated = dict(portfolio)
    positions = [dict(item) for item in updated.get("positions", []) if isinstance(item, dict)]
    positions = [item for item in positions if str(item.get("ticker", "")).upper() != symbol]
    updated["positions"] = positions
    updated["updated_at"] = _timestamp_now()
    return updated


def update_position(
    portfolio: dict[str, Any],
    ticker: str,
    shares: float | None = None,
    cost_basis: float | None = None,
    target_weight: float | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Update fields for an existing portfolio position."""
    symbol = str(ticker or "").strip().upper()
    updated = dict(portfolio)
    positions = [dict(item) for item in updated.get("positions", []) if isinstance(item, dict)]

    for item in positions:
        if str(item.get("ticker", "")).upper() != symbol:
            continue
        if shares is not None:
            item["shares"] = float(shares)
        if cost_basis is not None:
            item["cost_basis"] = float(cost_basis)
        if target_weight is not None:
            item["target_weight"] = float(target_weight)
        if notes is not None:
            item["notes"] = str(notes)
        break

    updated["positions"] = [item for item in positions if float(item.get("shares", 0.0) or 0.0) != 0.0]
    updated["updated_at"] = _timestamp_now()
    return updated


def _fetch_latest_prices(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}

    prices: dict[str, float] = {}
    try:
        history = yf.download(
            tickers=tickers,
            period="7d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return prices

    if history.empty:
        return prices

    if isinstance(history.columns, pd.MultiIndex):
        first_level = history.columns.get_level_values(0)
        second_level = history.columns.get_level_values(1)
        for symbol in tickers:
            symbol_history = pd.DataFrame()
            if symbol in first_level:
                symbol_history = history[symbol]
            elif symbol in second_level:
                symbol_history = history.xs(symbol, axis=1, level=1, drop_level=True)
            if symbol_history.empty:
                continue
            close_col = "Adj Close" if "Adj Close" in symbol_history.columns else "Close"
            if close_col not in symbol_history.columns:
                continue
            close = pd.to_numeric(symbol_history[close_col], errors="coerce").dropna()
            if not close.empty:
                prices[symbol] = float(close.iloc[-1])
    else:
        close_col = "Adj Close" if "Adj Close" in history.columns else "Close"
        if len(tickers) == 1 and close_col in history.columns:
            close = pd.to_numeric(history[close_col], errors="coerce").dropna()
            if not close.empty:
                prices[tickers[0]] = float(close.iloc[-1])

    return prices


def compute_live_values(portfolio: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Compute live value snapshot for portfolio positions.

    Returns:
    - holdings dataframe
    - summary dict with total value, pnl, and coverage stats
    """
    positions = [item for item in portfolio.get("positions", []) if isinstance(item, dict)]
    if not positions:
        empty = pd.DataFrame(
            columns=[
                "Ticker",
                "Shares",
                "CostBasis",
                "TargetWeight",
                "Price",
                "MarketValue",
                "CostValue",
                "PnL",
                "PnLPercent",
                "CurrentWeight",
            ]
        )
        return empty, {"TotalMarketValue": 0.0, "TotalCostValue": 0.0, "TotalPnL": 0.0, "PricedPositions": 0.0}

    tickers = sorted({str(item.get("ticker", "")).upper() for item in positions if str(item.get("ticker", "")).strip()})
    latest_prices = _fetch_latest_prices(tickers)

    rows: list[dict[str, Any]] = []
    for item in positions:
        ticker = str(item.get("ticker", "")).strip().upper()
        shares = float(item.get("shares", 0.0) or 0.0)
        cost_basis = item.get("cost_basis")
        target_weight = item.get("target_weight")
        price = latest_prices.get(ticker, np.nan)

        cost_value = np.nan if cost_basis is None else float(cost_basis) * shares
        market_value = np.nan if np.isnan(price) else float(price) * shares
        pnl = np.nan
        pnl_pct = np.nan
        if not np.isnan(market_value) and not np.isnan(cost_value):
            pnl = market_value - cost_value
            pnl_pct = (pnl / cost_value) if cost_value else np.nan

        rows.append(
            {
                "Ticker": ticker,
                "Shares": shares,
                "CostBasis": float(cost_basis) if cost_basis is not None else np.nan,
                "TargetWeight": float(target_weight) if target_weight is not None else np.nan,
                "Price": price,
                "MarketValue": market_value,
                "CostValue": cost_value,
                "PnL": pnl,
                "PnLPercent": pnl_pct,
            }
        )

    holdings = pd.DataFrame(rows)
    total_market_value = float(pd.to_numeric(holdings["MarketValue"], errors="coerce").sum(min_count=1) or 0.0)
    total_cost_value = float(pd.to_numeric(holdings["CostValue"], errors="coerce").sum(min_count=1) or 0.0)
    total_pnl = total_market_value - total_cost_value

    if total_market_value > 0:
        holdings["CurrentWeight"] = pd.to_numeric(holdings["MarketValue"], errors="coerce") / total_market_value
    else:
        holdings["CurrentWeight"] = np.nan

    summary = {
        "TotalMarketValue": total_market_value,
        "TotalCostValue": total_cost_value,
        "TotalPnL": total_pnl,
        "PricedPositions": float(holdings["Price"].notna().sum()),
    }
    return holdings, summary


def generate_rebalance_suggestions(
    holdings: pd.DataFrame,
    tolerance: float = 0.03,
) -> pd.DataFrame:
    """
    Generate simple rebalance suggestions from current vs target weights.

    Suggests buy/sell notional amounts based on deviation beyond tolerance.
    """
    if holdings.empty:
        return pd.DataFrame(
            columns=["Ticker", "CurrentWeight", "TargetWeight", "WeightGap", "Action", "ValueAdjustment"]
        )

    data = holdings.copy()
    if "CurrentWeight" not in data.columns or "TargetWeight" not in data.columns:
        return pd.DataFrame(
            columns=["Ticker", "CurrentWeight", "TargetWeight", "WeightGap", "Action", "ValueAdjustment"]
        )

    data["CurrentWeight"] = pd.to_numeric(data["CurrentWeight"], errors="coerce")
    data["TargetWeight"] = pd.to_numeric(data["TargetWeight"], errors="coerce")
    data = data.dropna(subset=["CurrentWeight", "TargetWeight"])
    if data.empty:
        return pd.DataFrame(
            columns=["Ticker", "CurrentWeight", "TargetWeight", "WeightGap", "Action", "ValueAdjustment"]
        )

    total_value = float(pd.to_numeric(data["MarketValue"], errors="coerce").sum(min_count=1) or 0.0)
    suggestions: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        gap = float(row["CurrentWeight"] - row["TargetWeight"])
        if abs(gap) <= tolerance:
            continue

        action = "Trim" if gap > 0 else "Add"
        value_adjustment = abs(gap) * total_value if total_value > 0 else np.nan
        suggestions.append(
            {
                "Ticker": row["Ticker"],
                "CurrentWeight": row["CurrentWeight"],
                "TargetWeight": row["TargetWeight"],
                "WeightGap": gap,
                "Action": action,
                "ValueAdjustment": value_adjustment,
            }
        )

    result = pd.DataFrame(suggestions)
    if not result.empty:
        result = result.sort_values(by="WeightGap", key=lambda value: value.abs(), ascending=False).reset_index(drop=True)
    return result

