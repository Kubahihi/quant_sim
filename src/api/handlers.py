"""
API endpoint handlers.

Each handler function processes a specific API endpoint request
and returns a standardized APIResponse.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import json
import math

import numpy as np
import pandas as pd

from .responses import (
    APIResponse,
    serialize_position,
)


def _clean_float(value: Any) -> Any:
    """Clean float values for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _utc_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _get_user_id(user: Optional[dict[str, Any]]) -> Optional[int]:
    """Extract user ID from user object."""
    if user and isinstance(user, dict):
        return user.get("id")
    return None


def _default_watchlist() -> list[str]:
    """Default watchlist used when no user-specific snapshot is available."""
    return ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]


def _latest_run_universe_tickers(user_id: Optional[int]) -> list[str]:
    """Extract tickers from the latest run history universe snapshot."""
    project_root = Path(__file__).resolve().parents[2]
    history_dir = (
        project_root / "data" / "users" / str(user_id) / "run_history"
        if user_id is not None
        else project_root / "data" / "run_history"
    )

    if not history_dir.exists():
        return []

    for file_path in sorted(history_dir.glob("*.json"), reverse=True):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            universe = payload.get("universe", [])
            if not isinstance(universe, list):
                continue
            tickers = []
            for value in universe:
                symbol = str(value).strip().upper()
                if symbol and symbol.isascii():
                    tickers.append(symbol)
            # Deduplicate while preserving order
            deduped = list(dict.fromkeys(tickers))
            if deduped:
                return deduped[:20]
        except Exception:
            continue

    return []


def _extract_symbol_history(history: pd.DataFrame, symbol: str, total_symbols: int) -> pd.DataFrame:
    """Extract single-symbol history from yfinance download payload."""
    if history.empty:
        return pd.DataFrame()

    if isinstance(history.columns, pd.MultiIndex):
        first_level = history.columns.get_level_values(0)
        second_level = history.columns.get_level_values(1)
        if symbol in first_level:
            data = history[symbol]
            return data if isinstance(data, pd.DataFrame) else data.to_frame()
        if symbol in second_level:
            return history.xs(symbol, axis=1, level=1, drop_level=True)
        return pd.DataFrame()

    if total_symbols == 1:
        return history
    return pd.DataFrame()


def _fetch_price_snapshot(symbols: list[str], period: str = "5d") -> dict[str, dict[str, Any]]:
    """Fetch latest price, change, and volume for a symbol list."""
    if not symbols:
        return {}

    clean_symbols = [str(item).strip().upper() for item in symbols if str(item).strip()]
    clean_symbols = list(dict.fromkeys(clean_symbols))
    if not clean_symbols:
        return {}

    try:
        import yfinance as yf

        history = yf.download(
            tickers=clean_symbols,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
            timeout=8,
        )
    except Exception:
        return {}

    snapshots: dict[str, dict[str, Any]] = {}
    for symbol in clean_symbols:
        try:
            symbol_history = _extract_symbol_history(history, symbol, len(clean_symbols))
            if symbol_history.empty:
                continue

            close_col = "Adj Close" if "Adj Close" in symbol_history.columns else "Close"
            if close_col not in symbol_history.columns:
                continue

            close = pd.to_numeric(symbol_history[close_col], errors="coerce").dropna()
            if close.empty:
                continue

            current_price = _clean_float(close.iloc[-1])
            prev_close = _clean_float(close.iloc[-2] if len(close) > 1 else close.iloc[-1])
            change = _clean_float(current_price - prev_close) if current_price is not None and prev_close is not None else None
            change_pct = _clean_float((change / prev_close) * 100) if change is not None and prev_close and prev_close > 0 else None

            volume = 0
            if "Volume" in symbol_history.columns:
                volume_values = pd.to_numeric(symbol_history["Volume"], errors="coerce").dropna()
                if not volume_values.empty:
                    volume = int(volume_values.iloc[-1])

            snapshots[symbol] = {
                "price": current_price,
                "prev_close": prev_close,
                "change": change,
                "change_percent": change_pct,
                "volume": volume,
            }
        except Exception:
            continue

    return snapshots


def handle_summary(user: Optional[dict[str, Any]] = None) -> APIResponse:
    """
    GET /api/v1/summary
    
    Returns a high-level portfolio summary including total value,
    P&L, and key metrics.
    
    Sample response:
    {
        "success": true,
        "timestamp": "2026-05-22T20:00:00Z",
        "data": {
            "total_value": 125000.00,
            "total_cost": 100000.00,
            "total_pnl": 25000.00,
            "total_pnl_percent": 25.0,
            "positions_count": 12,
            "open_trades_count": 3,
            "last_updated": "2026-05-22T19:30:00Z"
        }
    }
    """
    try:
        user_id = _get_user_id(user)
        
        # Load portfolio
        from src.portfolio_tracker.manager import load_portfolio, compute_live_values
        
        portfolio = load_portfolio("default", user_id=user_id)
        positions = portfolio.get("positions", [])
        
        if not positions:
            return APIResponse.ok({
                "total_value": 0.0,
                "total_cost": 0.0,
                "total_pnl": 0.0,
                "total_pnl_percent": 0.0,
                "positions_count": 0,
                "open_trades_count": 0,
                "last_updated": portfolio.get("updated_at", _utc_iso()),
            })
        
        # Compute live values
        holdings_df, summary = compute_live_values(portfolio)
        
        # Count open trades
        from src.swing_tracker.manager import load_trade_book
        trades = load_trade_book(user_id=user_id)
        open_trades = [t for t in trades if t.status in ("open", "overdue")]
        
        total_market_value = _clean_float(summary.get("TotalMarketValue", 0.0))
        total_cost_value = _clean_float(summary.get("TotalCostValue", 0.0))
        total_pnl = _clean_float(summary.get("TotalPnL", 0.0))
        
        pnl_percent = 0.0
        if total_cost_value and total_cost_value > 0:
            pnl_percent = _clean_float((total_pnl / total_cost_value) * 100)
        
        return APIResponse.ok({
            "total_value": total_market_value,
            "total_cost": total_cost_value,
            "total_pnl": total_pnl,
            "total_pnl_percent": pnl_percent,
            "positions_count": len(positions),
            "open_trades_count": len(open_trades),
            "last_updated": portfolio.get("updated_at", _utc_iso()),
        })
        
    except Exception as e:
        return APIResponse.error(str(e), "summary_error", 500)


def handle_portfolio(user: Optional[dict[str, Any]] = None) -> APIResponse:
    """
    GET /api/v1/portfolio
    
    Returns the full portfolio with positions and their current values.
    
    Sample response:
    {
        "success": true,
        "timestamp": "2026-05-22T20:00:00Z",
        "data": {
            "name": "default",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-05-22T19:30:00Z",
            "positions": [
                {
                    "ticker": "AAPL",
                    "shares": 50,
                    "cost_basis": 150.00,
                    "price": 175.00,
                    "market_value": 8750.00,
                    "cost_value": 7500.00,
                    "pnl": 1250.00,
                    "pnl_percent": 16.67,
                    "current_weight": 7.0
                }
            ],
            "total_value": 125000.00,
            "total_pnl": 25000.00
        }
    }
    """
    try:
        user_id = _get_user_id(user)
        
        from src.portfolio_tracker.manager import load_portfolio, compute_live_values
        
        portfolio = load_portfolio("default", user_id=user_id)
        
        if not portfolio.get("positions"):
            return APIResponse.ok({
                "name": portfolio.get("name", "default"),
                "created_at": portfolio.get("created_at", _utc_iso()),
                "updated_at": portfolio.get("updated_at", _utc_iso()),
                "positions": [],
                "total_value": 0.0,
                "total_pnl": 0.0,
            })
        
        # Compute live values
        holdings_df, summary = compute_live_values(portfolio)
        
        # Serialize positions
        serialized_positions = []
        for _, row in holdings_df.iterrows():
            pos_dict = row.to_dict()
            serialized = serialize_position(pos_dict)
            serialized_positions.append(serialized)
        
        return APIResponse.ok({
            "name": portfolio.get("name", "default"),
            "created_at": portfolio.get("created_at", _utc_iso()),
            "updated_at": portfolio.get("updated_at", _utc_iso()),
            "positions": serialized_positions,
            "total_value": _clean_float(summary.get("TotalMarketValue", 0.0)),
            "total_pnl": _clean_float(summary.get("TotalPnL", 0.0)),
        })
        
    except Exception as e:
        return APIResponse.error(str(e), "portfolio_error", 500)


def handle_positions(user: Optional[dict[str, Any]] = None) -> APIResponse:
    """
    GET /api/v1/positions
    
    Returns a list of current portfolio positions with detailed info.
    
    Sample response:
    {
        "success": true,
        "timestamp": "2026-05-22T20:00:00Z",
        "data": [
            {
                "ticker": "AAPL",
                "shares": 50,
                "cost_basis": 150.00,
                "target_weight": 10.0,
                "current_weight": 7.0,
                "price": 175.00,
                "market_value": 8750.00,
                "pnl": 1250.00,
                "pnl_percent": 16.67
            }
        ]
    }
    """
    try:
        user_id = _get_user_id(user)
        
        from src.portfolio_tracker.manager import load_portfolio, compute_live_values
        
        portfolio = load_portfolio("default", user_id=user_id)
        
        if not portfolio.get("positions"):
            return APIResponse.ok([])
        
        holdings_df, _ = compute_live_values(portfolio)
        
        positions = []
        for _, row in holdings_df.iterrows():
            pos_dict = row.to_dict()
            serialized = serialize_position(pos_dict)
            positions.append(serialized)
        
        return APIResponse.ok(positions)
        
    except Exception as e:
        return APIResponse.error(str(e), "positions_error", 500)


def handle_watchlist(user: Optional[dict[str, Any]] = None) -> APIResponse:
    """
    GET /api/v1/watchlist
    
    Returns the user's watchlist with current prices and changes.
    Falls back to a default set of popular tickers if no watchlist is configured.
    
    Sample response:
    {
        "success": true,
        "timestamp": "2026-05-22T20:00:00Z",
        "data": [
            {
                "ticker": "SPY",
                "price": 450.00,
                "change": 2.50,
                "change_percent": 0.56,
                "volume": 75000000
            }
        ]
    }
    """
    try:
        user_id = _get_user_id(user)

        watchlist_tickers = _latest_run_universe_tickers(user_id) or _default_watchlist()
        snapshots = _fetch_price_snapshot(watchlist_tickers, period="5d")

        watchlist_data = []
        for symbol in watchlist_tickers:
            if symbol not in snapshots:
                continue
            snapshot = snapshots[symbol]
            watchlist_data.append({
                "ticker": symbol,
                "price": snapshot.get("price"),
                "change": snapshot.get("change"),
                "change_percent": snapshot.get("change_percent"),
                "volume": snapshot.get("volume", 0),
            })

        return APIResponse.ok(watchlist_data)
        
    except Exception as e:
        return APIResponse.error(str(e), "watchlist_error", 500)


def handle_signals(user: Optional[dict[str, Any]] = None) -> APIResponse:
    """
    GET /api/v1/signals
    
    Returns active alerts and signals from the swing tracker.
    
    Sample response:
    {
        "success": true,
        "timestamp": "2026-05-22T20:00:00Z",
        "data": {
            "active_trades": [
                {
                    "id": "swing_abc123",
                    "ticker": "AAPL",
                    "direction": "long",
                    "entry_price": 170.00,
                    "current_price": 175.00,
                    "stop_loss": 165.00,
                    "target_price": 185.00,
                    "pnl_percent": 2.94,
                    "days_held": 5
                }
            ],
            "alerts": [
                {
                    "type": "stop_approaching",
                    "ticker": "TSLA",
                    "message": "Price approaching stop loss"
                }
            ]
        }
    }
    """
    try:
        user_id = _get_user_id(user)
        
        from src.swing_tracker.manager import load_trade_book, open_trade_rows
        
        trades = load_trade_book(user_id=user_id)
        open_trades = open_trade_rows(trades)
        signal_tickers = [item.ticker for item in open_trades[:10]]
        snapshots = _fetch_price_snapshot(signal_tickers, period="5d")
        
        # Serialize open trades with current prices
        active_trades = []
        alerts = []
        
        for trade in open_trades[:10]:  # Limit to 10
            current_price = snapshots.get(str(trade.ticker).upper(), {}).get("price")
            
            # Calculate PnL
            pnl_percent = None
            if current_price and trade.entry_price:
                if trade.direction == "long":
                    pnl_percent = _clean_float(((current_price - trade.entry_price) / trade.entry_price) * 100)
                else:
                    pnl_percent = _clean_float(((trade.entry_price - current_price) / trade.entry_price) * 100)
            
            # Calculate days held
            days_held = 0
            if trade.entry_date:
                entry_date = trade.entry_date
                if isinstance(entry_date, str):
                    entry_date = datetime.fromisoformat(entry_date).date()
                days_held = (datetime.now().date() - entry_date).days
            
            active_trades.append({
                "id": trade.id,
                "ticker": trade.ticker,
                "direction": trade.direction,
                "entry_price": _clean_float(trade.entry_price),
                "current_price": current_price,
                "stop_loss": _clean_float(trade.stop_loss),
                "target_price": _clean_float(trade.target_price),
                "pnl_percent": pnl_percent,
                "days_held": days_held,
                "status": trade.status,
            })
            
            # Check for alerts
            if current_price and trade.stop_loss:
                distance_to_stop = abs(current_price - trade.stop_loss)
                stop_distance_pct = (distance_to_stop / current_price) * 100 if current_price > 0 else 100
                
                if stop_distance_pct < 3:
                    alerts.append({
                        "type": "stop_approaching",
                        "ticker": trade.ticker,
                        "message": f"Price within {stop_distance_pct:.1f}% of stop loss",
                        "trade_id": trade.id,
                    })
        
        return APIResponse.ok({
            "active_trades": active_trades,
            "alerts": alerts,
            "total_open": len(open_trades),
        })
        
    except Exception as e:
        return APIResponse.error(str(e), "signals_error", 500)


def handle_recent_trades(user: Optional[dict[str, Any]] = None) -> APIResponse:
    """
    GET /api/v1/trades/recent
    
    Returns recent closed trades with performance metrics.
    
    Sample response:
    {
        "success": true,
        "timestamp": "2026-05-22T20:00:00Z",
        "data": [
            {
                "id": "swing_xyz789",
                "ticker": "MSFT",
                "direction": "long",
                "entry_price": 300.00,
                "exit_price": 320.00,
                "entry_date": "2026-05-10",
                "exit_date": "2026-05-20",
                "realized_pnl": 1000.00,
                "r_multiple": 2.0,
                "exit_reason": "target_reached"
            }
        ]
    }
    """
    try:
        user_id = _get_user_id(user)
        
        from src.swing_tracker.manager import load_trade_book, historical_trade_rows, trades_to_rows
        
        trades = load_trade_book(user_id=user_id)
        closed_trades = historical_trade_rows(trades)
        
        # Sort by exit date descending and limit to 20
        closed_trades.sort(key=lambda t: t.exit_date or datetime.min.date(), reverse=True)
        recent_trades = closed_trades[:20]
        
        # Serialize trades
        trade_rows = trades_to_rows(recent_trades)
        
        serialized_trades = []
        for tr in trade_rows:
            serialized_trades.append({
                "id": tr.get("ID", ""),
                "ticker": tr.get("Ticker", ""),
                "direction": tr.get("Direction", ""),
                "setup_type": tr.get("Setup", ""),
                "entry_price": _clean_float(tr.get("Entry")),
                "exit_price": _clean_float(tr.get("Exit")),
                "entry_date": tr.get("EntryDate", ""),
                "exit_date": tr.get("ExitDate", ""),
                "realized_pnl": _clean_float(tr.get("RealizedPnL")),
                "r_multiple": _clean_float(tr.get("RMultiple")),
                "exit_reason": tr.get("ExitReason", ""),
                "holding_days": tr.get("ActualHoldDays", 0),
            })
        
        return APIResponse.ok(serialized_trades)
        
    except Exception as e:
        return APIResponse.error(str(e), "trades_error", 500)


def handle_risk(user: Optional[dict[str, Any]] = None) -> APIResponse:
    """
    GET /api/v1/risk
    
    Returns risk metrics and analysis for the portfolio.
    
    Sample response:
    {
        "success": true,
        "timestamp": "2026-05-22T20:00:00Z",
        "data": {
            "portfolio_beta": 1.15,
            "portfolio_volatility": 18.5,
            "sharpe_ratio": 1.25,
            "max_drawdown": -12.3,
            "var_95": -2.5,
            "concentration_hhi": 0.15,
            "sector_exposure": {
                "Technology": 35.0,
                "Healthcare": 15.0,
                "Financials": 10.0
            },
            "risk_flags": []
        }
    }
    """
    try:
        user_id = _get_user_id(user)
        
        from src.portfolio_tracker.manager import load_portfolio, compute_live_values
        
        portfolio = load_portfolio("default", user_id=user_id)
        positions = portfolio.get("positions", [])
        
        if not positions:
            return APIResponse.ok({
                "portfolio_beta": None,
                "portfolio_volatility": None,
                "sharpe_ratio": None,
                "max_drawdown": None,
                "var_95": None,
                "concentration_hhi": None,
                "sector_exposure": {},
                "risk_flags": [],
                "message": "No positions to analyze",
            })
        
        holdings_df, summary = compute_live_values(portfolio)
        
        # Calculate basic risk metrics
        total_value = summary.get("TotalMarketValue", 0.0)
        
        # Calculate concentration (HHI)
        if "CurrentWeight" in holdings_df.columns:
            weights = pd.to_numeric(holdings_df["CurrentWeight"], errors="coerce").dropna()
            hhi = _clean_float(float((weights ** 2).sum()))
        else:
            hhi = None
        
        # Get sector exposure if available
        sector_exposure = {}
        try:
            from src.data.stock_universe import get_universe
            universe = get_universe()
            
            for _, row in holdings_df.iterrows():
                ticker = str(row.get("Ticker", "")).upper()
                if ticker in universe["Ticker"].values:
                    sector = universe[universe["Ticker"] == ticker]["Sector"].iloc[0]
                    if pd.notna(sector):
                        weight = float(row.get("CurrentWeight", 0) * 100) if "CurrentWeight" in row else 0
                        sector_exposure[str(sector)] = _clean_float(
                            sector_exposure.get(str(sector), 0) + weight
                        )
        except Exception:
            pass
        
        # Risk flags
        risk_flags = []
        if hhi and hhi > 0.25:
            risk_flags.append("High concentration - portfolio is not well diversified")
        
        if len(positions) < 5:
            risk_flags.append("Low number of positions - consider diversifying")
        
        # Check for large single positions
        if "CurrentWeight" in holdings_df.columns:
            max_weight = holdings_df["CurrentWeight"].max()
            if max_weight > 0.25:
                risk_flags.append(f"Single position exceeds 25% of portfolio")
        
        return APIResponse.ok({
            "portfolio_beta": None,  # Would require benchmark data
            "portfolio_volatility": None,  # Would require historical returns
            "sharpe_ratio": None,
            "max_drawdown": None,
            "var_95": None,
            "concentration_hhi": hhi,
            "effective_holdings": _clean_float(1.0 / hhi) if hhi and hhi > 0 else None,
            "sector_exposure": sector_exposure,
            "risk_flags": risk_flags,
            "total_value": _clean_float(total_value),
        })
        
    except Exception as e:
        return APIResponse.error(str(e), "risk_error", 500)


def handle_overview(user: Optional[dict[str, Any]] = None) -> APIResponse:
    """
    GET /api/v1/overview
    
    Returns a dashboard overview combining key metrics from all modules.
    
    Sample response:
    {
        "success": true,
        "timestamp": "2026-05-22T20:00:00Z",
        "data": {
            "portfolio": {
                "total_value": 125000.00,
                "total_pnl": 25000.00,
                "positions_count": 12
            },
            "trading": {
                "open_trades": 3,
                "closed_trades_30d": 5,
                "win_rate": 65.0
            },
            "market": {
                "regime": "bull",
                "spy_price": 450.00,
                "spy_change_percent": 0.56
            },
            "recent_activity": [
                {
                    "type": "trade_closed",
                    "ticker": "MSFT",
                    "pnl": 500.00,
                    "date": "2026-05-20"
                }
            ]
        }
    }
    """
    try:
        user_id = _get_user_id(user)
        
        # Portfolio summary
        from src.portfolio_tracker.manager import load_portfolio, compute_live_values
        
        portfolio = load_portfolio("default", user_id=user_id)
        holdings_df, summary = compute_live_values(portfolio) if portfolio.get("positions") else (pd.DataFrame(), {"TotalMarketValue": 0, "TotalPnL": 0})
        
        portfolio_data = {
            "total_value": _clean_float(summary.get("TotalMarketValue", 0.0)),
            "total_pnl": _clean_float(summary.get("TotalPnL", 0.0)),
            "positions_count": len(portfolio.get("positions", [])),
        }
        
        # Trading summary
        from src.swing_tracker.manager import load_trade_book, open_trade_rows, historical_trade_rows
        
        trades = load_trade_book(user_id=user_id)
        open_trades = open_trade_rows(trades)
        closed_trades = historical_trade_rows(trades)
        
        # Calculate win rate
        winning_trades = [t for t in closed_trades if t.realized_pnl and t.realized_pnl > 0]
        win_rate = _clean_float((len(winning_trades) / len(closed_trades)) * 100) if closed_trades else 0.0
        
        trading_data = {
            "open_trades": len(open_trades),
            "closed_trades_30d": len(closed_trades),  # Simplified - would filter by date
            "win_rate": win_rate,
        }
        
        # Market overview
        market_data = {"regime": "unknown", "spy_price": None, "spy_change_percent": None}
        spy_snapshot = _fetch_price_snapshot(["SPY"], period="5d").get("SPY", {})
        if spy_snapshot:
            change_pct = spy_snapshot.get("change_percent")
            market_data = {
                "regime": "bull" if (change_pct is not None and change_pct >= 0) else "bear",
                "spy_price": spy_snapshot.get("price"),
                "spy_change_percent": change_pct,
            }
        
        # Recent activity
        recent_activity = []
        
        # Add recent closed trades
        for trade in closed_trades[:5]:
            recent_activity.append({
                "type": "trade_closed",
                "ticker": trade.ticker,
                "pnl": _clean_float(trade.realized_pnl),
                "date": trade.exit_date.isoformat() if trade.exit_date else "",
            })
        
        # Sort by date
        recent_activity.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        return APIResponse.ok({
            "portfolio": portfolio_data,
            "trading": trading_data,
            "market": market_data,
            "recent_activity": recent_activity[:10],
        })
        
    except Exception as e:
        return APIResponse.error(str(e), "overview_error", 500)
