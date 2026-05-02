from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from src.analytics.scoring import compute_weighted_factor_score


def _normalize_text_series(data: pd.Series | pd.DataFrame) -> pd.Series:
    """Normalize arbitrary text-like column values into lowercase stripped strings."""
    series = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
    return series.map(lambda value: "" if pd.isna(value) else str(value).strip().lower())


def _chunked(values: list[str], size: int):
    """Yield fixed-size chunks from a list."""
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _ensure_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _apply_range(
    frame: pd.DataFrame,
    column: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> pd.DataFrame:
    if column not in frame.columns:
        return frame

    data = frame.copy()
    values = pd.to_numeric(data[column], errors="coerce")
    mask = pd.Series(True, index=data.index)
    if minimum is not None:
        mask &= values >= minimum
    if maximum is not None:
        mask &= values <= maximum
    mask &= values.notna()
    return data.loc[mask].copy()


def _apply_filter_map(
    frame: pd.DataFrame,
    filter_map: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> pd.DataFrame:
    if not filter_map:
        return frame
    output = frame.copy()
    for column, (minimum, maximum) in filter_map.items():
        output = _apply_range(output, column, minimum=minimum, maximum=maximum)
        if output.empty:
            return output
    return output


def apply_liquidity_filters(
    df: pd.DataFrame,
    min_avg_volume: float | None = None,
    min_market_cap: float | None = None,
    min_price: float | None = None,
) -> pd.DataFrame:
    """Apply cheap liquidity constraints suitable for full-universe screening."""
    output = _ensure_numeric(df, ["AvgVolume", "MarketCap", "Price"])
    if min_avg_volume is not None:
        output = _apply_range(output, "AvgVolume", minimum=min_avg_volume)
    if min_market_cap is not None:
        output = _apply_range(output, "MarketCap", minimum=min_market_cap)
    if min_price is not None:
        output = _apply_range(output, "Price", minimum=min_price)
    return output


def apply_valuation_filters(
    df: pd.DataFrame,
    valuation_filters: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> pd.DataFrame:
    """Apply valuation constraints (PE, ForwardPE, PEG, etc.)."""
    return _apply_filter_map(df, valuation_filters)


def apply_growth_filters(
    df: pd.DataFrame,
    growth_filters: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> pd.DataFrame:
    """Apply growth constraints (revenue, earnings growth)."""
    return _apply_filter_map(df, growth_filters)


def apply_quality_filters(
    df: pd.DataFrame,
    quality_filters: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> pd.DataFrame:
    """Apply quality constraints (ROE, ROA)."""
    return _apply_filter_map(df, quality_filters)


def apply_momentum_filters(
    df: pd.DataFrame,
    momentum_filters: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> pd.DataFrame:
    """Apply momentum constraints (e.g., 52-week return)."""
    return _apply_filter_map(df, momentum_filters)


def apply_classic_filters(
    df: pd.DataFrame,
    market_cap_range: tuple[float, float] | None = None,
    sectors: Sequence[str] | None = None,
    industries: Sequence[str] | None = None,
    exchanges: Sequence[str] | None = None,
    beta_range: tuple[float, float] | None = None,
    price_range: tuple[float, float] | None = None,
    min_avg_volume: float | None = None,
    valuation_filters: Mapping[str, tuple[float | None, float | None]] | None = None,
    growth_filters: Mapping[str, tuple[float | None, float | None]] | None = None,
    quality_filters: Mapping[str, tuple[float | None, float | None]] | None = None,
    momentum_filters: Mapping[str, tuple[float | None, float | None]] | None = None,
    dividend_filters: Mapping[str, tuple[float | None, float | None]] | None = None,
    liquidity_prefilter: bool = False,
) -> pd.DataFrame:
    """
    First-stage screener over the full cached universe.

    This stage should remain cheap: it only uses cached universe columns.
    """
    if df.empty:
        return df.copy()

    output = _ensure_numeric(
        df,
        [
            "MarketCap",
            "Beta",
            "Price",
            "AvgVolume",
            "PE",
            "ForwardPE",
            "PEG",
            "ROE",
            "ROA",
            "RevenueGrowth",
            "EarningsGrowth",
            "DividendYield",
            "Return52W",
        ],
    )

    if sectors and "Sector" in output.columns:
        sector_set = {item.strip().lower() for item in sectors if str(item).strip()}
        include_unknown = "unknown" in sector_set
        sector_set.discard("unknown")
        normalized_sector = _normalize_text_series(output["Sector"])
        sector_mask = normalized_sector.isin(sector_set) if sector_set else pd.Series(False, index=output.index)
        if include_unknown:
            sector_mask = sector_mask | normalized_sector.eq("")
        output = output[sector_mask]
    if industries and "Industry" in output.columns:
        industry_set = {item.strip().lower() for item in industries if str(item).strip()}
        include_unknown = "unknown" in industry_set
        industry_set.discard("unknown")
        normalized_industry = _normalize_text_series(output["Industry"])
        industry_mask = normalized_industry.isin(industry_set) if industry_set else pd.Series(False, index=output.index)
        if include_unknown:
            industry_mask = industry_mask | normalized_industry.eq("")
        output = output[industry_mask]
    if exchanges and "Exchange" in output.columns:
        exchange_set = {item.strip().lower() for item in exchanges if str(item).strip()}
        include_unknown = "unknown" in exchange_set
        exchange_set.discard("unknown")
        normalized_exchange = _normalize_text_series(output["Exchange"])
        exchange_mask = normalized_exchange.isin(exchange_set) if exchange_set else pd.Series(False, index=output.index)
        if include_unknown:
            exchange_mask = exchange_mask | normalized_exchange.eq("")
        output = output[exchange_mask]

    if market_cap_range:
        output = _apply_range(output, "MarketCap", market_cap_range[0], market_cap_range[1])
    if beta_range:
        output = _apply_range(output, "Beta", beta_range[0], beta_range[1])
    if price_range:
        output = _apply_range(output, "Price", price_range[0], price_range[1])

    if liquidity_prefilter:
        output = apply_liquidity_filters(
            output,
            min_avg_volume=min_avg_volume,
            min_market_cap=(market_cap_range[0] if market_cap_range else None),
            min_price=(price_range[0] if price_range else None),
        )
    elif min_avg_volume is not None:
        output = _apply_range(output, "AvgVolume", minimum=min_avg_volume)

    output = apply_valuation_filters(output, valuation_filters)
    output = apply_growth_filters(output, growth_filters)
    output = apply_quality_filters(output, quality_filters)
    output = apply_momentum_filters(output, momentum_filters)
    output = _apply_filter_map(output, dividend_filters)

    return output.reset_index(drop=True)


def _rank_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    ranks = numeric.rank(method="average", pct=True)
    if higher_is_better:
        score = ranks
    else:
        score = 1.0 - ranks
    return score.fillna(0.5)


def calculate_quant_score(
    df: pd.DataFrame,
    weight_preferences: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """
    Calculate transparent weighted stock score from available columns.

    Missing fields receive neutral ranks to preserve robustness.
    """
    if df.empty:
        output = df.copy()
        output["QuantScore"] = np.nan
        return output

    output = df.copy()
    factor_scores: dict[str, pd.Series] = {}

    factor_scores["value"] = pd.concat(
        [
            _rank_score(output.get("PE", pd.Series(dtype=float)), higher_is_better=False),
            _rank_score(output.get("ForwardPE", pd.Series(dtype=float)), higher_is_better=False),
            _rank_score(output.get("PEG", pd.Series(dtype=float)), higher_is_better=False),
        ],
        axis=1,
    ).mean(axis=1)

    factor_scores["growth"] = pd.concat(
        [
            _rank_score(output.get("RevenueGrowth", pd.Series(dtype=float)), higher_is_better=True),
            _rank_score(output.get("EarningsGrowth", pd.Series(dtype=float)), higher_is_better=True),
        ],
        axis=1,
    ).mean(axis=1)

    factor_scores["quality"] = pd.concat(
        [
            _rank_score(output.get("ROE", pd.Series(dtype=float)), higher_is_better=True),
            _rank_score(output.get("ROA", pd.Series(dtype=float)), higher_is_better=True),
        ],
        axis=1,
    ).mean(axis=1)

    factor_scores["momentum"] = pd.concat(
        [
            _rank_score(output.get("Return52W", pd.Series(dtype=float)), higher_is_better=True),
            _rank_score(output.get("RSI", pd.Series(dtype=float)), higher_is_better=True),
            _rank_score(output.get("MACD", pd.Series(dtype=float)), higher_is_better=True),
        ],
        axis=1,
    ).mean(axis=1)

    factor_scores["stability"] = pd.concat(
        [
            _rank_score(output.get("Beta", pd.Series(dtype=float)), higher_is_better=False),
            _rank_score(output.get("Volatility", pd.Series(dtype=float)), higher_is_better=False),
            _rank_score(output.get("Drawdown", pd.Series(dtype=float)).abs(), higher_is_better=False),
        ],
        axis=1,
    ).mean(axis=1)

    factor_scores["dividend"] = _rank_score(
        output.get("DividendYield", pd.Series(dtype=float)),
        higher_is_better=True,
    )

    default_weights = {
        "value": 1.0,
        "growth": 1.0,
        "quality": 1.0,
        "momentum": 1.0,
        "stability": 1.0,
        "dividend": 0.5,
    }
    if weight_preferences:
        for key, value in weight_preferences.items():
            if key in default_weights:
                default_weights[key] = max(0.0, float(value))

    weight_sum = float(sum(default_weights.values()))
    if weight_sum <= 0:
        default_weights = {key: 1.0 for key in default_weights}
        weight_sum = float(len(default_weights))

    normalized_weights = {key: value / weight_sum for key, value in default_weights.items()}
    quant_score = compute_weighted_factor_score(
        factors=factor_scores,
        weights=normalized_weights,
        neutral_value=0.5,
    )

    output["QuantScore"] = (quant_score * 100.0).round(2)
    return output


def rank_stocks(
    df: pd.DataFrame,
    sort_by: str = "QuantScore",
    ascending: bool = False,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Sort and rank stock candidates by preferred score/field."""
    if df.empty:
        output = df.copy()
        output["Rank"] = pd.Series(dtype="Int64")
        return output

    output = df.copy()
    if sort_by not in output.columns:
        sort_by = "QuantScore" if "QuantScore" in output.columns else output.columns[0]
    output = output.sort_values(by=[sort_by, "Ticker"], ascending=[ascending, True], na_position="last")
    if top_n is not None and top_n > 0:
        output = output.head(top_n)
    output = output.reset_index(drop=True)
    output["Rank"] = np.arange(1, len(output) + 1)
    return output


def _compute_rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return float(rsi.iloc[-1]) if not rsi.dropna().empty else np.nan


def _compute_macd(close: pd.Series) -> float:
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    return float(macd_line.iloc[-1]) if not macd_line.dropna().empty else np.nan


def _extract_history_for_symbol(history: pd.DataFrame, symbol: str, chunk_size: int) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    if isinstance(history.columns, pd.MultiIndex):
        first_level = history.columns.get_level_values(0)
        second_level = history.columns.get_level_values(1)
        if symbol in first_level:
            subset = history[symbol]
            return subset if isinstance(subset, pd.DataFrame) else subset.to_frame()
        if symbol in second_level:
            return history.xs(symbol, axis=1, level=1, drop_level=True)
        return pd.DataFrame()

    if chunk_size == 1:
        return history
    return pd.DataFrame()


def apply_technical_indicators(
    df_subset: pd.DataFrame,
    history_period: str = "1y",
    chunk_size: int = 120,
) -> pd.DataFrame:
    """
    Compute expensive technical indicators only for a pre-filtered subset.

    Indicators:
    - RSI(14)
    - MACD(12, 26)
    - Annualized volatility
    - Max drawdown
    """
    if df_subset.empty or "Ticker" not in df_subset.columns:
        output = df_subset.copy()
        for column in ["RSI", "MACD", "Volatility", "Drawdown"]:
            output[column] = np.nan
        return output

    output = df_subset.copy()
    normalized_tickers = output["Ticker"].map(
        lambda value: "" if pd.isna(value) else str(value).strip().upper()
    )
    tickers = normalized_tickers[normalized_tickers.ne("")].drop_duplicates().tolist()

    indicator_rows: list[dict[str, Any]] = []
    for chunk in _chunked(tickers, chunk_size):
        try:
            history = yf.download(
                tickers=chunk,
                period=history_period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception:
            history = pd.DataFrame()

        for symbol in chunk:
            row = {"Ticker": symbol, "RSI": np.nan, "MACD": np.nan, "Volatility": np.nan, "Drawdown": np.nan}
            try:
                symbol_history = _extract_history_for_symbol(history, symbol, len(chunk))
                close_column = None
                for candidate in ["Adj Close", "Close", "adj close", "close"]:
                    if candidate in symbol_history.columns:
                        close_column = candidate
                        break
                if close_column is None:
                    indicator_rows.append(row)
                    continue

                close = pd.to_numeric(symbol_history[close_column], errors="coerce").dropna()
                if len(close) < 30:
                    indicator_rows.append(row)
                    continue

                returns = close.pct_change().dropna()
                if returns.empty:
                    indicator_rows.append(row)
                    continue

                cumulative = (1 + returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = cumulative / running_max - 1.0

                row["RSI"] = _compute_rsi(close)
                row["MACD"] = _compute_macd(close)
                row["Volatility"] = float(returns.std() * np.sqrt(252))
                row["Drawdown"] = float(drawdown.min())
            except Exception:
                pass
            indicator_rows.append(row)

    indicator_frame = pd.DataFrame(indicator_rows).drop_duplicates(subset=["Ticker"], keep="first")
    merged = output.merge(indicator_frame, on="Ticker", how="left")
    return merged
