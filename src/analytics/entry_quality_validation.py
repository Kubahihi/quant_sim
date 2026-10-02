"""Descriptive forward-outcome study; does not tune weights or claim causality."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.entry_quality import calculate_entry_quality, _day, _prepare
from src.analytics.returns import calculate_returns
from src.analytics.risk_metrics import calculate_max_drawdown


@dataclass
class EntryValidationResult:
    observations: pd.DataFrame
    summary: pd.DataFrame
    warnings: list[str]


def validate_entry_quality(ticker, prices, benchmark_prices, *, dates=None,
                           benchmark="SPY", config=None):
    """Signal at date close, hypothetical entry next observed session close.

    Horizons contain 21/63/126 complete post-entry returns. Unfinished or invalid
    paths remain NaN, with separate counts in the summary. Default monthly sampling.
    """
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.has_duplicates:
        raise ValueError("Unique dated daily prices required.")
    p = prices.copy().sort_index()
    p.index = p.index.tz_localize(None).normalize()
    if p.index.has_duplicates or p.index.hasnans:
        raise ValueError("Unique valid session dates required.")
    if dates is None:
        dates = p.groupby(p.index.to_period("M")).tail(1).index
    rows = []
    for date in dates:
        date = _day(date)
        result = calculate_entry_quality(ticker, p, benchmark_prices, benchmark=benchmark,
                                         as_of=date, config=config)
        score = result.entry_score
        row = dict(date=date, entry_score=score, rating=result.rating,
                   bucket=None if score is None else ("0–39" if score < 40 else "40–59" if score < 60 else "60–79" if score < 80 else "80–100"),
                   warnings="; ".join(result.warnings))
        start = p.index.searchsorted(date, side="right")
        row['entry_date'] = p.index[start] if start < len(p) else pd.NaT
        for h in (21, 63, 126):
            for metric in ('return', 'max_drawdown', 'downside_deviation'):
                row[f'{metric}_{h}d'] = np.nan
            path = p.iloc[start:start + h + 1]
            if score is None or len(path) != h + 1:
                continue
            clean, issues = _prepare(path, path.index[-1], ticker, 7)
            if issues:
                row['warnings'] += "; Invalid forward path: " + "; ".join(issues)
                continue
            returns = calculate_returns(clean)
            row[f'return_{h}d'] = float(clean.iloc[-1] / clean.iloc[0] - 1)
            row[f'max_drawdown_{h}d'] = calculate_max_drawdown(returns)
            row[f'downside_deviation_{h}d'] = float(np.sqrt(np.mean(np.minimum(returns, 0.) ** 2)) * np.sqrt(252))
        rows.append(row)
    observations = pd.DataFrame(rows)
    metrics = [f'{metric}_{h}d' for h in (21, 63, 126)
               for metric in ('return', 'max_drawdown', 'downside_deviation')]
    summary = (observations.groupby('bucket')[metrics].agg(['count', 'mean', 'median'])
               .reindex(['0–39', '40–59', '60–79', '80–100'])) if rows else pd.DataFrame()
    return EntryValidationResult(observations, summary, [
        "Descriptive study only: overlapping horizons are dependent observations.",
        "Current adjusted histories can be revised; use point-in-time data and an unbiased security universe for report-grade evidence.",
        "Next-session close execution excludes costs, slippage, cash yield and opportunity cost; this is not a deployment-strategy backtest.",
    ])
