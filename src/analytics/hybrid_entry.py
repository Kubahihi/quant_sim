"""Report-ready comparison of lump sum, calendar DCA and a fixed hybrid entry rule.

This is an execution-risk study, not a price forecast.  It deliberately has no
parameter search: all thresholds are disclosed in ``HybridEntryConfig`` and the
same rule is applied to every historical starting point.
"""
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import pandas as pd

from src.analytics.deployment import day


@dataclass(frozen=True)
class HybridEntryConfig:
    first_fraction: float = .5
    minimum_wait_sessions: int = 1
    maximum_wait_sessions: int = 20
    fixed_second_session: int = 5
    forward_horizon_sessions: int = 63
    pullback_threshold: float = .05
    short_trend_window: int = 50
    long_trend_window: int = 200
    relative_strength_window: int = 20
    annual_cash_rate: float = .03
    cost_bps: float = 10.
    minimum_cases: int = 12

    def validate(self):
        if any(type(value) is not int for value in (
                self.minimum_wait_sessions, self.maximum_wait_sessions, self.fixed_second_session,
                self.forward_horizon_sessions,
                self.short_trend_window, self.long_trend_window,
                self.relative_strength_window, self.minimum_cases)):
            raise ValueError('Hybrid entry session counts must be integers.')
        if (not 1 <= self.minimum_wait_sessions < self.maximum_wait_sessions <= 63 or
                not 2 <= self.fixed_second_session <= self.maximum_wait_sessions or
                not 21 <= self.forward_horizon_sessions <= 252 or
                not 10 <= self.relative_strength_window <= self.short_trend_window < self.long_trend_window or
                not 4 <= self.minimum_cases <= 100):
            raise ValueError('Hybrid entry session settings are outside the supported range.')
        for value, name in ((self.first_fraction, 'first fraction'), (self.pullback_threshold, 'pullback threshold'),
                            (self.annual_cash_rate, 'annual cash rate'), (self.cost_bps, 'cost bps')):
            if not isfinite(value):
                raise ValueError(f'{name} must be finite.')
        if not 0 < self.first_fraction < 1 or not 0 < self.pullback_threshold <= .25 or not -1 <= self.annual_cash_rate <= 1 or not 0 <= self.cost_bps <= 1000:
            raise ValueError('Hybrid entry fractions, threshold, cash rate or costs are unsupported.')


@dataclass
class HybridEntrySignal:
    as_of: str
    action: str
    condition: str
    next_review: str | None
    forced_completion: str
    pullback_from_20d_high: float | None
    relative_return: float | None
    long_trend_confirmed: bool | None
    config: dict
    warnings: list[str]

    def to_dict(self):
        return asdict(self)


@dataclass
class HybridEntryStudy:
    summary: pd.DataFrame
    cases: pd.DataFrame
    current_signal: HybridEntrySignal
    warnings: list[str]
    config: dict


def summarize_hybrid_cases(cases):
    """Use the same result definitions for single-security and portfolio studies."""
    strategies = {'Lump sum': 'lump_sum_wealth', 'Fixed DCA': 'fixed_dca_wealth', 'Hybrid DCA': 'hybrid_wealth'}
    summary_rows = []
    for name, column in strategies.items():
        values = cases[column] if not cases.empty else pd.Series(dtype=float)
        waits = (pd.Series(0, index=cases.index) if name == 'Lump sum' else
                 cases['fixed_wait_sessions'] if name == 'Fixed DCA' else cases['wait_sessions']) if not cases.empty else pd.Series(dtype=float)
        summary_rows.append(dict(strategy=name, cases=len(values), non_overlapping_cases=len(values),
            mean_terminal_return=(float(values.mean() - 1) if len(values) else np.nan),
            median_terminal_return=(float(values.median() - 1) if len(values) else np.nan),
            worst_terminal_return=(float(values.min() - 1) if len(values) else np.nan),
            best_terminal_return=(float(values.max() - 1) if len(values) else np.nan),
            mean_wait_sessions=(float(waits.mean()) if len(waits) else np.nan),
            win_rate_vs_lump=(float((values > cases.lump_sum_wealth).mean())
                              if len(values) and name != 'Lump sum' else np.nan)))
    summary = pd.DataFrame(summary_rows)
    if not cases.empty:
        hybrid = summary.strategy == 'Hybrid DCA'
        summary.loc[hybrid, 'mean_advantage_vs_lump'] = float(cases.hybrid_vs_lump.mean())
        summary.loc[hybrid, 'mean_advantage_vs_fixed'] = float(cases.hybrid_vs_fixed.mean())
        summary.loc[hybrid, 'win_rate_vs_lump'] = float((cases.hybrid_vs_lump > 0).mean())
        summary.loc[hybrid, 'average_wait_sessions'] = float(cases.wait_sessions.mean())
    return summary


def _clean_pair(prices, benchmark_prices, cutoff):
    def clean(series, name):
        if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f'{name} adjusted close history is required.')
        data = pd.to_numeric(series.copy(), errors='coerce')
        data.index = data.index.tz_localize(None).normalize()
        data = data.loc[:cutoff].where(data > 0).dropna().sort_index()
        if data.index.has_duplicates:
            raise ValueError(f'{name} needs unique session dates.')
        return data
    security, benchmark = clean(prices, 'Security'), clean(benchmark_prices, 'Benchmark')
    common = security.index.intersection(benchmark.index)
    if len(common) < 300:
        raise ValueError('At least 300 common adjusted-price observations are required.')
    return security.loc[common], benchmark.loc[common]


def _trigger(prices, benchmark, index, config):
    if index < config.long_trend_window:
        return None, None, None, None
    price = float(prices.iloc[index])
    high = float(prices.iloc[index - config.relative_strength_window + 1:index + 1].max())
    pullback = price / high - 1
    relative = (float(prices.iloc[index] / prices.iloc[index - config.relative_strength_window] - 1) -
                float(benchmark.iloc[index] / benchmark.iloc[index - config.relative_strength_window] - 1))
    long_trend = price >= float(prices.iloc[index - config.long_trend_window + 1:index + 1].mean())
    short_trend = price >= float(prices.iloc[index - config.short_trend_window + 1:index + 1].mean())
    if long_trend and pullback <= -config.pullback_threshold:
        return 'Controlled pullback in long-term trend', pullback, relative, long_trend
    if long_trend and short_trend and relative >= 0:
        return 'Trend and relative-strength confirmation', pullback, relative, long_trend
    return None, pullback, relative, long_trend


def hybrid_entry_signal(prices, benchmark_prices, *, as_of, config=None):
    config = config or HybridEntryConfig()
    config.validate()
    today = day(as_of)
    security, benchmark = _clean_pair(prices, benchmark_prices, today)
    condition, pullback, relative, long_trend = _trigger(security, benchmark, len(security) - 1, config)
    if condition:
        action, next_review = 'Condition met for remaining-tranche review', None
    else:
        action = 'No condition; re-evaluate next session'
        next_review = str((today + pd.offsets.BDay(1)).date())
    return HybridEntrySignal(str(today.date()), action, condition or 'No entry condition today', next_review,
        f'By observed session {config.maximum_wait_sessions}, counting first execution as session 1',
        pullback, relative, long_trend,
        asdict(config), [
            'The rule is a disclosed execution discipline, not a forecast of lower prices.',
            'The forced completion date prevents indefinite cash drag and must remain subject to the approved thesis and portfolio checks.',
        ])


def study_hybrid_entry(prices, benchmark_prices, *, as_of, config=None):
    """Compare three execution choices using non-overlapping historical starting points.

    Each case has the same terminal date. The hybrid keeps the second fraction in
    cash until its first condition or its fixed deadline, and all strategies pay
    the same stated proportional execution cost.
    """
    config = config or HybridEntryConfig()
    config.validate()
    cutoff = day(as_of)
    security, benchmark = _clean_pair(prices, benchmark_prices, cutoff)
    current = hybrid_entry_signal(security, benchmark, as_of=cutoff, config=config)
    span = config.maximum_wait_sessions + config.forward_horizon_sessions + 1
    starts = range(config.long_trend_window, len(security) - span, span)
    fee, rows = config.cost_bps / 10_000, []
    for start in starts:
        initial_execution = start + 1
        deadline_execution = initial_execution + config.maximum_wait_sessions - 1
        fixed_execution = initial_execution + config.fixed_second_session - 1
        terminal = start + span
        condition, second_execution = 'Forced completion', deadline_execution
        for review in range(initial_execution, deadline_execution):
            if review < initial_execution + config.minimum_wait_sessions - 1:
                continue
            candidate, *_ = _trigger(security, benchmark, review, config)
            if candidate:
                condition, second_execution = candidate, review + 1
                break
        if terminal >= len(security) or second_execution >= terminal:
            continue
        initial_price = float(security.iloc[initial_execution])
        fixed_price = float(security.iloc[fixed_execution])
        hybrid_price = float(security.iloc[second_execution])
        terminal_price = float(security.iloc[terminal])
        wait_fixed = fixed_execution - initial_execution
        wait_hybrid = second_execution - initial_execution
        lump = (1 - fee) * terminal_price / initial_price
        fixed = config.first_fraction * (1 - fee) * terminal_price / initial_price + (1 - config.first_fraction) * (
            (1 + config.annual_cash_rate) ** (wait_fixed / 252) * (1 - fee) * terminal_price / fixed_price)
        hybrid = config.first_fraction * (1 - fee) * terminal_price / initial_price + (1 - config.first_fraction) * (
            (1 + config.annual_cash_rate) ** (wait_hybrid / 252) * (1 - fee) * terminal_price / hybrid_price)
        rows.append(dict(start_date=str(security.index[start].date()), first_entry_date=str(security.index[initial_execution].date()),
            fixed_second_entry_date=str(security.index[fixed_execution].date()), fixed_wait_sessions=wait_fixed,
            second_entry_date=str(security.index[second_execution].date()),
            terminal_date=str(security.index[terminal].date()), trigger=condition, wait_sessions=wait_hybrid,
            lump_sum_wealth=lump, fixed_dca_wealth=fixed, hybrid_wealth=hybrid,
            hybrid_vs_lump=hybrid / lump - 1, hybrid_vs_fixed=hybrid / fixed - 1))
    cases = pd.DataFrame(rows)
    summary = summarize_hybrid_cases(cases)
    enough = len(cases) >= config.minimum_cases
    warnings = [
        'Starting points are non-overlapping but remain a single-security historical study, not independent proof.',
        'The study uses adjusted close proxies, a stated cash rate and stated proportional costs; it excludes taxes, liquidity and market impact.',
        'Do not select this rule because it wins one chart. Use the same specification across candidates and retain unfavourable results.',
    ]
    if not enough:
        warnings.append(f'Only {len(cases)} complete non-overlapping cases; at least {config.minimum_cases} are required for report use.')
    return HybridEntryStudy(summary, cases, current, warnings, asdict(config))
