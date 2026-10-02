"""Coverage-aware portfolio implementation evidence.

The module deliberately separates an execution comparison from a claim that a
team can forecast prices.  It uses only constituents with enough *pre-existing*
history for the disclosed rule, reports the weight that this represents, and
never relabels a partial basket result as whole-portfolio evidence.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.analytics.deployment import day
from src.analytics.hybrid_entry import (
    HybridEntryConfig, HybridEntryStudy, _trigger, hybrid_entry_signal, summarize_hybrid_cases,
)


MIN_PORTFOLIO_HISTORY_SESSIONS = 423
MIN_COMMON_STUDY_SESSIONS = 300


@dataclass
class PortfolioEntryStudy:
    study: HybridEntryStudy | None
    coverage: pd.DataFrame
    evidence_weight: float
    full_portfolio_covered: bool
    warnings: list[str]
    config: dict
    latest_common_session: str | None = None


@dataclass(frozen=True)
class TrancheReview:
    status: str
    sessions_since_first: int | None
    explanation: str


def review_second_tranche(session_dates, *, as_of, first_execution_date=None,
                          first_execution_reference='', final_recorded=False,
                          final_execution_reference='',
                          condition_met=False, signal_fresh=True, policy_approved=False,
                          config=None):
    """Review a recorded first tranche against a fixed exchange-session deadline."""
    config = config or HybridEntryConfig(first_fraction=.75, maximum_wait_sessions=10,
                                         forward_horizon_sessions=21)
    config.validate()
    dates = pd.DatetimeIndex(session_dates).tz_localize(None).normalize().unique().sort_values()
    dates = dates[dates <= day(as_of)]
    if first_execution_date is None:
        return TrancheReview('First tranche not recorded', None,
                             'Record the initial execution and its official reference before reviewing the final 25%.')
    if not str(first_execution_reference).strip():
        return TrancheReview('Execution reference missing', None,
                             'Enter the official reference for the first execution before using this review.')
    if not policy_approved:
        return TrancheReview('Policy approval not recorded', None,
                             'Confirm the team approved these weights and the 75/25 rule before reviewing completion.')
    first = day(first_execution_date)
    if first not in dates:
        return TrancheReview('Execution date needs verification', None,
                             'The entered date is not in this ticker\'s observed sessions; verify the trade and price history.')
    elapsed = int((dates > first).sum())
    if final_recorded:
        if not str(final_execution_reference).strip():
            return TrancheReview('Final execution reference missing', elapsed,
                                 'Enter the official reference before marking the final tranche complete.')
        return TrancheReview('Final tranche recorded', elapsed,
                             'No further entry review is due for this position.')
    if elapsed + 1 < config.minimum_wait_sessions:
        return TrancheReview('Wait for separate session', elapsed,
                             f'The second execution cannot occur until session '
                             f'{config.minimum_wait_sessions} after the first execution.')
    if elapsed >= config.maximum_wait_sessions - 2:
        status = ('Completion review overdue' if elapsed >= config.maximum_wait_sessions - 1
                  else 'Completion review due next session')
        return TrancheReview(status, elapsed,
                             'The session-ten execution deadline has arrived or passed. Verify thesis, cash, limits and the '
                             'official ledger before deciding on the remaining allocation.')
    if not signal_fresh:
        return TrancheReview('Await current market data', elapsed,
                             'The common basket condition is incomplete or stale; refresh data before a conditional review.')
    if condition_met:
        return TrancheReview('Conditional review due next session', elapsed,
                             'The disclosed basket condition is met. Verify thesis, cash, limits and the '
                             'official ledger before deciding on the remaining allocation.')
    return TrancheReview('Wait and review next session', elapsed,
                         'No condition is met; retain the final 25% until a later review or session ten.')


def _clean_history(prices, cutoff):
    if not isinstance(prices, pd.Series) or not isinstance(prices.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    data = pd.to_numeric(prices.copy(), errors='coerce')
    data.index = data.index.tz_localize(None).normalize()
    return data.loc[:cutoff].where(data > 0).dropna().sort_index()


def deployment_path(cases, *, first_fraction, maximum_wait_sessions, fixed_second_session=5):
    """Show the fixed paths and the observed average hybrid deployment by session."""
    days = np.arange(maximum_wait_sessions + 1)
    waits = cases['wait_sessions'].to_numpy(dtype=float) if not cases.empty else np.array([])
    hybrid = (first_fraction + (1 - first_fraction) *
              np.array([(waits <= review).mean() for review in days])) if len(waits) else np.full(len(days), np.nan)
    return pd.DataFrame({
        'session': days,
        'lump_sum': np.ones(len(days)),
        'fixed_dca': np.where(days >= fixed_second_session - 1, 1.0, first_fraction),
        'hybrid_historical_average': hybrid,
    })


def study_portfolio_entry(histories, weights, benchmark_prices, *, as_of, quote_currencies,
                          fx_to_usd, benchmark_currency='USD', config=None):
    """Study a fixed weighted basket, while disclosing incomplete coverage.

    ``histories`` maps ticker to adjusted-close history and ``weights`` maps the
    same tickers to target weights. Quote histories are converted to USD with
    same-day or earlier FX observations before portfolio returns are built.
    A fixed-weight daily return index supplies the shared entry condition. Each
    historical result buys and holds the individual securities at the specified
    target weights, without rebalancing after entry.
    """
    config = config or HybridEntryConfig(
        first_fraction=.75, maximum_wait_sessions=10, forward_horizon_sessions=21,
    )
    config.validate()
    cutoff = day(as_of)
    if not isinstance(histories, dict) or not histories:
        raise ValueError('At least one portfolio history is required.')
    symbols = [str(symbol).upper() for symbol in histories]
    if set(symbols) != {str(symbol).upper() for symbol in weights}:
        raise ValueError('Portfolio histories and weights must use the same tickers.')
    currencies = {str(symbol).upper(): str(currency).upper() for symbol, currency in quote_currencies.items()}
    if set(symbols) != set(currencies):
        raise ValueError('A quote currency is required for every portfolio ticker.')
    if benchmark_currency.upper() not in {'USD', 'GBP', 'EUR'} or any(
            currency not in {'USD', 'GBP', 'EUR'} for currency in currencies.values()):
        raise ValueError('Only USD, GBP and EUR quote currencies are supported in this comparison.')
    numeric_weights = {str(symbol).upper(): float(value) for symbol, value in weights.items()}
    if any(not np.isfinite(value) or value <= 0 for value in numeric_weights.values()):
        raise ValueError('Every portfolio target weight must be positive and finite.')
    total = sum(numeric_weights.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError('Portfolio target weights must sum to 100%.')

    cleaned_fx = {str(currency).upper(): _clean_history(series, cutoff)
                  for currency, series in fx_to_usd.items()}
    required_fx = {currency for currency in [*currencies.values(), benchmark_currency.upper()] if currency != 'USD'}
    missing_fx = required_fx - set(cleaned_fx)
    if missing_fx or any(cleaned_fx[currency].empty for currency in required_fx):
        raise ValueError('Missing USD conversion history for: ' + ', '.join(sorted(missing_fx or required_fx)) + '.')

    def to_usd(series, currency):
        cleaned = _clean_history(series, cutoff)
        if currency == 'USD' or cleaned.empty:
            return cleaned
        rates = cleaned_fx[currency].reindex(cleaned.index, method='ffill', tolerance=pd.Timedelta(days=4))
        return cleaned.mul(rates).dropna()

    raw_clean = {str(symbol).upper(): _clean_history(series, cutoff)
                 for symbol, series in histories.items()}
    clean = {symbol: to_usd(series, currencies[symbol]) for symbol, series in raw_clean.items()}
    rows = []
    eligible = []
    for symbol, series in clean.items():
        observations = len(raw_clean[symbol])
        included = observations >= MIN_PORTFOLIO_HISTORY_SESSIONS
        rows.append(dict(ticker=symbol, quote_currency=currencies[symbol], target_weight=numeric_weights[symbol],
                         observations=observations, usable_usd_sessions=len(series),
                         required_observations=MIN_PORTFOLIO_HISTORY_SESSIONS, included_in_evidence=included,
                         evidence_status='Included' if included else 'Data-limited'))
        if included:
            eligible.append(symbol)
    coverage = pd.DataFrame(rows).sort_values('ticker').reset_index(drop=True)
    evidence_weight = float(sum(numeric_weights[symbol] for symbol in eligible))
    full_covered = bool(len(eligible) == len(symbols))
    warnings = [
        'This is an implementation comparison, not a forecast or proof of timing alpha.',
        f'{MIN_PORTFOLIO_HISTORY_SESSIONS} observed sessions is an inclusion threshold, not the '
        f'{config.minimum_cases}-case threshold for report evidence.',
        'The historical basket uses today\'s selected holdings and target weights, so it is exposed to selection bias.',
        'Quote histories are translated into USD with contemporaneous or earlier daily FX closes; actual execution FX rates and spreads may differ.',
        'The comparison uses adjusted-close proxies, a stated cash rate and proportional costs; it excludes taxes, liquidity and market impact.',
    ]
    if not full_covered:
        omitted = ', '.join(coverage.loc[~coverage.included_in_evidence, 'ticker'])
        warnings.append(f'Data-limited holdings are not included in the historical basket: {omitted}. '
                        f'The evidence covers {evidence_weight:.1%} of entered target weight, not the full portfolio.')
    if not eligible:
        warnings.append('No holding has enough history for the disclosed non-overlapping comparison.')
        return PortfolioEntryStudy(None, coverage, evidence_weight, full_covered, warnings, asdict(config))

    aligned = pd.concat({symbol: clean[symbol] for symbol in eligible}, axis=1, join='inner').dropna()
    if len(aligned) < MIN_COMMON_STUDY_SESSIONS:
        warnings.append(f'Eligible holdings have fewer than {MIN_COMMON_STUDY_SESSIONS} overlapping USD sessions '
                        'for a common portfolio comparison.')
        return PortfolioEntryStudy(None, coverage, evidence_weight, full_covered, warnings, asdict(config))
    selected_weights = pd.Series({symbol: numeric_weights[symbol] for symbol in eligible}, dtype=float)
    selected_weights /= selected_weights.sum()
    returns = aligned.pct_change().fillna(0.0)
    signal_index = 100.0 * (1.0 + returns.mul(selected_weights, axis=1).sum(axis=1)).cumprod()
    benchmark = to_usd(benchmark_prices, benchmark_currency.upper()).reindex(signal_index.index).dropna()
    signal_index = signal_index.reindex(benchmark.index)
    aligned = aligned.reindex(benchmark.index)
    if len(signal_index) < MIN_COMMON_STUDY_SESSIONS:
        warnings.append(f'The benchmark has fewer than {MIN_COMMON_STUDY_SESSIONS} sessions aligned with the USD basket.')
        return PortfolioEntryStudy(None, coverage, evidence_weight, full_covered, warnings, asdict(config))
    current = hybrid_entry_signal(signal_index, benchmark, as_of=cutoff, config=config)
    span = config.maximum_wait_sessions + config.forward_horizon_sessions + 1
    fee, rows = config.cost_bps / 10_000, []
    for start in range(config.long_trend_window, len(signal_index) - span, span):
        initial = start + 1
        fixed_date = initial + config.fixed_second_session - 1
        deadline_date = initial + config.maximum_wait_sessions - 1
        terminal = start + span
        trigger, second_date = 'Forced completion', deadline_date
        for review in range(initial + config.minimum_wait_sessions - 1, deadline_date):
            condition, *_ = _trigger(signal_index, benchmark, review, config)
            if condition:
                trigger, second_date = condition, review + 1
                break
        initial_prices = aligned.iloc[initial]
        fixed_prices = aligned.iloc[fixed_date]
        second_prices = aligned.iloc[second_date]
        terminal_prices = aligned.iloc[terminal]
        wait_hybrid = second_date - initial
        wait_fixed = fixed_date - initial
        cash_fixed = (1 + config.annual_cash_rate) ** (wait_fixed / 252)
        cash_hybrid = (1 + config.annual_cash_rate) ** (wait_hybrid / 252)
        initial_wealth = (1 - fee) * terminal_prices / initial_prices
        fixed_wealth = (config.first_fraction * initial_wealth + (1 - config.first_fraction) *
                        cash_fixed * (1 - fee) * terminal_prices / fixed_prices)
        hybrid_wealth = (config.first_fraction * initial_wealth + (1 - config.first_fraction) *
                         cash_hybrid * (1 - fee) * terminal_prices / second_prices)
        lump = float(initial_wealth.dot(selected_weights))
        fixed = float(fixed_wealth.dot(selected_weights))
        hybrid = float(hybrid_wealth.dot(selected_weights))
        rows.append(dict(start_date=str(signal_index.index[start].date()),
            first_entry_date=str(signal_index.index[initial].date()),
            fixed_second_entry_date=str(signal_index.index[fixed_date].date()),
            fixed_wait_sessions=wait_fixed,
            second_entry_date=str(signal_index.index[second_date].date()),
            terminal_date=str(signal_index.index[terminal].date()), trigger=trigger,
            wait_sessions=wait_hybrid, lump_sum_wealth=lump, fixed_dca_wealth=fixed,
            hybrid_wealth=hybrid, hybrid_vs_lump=hybrid / lump - 1,
            hybrid_vs_fixed=hybrid / fixed - 1))
    cases = pd.DataFrame(rows)
    summary = summarize_hybrid_cases(cases)
    warnings.append('The common condition uses a fixed-weight USD basket; terminal wealth uses target-weighted '
                    'buy-and-hold positions in each ETF, with no daily rebalancing.')
    if len(cases) < config.minimum_cases:
        warnings.append(f'Only {len(cases)} non-overlapping completed starts; at least '
                        f'{config.minimum_cases} are required for a report-level performance claim.')
    study = HybridEntryStudy(summary, cases, current, warnings.copy(), asdict(config))
    return PortfolioEntryStudy(study, coverage, evidence_weight, full_covered, warnings,
                               asdict(config), str(signal_index.index[-1].date()))
