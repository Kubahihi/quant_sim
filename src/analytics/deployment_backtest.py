"""Equal-capital, next-session-close comparison of prespecified deployment policies."""
from dataclasses import dataclass, replace
import hashlib

import numpy as np
import pandas as pd

from src.analytics.deployment import DeploymentPlan, DeploymentState, day, propose_deployment
from src.analytics.entry_quality import calculate_entry_quality, _prepare
from src.analytics.returns import calculate_returns
from src.analytics.risk_metrics import calculate_max_drawdown

BACKTEST_POLICIES = ('Immediate', 'Fixed schedule', 'Score-adjusted')


@dataclass
class DeploymentComparison:
    summary: pd.DataFrame
    equity: pd.DataFrame
    decisions: pd.DataFrame
    assumptions: dict
    warnings: list[str]


def compare_deployment_policies(ticker, prices, benchmark_prices, *, start_date,
        benchmark='SPY', capital=25_000., reviews=5, interval_days=7, horizon_sessions=126,
        cost_bps=10., annual_cash_rate=.03, config=None) -> DeploymentComparison:
    """Simulate a ring-fenced cash/security sleeve, not the entire client portfolio.

    Fractional adjusted units represent a total-return proxy, not executable share
    counts. Signals use review-date information; fills use next observed close.
    Capital, costs and horizon are identical across policies. No weight optimization.
    """
    if (not np.isfinite(capital) or capital <= 0 or not np.isfinite(cost_bps) or
            not 0 <= cost_bps <= 1000 or not np.isfinite(annual_cash_rate) or
            not -1 < annual_cash_rate <= 1 or type(horizon_sessions) is not int or horizon_sessions < 1):
        raise ValueError('Invalid capital, costs, cash rate or observation horizon.')
    start = day(start_date)
    if not isinstance(prices, pd.Series) or not isinstance(prices.index, pd.DatetimeIndex) or prices.empty:
        raise ValueError('Dated security history required.')
    if prices.index.hasnans:
        raise ValueError('Valid security session dates required.')
    ordered = prices.copy()
    ordered.index = ordered.index.tz_localize(None).normalize()
    ordered = ordered.sort_index()
    first = ordered.index.searchsorted(start, side='right')
    if first == 0 or (start - ordered.index[first - 1]).days > 7:
        raise ValueError('A recent observed security price at or before the first review is required.')
    end = first + horizon_sessions
    if first >= len(ordered) or end >= len(ordered):
        raise ValueError('A complete forward evaluation horizon is required.')
    # Unused future observations cannot determine whether this historical study runs.
    cutoff = ordered.index[end]
    p, issues = _prepare(ordered.loc[:cutoff], cutoff, ticker, 7)
    if issues:
        raise ValueError('; '.join(issues))
    sessions = p.index[first:end + 1]
    template = DeploymentPlan(ticker, benchmark, 1., float(capital), str(start.date()), reviews,
        interval_days, thesis='Prespecified historical study', invalidation='Not modeled in this price-only study',
        horizon=f'{horizon_sessions} sessions', mandate_reference='Equal-capital sleeve study', cost_bps=cost_bps)
    review_dates = template.dates()
    if review_dates[-1] >= sessions[-1]:
        raise ValueError('The evaluation horizon must extend beyond the final review execution.')
    execution_reviews = {}
    for review_date in review_dates:
        position = p.index.searchsorted(review_date, side='right')
        execution_reviews.setdefault(p.index[position], []).append(review_date)
    # Different calendar reviews can map to the same execution session after holidays.
    # Catch up once using the latest due review, avoiding duplicate orders.
    scores = {d: calculate_entry_quality(ticker, p, benchmark_prices, benchmark=benchmark,
                as_of=d, config=config) for ds in execution_reviews.values() for d in ds}
    used_scores = [scores[ds[-1]] for ds in execution_reviews.values()]
    score_coverage = sum(s.entry_score is not None for s in used_scores) / len(used_scores)
    fee = cost_bps / 10_000
    all_curves, rows, summary = {}, [], []
    for policy in BACKTEST_POLICIES:
        plan = replace(template, policy=policy)
        cash, units, spent, costs = float(capital), 0., 0., 0.
        previous_date, completion, last_review = start, None, None
        curve = {start: float(capital)}
        underinvestment = []
        for session in sessions:
            cash *= (1 + annual_cash_rate) ** ((session - previous_date).days / 365.25)
            current_price = float(p.loc[session])
            holding = units * current_price
            if session in execution_reviews:
                review_date = execution_reviews[session][-1]
                historical = p.loc[:review_date]
                reference_price = float(historical.iloc[-1]) if not historical.empty else current_price
                # Proposal valuation at the review close; final cash/gap caps rechecked at fill.
                review_cash = cash / (1 + annual_cash_rate) ** ((session - review_date).days / 365.25)
                state = DeploymentState(review_cash + units * reference_price, units * reference_price,
                    review_cash, spent=spent, snapshot_date=str(review_date.date()),
                    snapshot_reference='Historical adjusted-close sleeve', last_review_date=last_review)
                decision = propose_deployment(plan, state, as_of=review_date, entry=scores[review_date])
                notional = min(decision.proposed_purchase, cash / (1 + fee))
                charge = notional * fee
                units += notional / current_price
                cash = max(0., cash - notional - charge)  # remove floating-point dust only
                spent += notional
                costs += charge
                last_review = decision.scheduled_review
                rows.append(dict(policy=policy, review_date=str(review_date.date()),
                    execution_date=str(session.date()), score=scores[review_date].entry_score,
                    proposed=decision.proposed_purchase, executed=notional, costs=charge,
                    status=decision.status, reason=decision.explanation,
                    score_warnings='; '.join(scores[review_date].warnings)))
            wealth = cash + units * current_price
            curve[session] = wealth
            underinvestment.append(cash / wealth if wealth > 0 else 0.)
            if completion is None and spent + costs >= .99 * capital:
                completion = session
            previous_date = session
        equity = pd.Series(curve)
        returns = calculate_returns(equity)
        all_curves[policy] = equity
        summary.append(dict(policy=policy, ending_wealth=float(equity.iloc[-1]),
            total_return=float(equity.iloc[-1] / capital - 1), max_drawdown=calculate_max_drawdown(returns),
            downside_deviation=float(np.sqrt(252 * np.mean(np.minimum(returns, 0.) ** 2))),
            mean_cash_fraction=float(np.mean(underinvestment)), turnover=spent / capital,
            costs=costs, remaining_cash=cash,
            completion_days=(completion - start).days if completion is not None else None,
            completed=completion is not None, score_coverage=score_coverage))
    summary = pd.DataFrame(summary).set_index('policy')
    summary['wealth_vs_immediate'] = summary.ending_wealth - summary.loc['Immediate', 'ending_wealth']
    return DeploymentComparison(summary, pd.DataFrame(all_curves), pd.DataFrame(rows),
        dict(ticker=ticker, benchmark=benchmark, start_date=str(start.date()), capital=capital,
             reviews=reviews, interval_days=interval_days, horizon_sessions=horizon_sessions,
             cost_bps=cost_bps, annual_cash_rate=annual_cash_rate, policy_version=template.version,
             score_config=as_score_config(config),
             security_history_sha256=hashlib.sha256(p.to_csv().encode()).hexdigest(),
             benchmark_history_sha256=(hashlib.sha256(benchmark_prices.to_csv().encode()).hexdigest()
                                       if benchmark_prices is not None else None)), [
        'Single-security total-return sleeve study; excludes other holdings, taxes, liquidity and thesis changes.',
        'Adjusted fractional units and next-close fills are analytical proxies, not historical executable orders.',
        'No parameter optimization. Revised prices and survivor selection can bias results; use held-out point-in-time evidence.',
        'Score-adjusted missing assessments pause purchases, including the deadline; check decision warnings and completion.',
    ])


def as_score_config(config):
    from dataclasses import asdict
    from src.analytics.entry_quality import EntryQualityConfig
    return asdict(config or EntryQualityConfig())


def compare_deployment_windows(ticker, prices, benchmark_prices, *, start_dates, **kwargs):
    """Paired window outcomes, explicitly excluding incomplete forward windows.

    No confidence interval is asserted: windows may overlap and be serially dependent.
    Dispersion (10th/90th percentiles) is descriptive, not an inference interval.
    """
    rows, excluded = [], []
    dates = sorted({day(d) for d in start_dates})
    for date in dates:
        try:
            result = compare_deployment_policies(ticker, prices, benchmark_prices, start_date=date, **kwargs)
        except ValueError as exc:
            excluded.append(dict(start_date=str(date.date()), reason=str(exc)))
            continue
        for policy, row in result.summary.iterrows():
            rows.append(dict(start_date=str(date.date()), policy=policy, **row.to_dict()))
    observations = pd.DataFrame(rows)
    if observations.empty:
        return observations, pd.DataFrame(), pd.DataFrame(excluded)
    grouped = observations.groupby('policy')['wealth_vs_immediate']
    summary = grouped.agg(['count', 'mean', 'median', 'min', 'max'])
    summary['p10'] = grouped.quantile(.1)
    summary['p90'] = grouped.quantile(.9)
    summary['win_rate_vs_immediate'] = grouped.apply(lambda s: float((s > 0).mean()))
    return observations, summary, pd.DataFrame(excluded)
