"""Frozen inputs and report tables for the prespecified portfolio entry comparison."""

from dataclasses import asdict, replace
from hashlib import sha256
import json

import pandas as pd

from src.analytics.hybrid_entry import HybridEntryConfig
from src.analytics.portfolio_entry import study_portfolio_entry


def _points(series):
    clean = series.dropna().sort_index()
    return [[str(pd.Timestamp(date).date()), float(value)] for date, value in clean.items()]


def make_entry_snapshot(histories, weights, benchmark_prices, *, as_of,
                        quote_currencies, fx_to_usd, benchmark_currency='USD', config=None):
    """Save the actual adjusted closes and FX used, not merely a vendor name."""
    config = config or HybridEntryConfig(first_fraction=.75, maximum_wait_sessions=10,
                                         fixed_second_session=5, forward_horizon_sessions=21)
    config.validate()
    snapshot = dict(schema='portfolio-entry-input-v1', as_of=str(pd.Timestamp(as_of).date()),
                    weights=weights, quote_currencies=quote_currencies,
                    benchmark_currency=benchmark_currency, config=asdict(config),
                    histories={key: _points(value) for key, value in histories.items()},
                    benchmark=_points(benchmark_prices),
                    fx_to_usd={key: _points(value) for key, value in fx_to_usd.items()},
                    price_basis='vendor dividend/split adjusted daily closes; USD FX closes',
                    approval='entered weights; approval not independently verified',
                    source='Yahoo Finance via QuantSim YahooFetcher unless separately documented')
    payload = json.dumps(snapshot, sort_keys=True, separators=(',', ':'), allow_nan=False)
    snapshot['input_sha256'] = sha256(payload.encode()).hexdigest()
    return snapshot


def _series(points):
    return pd.Series([value for _, value in points],
                     index=pd.DatetimeIndex([date for date, _ in points]), dtype=float)


def analyze_entry_snapshot(snapshot):
    """Recompute nine fixed cost/cash assumptions on exactly the same saved data."""
    provided_hash = snapshot.get('input_sha256')
    payload = {key: value for key, value in snapshot.items() if key != 'input_sha256'}
    actual_hash = sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()
    if provided_hash != actual_hash:
        raise ValueError('Input snapshot fingerprint does not match its contents.')
    config = HybridEntryConfig(**snapshot['config'])
    config.validate()
    histories = {key: _series(value) for key, value in snapshot['histories'].items()}
    benchmark = _series(snapshot['benchmark'])
    fx = {key: _series(value) for key, value in snapshot['fx_to_usd'].items()}
    base = dict(as_of=snapshot['as_of'], quote_currencies=snapshot['quote_currencies'],
                fx_to_usd=fx, benchmark_currency=snapshot['benchmark_currency'])

    def run(cfg):
        return study_portfolio_entry(histories, snapshot['weights'], benchmark, config=cfg, **base)

    study = run(config)
    if study.study is None:
        return study, pd.DataFrame()
    expected_starts = study.study.cases.start_date.tolist()
    rows = []
    for cost_bps in (0., 10., 25.):
        for cash_rate in (0., .03, .05):
            changed = run(replace(config, cost_bps=cost_bps, annual_cash_rate=cash_rate))
            cases = changed.study.cases
            if cases.start_date.tolist() != expected_starts:
                raise AssertionError('Sensitivity changed the historical starting points.')
            rows.append(dict(cost_bps=cost_bps, annual_cash_rate=cash_rate,
                             cases=len(cases), mean_lump_return=float(cases.lump_sum_wealth.mean() - 1),
                             mean_fixed_return=float(cases.fixed_dca_wealth.mean() - 1),
                             mean_conditional_return=float(cases.hybrid_wealth.mean() - 1),
                             conditional_vs_lump=float(cases.hybrid_vs_lump.mean()),
                             conditional_vs_fixed=float(cases.hybrid_vs_fixed.mean())))
    return study, pd.DataFrame(rows)
