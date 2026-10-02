"""Secondary scenario using the existing 2015–2026 replacement ETF export."""

import json
from pathlib import Path

import pandas as pd

from src.analytics.hybrid_entry import HybridEntryConfig
from src.analytics.portfolio_entry_export import make_entry_snapshot, analyze_entry_snapshot


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'proxy_backtest_20260926'
OUT = ROOT / 'outputs' / 'entry_strategy_20260930' / 'proxy_scenario'


def main():
    prices = pd.read_json(SOURCE / 'adjusted_prices_usd.json', orient='table')
    daily = pd.read_json(SOURCE / 'daily_results.json', orient='table')
    benchmark = pd.Series(index=prices.index, dtype=float)
    benchmark.iloc[0] = 100.0
    benchmark.iloc[1:] = 100.0 * (1.0 + daily.spy_return).cumprod().to_numpy()
    histories = {symbol: prices[symbol] for symbol in prices}
    weights = dict.fromkeys(histories, .25)
    snapshot = make_entry_snapshot(histories, weights, benchmark, as_of=prices.index[-1],
        quote_currencies=dict.fromkeys(histories, 'USD'), fx_to_usd={},
        config=HybridEntryConfig(first_fraction=.75, maximum_wait_sessions=10,
            fixed_second_session=5, forward_horizon_sessions=21))
    snapshot['source'] = 'Saved QuantSim replacement ETF USD price export, 2026-09-26; SPY index reconstructed from saved daily returns'
    snapshot['approval'] = 'Scenario using replacement ETFs and entered equal weights; no portfolio approval attested'
    # The amended provenance is itself covered by the input fingerprint.
    from hashlib import sha256
    payload = {key: value for key, value in snapshot.items() if key != 'input_sha256'}
    snapshot['input_sha256'] = sha256(json.dumps(payload, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    result, sensitivity = analyze_entry_snapshot(snapshot)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'inputs.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    result.coverage.to_csv(OUT / 'coverage.csv', index=False)
    result.study.cases.to_csv(OUT / 'starts.csv', index=False)
    result.study.summary.to_csv(OUT / 'summary.csv', index=False)
    sensitivity.to_csv(OUT / 'sensitivity.csv', index=False)
    print(result.study.summary.to_string(index=False))
    print(sensitivity[['cost_bps', 'annual_cash_rate', 'conditional_vs_lump', 'conditional_vs_fixed']].to_string(index=False))


if __name__ == '__main__':
    main()
