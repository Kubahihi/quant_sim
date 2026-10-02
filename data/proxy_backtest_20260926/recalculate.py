from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.price_alignment import align_daily_prices
from src.analytics.portfolio_metrics import calculate_portfolio_daily_returns, calculate_portfolio_core_metrics

out = Path(__file__).resolve().parent
raw = pd.read_pickle(ROOT / 'data/cache/proxy_history_20260926.pkl')['Close']
symbols = ['XCEM', 'VEA', 'VT', 'MVEU.L']
required = symbols + ['EURUSD=X', 'SPY']
assert all(raw[s].notna().sum() > 1000 for s in required)
start = max(raw[s].first_valid_index() for s in required)
end = min(raw[s].last_valid_index() for s in required)
# Limit to US sessions for comparable annualization; fill exchange holidays
# on the full source calendar first, using QuantSim's bounded policy.
aligned = align_daily_prices(raw.loc[start:end, required])
aligned = aligned.loc[raw.loc[start:end, 'SPY'].dropna().index].dropna()
prices = aligned[symbols].copy()
prices['MVEU.L'] *= aligned['EURUSD=X']
returns = prices.pct_change(fill_method=None).dropna()
p = calculate_portfolio_daily_returns(returns, np.full(4, 0.25))
b = aligned['SPY'].pct_change(fill_method=None).reindex(p.index)
assert b.notna().all() and np.isfinite(p).all()
pm = calculate_portfolio_core_metrics(p, risk_free_rate=0.03)
bm = calculate_portfolio_core_metrics(b, risk_free_rate=0.03)
years = (aligned.index[-1] - aligned.index[0]).days / 365.25
for m in [pm, bm]:
    m['calendar_cagr'] = (1 + m['total_return']) ** (1 / years) - 1
    m['final_value_usd'] = 100000 * (1 + m['total_return'])
growth = (1 + p).cumprod()
assert np.isclose(growth.iloc[-1] - 1, pm['total_return'])
assert np.isclose((1 + p).prod(), np.exp(np.log1p(p).sum()))
report = {
    'start': str(aligned.index[0].date()), 'end': str(aligned.index[-1].date()),
    'observations': len(p), 'currency': 'USD', 'initial_value': 100000,
    'weights': dict.fromkeys(symbols, 0.25), 'benchmark': 'SPY',
    'risk_free_rate_for_sharpe': 0.03,
    'method': 'Daily constant weights; dividend/split adjusted Yahoo prices; MVEU.L EUR converted to USD using EURUSD=X. US sessions, bounded holiday forward fill. No transaction costs or taxes.',
    'proxy_limitations': 'XCEM uses a different ex-China EM index; VEA includes small caps and FTSE country classifications (including South Korea). MVEU.L is an older accumulating Europe minimum-volatility share class. No pre-inception prices fabricated. This is a replacement-portfolio backtest, not the original funds actual historical performance.',
    'portfolio': pm, 'benchmark_metrics': bm,
    'source_coverage': {s: {'start': str(raw[s].first_valid_index().date()), 'end': str(raw[s].last_valid_index().date()), 'count': int(raw[s].count())} for s in required},
}
(out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
prices.to_json(out / 'adjusted_prices_usd.json', orient='table', date_format='iso')
pd.DataFrame({'portfolio_return': p, 'spy_return': b, 'portfolio_value_usd': 100000 * growth}).to_json(out / 'daily_results.json', orient='table', date_format='iso')
print(json.dumps(report, indent=2))
