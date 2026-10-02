import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from ui import entry_quality


APP = "from ui.entry_quality import render_entry_quality\nrender_entry_quality('TEST', key_prefix='test')"


def test_panel_is_opt_in(monkeypatch):
    def fail(*args):
        raise AssertionError('Unexpected fetch')
    monkeypatch.setattr(entry_quality, '_entry_history', fail)
    app = AppTest.from_string(APP).run()
    assert not app.exception
    assert not app.metric


def test_panel_scores_and_recalculates_target(monkeypatch):
    p = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(3).normal(.001, .01, 800))),
                  index=pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=800))
    monkeypatch.setattr(entry_quality, '_entry_history', lambda *args: p)
    app = AppTest.from_string(APP).run()
    next(box for box in app.checkbox if box.label == 'Calculate entry quality').check().run()
    assert not app.exception
    assert len(app.metric) == 1  # default competition view excludes unvalidated sizing
    next(w for w in app.checkbox if w.label == 'Research mode — unvalidated sizing and timing').check().run()
    assert len(app.metric) == 3
    app.number_input[0].set_value(10.).run()
    assert not app.exception
    assert len(app.dataframe) == 1


def test_failed_api_is_incomplete(monkeypatch):
    def fail(*args):
        raise RuntimeError('network unavailable')
    monkeypatch.setattr(entry_quality, '_entry_history', fail)
    app = AppTest.from_string(APP).run()
    next(box for box in app.checkbox if box.label == 'Calculate entry quality').check().run()
    assert not app.exception
    assert 'Incomplete' in app.info[0].value
    assert app.warning
    assert not app.metric


def test_fetch_uses_existing_adjusted_source_and_excludes_today(monkeypatch):
    from src.data.fetchers.yahoo_fetcher import FetchResult, YahooFetcher

    calls = []
    def fetch(self, ticker, start, end):
        calls.append((ticker, start, end))
        return FetchResult(pd.DataFrame(), success=False, error='Unavailable')
    monkeypatch.setattr(YahooFetcher, 'fetch_prices', fetch)
    entry_quality._entry_history.clear()
    assert entry_quality._entry_history('FAIL', '2026-09-25') is None
    assert calls[0][2] == pd.Timestamp('2026-09-25')
    assert (calls[0][2] - calls[0][1]).days == 1500
    entry_quality._entry_history.clear()


def test_portfolio_panel_shows_exposure_and_case_charts(monkeypatch):
    index = pd.bdate_range('2022-01-03', periods=900)
    histories = {}
    for seed, ticker in enumerate(('EXCS.L', 'XUSE.L', 'VT', 'MVED.L', 'SPY'), 1):
        rng = np.random.default_rng(seed)
        histories[ticker] = pd.Series(100 * np.exp(np.cumsum(rng.normal(.0003, .01, len(index)))), index)
    histories['GBPUSD=X'] = pd.Series(1.3, index)
    histories['EURUSD=X'] = pd.Series(1.1, index)
    monkeypatch.setattr(entry_quality, '_entry_history', lambda ticker, as_of: histories[ticker])
    app = AppTest.from_string(
        "from ui.entry_quality import _render_portfolio_entry_framework\n"
        "_render_portfolio_entry_framework(as_of='2026-09-29', key_prefix='portfolio')"
    ).run()
    next(box for box in app.checkbox if box.label == 'Evaluate portfolio implementation evidence').check().run(timeout=15)
    assert not app.exception
    assert next(metric for metric in app.metric if metric.label.startswith('Hybrid vs lump sum'))
    assert len(app.get('plotly_chart')) == 2


def test_partial_portfolio_evidence_does_not_display_a_buy_instruction(monkeypatch):
    index = pd.bdate_range('2022-01-03', periods=900)
    histories = {}
    for seed, ticker in enumerate(('EXCS.L', 'XUSE.L', 'VT', 'MVED.L', 'SPY'), 1):
        rng = np.random.default_rng(seed)
        series = pd.Series(100 * np.exp(np.cumsum(rng.normal(.0003, .01, len(index)))), index)
        histories[ticker] = series.iloc[-400:] if ticker == 'XUSE.L' else series
    histories['GBPUSD=X'] = pd.Series(1.3, index)
    histories['EURUSD=X'] = pd.Series(1.1, index)
    monkeypatch.setattr(entry_quality, '_entry_history', lambda ticker, as_of: histories[ticker])
    app = AppTest.from_string(
        "from ui.entry_quality import _render_portfolio_entry_framework\n"
        "_render_portfolio_entry_framework(as_of='2026-09-29', key_prefix='partial')"
    ).run()
    next(box for box in app.checkbox if box.label == 'Evaluate portfolio implementation evidence').check().run(timeout=15)
    assert not app.exception
    assert any('renormalized covered basket' in warning.value for warning in app.warning)
    assert all('Buy remaining tranche' not in str(metric.value) for metric in app.metric)
    assert next(metric for metric in app.metric if metric.label == 'Condition on covered basket')


def test_423_session_listing_is_included_in_portfolio_panel(monkeypatch):
    index = pd.bdate_range('2022-01-03', periods=900)
    histories = {}
    for seed, ticker in enumerate(('EXCS.L', 'XUSE.L', 'VT', 'MVED.L', 'SPY'), 1):
        rng = np.random.default_rng(seed)
        series = pd.Series(100 * np.exp(np.cumsum(rng.normal(.0003, .01, len(index)))), index)
        histories[ticker] = series.iloc[-423:] if ticker == 'XUSE.L' else series
    histories['GBPUSD=X'] = pd.Series(1.3, index)
    histories['EURUSD=X'] = pd.Series(1.1, index)
    monkeypatch.setattr(entry_quality, '_entry_history', lambda ticker, as_of: histories[ticker])
    app = AppTest.from_string(
        "from ui.entry_quality import _render_portfolio_entry_framework\n"
        "_render_portfolio_entry_framework(as_of='2026-09-29', key_prefix='at_threshold')"
    ).run()
    next(box for box in app.checkbox if box.label == 'Evaluate portfolio implementation evidence').check().run(timeout=15)
    assert not app.exception
    coverage = app.dataframe[0].value.set_index('Ticker')
    assert coverage.loc['XUSE.L', 'Observed sessions'] == 423
    assert coverage.loc['XUSE.L', 'Status'] == 'Included'
    assert all('covers 75.0%' not in warning.value for warning in app.warning)
    assert any('completed starts is insufficient' in warning.value for warning in app.warning)


def test_recorded_first_tranche_has_a_session_based_completion_review(monkeypatch):
    index = pd.bdate_range(end='2026-09-28', periods=900)
    histories = {}
    for seed, ticker in enumerate(('EXCS.L', 'XUSE.L', 'VT', 'MVED.L', 'SPY'), 1):
        rng = np.random.default_rng(seed)
        histories[ticker] = pd.Series(100 * np.exp(np.cumsum(rng.normal(.0003, .01, len(index)))), index)
    histories['GBPUSD=X'] = pd.Series(1.3, index)
    histories['EURUSD=X'] = pd.Series(1.1, index)
    monkeypatch.setattr(entry_quality, '_entry_history', lambda ticker, as_of: histories[ticker])
    app = AppTest.from_string(
        "from ui.entry_quality import _render_portfolio_entry_framework\n"
        "_render_portfolio_entry_framework(as_of='2026-09-29', key_prefix='review')"
    ).run()
    next(box for box in app.checkbox if box.label == 'Evaluate portfolio implementation evidence').check().run(timeout=15)
    next(box for box in app.checkbox if box.label == 'Team approved these portfolio weights and the 75/25 rule').check().run(timeout=15)
    next(box for box in app.checkbox if box.label == 'First 75% recorded').check().run(timeout=15)
    next(field for field in app.date_input if field.label == 'First execution date').set_value(index[-10].date())
    next(field for field in app.text_input if field.label == 'Official execution reference').set_value('WINS-EXCS-1')
    app.run(timeout=15)
    assert not app.exception
    assert any('Completion review overdue' in item.value for item in app.markdown)
