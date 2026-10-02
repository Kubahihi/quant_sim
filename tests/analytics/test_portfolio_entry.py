import numpy as np
import pandas as pd

from src.analytics.portfolio_entry import deployment_path, review_second_tranche, study_portfolio_entry
from src.analytics.hybrid_entry import HybridEntryConfig
from src.analytics.portfolio_entry_export import make_entry_snapshot, analyze_entry_snapshot


def _history(n, seed):
    index = pd.bdate_range('2020-01-02', periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(.0003, .01, n))), index=index)


def test_portfolio_study_discloses_data_limited_weight_without_using_it_as_evidence():
    long_a, long_b = _history(1_000, 1), _history(1_000, 2)
    short = _history(200, 3)
    result = study_portfolio_entry(
        {'AAA': long_a, 'BBB': long_b, 'NEW': short},
        {'AAA': .4, 'BBB': .4, 'NEW': .2}, _history(1_000, 4), as_of=long_a.index[-1],
        quote_currencies={'AAA': 'USD', 'BBB': 'USD', 'NEW': 'USD'}, fx_to_usd={},
    )
    assert result.study is not None
    assert result.evidence_weight == .8
    assert not result.full_portfolio_covered
    assert result.coverage.set_index('ticker').loc['NEW', 'evidence_status'] == 'Data-limited'
    assert any('80.0%' in warning for warning in result.warnings)


def test_portfolio_study_marks_full_coverage_when_every_holding_is_eligible():
    first, second, benchmark = _history(1_000, 1), _history(1_000, 2), _history(1_000, 3)
    result = study_portfolio_entry({'AAA': first, 'BBB': second}, {'AAA': .5, 'BBB': .5}, benchmark,
                                   as_of=first.index[-1],
                                   quote_currencies={'AAA': 'USD', 'BBB': 'USD'}, fx_to_usd={})
    assert result.study is not None
    assert result.full_portfolio_covered
    assert result.evidence_weight == 1.0
    case = result.study.cases.iloc[0]
    first_date = first.index[first.index.get_loc(pd.Timestamp(case.start_date)) + 1]
    terminal_date = pd.Timestamp(case.terminal_date)
    expected_lump = .5 * .999 * first.loc[terminal_date] / first.loc[first_date] + \
                    .5 * .999 * second.loc[terminal_date] / second.loc[first_date]
    assert np.isclose(case.lump_sum_wealth, expected_lump)
    assert result.study.cases.wait_sessions.ge(1).all()


def test_exactly_423_sessions_includes_the_short_listing_without_claiming_twelve_cases():
    long_a, long_b, long_c, short, benchmark = [_history(1_000, seed) for seed in range(1, 6)]
    long_b = long_b.drop(long_b.index[-420:-412])  # different exchange calendar
    result = study_portfolio_entry(
        {'AAA': long_a, 'BBB': long_b, 'CCC': long_c, 'NEW': short.iloc[-423:]},
        {'AAA': .25, 'BBB': .25, 'CCC': .25, 'NEW': .25}, benchmark,
        as_of=benchmark.index[-1],
        quote_currencies={'AAA': 'USD', 'BBB': 'USD', 'CCC': 'USD', 'NEW': 'USD'}, fx_to_usd={},
    )
    assert result.coverage.set_index('ticker').loc['NEW', 'observations'] == 423
    assert result.coverage.included_in_evidence.all()
    assert result.full_portfolio_covered
    assert result.evidence_weight == 1.0
    assert result.study is not None
    assert 0 < len(result.study.cases) < result.config['minimum_cases']


def test_deployment_chart_shows_early_hybrid_completion_and_fixed_deadline():
    cases = pd.DataFrame({'wait_sessions': [0, 5, 10]})
    path = deployment_path(cases, first_fraction=.75, maximum_wait_sessions=10)
    assert path.loc[0, 'lump_sum'] == 1.0
    assert path.loc[0, 'fixed_dca'] == .75
    assert np.isclose(path.loc[0, 'hybrid_historical_average'], .75 + .25 / 3)
    assert path.loc[3, 'fixed_dca'] == .75
    assert path.loc[4, 'fixed_dca'] == 1.0
    assert path.loc[10, 'fixed_dca'] == 1.0
    assert path.loc[10, 'hybrid_historical_average'] == 1.0


def test_portfolio_study_requires_fx_for_non_usd_listing():
    import pytest

    history = _history(900, 1)
    with pytest.raises(ValueError, match='Missing USD conversion'):
        study_portfolio_entry({'AAA': history}, {'AAA': 1.0}, history, as_of=history.index[-1],
                              quote_currencies={'AAA': 'GBP'}, fx_to_usd={})


def test_portfolio_study_translates_quote_returns_into_usd():
    index = pd.bdate_range('2022-01-03', periods=900)
    flat = pd.Series(100.0, index)
    fx = pd.Series(np.linspace(1.0, 1.3, len(index)), index)
    translated = study_portfolio_entry({'AAA': flat}, {'AAA': 1.0}, flat, as_of=index[-1],
        quote_currencies={'AAA': 'GBP'}, fx_to_usd={'GBP': fx})
    unconverted = study_portfolio_entry({'AAA': flat}, {'AAA': 1.0}, flat, as_of=index[-1],
        quote_currencies={'AAA': 'USD'}, fx_to_usd={})
    assert translated.study.summary.iloc[0].mean_terminal_return > unconverted.study.summary.iloc[0].mean_terminal_return


def test_second_tranche_review_requires_a_record_and_separate_session():
    dates = pd.bdate_range('2026-09-01', periods=20)
    assert review_second_tranche(dates, as_of=dates[10], condition_met=True).status == 'First tranche not recorded'
    common = dict(session_dates=dates, first_execution_date=dates[10],
                  first_execution_reference='WINS-123', policy_approved=True, condition_met=True)
    assert review_second_tranche(as_of=dates[10], **common).status == 'Conditional review due next session'
    assert review_second_tranche(as_of=dates[11], **common).status == 'Conditional review due next session'
    assert review_second_tranche(as_of=dates[-1], final_recorded=True, **common).status == 'Final execution reference missing'
    assert review_second_tranche(as_of=dates[-1], final_recorded=True,
                                  final_execution_reference='WINS-456', **common).status == 'Final tranche recorded'


def test_second_tranche_deadline_does_not_depend_on_a_positive_signal():
    dates = pd.bdate_range('2026-09-01', periods=20)
    review = review_second_tranche(dates, as_of=dates[13], first_execution_date=dates[4],
                                   first_execution_reference='WINS-123', policy_approved=True,
                                   condition_met=False, signal_fresh=False)
    assert review.sessions_since_first == 9
    assert review.status == 'Completion review overdue'
    assert review_second_tranche(dates, as_of=dates[14], first_execution_date=dates[4],
        first_execution_reference='WINS-123', policy_approved=True).status == 'Completion review overdue'


def test_frozen_input_reproduces_prespecified_sessions_and_detects_tampering():
    import copy

    price, benchmark = _history(900, 1), _history(900, 2)
    config = HybridEntryConfig(first_fraction=.75, maximum_wait_sessions=10,
                               fixed_second_session=5, forward_horizon_sessions=21)
    snapshot = make_entry_snapshot({'AAA': price}, {'AAA': 1.0}, benchmark,
        as_of=price.index[-1], quote_currencies={'AAA': 'USD'}, fx_to_usd={}, config=config)
    result, sensitivity = analyze_entry_snapshot(snapshot)
    cases = result.study.cases
    dates = price.index
    for row in cases.itertuples():
        first = dates.get_loc(pd.Timestamp(row.first_entry_date))
        assert dates[first + 4] == pd.Timestamp(row.fixed_second_entry_date)
        assert first < dates.get_loc(pd.Timestamp(row.second_entry_date)) <= first + 9
        assert dates.get_loc(pd.Timestamp(row.terminal_date)) > first + 9
    assert len(sensitivity) == 9
    changed = copy.deepcopy(snapshot)
    changed['histories']['AAA'][0][1] *= 2
    import pytest
    with pytest.raises(ValueError, match='fingerprint'):
        analyze_entry_snapshot(changed)


def test_portfolio_historical_decision_does_not_use_later_prices():
    security, benchmark = _history(900, 9), _history(900, 10)
    kwargs = dict(as_of=security.index[-1], quote_currencies={'AAA': 'USD'}, fx_to_usd={},
                  config=HybridEntryConfig(first_fraction=.75, maximum_wait_sessions=10,
                                           fixed_second_session=5, forward_horizon_sessions=21))
    original = study_portfolio_entry({'AAA': security}, {'AAA': 1.0}, benchmark, **kwargs)
    first = original.study.cases.iloc[0]
    changed_security, changed_benchmark = security.copy(), benchmark.copy()
    later = security.index > pd.Timestamp(first.terminal_date)
    changed_security.loc[later] *= 3
    changed_benchmark.loc[later] *= .5
    changed = study_portfolio_entry({'AAA': changed_security}, {'AAA': 1.0}, changed_benchmark, **kwargs)
    assert changed.study.cases.iloc[0].to_dict() == first.to_dict()
