from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analytics.deployment import DeploymentPlan, DeploymentState, propose_deployment
from src.analytics.entry_quality import calculate_entry_quality
from src.analytics.deployment_backtest import compare_deployment_policies, compare_deployment_windows


def plan(**kwargs):
    return replace(DeploymentPlan('TEST', 'SPY', .05, 25_000., '2025-01-06',
        thesis='Durable earnings', invalidation='Margin deterioration', horizon='Five years',
        mandate_reference='Committee 1', cost_bps=0.), **kwargs)


def state(**kwargs):
    return replace(DeploymentState(500_000., 0., 25_000., snapshot_date='2025-01-06',
        snapshot_reference='Reconciled snapshot 1'), **kwargs)


def prices(n=1000, seed=5):
    return pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(.0004, .01, n))),
        index=pd.bdate_range('2022-01-03', periods=n))


def entry(score=90, date='2025-01-06'):
    result = calculate_entry_quality('TEST', prices(), prices(seed=9), as_of=date)
    assert result.data_quality['complete']
    return replace(result, entry_score=score)


@pytest.mark.parametrize('policy,amount', [('Immediate', 25_000), ('Fixed schedule', 5_000), ('Score-adjusted', 7_500)])
def test_initial_purchases(policy, amount):
    decision = propose_deployment(plan(policy=policy), state(), as_of='2025-01-06', entry=entry())
    assert decision.proposed_purchase == amount


@pytest.mark.parametrize('score,amount', [(39.999, 2500), (40., 5000), (79.999, 5000), (80., 7500)])
def test_score_band_boundaries(score, amount):
    assert propose_deployment(plan(policy='Score-adjusted'), state(), as_of='2025-01-06',
                              entry=entry(score)).proposed_purchase == amount


def test_acceleration_is_not_repeated_and_deferral_catches_up():
    p = plan(policy='Score-adjusted')
    s = state(spent=7_500, holding_value=7_500, cash=17_500, snapshot_date='2025-01-13')
    d = propose_deployment(p, s, as_of='2025-01-13', entry=entry(90, '2025-01-13'))
    assert d.proposed_purchase == 5_000
    low = propose_deployment(p, state(), as_of='2025-01-06', entry=entry(20))
    assert low.proposed_purchase == 2_500
    s = state(spent=2_500, holding_value=2_500, cash=22_500, snapshot_date='2025-01-13')
    assert propose_deployment(p, s, as_of='2025-01-13', entry=entry(60, '2025-01-13')).proposed_purchase == 7_500


def test_two_stage_plan_calculates_the_completion_review_and_remaining_purchase():
    p = plan(policy='Fixed schedule', reviews=2, interval_days=14)
    assert p.is_two_stage()
    first = propose_deployment(p, state(), as_of='2025-01-06')
    assert first.proposed_purchase == 12_500
    assert first.next_review == '2025-01-20'
    second_state = state(spent=12_500, holding_value=12_500, cash=12_500,
                         snapshot_date='2025-01-20')
    second = propose_deployment(p, second_state, as_of='2025-01-20')
    assert second.proposed_purchase == 12_500
    assert second.next_review is None


def test_signal_gated_plan_buys_half_then_only_completes_on_a_qualified_review_or_deadline():
    p = plan(policy='Signal-gated', reviews=5, interval_days=5)
    first = propose_deployment(p, state(), as_of='2025-01-06')
    assert first.proposed_purchase == 12_500
    deferred = SimpleNamespace(as_of='2025-01-13', status='research_ready', action='Wait for next model review')
    s = state(spent=12_500, holding_value=12_500, cash=12_500, snapshot_date='2025-01-13')
    assert propose_deployment(p, s, as_of='2025-01-13', signal=deferred).proposed_purchase == 0
    qualified = SimpleNamespace(as_of='2025-01-13', status='research_ready', action='Buy remaining tranche now')
    assert propose_deployment(p, s, as_of='2025-01-13', signal=qualified).proposed_purchase == 12_500


def test_pending_orders_and_limits():
    s = state(holding_value=20_000, pending_position=3_000, pending_plan=2_000, reserved_cash=24_000)
    d = propose_deployment(plan(policy='Immediate'), s, as_of='2025-01-06')
    assert d.allocation_gap == 2_000
    assert d.remaining_budget == 23_000
    assert d.proposed_purchase == 1_000
    assert d.status == 'Limited'
    assert propose_deployment(plan(max_purchase=100), state(), as_of='2025-01-06').proposed_purchase == 100


def test_fee_reservation_and_post_fee_target_limit():
    p = plan(policy='Immediate', cost_bps=100)
    s = state(cash=50_000)
    d = propose_deployment(p, s, as_of='2025-01-06')
    assert d.proposed_purchase + d.estimated_cost <= s.cash
    assert d.proposed_purchase == pytest.approx(.05 * (s.portfolio_value - d.estimated_cost))


def test_completed_budget_does_not_authorize_extra_capital_or_sales():
    d = propose_deployment(plan(), state(spent=25_000, holding_value=10_000), as_of='2025-01-06')
    assert d.proposed_purchase == 0
    d = propose_deployment(plan(), state(holding_value=30_000), as_of='2025-01-06')
    assert d.proposed_purchase == 0


@pytest.mark.parametrize('changes', [dict(thesis_valid=False), dict(constraints_valid=False),
    dict(snapshot_date='2025-01-05'), dict(snapshot_date='2025-01-07'), dict(last_review_date='2025-01-06')])
def test_review_gates(changes):
    assert propose_deployment(plan(), state(**changes), as_of='2025-01-06').proposed_purchase == 0


def test_deadline_and_missing_scores():
    p, s = plan(policy='Score-adjusted'), state(snapshot_date='2025-02-03')
    assert propose_deployment(p, s, as_of='2025-02-03', entry=entry(0, '2025-02-03')).proposed_purchase == 25_000
    assert propose_deployment(p, s, as_of='2025-02-03').proposed_purchase == 0
    assert propose_deployment(p, s, as_of='2025-02-03', entry=entry()).status == 'Review required'
    assert propose_deployment(plan(), state(snapshot_date='2025-01-05'), as_of='2025-01-05').status == 'Not due'


@pytest.mark.parametrize('changes', [dict(target_weight=.2, max_position_weight=.1), dict(budget=np.nan),
    dict(reviews=0), dict(interval_days=0), dict(adjustment=.6), dict(thesis=''), dict(policy='Magic')])
def test_invalid_plan(changes):
    with pytest.raises(ValueError):
        plan(**changes).validate()


@pytest.mark.parametrize('changes', [dict(cash=np.nan), dict(holding_value=-1), dict(portfolio_value=1),
    dict(pending_position=10), dict(pending_plan=1), dict(snapshot_reference='')])
def test_invalid_snapshot(changes):
    with pytest.raises(ValueError):
        state(**changes).validate()


def test_backtest_flat_price_cash_and_cost_accounting():
    p = prices() * 0 + 100
    result = compare_deployment_policies('TEST', p, p, start_date=p.index[400], annual_cash_rate=0, cost_bps=10)
    immediate = result.summary.loc['Immediate']
    assert immediate.ending_wealth == pytest.approx(25_000 / 1.001)
    assert immediate.ending_wealth + immediate.costs == pytest.approx(25_000)
    assert immediate.max_drawdown == pytest.approx(-immediate.costs / 25_000)
    assert result.summary.loc['Score-adjusted', 'ending_wealth'] == 25_000  # flat data unavailable
    assert not result.summary.loc['Score-adjusted', 'completed']
    assert result.decisions.iloc[0].execution_date == str(p.index[401].date())
    assert result.equity.iloc[0].eq(25_000).all()


def test_fixed_schedule_equal_tranches_and_cash_interest():
    p = prices() * 0 + 100
    result = compare_deployment_policies('TEST', p, p, start_date=p.index[400], annual_cash_rate=.05, cost_bps=0)
    fixed = result.decisions.query("policy == 'Fixed schedule'")
    assert fixed.executed.tolist() == [5000] * 5
    assert result.summary.loc['Score-adjusted', 'ending_wealth'] > result.summary.loc['Fixed schedule', 'ending_wealth']
    assert result.summary.loc['Fixed schedule', 'ending_wealth'] > result.summary.loc['Immediate', 'ending_wealth']


def test_backtest_future_changes_do_not_change_earlier_decisions():
    p, b = prices(), prices(seed=9)
    a = compare_deployment_policies('TEST', p, b, start_date=p.index[400])
    p.iloc[420:] *= 1.5
    b.iloc[420:] *= .5
    changed = compare_deployment_policies('TEST', p, b, start_date=p.index[400])
    cutoff = str(p.index[420].date())
    pd.testing.assert_frame_equal(a.decisions[a.decisions.execution_date < cutoff].reset_index(drop=True),
        changed.decisions[changed.decisions.execution_date < cutoff].reset_index(drop=True))


def test_backtest_rejects_incomplete_forward_horizon_and_reports_exclusions():
    p = prices()
    with pytest.raises(ValueError, match='complete forward'):
        compare_deployment_policies('TEST', p, p, start_date=p.index[-5])
    obs, summary, excluded = compare_deployment_windows('TEST', p, p,
        start_dates=[p.index[400], p.index[600], p.index[-5]])
    assert len(obs) == 6
    assert len(excluded) == 1
    assert summary['count'].eq(2).all()


def test_same_inputs_reproduce_evidence():
    p = prices()
    a = compare_deployment_policies('TEST', p, p, start_date=p.index[400])
    b = compare_deployment_policies('TEST', p, p, start_date=p.index[400])
    pd.testing.assert_frame_equal(a.summary, b.summary)
    pd.testing.assert_frame_equal(a.decisions, b.decisions)


def test_unused_future_corruption_does_not_invalidate_completed_study():
    p, b = prices(), prices(seed=9)
    original = compare_deployment_policies('TEST', p, b, start_date=p.index[400])
    p.iloc[600:] = np.nan
    b.iloc[600:] = np.nan
    changed = compare_deployment_policies('TEST', p, b, start_date=p.index[400])
    pd.testing.assert_frame_equal(original.summary, changed.summary)
    pd.testing.assert_frame_equal(original.decisions, changed.decisions)
