import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.analytics.deployment import DeploymentPlan, DeploymentState
from src.analytics.deployment_timing import TimingEstimate
from src.visualization.deployment_chart import build_deployment_chart


def plan():
    return DeploymentPlan('TEST', 'SPY', .05, 25000, '2025-01-06', thesis='Quality',
        invalidation='Margins', horizon='5 years', mandate_reference='Committee')


def prices():
    return pd.Series(np.linspace(100, 110, 300), index=pd.bdate_range(end='2025-01-20', periods=300))


def events():
    return [dict(event_id='review1', kind='review', payload=dict(entry={'entry_score': 70},
            decision={'explanation': 'Calendar tranche'})),
            dict(event_id='fill1', kind='execution', payload=dict(execution_date='2025-01-07',
                review_id='review1', execution_reference='OFFICIAL1', notional=5000, override_reason=''))]


def test_actuals_future_and_calendar_are_separate():
    fig = build_deployment_chart(plan(), prices(), events(), as_of='2025-01-10')
    fig.to_json()
    observed = next(t for t in fig.data if t.name == 'Observed adjusted close')
    assert max(observed.x) <= pd.Timestamp('2025-01-10')
    actual = next(t for t in fig.data if t.name == 'Recorded executed notional')
    assert actual.y[-1] == 5000
    assert max(actual.x) == pd.Timestamp('2025-01-10')
    assert any('not executions' in (t.name or '') for t in fig.data)
    assert 'OFFICIAL1' in fig.to_json()


def test_unknown_snapshot_history_is_not_invented():
    state = DeploymentState(500000, 10000, 25000, spent=10000, snapshot_date='2025-01-10', snapshot_reference='WInS')
    fig = build_deployment_chart(plan(), prices(), events(), as_of='2025-01-10', state=state)
    actual = next(t for t in fig.data if t.name == 'Recorded executed notional')
    assert actual.y[-1] == 5000
    assert any('incomplete execution history' in (t.name or '') for t in fig.data)


def test_shading_only_for_supported_matching_estimate():
    timing = TimingEstimate('2025-01-10', 'estimated', 'Condition', sample_count=12,
        probabilities={'5': .25, '10': .5, '20': .75}, no_hit_probability=.25)
    fig = build_deployment_chart(plan(), prices(), [], as_of=timing.as_of, timing=timing)
    assert sum(s.type == 'rect' for s in fig.layout.shapes) == 3
    timing.status = 'insufficient'
    fig = build_deployment_chart(plan(), prices(), [], as_of=timing.as_of, timing=timing)
    assert not any(s.type == 'rect' for s in fig.layout.shapes)


def test_closed_plan_has_no_future_proposals_or_windows():
    fig = build_deployment_chart(plan(), prices(), events(), as_of='2025-01-10', closed=True)
    assert not any('calendar' in (t.name or '').lower() for t in fig.data)
    assert not any(s.type == 'rect' for s in fig.layout.shapes)


def test_no_price_data_still_shows_reported_deployment():
    fig = build_deployment_chart(plan(), None, events(), as_of='2025-01-10')
    assert next(t for t in fig.data if t.name == 'Recorded executed notional').y[-1] == 5000
    assert json.loads(fig.to_json())['data']


def test_signal_gated_chart_marks_the_security_specific_model_action():
    signal_plan = DeploymentPlan('TEST', 'SPY', .05, 25000, '2025-01-06', reviews=5, interval_days=5,
        policy='Signal-gated', thesis='Quality', invalidation='Margins', horizon='5 years', mandate_reference='Committee')
    signal = SimpleNamespace(status='research_ready', action='Wait for next model review',
        next_review='2025-01-15', predicted_excess_return=.012)
    fig = build_deployment_chart(signal_plan, prices(), [], as_of='2025-01-10', signal=signal)
    assert any(t.name == 'Model: re-evaluate remaining tranche' for t in fig.data)
    assert 'Model: wait until 2025-01-15' in fig.to_json()
