from dataclasses import replace
import sqlite3
from types import SimpleNamespace

import pytest

from src.analytics.deployment import DeploymentPlan, DeploymentState
from src.portfolio_tracker.deployment_store import (
    save_deployment_plan, list_deployment_plans, record_deployment_review,
    record_deployment_execution, deployment_events,
    close_deployment_plan,
)


@pytest.fixture
def store():
    conn = sqlite3.connect(':memory:')
    p = DeploymentPlan('TEST', 'SPY', .05, 25_000, '2025-01-06', thesis='Quality',
        invalidation='Earnings failure', horizon='5y', mandate_reference='Committee approval')
    pid = save_deployment_plan(conn, p, scope='team', actor='analyst')
    s = DeploymentState(500_000, 0, 25_000, snapshot_date='2025-01-06', snapshot_reference='WInS 1')
    yield conn, pid, s
    conn.close()


def test_scope_isolation(store):
    conn, pid, _ = store
    assert len(list_deployment_plans(conn, scope='team')) == 1
    assert list_deployment_plans(conn, scope='other') == []
    with pytest.raises(ValueError, match='scope'):
        deployment_events(conn, scope='other', plan_id=pid)
    with pytest.raises(ValueError, match='scope'):
        list_deployment_plans(conn, scope=None)


def test_review_idempotent_and_immutable(store):
    conn, pid, state = store
    kwargs = dict(scope='team', actor='analyst', plan_id=pid, state=state, as_of='2025-01-06')
    rid = record_deployment_review(conn, **kwargs)
    assert record_deployment_review(conn, **kwargs) == rid
    with pytest.raises(ValueError, match='already saved'):
        record_deployment_review(conn, **{**kwargs, 'state': replace(state, cash=20_000)})
    event = deployment_events(conn, scope='team', plan_id=pid)[0]
    assert event['payload']['state']['snapshot_reference'] == 'WInS 1'
    assert event['payload']['plan']['version'] == '1.0'


def test_execution_override_and_duplicate_protection(store):
    conn, pid, state = store
    rid = record_deployment_review(conn, scope='team', actor='analyst', plan_id=pid, state=state, as_of='2025-01-06')
    args = dict(scope='team', actor='analyst', plan_id=pid, review_id=rid,
        execution_reference='fill-1', execution_date='2025-01-07', notional=6_000)
    with pytest.raises(ValueError, match='override'):
        record_deployment_execution(conn, **args)
    eid = record_deployment_execution(conn, **args, override_reason='Committee approved acceleration')
    assert record_deployment_execution(conn, **args, override_reason='Committee approved acceleration') == eid
    with pytest.raises(ValueError, match='different values'):
        record_deployment_execution(conn, **{**args, 'notional': 5_000})
    assert len(deployment_events(conn, scope='team', plan_id=pid)) == 2
    with pytest.raises(ValueError, match='below the executions'):
        record_deployment_review(conn, scope='team', actor='analyst', plan_id=pid,
            state=replace(state, snapshot_date='2025-01-13'), as_of='2025-01-13')


def test_cannot_record_execution_before_review(store):
    conn, pid, state = store
    rid = record_deployment_review(conn, scope='team', actor='analyst', plan_id=pid, state=state, as_of='2025-01-06')
    with pytest.raises(ValueError, match='predate'):
        record_deployment_execution(conn, scope='team', actor='analyst', plan_id=pid,
            review_id=rid, execution_reference='fill-1', execution_date='2025-01-05', notional=1_000)


def test_plan_closure_preserves_evidence_and_stops_new_reviews(store):
    conn, pid, state = store
    with pytest.raises(ValueError, match='reason'):
        close_deployment_plan(conn, scope='team', actor='analyst', plan_id=pid, reason='')
    eid = close_deployment_plan(conn, scope='team', actor='analyst', plan_id=pid,
                               reason='Unfunded balance cancelled after thesis review')
    assert eid == close_deployment_plan(conn, scope='team', actor='analyst', plan_id=pid,
                               reason='Unfunded balance cancelled after thesis review')
    with pytest.raises(ValueError, match='closed'):
        record_deployment_review(conn, scope='team', actor='analyst', plan_id=pid,
                                  state=state, as_of='2025-01-06')
    assert len(list_deployment_plans(conn, scope='team')) == 1


def test_signal_gated_review_retains_the_same_date_algorithm_evidence(store):
    conn, _, _ = store
    plan = DeploymentPlan('MODEL', 'SPY', .05, 25_000, '2025-01-06', reviews=5, interval_days=5,
        policy='Signal-gated', thesis='Quality', invalidation='Earnings failure', horizon='5y',
        mandate_reference='Committee approval')
    plan_id = save_deployment_plan(conn, plan, scope='team', actor='analyst')
    initial = DeploymentState(500_000, 0, 25_000, snapshot_date='2025-01-06', snapshot_reference='WInS 1')
    record_deployment_review(conn, scope='team', actor='analyst', plan_id=plan_id, state=initial, as_of='2025-01-06')
    signal = SimpleNamespace(as_of='2025-01-13', status='research_ready', action='Wait for next model review',
        to_dict=lambda: {'as_of': '2025-01-13', 'action': 'Wait for next model review', 'fixed_model': True})
    second = DeploymentState(500_000, 12_500, 12_500, spent=12_500, snapshot_date='2025-01-13',
        snapshot_reference='WInS 2', last_review_date='2025-01-06')
    record_deployment_review(conn, scope='team', actor='analyst', plan_id=plan_id, state=second,
                             as_of='2025-01-13', signal=signal)
    event = deployment_events(conn, scope='team', plan_id=plan_id)[-1]
    assert event['payload']['entry_algorithm_signal']['fixed_model']
