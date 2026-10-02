"""Scoped, append-only deployment plans and review/execution evidence.

Uses the application's SQLite/libSQL connection. Callers supply an authenticated
scope and actor; this module is not an authorization boundary or an execution ledger.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from src.analytics.deployment import DeploymentPlan, DeploymentState, day, propose_deployment
from src.portfolio_tracker.strategy_store import _commit_and_sync, _row_value


def _encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _schema(connection):
    connection.execute('''CREATE TABLE IF NOT EXISTS deployment_plans (
        scope TEXT NOT NULL, plan_id TEXT NOT NULL, created_at TEXT NOT NULL,
        actor TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(scope, plan_id))''')
    connection.execute('''CREATE TABLE IF NOT EXISTS deployment_events (
        scope TEXT NOT NULL, plan_id TEXT NOT NULL, event_id TEXT NOT NULL,
        kind TEXT NOT NULL, review_date TEXT, created_at TEXT NOT NULL,
        actor TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(scope, event_id))''')
    connection.execute('''CREATE UNIQUE INDEX IF NOT EXISTS deployment_review_once
        ON deployment_events(scope, plan_id, review_date) WHERE kind = 'review' ''')


def _identity(scope, actor):
    if not isinstance(scope, str) or not scope.strip() or not isinstance(actor, str) or not actor.strip():
        raise ValueError('Authenticated scope and actor required.')


def _now():
    return datetime.now(timezone.utc).isoformat()


def save_deployment_plan(connection, plan: DeploymentPlan, *, scope, actor):
    _identity(scope, actor)
    plan.validate()
    _schema(connection)
    plan_id = uuid4().hex
    connection.execute('INSERT INTO deployment_plans VALUES (?, ?, ?, ?, ?)',
                       (scope, plan_id, _now(), actor, _encode(plan.to_dict())))
    _commit_and_sync(connection)
    return plan_id


def list_deployment_plans(connection, *, scope, ticker=None):
    _identity(scope, 'reader')
    _schema(connection)
    rows = connection.execute('SELECT plan_id, created_at, actor, payload FROM deployment_plans '
                              'WHERE scope = ? ORDER BY created_at DESC, plan_id', (scope,)).fetchall()
    records = [dict(plan_id=_row_value(r, 'plan_id', 0), created_at=_row_value(r, 'created_at', 1),
                    actor=_row_value(r, 'actor', 2), plan=json.loads(_row_value(r, 'payload', 3))) for r in rows]
    return [r for r in records if ticker is None or r['plan']['ticker'].upper() == ticker.upper()]


def _plan(connection, scope, plan_id):
    rows = [r for r in list_deployment_plans(connection, scope=scope) if r['plan_id'] == plan_id]
    if not rows:
        raise ValueError('Plan not found in this scope.')
    return DeploymentPlan(**rows[0]['plan'])


def deployment_events(connection, *, scope, plan_id):
    _plan(connection, scope, plan_id)
    rows = connection.execute('SELECT event_id, kind, created_at, actor, payload FROM deployment_events '
        'WHERE scope = ? AND plan_id = ? ORDER BY created_at, event_id', (scope, plan_id)).fetchall()
    return [dict(event_id=_row_value(r, 'event_id', 0), kind=_row_value(r, 'kind', 1),
                 created_at=_row_value(r, 'created_at', 2), actor=_row_value(r, 'actor', 3),
                 payload=json.loads(_row_value(r, 'payload', 4))) for r in rows]


def record_deployment_review(connection, *, scope, actor, plan_id, state: DeploymentState,
                             as_of, entry=None, signal=None):
    _identity(scope, actor)
    plan = _plan(connection, scope, plan_id)
    events = deployment_events(connection, scope=scope, plan_id=plan_id)
    if any(e['kind'] == 'closure' for e in events):
        raise ValueError('This plan is closed. Create a new approved plan for further deployment.')
    executed = sum(e['payload']['notional'] for e in events if e['kind'] == 'execution')
    if state.spent + .01 < executed:
        raise ValueError('Snapshot spending is below the executions already recorded for this plan.')
    decision = propose_deployment(plan, state, as_of=as_of, entry=entry, signal=signal)
    if decision.scheduled_review is None:
        raise ValueError('No scheduled review is due.')
    payload = dict(plan=plan.to_dict(), state=asdict(state), decision=decision.to_dict(),
                   entry=entry.to_dict() if entry else None,
                   entry_algorithm_signal=signal.to_dict() if signal else None)
    event_id = hashlib.sha256((plan_id + _encode(payload)).encode()).hexdigest()
    existing = next((e for e in events if e['kind'] == 'review' and
                     e['payload']['decision']['scheduled_review'] == decision.scheduled_review), None)
    if existing:
        if existing['event_id'] == event_id:
            return event_id
        raise ValueError('This calendar review is already saved. Existing evidence cannot be overwritten.')
    latest = max((e['payload']['decision']['scheduled_review'] for e in events if e['kind'] == 'review'), default=None)
    if latest and decision.scheduled_review < latest:
        raise ValueError('A review cannot be added before an already recorded review.')
    connection.execute('INSERT INTO deployment_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (scope, plan_id, event_id, 'review', decision.scheduled_review, _now(), actor, _encode(payload)))
    _commit_and_sync(connection)
    return event_id


def close_deployment_plan(connection, *, scope, actor, plan_id, reason):
    """Preserve an unfunded-completion/cancellation exception without deleting evidence."""
    _identity(scope, actor)
    if not reason.strip():
        raise ValueError('A closure / completion exception reason is required.')
    events = deployment_events(connection, scope=scope, plan_id=plan_id)
    existing = next((e for e in events if e['kind'] == 'closure'), None)
    if existing:
        if existing['payload']['reason'] == reason.strip():
            return existing['event_id']
        raise ValueError('Plan closure has already been recorded.')
    event_id = hashlib.sha256((plan_id + ':closure').encode()).hexdigest()
    connection.execute('INSERT INTO deployment_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (scope, plan_id, event_id, 'closure', None, _now(), actor, _encode({'reason': reason.strip()})))
    _commit_and_sync(connection)
    return event_id


def record_deployment_execution(connection, *, scope, actor, plan_id, review_id,
        execution_reference, execution_date, notional, fees=0., override_reason=''):
    from src.analytics.deployment import nonnegative
    _identity(scope, actor)
    for value, name in ((notional, 'notional'), (fees, 'fees')):
        nonnegative(value, name)
    if notional <= 0 or not execution_reference.strip():
        raise ValueError('Positive executed notional and a unique external execution reference are required.')
    events = deployment_events(connection, scope=scope, plan_id=plan_id)
    review = next((e for e in events if e['event_id'] == review_id and e['kind'] == 'review'), None)
    if review is None:
        raise ValueError('Saved review not found.')
    if day(execution_date) < day(review['payload']['decision']['as_of']):
        raise ValueError('Execution cannot predate its review.')
    payload = dict(review_id=review_id, execution_reference=execution_reference.strip(),
                   execution_date=str(day(execution_date).date()), notional=float(notional),
                   fees=float(fees), override_reason=override_reason.strip(),
                   source='Manually reported; reconcile with the official execution ledger.')
    event_id = hashlib.sha256((plan_id + ':execution:' + execution_reference.strip()).encode()).hexdigest()
    existing = next((e for e in events if e['event_id'] == event_id), None)
    if existing:
        if existing['payload'] == payload:
            return event_id
        raise ValueError('Execution reference already exists with different values.')
    previous = sum(e['payload']['notional'] for e in events if e['kind'] == 'execution' and
                   e['payload']['review_id'] == review_id)
    if previous + notional > review['payload']['decision']['proposed_purchase'] + .01 and not override_reason.strip():
        raise ValueError('Execution exceeds the proposal; document an override reason.')
    connection.execute('INSERT INTO deployment_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (scope, plan_id, event_id, 'execution', None, _now(), actor, _encode(payload)))
    _commit_and_sync(connection)
    return event_id
