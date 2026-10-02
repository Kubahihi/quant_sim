from dataclasses import replace

from src.analytics.deployment import DeploymentPlan
from src.analytics.deployment_evidence import REQUIRED_CONTEXT, build_deployment_evidence, evidence_markdown


def record(**kwargs):
    plan = DeploymentPlan('TEST', 'SPY', .05, 25000, '2026-10-01', thesis='Quality',
        invalidation='Margin deterioration', horizon='5 years', mandate_reference='IPS 1', **kwargs)
    return dict(plan_id='saved-plan', created_at='2026-09-27T12:00:00+00:00', actor='student', plan=plan.to_dict())


def test_missing_rationale_cannot_be_reported_as_documented():
    evidence = build_deployment_evidence(record(), [], as_of='2026-10-01')
    assert evidence['evidence_status'] == 'Documentation incomplete'
    assert len(evidence['missing_evidence']) == len(REQUIRED_CONTEXT) + 1
    text = evidence_markdown(evidence)
    assert '[Not supplied' in text
    assert 'Codex' in text
    assert 'SurveyMonkey Apply' in text
    assert 'https://globalyouth.wharton.upenn.edu/' in text


def test_documented_process_never_certifies_performance_or_compliance():
    context = {k: 'Student-supplied explanation' for k in REQUIRED_CONTEXT}
    event = dict(kind='review', payload=dict(state={'snapshot_reference': 'WInS snapshot'}))
    evidence = build_deployment_evidence(record(competition_context=context), [event], as_of='2026-10-01')
    assert evidence['evidence_status'] == 'Process documented — student verification required'
    assert not evidence['missing_evidence']
    assert any('superior returns' in v for v in evidence['unsupported_claims'])
    assert 'not an automated check' in evidence['annual_rules_scope']


def test_research_policy_remains_unvalidated_and_retrospective_plan_flagged():
    r = record(policy='Score-adjusted')
    r['created_at'] = '2026-10-02'
    evidence = build_deployment_evidence(r, [], as_of='2026-10-03')
    assert 'unvalidated' in evidence['evidence_status']
    assert 'retrospective' in evidence['policy_chronology']


def test_signal_gated_policy_remains_unvalidated():
    evidence = build_deployment_evidence(record(policy='Signal-gated'), [], as_of='2026-10-01')
    assert 'unvalidated' in evidence['evidence_status']


def test_policy_hash_changes_with_student_rationale():
    a = record()
    b = record(competition_context={'client_objective': 'Liquidity reserve'})
    assert build_deployment_evidence(a, [], as_of='2026-10-01')['policy_sha256'] != build_deployment_evidence(b, [], as_of='2026-10-01')['policy_sha256']
