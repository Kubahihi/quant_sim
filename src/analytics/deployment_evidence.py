"""Evidence inventory and bounded claims, never a competition-compliance certificate."""
from dataclasses import asdict
from hashlib import sha256
import json

from src.analytics.deployment import DeploymentPlan, day


OFFICIAL_SOURCES = [
    {'title': 'Wharton competition overview',
     'url': 'https://globalyouth.wharton.upenn.edu/competitions/investment-competition/'},
    {'title': 'Wharton general rules and AI policy',
     'url': 'https://globalyouth.wharton.upenn.edu/competitions/investment-competition/rules-roles/'},
    {'title': 'Wharton FAQ: annual materials and judging',
     'url': 'https://globalyouth.wharton.upenn.edu/competitions/investment-competition/faq/'},
]
REQUIRED_CONTEXT = {
    'client_objective': 'Link to the client objective and IPS',
    'deployment_rationale': 'Why this schedule instead of immediate deployment, including cash opportunity cost',
    'benchmark_rationale': 'Why this benchmark and currency are appropriate',
    'annual_rules_reference': 'Reference to the team-verified annual competition materials',
    'ai_disclosure': 'Student-reviewed disclosure and citation of AI assistance',
}


def build_deployment_evidence(plan_record, events, *, as_of):
    """Build a factual working paper; never invent rationale, approvals or outcomes."""
    plan = DeploymentPlan(**plan_record['plan'])
    plan.validate()
    context = plan.competition_context
    missing = [description for key, description in REQUIRED_CONTEXT.items() if not context.get(key, '').strip()]
    reviews = [e for e in events if e.get('kind') == 'review']
    executions = [e for e in events if e.get('kind') == 'execution']
    if not plan_record.get('plan_id'):
        missing.append('Persisted approved plan identifier')
    created = plan_record.get('created_at')
    chronology = 'Not established'
    if created:
        try:
            chronology = ('Recorded by first review date; intraday ordering unverified'
                          if day(created) <= day(plan.start_date) else 'Recorded after first review date: retrospective plan')
        except (ValueError, TypeError):
            chronology = 'Invalid creation timestamp'
    if chronology != 'Recorded by first review date; intraday ordering unverified':
        missing.append('Evidence that policy was recorded before implementation')
    if not reviews:
        missing.append('Recorded review with a reconciled portfolio reference')
    else:
        if any(not r['payload'].get('state', {}).get('snapshot_reference', '').strip() for r in reviews):
            missing.append('Portfolio source reference for every review')
    research_only = plan.policy in {'Score-adjusted', 'Signal-gated'}
    payload = {
        'document_type': 'Analytical working paper for independent student review; not a submission',
        'as_of': str(day(as_of).date()),
        'evidence_status': ('Research policy — performance unvalidated' if research_only else
                            'Documentation incomplete' if missing else 'Process documented — student verification required'),
        'plan_record': plan_record, 'events': events,
        'missing_evidence': missing, 'policy_chronology': chronology,
        'recorded_reviews': len(reviews), 'manually_reported_executions': len(executions),
        'recorded_notional': sum(float(e['payload']['notional']) for e in executions),
        'defensible_scope': [
            'A documented implementation process for a separately selected security and target allocation.',
            'A prespecified calendar, cash and position limits, review records, and explicit exceptions.',
            'Historical comparisons describe the stated sample and assumptions only.',
        ],
        'unsupported_claims': [
            'The score predicts prices, identifies optimal purchase dates, or proves superior returns.',
            'Analogue frequencies are calibrated probabilities or authorize purchases.',
            'Passing software tests demonstrates investment effectiveness.',
            'This working paper certifies compliance with current private competition requirements.',
        ],
        'known_limitations': [
            'Portfolio and execution evidence is manually supplied and requires official-ledger reconciliation.',
            'Future chart dates use weekdays rather than an exchange holiday calendar.',
            'Adjusted histories may be revised; selection, regime and dependence biases remain.',
            'Current strategy backtests do not trade on the analogue timing estimator.',
        ],
        'ai_provenance': 'Implementation and methodology text were developed with OpenAI Codex assistance. '
                         'Students must independently verify the analysis and cite AI-generated material included in their report.',
        'public_rules_checked_on': '2026-09-27', 'sources': OFFICIAL_SOURCES,
        'annual_rules_scope': 'Detailed 2026–2027 materials are distributed through SurveyMonkey Apply. '
                              'They were not supplied to this module; the reference is a team attestation, not an automated check.',
    }
    payload['policy_sha256'] = sha256(json.dumps(asdict(plan), sort_keys=True, allow_nan=False).encode()).hexdigest()
    return payload


def evidence_markdown(evidence):
    """No first-person team narrative and no fabricated investment conclusion."""
    plan = evidence['plan_record']['plan']
    lines = ['# Deployment evidence working paper', '', evidence['document_type'], '',
             f"Status: {evidence['evidence_status']}", f"Security: {plan['ticker']}",
             f"Policy: {plan['policy']}; budget: {plan['budget']:,.2f}; target: {plan['target_weight']:.1%}",
             f"Policy chronology: {evidence['policy_chronology']}", '', '## Team-supplied rationale', '']
    for key, label in REQUIRED_CONTEXT.items():
        value = plan.get('competition_context', {}).get(key, '').strip()
        lines.extend([f'**{label}**', '', value or '[Not supplied — student input required]', ''])
    for key, heading in [('missing_evidence', 'Missing evidence'), ('defensible_scope', 'Scope of support'),
                         ('unsupported_claims', 'Claims not supported'), ('known_limitations', 'Limitations')]:
        lines.extend([f'## {heading}', '', *[f'- {v}' for v in evidence[key]], ''])
    lines.extend(['## Attribution and source scope', '', evidence['ai_provenance'], '', evidence['annual_rules_scope'], '',
                  *[f"- [{s['title']}]({s['url']})" for s in evidence['sources']], '',
                  f"Policy fingerprint: {evidence['policy_sha256']}"])
    return '\n'.join(lines)
