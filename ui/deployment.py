"""Reviewable deployment plans and comparative evidence within company research."""
from contextlib import closing
from dataclasses import asdict
from io import BytesIO
import json
import zipfile

import pandas as pd
import streamlit as st

from src.analytics.deployment import DeploymentPlan, DeploymentState, POLICIES, propose_deployment
from src.analytics.deployment_backtest import compare_deployment_policies, compare_deployment_windows
from src.analytics.deployment_evidence import build_deployment_evidence, evidence_markdown, REQUIRED_CONTEXT
from src.portfolio_tracker.deployment_store import (
    deployment_events, list_deployment_plans, record_deployment_execution,
    record_deployment_review, save_deployment_plan,
    close_deployment_plan,
)


def render_deployment_planner(ticker, entry, prices, benchmark_prices, *, key_prefix,
                              connection_factory=None, scope=None, actor=None, can_edit=True,
                              research_mode=None):
    prefix = f'{key_prefix}_{ticker}_deployment'
    if not st.checkbox('Open deployment planner', key=f'{prefix}_open'):
        return
    if research_mode is None:
        research_mode = st.checkbox('Research mode — unvalidated sizing and timing', key=f'{prefix}_research')
    st.markdown('### Deployment plan')
    if not research_mode:
        st.info('Competition view: document a client-linked implementation process. '
                'Experimental timing and score-based purchase sizing are excluded from this view.')
    st.caption('Plan purchases for an approved thesis. Future score adjustments are conditional. '
               'No orders are submitted. Portfolio inputs below must be reconciled with the official ledger.')
    persist = connection_factory is not None and bool(scope) and bool(actor)
    saved = []
    if persist:
        try:
            with closing(connection_factory()) as conn:
                saved = list_deployment_plans(conn, scope=scope, ticker=ticker)
        except Exception as exc:
            st.error(f'Could not load deployment plans: {exc}')
            return
    options = ['New plan'] + [r['plan_id'] for r in saved]
    if f'{prefix}_next_selection' in st.session_state:
        st.session_state[f'{prefix}_selected'] = st.session_state.pop(f'{prefix}_next_selection')
    selected = st.selectbox('Deployment plan', options, key=f'{prefix}_selected',
        format_func=lambda x: x if x == 'New plan' else next(
            f"{r['plan']['start_date']} · {r['plan']['policy']} · {x[:8]}" for r in saved if r['plan_id'] == x))
    if st.session_state.get(f'{prefix}_previous_selection') != selected:
        # Attestation belongs to this plan; never carry an approval across plans.
        for suffix in ('valid', 'constraints', 'reference'):
            st.session_state.pop(f'{prefix}_{suffix}', None)
        st.session_state[f'{prefix}_previous_selection'] = selected
    events = []
    if selected == 'New plan':
        left, right = st.columns(2)
        source_target = entry.target_weight * 100 if entry.target_weight and entry.target_weight > 0 else 5.
        policy = left.selectbox('Purchase policy', POLICIES if research_mode else POLICIES[:2], index=1, key=f'{prefix}_policy')
        budget = right.number_input('Approved incremental budget', min_value=1., value=25_000., key=f'{prefix}_budget')
        start = left.date_input('First review date', value=pd.Timestamp(entry.as_of).date(), key=f'{prefix}_start')
        target_weight = right.number_input('Target portfolio weight (%)', min_value=.1, max_value=100.,
                                           value=float(source_target), step=.1, key=f'{prefix}_target_weight')
        pattern = right.selectbox('Deployment pattern', ['Equal schedule', 'Two-stage: 50% now / 50% later'],
                                  key=f'{prefix}_pattern', disabled=policy == 'Signal-gated')
        if policy == 'Signal-gated':
            interval = left.number_input('Business-day proxy between model reviews', min_value=1, max_value=21,
                                         value=5, key=f'{prefix}_signal_interval')
            max_reviews = min(13, 63 // int(interval) + 1)
            reviews = right.number_input('Model review count including completion', min_value=3, max_value=max_reviews,
                                         value=min(5, max_reviews), key=f'{prefix}_signal_reviews')
            st.caption('Research policy: buy 50% at the first review. The model evaluates the remaining 50% '
                       'at each later review and forces completion at the final review.')
        elif pattern == 'Two-stage: 50% now / 50% later':
            reviews = 2
            interval = left.number_input('Days until second-half review', min_value=1, max_value=365,
                                         value=14, key=f'{prefix}_second_half_days')
            st.caption('The remaining half is reviewed on a fixed date. It is not triggered by a price forecast.')
        else:
            reviews = right.number_input('Number of reviews', min_value=1, max_value=52, value=5, key=f'{prefix}_reviews')
            interval = left.number_input('Days between reviews', min_value=1, max_value=365, value=7, key=f'{prefix}_interval')
        max_weight = right.number_input('Maximum position weight (%)', min_value=float(target_weight), max_value=100.,
                                        value=max(10., float(target_weight)), key=f'{prefix}_max_weight')
        max_purchase = left.number_input('Maximum purchase per review', min_value=1., value=25_000., key=f'{prefix}_max_purchase')
        cost_bps = right.number_input('Estimated execution costs (basis points)', min_value=0., max_value=1000.,
                                      value=10., key=f'{prefix}_cost_bps')
        thesis = st.text_area('Approved investment thesis', key=f'{prefix}_thesis')
        invalidation = st.text_area('Conditions that invalidate the thesis', key=f'{prefix}_invalidation')
        horizon = st.text_input('Investment horizon', key=f'{prefix}_horizon')
        mandate = st.text_input('Mandate / approval reference', key=f'{prefix}_mandate')
        context = {}
        with st.expander('Team rationale and competition evidence'):
            st.caption('Write your own reasoning. Empty fields remain explicitly missing in the evidence export; '
                       'they are never generated on the team’s behalf. Verify current private annual rules.')
            for field, label in REQUIRED_CONTEXT.items():
                context[field] = st.text_area(label, key=f'{prefix}_context_{field}')
        plan = DeploymentPlan(ticker, entry.benchmark, target_weight / 100, budget, str(start),
            int(reviews), int(interval), policy, thesis, invalidation, horizon, mandate,
            max_weight / 100, max_purchase, cost_bps=cost_bps, competition_context=context)
        if st.button('Save approved plan', key=f'{prefix}_save', disabled=not (persist and can_edit)):
            try:
                with closing(connection_factory()) as conn:
                    plan_id = save_deployment_plan(conn, plan, scope=scope, actor=actor)
                st.session_state[f'{prefix}_next_selection'] = plan_id
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f'Plan could not be saved: {exc}')
        if not persist:
            st.caption('Preview mode: saving requires an authenticated application workspace.')
    else:
        plan = DeploymentPlan(**next(r['plan'] for r in saved if r['plan_id'] == selected))
        with closing(connection_factory()) as conn:
            events = deployment_events(conn, scope=scope, plan_id=selected)
        st.write(f'**{plan.policy}** · Approved budget {plan.budget:,.2f} · Target {plan.target_weight:.1%}')
        st.caption(f'Thesis: {plan.thesis} | Invalidation: {plan.invalidation} | Horizon: {plan.horizon}')
    try:
        plan.validate()
    except ValueError as exc:
        st.info(str(exc))
        _render_comparison(ticker, entry.benchmark, prices, benchmark_prices, prefix, research_mode=research_mode)
        return
    if selected != 'New plan':
        record = next(r for r in saved if r['plan_id'] == selected)
        _render_evidence(record, events, entry.as_of, prefix)
    if not research_mode and plan.policy not in POLICIES[:2]:
        st.warning('This saved plan uses an unvalidated research policy. It has not been converted or overwritten. '
                   'Enable Research mode to inspect its proposals; use a separately approved calendar plan for the competition process.')
        return
    schedule_rows = [dict(review=i + 1, date=str(d.date()),
        baseline_cumulative_budget=(plan.budget if plan.policy == 'Immediate' else
                                    (plan.budget / 2 if plan.policy == 'Signal-gated' and i == 0 else
                                     plan.budget if plan.policy == 'Signal-gated' and i == plan.reviews - 1 else
                                     (i + 1) * plan.budget / plan.reviews)),
        execution='Next eligible session after review') for i, d in enumerate(plan.dates())]
    if plan.is_two_stage():
        schedule_rows[0]['stage'] = 'First half (50%)'
        schedule_rows[1]['stage'] = 'Second half (50%) — completion review'
        st.info(f"Two-stage plan: first half at {plan.dates()[0].date()}, then review the remaining half on "
                f"{plan.dates()[1].date()}. The second date is fixed when the plan is saved.")
    if plan.policy == 'Signal-gated':
        schedule_rows[0]['stage'] = 'Initial half (50%)'
        for row in schedule_rows[1:-1]:
            row['stage'] = 'Model review of remaining half'
        schedule_rows[-1]['stage'] = 'Forced completion review'
        st.warning('Research policy: its model output is a candidate signal, not established investment edge. '
                   'The saved final review prevents indefinite deferral.')
    st.dataframe(pd.DataFrame(schedule_rows),
        hide_index=True, use_container_width=True)
    st.caption('Review dates are calendar dates. A current eligible quote, constraints check and human execution review remain necessary.')
    _render_hybrid_entry_framework(plan, prices, benchmark_prices, entry.as_of, prefix)
    st.markdown('#### Current portfolio snapshot')
    a, b = st.columns(2)
    value = a.number_input('Portfolio value', min_value=1., value=500_000., key=f'{prefix}_value')
    holding = b.number_input('Current security holding value', min_value=0., value=0., key=f'{prefix}_holding')
    cash = a.number_input('Total cash', min_value=0., value=25_000., key=f'{prefix}_cash')
    recorded_spent = sum(e['payload']['notional'] for e in events if e['kind'] == 'execution')
    spent = b.number_input('Executed notional under this plan', min_value=float(recorded_spent),
                           value=float(recorded_spent), key=f'{prefix}_{selected}_spent')
    pending = a.number_input('Pending purchases for this security', min_value=0., value=0., key=f'{prefix}_pending')
    pending_plan = b.number_input('Pending purchases under this plan', min_value=0., value=0., key=f'{prefix}_pending_plan')
    reserved = a.number_input('Cash reserved for all pending orders', min_value=0., value=0., key=f'{prefix}_reserved')
    snapshot_date = b.date_input('Portfolio snapshot date', value=pd.Timestamp(entry.as_of).date(), key=f'{prefix}_snapshot_date')
    reference = st.text_input('Portfolio snapshot / reconciliation reference', key=f'{prefix}_reference')
    thesis_valid = st.checkbox('Thesis remains valid', key=f'{prefix}_valid')
    constraints_valid = st.checkbox('Portfolio limits, liquidity, cash for fees and mandate checks passed', key=f'{prefix}_constraints')
    review_events = [e for e in events if e['kind'] == 'review']
    closed = any(e['kind'] == 'closure' for e in events)
    if closed:
        st.info('This plan is closed. Its evidence remains available; further deployment needs a new approved plan.')
    last_review = max((e['payload']['decision']['scheduled_review'] for e in review_events), default=None)
    state = DeploymentState(value, holding, cash, spent, pending, pending_plan, reserved,
        thesis_valid, constraints_valid and not closed, str(snapshot_date), reference, last_review)
    signal = None
    if plan.policy == 'Signal-gated':
        signal = _render_entry_algorithm(plan, prices, benchmark_prices, entry.as_of, prefix)
    try:
        decision = propose_deployment(plan, state, as_of=entry.as_of, entry=entry, signal=signal)
    except ValueError as exc:
        st.warning(str(exc))
        decision = None
    if decision:
        a, b, c = st.columns(3)
        a.metric('Proposed purchase', f'{decision.proposed_purchase:,.2f}')
        b.metric('Uncommitted plan budget', f'{decision.remaining_budget:,.2f}')
        c.metric('Next review', decision.next_review or 'Completion review reached')
        if plan.is_two_stage() and decision.next_review:
            st.caption(f'Second-half review: {decision.next_review}. It proposes the remaining budget only if '
                       'the saved thesis, mandate, cash and portfolio checks pass.')
        st.write(f'**{decision.status}** — {decision.explanation}')
        st.caption(f'Estimated execution costs: {decision.estimated_cost:,.2f}; cash cap includes these costs.')
        if selected != 'New plan' and st.button('Record review', key=f'{prefix}_review', disabled=not can_edit or closed):
            try:
                with closing(connection_factory()) as conn:
                    record_deployment_review(conn, scope=scope, actor=actor, plan_id=selected,
                        state=state, as_of=entry.as_of, entry=entry, signal=signal)
                st.rerun()
            except Exception as exc:
                st.error(f'Review could not be recorded: {exc}')
        st.download_button('Download current proposal', json.dumps(dict(plan=plan.to_dict(),
            state=asdict(state), decision=decision.to_dict(), entry=entry.to_dict()), indent=2),
            file_name=f'{ticker}_deployment_proposal.json', mime='application/json', key=f'{prefix}_proposal')
    _render_deployment_visual(plan, prices, benchmark_prices, entry, events,
        state if decision is not None else None, decision, closed, prefix, signal=signal, research_mode=research_mode)
    if review_events:
        with st.expander('Record an actual execution / override'):
            st.caption('Manual evidence only. This does not update portfolio holdings or submit an order. '
                       'Refresh the portfolio snapshot before the next review.')
            review_id = st.selectbox('Saved review', [e['event_id'] for e in review_events], key=f'{prefix}_review_id')
            execution_ref = st.text_input('Unique official execution reference', key=f'{prefix}_exec_ref')
            execution_date = st.date_input('Actual execution date', value=pd.Timestamp(entry.as_of).date(),
                                           key=f'{prefix}_exec_date')
            notional = st.number_input('Actual executed notional', min_value=0., key=f'{prefix}_notional')
            fees = st.number_input('Actual fees', min_value=0., key=f'{prefix}_fees')
            override = st.text_area('Override reason (required when exceeding the proposal)', key=f'{prefix}_override')
            if st.button('Record execution evidence', key=f'{prefix}_execution', disabled=not can_edit):
                try:
                    with closing(connection_factory()) as conn:
                        record_deployment_execution(conn, scope=scope, actor=actor, plan_id=selected,
                            review_id=review_id, execution_reference=execution_ref, execution_date=execution_date,
                            notional=notional, fees=fees, override_reason=override)
                    st.rerun()
                except Exception as exc:
                    st.error(f'Execution could not be recorded: {exc}')
    if selected != 'New plan':
        if not closed:
            with st.expander('Close plan / record completion exception'):
                reason = st.text_area('Closure reason and treatment of any undeployed allocation', key=f'{prefix}_close_reason')
                if st.button('Close deployment plan', key=f'{prefix}_close', disabled=not can_edit):
                    try:
                        with closing(connection_factory()) as conn:
                            close_deployment_plan(conn, scope=scope, actor=actor, plan_id=selected, reason=reason)
                        st.rerun()
                    except Exception as exc:
                        st.error(f'Plan could not be closed: {exc}')
        plan_record = next(r for r in saved if r['plan_id'] == selected)
        st.download_button('Download plan and audit trail', json.dumps(dict(plan_record=plan_record, events=events), indent=2),
            file_name=f'{ticker}_deployment_audit.json', mime='application/json', key=f'{prefix}_audit')
        with st.expander('Saved decision history'):
            st.json(events)
    _render_comparison(ticker, entry.benchmark, prices, benchmark_prices, prefix, research_mode=research_mode)


def _render_evidence(record, events, as_of, prefix):
    evidence = build_deployment_evidence(record, events, as_of=as_of)
    with st.expander('Competition evidence working paper'):
        st.write(evidence['evidence_status'])
        for missing in evidence['missing_evidence']:
            st.caption(f'Missing: {missing}')
        st.caption('This is an evidence inventory, not competition approval or a performance certification.')
        st.caption(evidence['ai_provenance'])
        archive = BytesIO()
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr('working_paper.md', evidence_markdown(evidence))
            bundle.writestr('evidence.json', json.dumps(evidence, indent=2))
        st.download_button('Download competition evidence working paper', archive.getvalue(),
            file_name=f"{record['plan']['ticker']}_competition_evidence.zip", mime='application/zip',
            key=f'{prefix}_competition_export')


def _render_hybrid_entry_framework(plan, prices, benchmark_prices, as_of, prefix):
    """A concise, exportable comparison for explaining entry discipline in a report."""
    from src.analytics.hybrid_entry import HybridEntryConfig, study_hybrid_entry
    import plotly.graph_objects as go

    with st.expander('Report-ready entry framework: lump sum vs DCA vs hybrid', expanded=True):
        st.caption('Hybrid DCA invests 50% at the first eligible session. The remaining 50% is invested at '
                   'the first of two disclosed conditions: a 5% pullback within a long-term trend, or trend plus '
                   'relative-strength confirmation. It is forced by day 20 if neither appears.')
        config = HybridEntryConfig(cost_bps=plan.cost_bps)
        try:
            with st.spinner('Comparing the three entry methods on non-overlapping completed historical cases…'):
                study = study_hybrid_entry(prices, benchmark_prices, as_of=as_of, config=config)
        except ValueError as exc:
            st.warning(f'Entry framework unavailable: {exc}')
            return
        signal = study.current_signal
        a, b, c = st.columns(3)
        a.metric('Condition today', signal.action)
        b.metric('20-day pullback from high', 'Unavailable' if signal.pullback_from_20d_high is None else
                 f'{signal.pullback_from_20d_high:.1%}')
        c.metric('20-day return versus benchmark', 'Unavailable' if signal.relative_return is None else
                 f'{signal.relative_return:.1%}')
        st.caption('This describes what the fixed rule would do with the remaining tranche today. '
                   'It is separate from the historical evidence below and is not a price forecast.')
        if signal.next_review:
            st.info(f'{signal.condition}. Check again on {signal.next_review}. Once the first tranche is '
                    f'recorded, the completion limit is {signal.forced_completion}.')
        else:
            st.info(f'{signal.condition}. Review the remaining tranche only after a recorded first execution '
                    'and a separate market session, subject to the approved thesis and portfolio checks.')
        hybrid_row = study.summary.loc[study.summary.strategy == 'Hybrid DCA'].iloc[0]
        cases = int(hybrid_row['cases'])
        minimum = study.config['minimum_cases']
        advantage = hybrid_row.get('mean_advantage_vs_lump')
        if cases < minimum:
            st.warning(f'Not report-ready: only {cases} independent historical cases are available; at least {minimum} are required. '
                       'Do not use today\'s rule status as evidence that Hybrid DCA is better.')
        elif pd.isna(advantage):
            st.warning('Not report-ready: the historical comparison has no complete Hybrid DCA result.')
        elif advantage <= 0:
            st.warning(f'Hybrid DCA did not beat lump sum on average in this study ({advantage:.2%}). '
                       'Use lump sum as the historical reference; do not claim a hybrid-entry advantage.')
        else:
            st.info(f'Hybrid DCA beat lump sum by {advantage:.2%} on average in this fixed historical study. '
                    'This is descriptive evidence, not a future-return claim.')
        display_summary = study.summary.copy()
        percentage_columns = ('mean_terminal_return', 'median_terminal_return', 'worst_terminal_return',
                              'mean_advantage_vs_lump', 'mean_advantage_vs_fixed', 'win_rate_vs_lump')
        for column in percentage_columns:
            display_summary[column] = display_summary[column].map(
                lambda value: '—' if pd.isna(value) else f'{value:.2%}')
        display_summary['average_wait_sessions'] = display_summary['average_wait_sessions'].map(
            lambda value: '—' if pd.isna(value) else f'{value:.1f} sessions')
        display_summary = display_summary.rename(columns={
            'strategy': 'Method', 'cases': 'Historical cases',
            'mean_terminal_return': 'Average return to common end date',
            'median_terminal_return': 'Median return', 'worst_terminal_return': 'Worst return',
            'mean_advantage_vs_lump': 'Hybrid average vs lump sum',
            'mean_advantage_vs_fixed': 'Hybrid average vs fixed DCA',
            'win_rate_vs_lump': 'Hybrid wins vs lump sum',
            'average_wait_sessions': 'Average wait for second tranche',
        })
        st.dataframe(display_summary, hide_index=True, use_container_width=True)
        st.caption('How to read the chart: bars are average returns across the historical cases; white dots are median returns. '
                   'The rule status above is a separate, present-day decision rule and is not evidence of future performance.')
        figure = go.Figure()
        colors = {'Lump sum': '#f59e0b', 'Fixed DCA': '#a78bfa', 'Hybrid DCA': '#38bdf8'}
        figure.add_trace(go.Bar(x=study.summary.strategy, y=study.summary.mean_terminal_return,
            marker_color=[colors[name] for name in study.summary.strategy], name='Mean terminal return',
            hovertemplate='%{x}<br>Mean terminal return: %{y:.2%}<extra></extra>'))
        figure.add_trace(go.Scatter(x=study.summary.strategy, y=study.summary.median_terminal_return,
            mode='markers', marker=dict(color='#e2e8f0', size=10), name='Median terminal return',
            hovertemplate='%{x}<br>Median terminal return: %{y:.2%}<extra></extra>'))
        figure.update_layout(title='Same-terminal-date historical comparison', height=360,
            yaxis_title='Terminal return', yaxis_tickformat='.0%', hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(figure, use_container_width=True, key=f'{prefix}_hybrid_chart')
        for warning in study.warnings:
            st.caption(warning)
        summary_lines = ['| Strategy | Cases | Mean terminal return | Median terminal return | Worst terminal return |',
                         '| --- | ---: | ---: | ---: | ---: |']
        for row in study.summary.to_dict('records'):
            percent = lambda value: 'Unavailable' if pd.isna(value) else f'{value:.2%}'
            summary_lines.append(f"| {row['strategy']} | {row['cases']} | {percent(row['mean_terminal_return'])} | "
                                 f"{percent(row['median_terminal_return'])} | {percent(row['worst_terminal_return'])} |")
        report = '\n'.join([
            '# Entry execution rationale', '',
            'This is an analytical working paper, not a claim of market-timing alpha.', '',
            '## Current rule state', '',
            f'- Condition status: {signal.action}', f'- Condition: {signal.condition}',
            f'- Next review: {signal.next_review or "Not applicable"}',
            f'- Forced completion: {signal.forced_completion}', '',
            '## Historical same-terminal comparison', '', *summary_lines, '',
            '## Method', '',
            'Lump sum invests at the first eligible session. Fixed DCA invests 50% then the remaining 50% after 20 sessions. '
            'Hybrid DCA invests 50% initially and the remainder at the first disclosed condition or the same 20-session deadline. '
            'All strategies use the same terminal date, stated proportional costs and cash yield.', '',
            '## Limitations', '', *[f'- {warning}' for warning in study.warnings],
        ])
        archive = BytesIO()
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr('entry_execution_rationale.md', report)
            bundle.writestr('summary.csv', study.summary.to_csv(index=False))
            bundle.writestr('historical_cases.csv', study.cases.to_csv(index=False))
            bundle.writestr('methodology.json', json.dumps(dict(config=study.config, signal=signal.to_dict(),
                warnings=study.warnings), indent=2))
        st.download_button('Download entry-execution report evidence', archive.getvalue(),
            file_name=f'{plan.ticker}_entry_execution_evidence.zip', mime='application/zip',
            key=f'{prefix}_hybrid_export')


def _render_entry_algorithm(plan, prices, benchmark_prices, as_of, prefix):
    """Render a fixed-model research output without turning it into an order."""
    from src.analytics.entry_algorithm import EntryAlgorithmConfig, entry_algorithm_signal, validate_entry_algorithm

    with st.expander('Signal-gated entry algorithm (research)', expanded=True):
        st.caption('Fixed ridge model of 21-session security return relative to the benchmark. '
                   'It uses only prices known at each review, buys the remaining half on a qualifying review, '
                   'and forces the final review. It is not a price target or order instruction.')
        config = EntryAlgorithmConfig(review_interval_sessions=plan.interval_days,
            maximum_wait_sessions=plan.interval_days * (plan.reviews - 1), cost_bps=plan.cost_bps)
        try:
            with st.spinner('Fitting past-only entry model and running non-overlapping walk-forward checks…'):
                signal = entry_algorithm_signal(prices, benchmark_prices, as_of=as_of, config=config)
                validation = validate_entry_algorithm(prices, benchmark_prices, as_of=as_of, config=config)
        except ValueError as exc:
            st.warning(f'Entry algorithm unavailable: {exc}')
            return None
        left, middle, right = st.columns(3)
        left.metric('Model action', signal.action)
        middle.metric('Predicted 21-session excess return',
                      'Unavailable' if signal.predicted_excess_return is None else f'{signal.predicted_excess_return:.2%}')
        right.metric('Walk-forward net advantage',
                     'Unavailable' if signal.validation_mean_advantage is None else f'{signal.validation_mean_advantage:.2%}')
        st.caption(f"Validation cases: {signal.validation_cases}; win rate versus immediate entry: "
                   f"{'Unavailable' if signal.validation_hit_rate is None else f'{signal.validation_hit_rate:.0%}'}. ")
        if signal.status == 'research_ready':
            if signal.next_review:
                st.info(f'Model deferred the remainder. Recalculate it at the saved next review: {signal.next_review}. '
                        f'Completion is forced by {signal.completion_deadline}.')
            else:
                st.info('The model qualifies the remaining tranche for this saved review, subject to the portfolio checks below.')
        else:
            st.warning('The research gate is not met. The planner keeps calendar control and will not use this model '
                       'to defer or complete an intermediate remaining tranche.')
        with st.expander('Model inputs, walk-forward cases and limitations'):
            st.json(dict(signal=signal.to_dict(), summary=validation.summary, warnings=validation.warnings))
            st.dataframe(validation.observations, hide_index=True, use_container_width=True)
        archive = BytesIO()
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr('signal.json', json.dumps(signal.to_dict(), indent=2))
            bundle.writestr('walk_forward_cases.csv', validation.observations.to_csv(index=False))
            bundle.writestr('methodology.json', json.dumps(dict(summary=validation.summary,
                warnings=validation.warnings, feature_names=list(signal.model_coefficients)), indent=2))
        st.download_button('Download entry-algorithm evidence', archive.getvalue(),
            file_name=f'{plan.ticker}_entry_algorithm_evidence.zip', mime='application/zip',
            key=f'{prefix}_entry_algorithm_export')
        return signal


@st.cache_data(ttl=900, show_spinner=False)
def _timing_estimate(ticker, prices, benchmark_prices, benchmark, as_of):
    from src.analytics.deployment_timing import estimate_deployment_timing

    return estimate_deployment_timing(ticker, prices, benchmark_prices, benchmark=benchmark, as_of=as_of)


def _render_deployment_visual(plan, prices, benchmark_prices, entry, events, state, decision, closed, prefix,
                              signal=None, research_mode=False):
    from src.visualization.deployment_chart import build_deployment_chart

    st.markdown('#### Purchase timeline')
    timing = None
    if research_mode and not closed and st.checkbox('Estimate next entry window from historical analogues', key=f'{prefix}_timing'):
        if entry.benchmark.upper() != plan.benchmark.upper():
            st.warning(f'Select the saved plan benchmark ({plan.benchmark}) above before estimating timing.')
        else:
            with st.spinner('Comparing completed historical episodes…'):
                timing = _timing_estimate(plan.ticker, prices, benchmark_prices, plan.benchmark, entry.as_of)
            if timing.status == 'ready':
                st.info('The observable entry conditions are already met in the latest data. '
                        'The purchase still requires the saved plan and portfolio checks to pass.')
            elif timing.status == 'estimated':
                a, b, c = st.columns(3)
                for column, h in zip((a, b, c), (5, 10, 20)):
                    column.metric(f'Conditions within {h} sessions', f'{timing.probabilities[str(h)]:.0%}')
                st.caption(f'{timing.sample_count} comparable completed episodes · '
                           f'No qualifying condition within 20 sessions: {timing.no_hit_probability:.0%}. '
                           'These are experimental historical frequencies, not calibrated probabilities.')
                wait = (f'{timing.median_sessions} trading sessions' if timing.median_sessions is not None
                        else 'Beyond 20 sessions / not established')
                st.write(f'**Estimated median waiting time:** {wait}')
            else:
                st.info('A supported timing estimate is unavailable. The calendar schedule remains visible.')
            with st.expander('Timing evidence and limitations'):
                st.write(timing.condition)
                for warning in timing.warnings:
                    st.caption(warning)
                st.json(timing.to_dict())
                st.download_button('Download timing evidence', json.dumps(timing.to_dict(), indent=2),
                    file_name=f'{plan.ticker}_timing_evidence.json', mime='application/json', key=f'{prefix}_timing_export')
            _render_timing_validation(plan, prices, benchmark_prices, entry.as_of, prefix)
    figure = build_deployment_chart(plan, prices, events, as_of=entry.as_of, state=state,
                                     decision=decision, timing=timing, signal=signal, closed=closed)
    st.plotly_chart(figure, use_container_width=True, key=f'{prefix}_chart')
    st.caption('Solid lines and green markers show observed prices and manually reported executions. '
               'Dashed paths are calendar targets, not promised purchases. Shading, when available, shows '
               'historical condition-arrival frequencies; it does not forecast price or authorize a trade. '
               'Future sessions are weekday proxies and exclude no exchange holidays. '
               'The orange budget line is not portfolio cash; undocumented snapshot spending is a separate marker.')
    if plan.policy == 'Signal-gated':
        st.caption('The blue model marker or label is the current security-specific algorithm output. '
                   'The grey/purple schedule is shared plan geometry and is not the recommendation.')
    if not closed:
        st.caption(f'Final scheduled review: {plan.dates()[-1].date()} · '
                   'Amounts and dates remain subject to data, thesis, cash and portfolio constraints.')
    st.download_button('Download interactive purchase timeline', figure.to_html(include_plotlyjs=True, full_html=True),
        file_name=f'{plan.ticker}_purchase_timeline.html', mime='text/html', key=f'{prefix}_chart_export')


def _render_comparison(ticker, benchmark, prices, benchmark_prices, prefix, *, research_mode=False):
    with st.expander('Compare deployment policies'):
        st.caption('Equal-capital historical sleeve study. Parameters are fixed before running; '
                   'results do not validate the investment thesis or establish predictive skill.')
        if prices is None or len(prices) < 400:
            st.info('More adjusted price history is required for a meaningful complete comparison.')
            return
        start = st.date_input('Historical first review', value=prices.index[-200].date(), key=f'{prefix}_bt_start')
        capital = st.number_input('Study starting cash', min_value=1., value=25_000., key=f'{prefix}_bt_capital')
        reviews = st.number_input('Study review count', min_value=1, max_value=26, value=5, key=f'{prefix}_bt_reviews')
        interval = st.number_input('Study review interval (calendar days)', min_value=1, max_value=30, value=7, key=f'{prefix}_bt_interval')
        cost = st.number_input('Execution costs including slippage (basis points)', min_value=0., max_value=1000., value=10., key=f'{prefix}_bt_cost')
        rate = st.number_input('Annual cash return (%)', min_value=-99., max_value=100., value=3., key=f'{prefix}_bt_rate')
        horizon = st.number_input('Forward evaluation sessions', min_value=21, max_value=504, value=126, key=f'{prefix}_bt_horizon')
        rolling = st.checkbox('Include repeated historical windows', key=f'{prefix}_bt_rolling')
        if not st.button('Run policy comparison', key=f'{prefix}_bt_run'):
            return
        try:
            kwargs = dict(benchmark=benchmark, capital=capital, reviews=int(reviews), interval_days=int(interval),
                horizon_sessions=int(horizon), cost_bps=cost, annual_cash_rate=rate / 100)
            result = compare_deployment_policies(ticker, prices, benchmark_prices, start_date=start, **kwargs)
            included = ['Immediate', 'Fixed schedule', 'Score-adjusted'] if research_mode else list(POLICIES[:2])
            result.assumptions['included_policies'] = included
            result.assumptions['presentation'] = 'Research' if research_mode else 'Competition process comparison'
            result.assumptions['claim_scope'] = 'Descriptive sample comparison; no automatic superiority or compliance claim.'
            if research_mode:
                result.assumptions['signal_gated_scope'] = (
                    'Signal-gated policy has a separate common-terminal walk-forward study in its plan panel.')
            summary_view = result.summary.loc[included]
            equity_view = result.equity[included]
            decisions_view = result.decisions[result.decisions.policy.isin(included)]
            st.dataframe(summary_view, use_container_width=True)
            st.line_chart(equity_view)
            for warning in result.warnings:
                st.caption(warning)
            files = {'summary.csv': summary_view.to_csv(), 'equity.csv': equity_view.to_csv(),
                     'decisions.csv': decisions_view.to_csv(index=False),
                     'methodology.json': json.dumps(dict(assumptions=result.assumptions, warnings=result.warnings), indent=2)}
            if rolling:
                # Disjoint return windows; shared estimation histories can still induce dependence.
                dates = prices.index[252:len(prices) - int(horizon) - 1:int(horizon) + 1]
                obs, summary, excluded = compare_deployment_windows(ticker, prices, benchmark_prices,
                    start_dates=dates, **kwargs)
                if not summary.empty:
                    summary = summary.loc[summary.index.isin(included)]
                    obs = obs[obs.policy.isin(included)]
                st.dataframe(summary, use_container_width=True)
                st.caption('Repeated windows are spaced by the evaluation horizon. p10/p90 describe observed '
                           'dispersion, not confidence intervals. Small samples and shared histories limit inference.')
                if not excluded.empty:
                    st.warning(f'{len(excluded)} windows excluded; reasons are included in the export.')
                files.update({'windows.csv': obs.to_csv(index=False), 'window_summary.csv': summary.to_csv(),
                              'excluded_windows.csv': excluded.to_csv(index=False)})
            archive = BytesIO()
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
                for name, content in files.items():
                    bundle.writestr(name, content)
            st.download_button('Download comparison evidence', archive.getvalue(),
                file_name=f'{ticker}_deployment_evidence.zip', mime='application/zip', key=f'{prefix}_bt_download')
        except ValueError as exc:
            st.warning(str(exc))


def _render_timing_validation(plan, prices, benchmark_prices, as_of, prefix):
    with st.expander('Walk-forward timing diagnostic (research only)'):
        st.caption('Each forecast uses only then-available data. This tests arrival of the defined condition, '
                   'not purchase profitability. Retrospective date selection does not constitute prospective validation.')
        if prices is None or len(prices) < 400:
            st.info('Insufficient history for this diagnostic.')
            return
        start = st.date_input('Diagnostic evaluation start', value=prices.index[-180].date(), key=f'{prefix}_validation_start')
        end = st.date_input('Diagnostic evaluation end', value=prices.index[-1].date(), key=f'{prefix}_validation_end')
        if not st.button('Run walk-forward diagnostic', key=f'{prefix}_validation_run'):
            return
        from src.analytics.deployment_timing_validation import validate_deployment_timing
        try:
            with st.spinner('Evaluating historical forecasts and a past-only base rate…'):
                result = validate_deployment_timing(plan.ticker, prices, benchmark_prices,
                    benchmark=plan.benchmark, evaluation_start=start, evaluation_end=end, data_as_of=as_of)
            st.dataframe(result.summary, hide_index=True, use_container_width=True)
            st.caption(f"Estimable origins: {result.metadata['estimable_origins']} / {result.metadata['requested_origins']}. "
                       'Zero evaluated outcomes means no calibration evidence, not a perfect result.')
            st.caption(result.metadata['interpretation'])
            st.caption(result.metadata['limitations'])
            archive = BytesIO()
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr('forecasts.csv', result.observations.to_csv(index=False))
                bundle.writestr('diagnostic.csv', result.summary.to_csv(index=False))
                bundle.writestr('methodology.json', json.dumps(result.metadata, indent=2))
            st.download_button('Download timing diagnostic', archive.getvalue(),
                file_name=f'{plan.ticker}_timing_diagnostic.zip', mime='application/zip', key=f'{prefix}_validation_export')
        except ValueError as exc:
            st.warning(str(exc))
