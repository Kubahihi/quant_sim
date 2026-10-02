"""Compact, opt-in deployment assessment shared across research screens."""
from datetime import timedelta

import pandas as pd
import streamlit as st

from src.analytics.entry_quality import calculate_entry_quality


@st.cache_data(ttl=900, show_spinner=False)
def _entry_history(ticker: str, as_of: str):
    from src.data.fetchers.yahoo_fetcher import YahooFetcher

    date = pd.Timestamp(as_of).to_pydatetime()
    # Yahoo's end is exclusive: omit today's potentially provisional daily bar.
    result = YahooFetcher().fetch_prices(ticker, date - timedelta(days=1500), date)
    return result.data['close'] if result.success and 'close' in result.data else None


def _parse_portfolio_targets(tickers_text: str, weights_text: str):
    tickers = [value.strip().upper() for value in tickers_text.replace(',', ' ').split() if value.strip()]
    raw_weights = [value.strip() for value in weights_text.replace(',', ' ').split() if value.strip()]
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError('Enter each approved ticker once.')
    if len(raw_weights) != len(tickers):
        raise ValueError('Enter one target weight for every ticker.')
    weights = [float(value) / 100 for value in raw_weights]
    if any(weight <= 0 for weight in weights) or abs(sum(weights) - 1) > 1e-6:
        raise ValueError('Target weights must be positive and sum to 100%.')
    return tickers, dict(zip(tickers, weights))


def _parse_quote_currencies(tickers, currencies_text):
    values = [value.strip().upper() for value in currencies_text.replace(',', ' ').split() if value.strip()]
    if len(values) != len(tickers) or any(value not in {'USD', 'GBP', 'EUR'} for value in values):
        raise ValueError('Enter USD, GBP or EUR for every ticker, in the same order.')
    return dict(zip(tickers, values))


def _markdown_table(frame):
    """Small dependency-free GFM table for the downloaded working paper."""
    columns = list(frame.columns)
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    header = '| ' + ' | '.join(map(cell, columns)) + ' |'
    rule = '| ' + ' | '.join('---' for _ in columns) + ' |'
    rows = ['| ' + ' | '.join(cell(value) for value in row) + ' |'
            for row in frame.itertuples(index=False, name=None)]
    return '\n'.join([header, rule, *rows])


def _render_portfolio_entry_framework(*, as_of: str, key_prefix: str):
    """Render a basket-level implementation appendix for a pre-selected portfolio."""
    with st.expander('Portfolio entry study · immediate, fixed and conditional', expanded=True):
        st.markdown('**Compared rules:** 100% at the first eligible session; 75% then 25% on session 5; '
                    'or 75% then a conditional 25% with execution no later than session 10. '
                    'Sessions count the first purchase as session 1.')
        st.caption('The fixed session 5 was specified before this comparison. Enter the proposed portfolio '
                   'weights below; the suggested 25% weights are placeholders until your team approves them.')
        tickers_text = st.text_input('Portfolio tickers', value='EXCS.L, XUSE.L, VT, MVED.L',
                                     key=f'{key_prefix}_portfolio_entry_tickers')
        weights_text = st.text_input('Target weights (%) in the same order', value='25, 25, 25, 25',
                                     key=f'{key_prefix}_portfolio_entry_weights')
        currencies_text = st.text_input('Listing currencies in the same order', value='GBP, GBP, USD, EUR',
                                         key=f'{key_prefix}_portfolio_entry_currencies')
        st.caption('Verify listing currencies if you change tickers. The suggested four listings are GBP, GBP, USD and EUR.')
        portfolio_benchmark = st.text_input('Portfolio comparison benchmark', value='SPY',
                                             key=f'{key_prefix}_portfolio_entry_benchmark').strip().upper()
        benchmark_currency = st.selectbox('Benchmark listing currency', ['USD', 'GBP', 'EUR'],
                                           key=f'{key_prefix}_portfolio_entry_benchmark_currency')
        client_rationale = st.text_area('Client / IPS reason for using a staged entry',
                                        key=f'{key_prefix}_portfolio_entry_client_rationale',
                                        help='Write the team\'s actual risk and opportunity-cost reasoning for this client.')
        if not st.checkbox('Evaluate portfolio implementation evidence', key=f'{key_prefix}_portfolio_entry_run'):
            return
        try:
            tickers, weights = _parse_portfolio_targets(tickers_text, weights_text)
            quote_currencies = _parse_quote_currencies(tickers, currencies_text)
        except ValueError as exc:
            st.warning(str(exc))
            return
        signature_key = f'{key_prefix}_portfolio_entry_approval_signature'
        signature = (tuple(tickers), tuple(weights.items()), tuple(quote_currencies.items()),
                     portfolio_benchmark, benchmark_currency, client_rationale.strip())
        if st.session_state.get(signature_key) != signature:
            st.session_state.pop(f'{key_prefix}_portfolio_entry_policy_approved', None)
            for ticker in tickers:
                for suffix in ('portfolio_first_recorded', 'portfolio_first_date',
                               'portfolio_first_reference', 'portfolio_final_recorded',
                               'portfolio_final_reference'):
                    st.session_state.pop(f'{key_prefix}_{ticker}_{suffix}', None)
            st.session_state[signature_key] = signature
        histories, failed = {}, []
        for ticker in tickers:
            try:
                history = _entry_history(ticker, as_of)
            except Exception:
                history = None
            if history is None or history.empty:
                failed.append(ticker)
            else:
                histories[ticker] = history
        benchmark_history = _entry_history(portfolio_benchmark, as_of) if portfolio_benchmark else None
        if failed:
            st.warning('No usable adjusted-close history was returned for: ' + ', '.join(failed) + '.')
        if benchmark_history is None or benchmark_history.empty:
            st.warning(f'No usable benchmark history was returned for {portfolio_benchmark}.')
            return
        fx_histories = {}
        for currency in sorted({*quote_currencies.values(), benchmark_currency} - {'USD'}):
            fx_symbol = f'{currency}USD=X'
            try:
                fx_histories[currency] = _entry_history(fx_symbol, as_of)
            except Exception:
                fx_histories[currency] = None
            if fx_histories[currency] is None or fx_histories[currency].empty:
                st.warning(f'USD conversion history for {currency} is unavailable ({fx_symbol}). '
                           'The portfolio comparison is paused until all quote currencies can be aligned.')
                return
        # Empty histories are passed explicitly so coverage shows the holding rather than silently dropping it.
        histories = {ticker: histories.get(ticker, pd.Series(dtype=float)) for ticker in tickers}
        from src.analytics.portfolio_entry import deployment_path, review_second_tranche, study_portfolio_entry
        from src.analytics.hybrid_entry import HybridEntryConfig
        try:
            study = study_portfolio_entry(
                histories, weights, benchmark_history, as_of=as_of,
                quote_currencies=quote_currencies, fx_to_usd=fx_histories,
                benchmark_currency=benchmark_currency,
                config=HybridEntryConfig(first_fraction=.75, maximum_wait_sessions=10, fixed_second_session=5,
                                         forward_horizon_sessions=21),
            )
        except ValueError as exc:
            st.warning(f'Portfolio framework unavailable: {exc}')
            return
        st.markdown('**1 · Data coverage**')
        coverage = study.coverage.copy()
        coverage['target_weight'] = coverage['target_weight'].map('{:.1%}'.format)
        coverage = coverage.rename(columns={'ticker': 'Ticker', 'target_weight': 'Target weight',
                                            'quote_currency': 'Listing currency',
                                            'observations': 'Observed sessions',
                                            'usable_usd_sessions': 'Usable USD sessions',
                                            'required_observations': 'Inclusion threshold',
                                            'included_in_evidence': 'Included in basket',
                                            'evidence_status': 'Status'})
        st.dataframe(coverage, hide_index=True, use_container_width=True)
        if not study.full_portfolio_covered:
            missing = ', '.join(study.coverage.loc[~study.coverage.included_in_evidence, 'ticker'])
            st.warning(f'Historical result covers {study.evidence_weight:.1%} of the entered portfolio weight. '
                       f'{missing} lacks sufficient price history. Returns below are for a renormalized covered '
                       'basket, not the entered four-holding portfolio.')
        if study.study is None:
            for warning in study.warnings:
                st.caption(warning)
            return
        result = study.study
        from src.analytics.portfolio_entry_export import make_entry_snapshot, analyze_entry_snapshot
        snapshot = make_entry_snapshot(histories, weights, benchmark_history, as_of=as_of,
            quote_currencies=quote_currencies, fx_to_usd=fx_histories,
            benchmark_currency=benchmark_currency, config=HybridEntryConfig(**result.config))
        import json
        st.download_button('Download reproducible entry inputs (JSON)',
            json.dumps(snapshot, indent=2), file_name='portfolio_entry_inputs.json',
            mime='application/json', key=f'{key_prefix}_portfolio_entry_inputs_export')
        _, sensitivity = analyze_entry_snapshot(snapshot)
        st.download_button('Download cost and cash sensitivity (CSV)', sensitivity.to_csv(index=False),
            file_name='portfolio_entry_sensitivity.csv', mime='text/csv',
            key=f'{key_prefix}_portfolio_entry_sensitivity_export')
        hybrid = result.summary.loc[result.summary.strategy == 'Hybrid DCA'].iloc[0]
        advantage = hybrid['mean_advantage_vs_lump']
        enough_cases = int(hybrid['cases']) >= result.config['minimum_cases']
        advantage_fixed = hybrid['mean_advantage_vs_fixed']
        scope = 'whole entered portfolio' if study.full_portfolio_covered else f'{study.evidence_weight:.1%} covered weight only'
        evidence_ready = study.full_portfolio_covered and enough_cases
        st.markdown('**2 · Historical result**')
        st.metric('Hybrid vs lump sum · mean change in terminal value', f'{advantage:+.2%}')
        st.caption(f'Hybrid vs fixed DCA: {advantage_fixed:+.2%} · {int(hybrid["cases"])} non-overlapping starts · '
                   f'{scope}. Positive means the hybrid ended with more value; negative means less.')
        if not evidence_ready:
            st.warning('Report conclusion: the hybrid has no demonstrated whole-portfolio advantage. '
                       'This comparison is exploratory because price coverage or the number of completed starts '
                       'is insufficient. State the observed result only for the covered basket.')
        elif advantage <= 0 or advantage_fixed <= 0:
            st.warning('Report conclusion: this sample does not support claiming an execution advantage for the '
                       'rapid hybrid over both alternatives. If the team keeps the rule for risk or behavioural '
                       'reasons, report its measured cost and the client rationale.')
        else:
            st.info('Report conclusion: the hybrid had a positive average difference against both alternatives '
                    'in this historical sample. The case chart shows whether this depended on a few unusual starts.')
        evidence_label = ('the full entered portfolio' if study.full_portfolio_covered else
                          f'the covered {study.evidence_weight:.0%} of entered target weight')
        interpretation = ('The sample is too short for a performance-advantage claim.' if not enough_cases else
                          'The sample is descriptive; it does not establish future timing ability.')
        report_draft = (
            'We evaluated a rapid staged entry: 75% at the first eligible session and the final 25% '
            'on a pre-specified condition no earlier than a separate session, with completion by session ten. '
            f'For {evidence_label}, {int(hybrid["cases"])} non-overlapping historical starts produced a mean '
            f'terminal-value difference of {advantage:+.2%} versus lump sum and {advantage_fixed:+.2%} '
            f'versus fixed 75/25 DCA on session 5. {interpretation} '
            'Our choice must also match the client\'s IPS, and actual purchases are documented separately.'
        )
        with st.expander('Draft wording for the competition report'):
            st.write(report_draft)
            if not client_rationale.strip():
                st.warning('Add the client / IPS reason above before using this wording in the report.')
        signal = result.current_signal
        st.markdown('**3 · What the rule sees today**')
        st.caption('These indicators describe only the covered synthetic basket. The current condition does not '
                   'establish that the hybrid adds value or authorize an order for any holding.')
        left, middle, right = st.columns(3)
        condition_met = signal.condition != 'No entry condition today'
        left.metric('Condition on covered basket', 'Met' if condition_met else 'Not met')
        middle.metric('20-day basket pullback', f'{signal.pullback_from_20d_high:.1%}')
        right.metric('20-day basket return versus benchmark', f'{signal.relative_return:.1%}')
        st.caption(f'Condition: {signal.condition}. The ten-session completion rule is a property of the candidate '
                   'policy. Actual orders require a separately approved plan and portfolio checks.')
        review_rows = []
        with st.expander('Review the final 25% against recorded first purchases', expanded=False):
            st.caption('Enter execution details from the official ledger. These entries stay in this browser session '
                       'and do not place orders or verify a WInS transaction.')
            approved = st.checkbox('Team approved these portfolio weights and the 75/25 rule',
                                   key=f'{key_prefix}_portfolio_entry_policy_approved')
            latest_common = pd.Timestamp(study.latest_common_session)
            signal_available = study.full_portfolio_covered and (pd.Timestamp(as_of) - latest_common).days <= 4
            st.caption(f'Common basket prices through {latest_common.date()}. '
                       'The conditional review is available only with complete, recent basket data.')
            for ticker in tickers:
                with st.container(border=True):
                    st.write(f'**{ticker}**')
                    first_recorded = st.checkbox('First 75% recorded',
                        key=f'{key_prefix}_{ticker}_portfolio_first_recorded')
                    if first_recorded:
                        left, middle, right = st.columns(3)
                        first_date = left.date_input('First execution date', value=pd.Timestamp(as_of).date(),
                            key=f'{key_prefix}_{ticker}_portfolio_first_date')
                        reference = middle.text_input('Official execution reference',
                            key=f'{key_prefix}_{ticker}_portfolio_first_reference')
                        final_recorded = right.checkbox('Final 25% recorded',
                            key=f'{key_prefix}_{ticker}_portfolio_final_recorded')
                        final_reference = (st.text_input('Official final execution reference',
                            key=f'{key_prefix}_{ticker}_portfolio_final_reference') if final_recorded else '')
                    else:
                        first_date, reference, final_recorded, final_reference = None, '', False, ''
                    review = review_second_tranche(
                        histories[ticker].index, as_of=as_of, first_execution_date=first_date,
                        first_execution_reference=reference, final_recorded=final_recorded,
                        final_execution_reference=final_reference,
                        condition_met=condition_met, signal_fresh=signal_available,
                        policy_approved=approved,
                        config=HybridEntryConfig(**result.config),
                    )
                    st.write(f'**{review.status}** — {review.explanation}')
                    review_rows.append(dict(ticker=ticker, first_execution_date=str(first_date or ''),
                        first_execution_reference=reference, final_recorded=final_recorded,
                        final_execution_reference=final_reference,
                        sessions_since_first=review.sessions_since_first, review_status=review.status,
                        explanation=review.explanation))
            st.download_button('Download entered tranche reviews (CSV)',
                pd.DataFrame(review_rows).to_csv(index=False), file_name='portfolio_tranche_reviews.csv',
                mime='text/csv', key=f'{key_prefix}_portfolio_tranche_reviews_export')
        summary = result.summary.copy()
        for column in ('mean_terminal_return', 'median_terminal_return', 'worst_terminal_return',
                       'best_terminal_return', 'mean_advantage_vs_lump',
                       'mean_advantage_vs_fixed', 'win_rate_vs_lump'):
            summary[column] = summary[column].map(lambda value: '—' if pd.isna(value) else f'{value:.2%}')
        summary['mean_wait_sessions'] = summary['mean_wait_sessions'].map(lambda value: f'{value:.1f}')
        summary['average_wait_sessions'] = summary['average_wait_sessions'].map(
            lambda value: '—' if pd.isna(value) else f'{value:.1f} sessions')
        summary = summary.rename(columns={'strategy': 'Method', 'cases': 'Historical cases',
            'mean_terminal_return': 'Average return to common end date', 'median_terminal_return': 'Median return',
            'worst_terminal_return': 'Worst return', 'best_terminal_return': 'Best return',
            'mean_wait_sessions': 'Average wait (sessions)',
            'non_overlapping_cases': 'Non-overlapping starts',
            'mean_advantage_vs_lump': 'Rapid hybrid vs lump sum',
            'mean_advantage_vs_fixed': 'Rapid hybrid vs front-loaded DCA',
            'win_rate_vs_lump': 'Wins vs lump sum',
            'average_wait_sessions': 'Average wait for final 25%'})
        st.dataframe(summary, hide_index=True, use_container_width=True)
        import plotly.graph_objects as go
        st.markdown('**4 · What the entry rule changes**')
        path = deployment_path(result.cases, first_fraction=result.config['first_fraction'],
                               maximum_wait_sessions=result.config['maximum_wait_sessions'],
                               fixed_second_session=result.config['fixed_second_session'])
        exposure_chart = go.Figure()
        for column, label, color, dash in (
            ('lump_sum', 'Lump sum', '#f59e0b', 'solid'),
            ('fixed_dca', 'Fixed 75/25 DCA', '#a78bfa', 'dash'),
            ('hybrid_historical_average', 'Rapid hybrid · historical average', '#38bdf8', 'solid'),
        ):
            exposure_chart.add_trace(go.Scatter(x=path.session, y=path[column], mode='lines',
                line=dict(color=color, width=3, dash=dash, shape='hv'), name=label,
                hovertemplate='Session %{x}<br>Capital deployed: %{y:.0%}<extra></extra>'))
        exposure_chart.update_layout(title='Share of covered basket capital invested', height=320,
            xaxis_title='Sessions after first purchase', yaxis_title='Covered basket capital invested',
            yaxis=dict(range=[0.68, 1.04], tickformat='.0%'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(exposure_chart, use_container_width=True, key=f'{key_prefix}_portfolio_entry_exposure')
        immediate = int((result.cases.wait_sessions == 0).sum())
        forced = int((result.cases.trigger == 'Forced completion').sum())
        st.caption(f'The blue line is the average of {len(result.cases)} historical execution paths: '
                   f'{immediate} completed at the first purchase and {forced} reached the deadline. '
                   'This chart shows capital deployment, not investment performance.')

        cases_chart = go.Figure()
        case_table = result.cases.sort_values('start_date')
        for column, label, color in (
            ('hybrid_vs_lump', 'Hybrid vs lump sum', '#38bdf8'),
            ('hybrid_vs_fixed', 'Hybrid vs fixed DCA', '#a78bfa'),
        ):
            cases_chart.add_trace(go.Bar(x=case_table.start_date, y=case_table[column],
                marker_color=color, name=label,
                customdata=case_table[['wait_sessions', 'trigger', 'second_entry_date']].to_numpy(),
                hovertemplate='Start %{x}<br>Change in terminal value: %{y:+.2%}<br>'
                              'Wait: %{customdata[0]} sessions<br>Trigger: %{customdata[1]}<br>'
                              'Second purchase: %{customdata[2]}<extra></extra>'))
        cases_chart.add_hline(y=0, line_color='#94a3b8', line_width=1)
        cases_chart.update_layout(title='Did waiting for the final 25% help in the covered basket?', height=360, barmode='group',
            xaxis_title='Historical start date', yaxis_title='Change in terminal value',
            yaxis_tickformat='+.1%', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(cases_chart, use_container_width=True, key=f'{key_prefix}_portfolio_entry_cases')
        wins = int((result.cases.hybrid_vs_lump > 0).sum())
        st.caption(f'Each pair of bars is one completed historical start; above zero favours the rapid hybrid. '
                   f'It beat lump sum in {wins} of {len(result.cases)} starts. All methods have the same terminal '
                   'date, target weights, stated costs and cash yield.')
        verdict = ('The full entered portfolio had enough completed starts for a descriptive comparison.'
                   if evidence_ready else
                   'This result cannot support a whole-portfolio performance claim: price coverage or the '
                   'number of completed historical starts is insufficient.')
        report = '\n'.join([
            '# Portfolio entry strategy — working paper', '',
            f'As of {as_of}. Entered weights must be checked against the approved portfolio and actual trades '
            'against the official execution record.', '',
            '## Client and policy alignment',
            f'- Team-entered IPS rationale: {client_rationale.strip() or "Missing — team must supply before report use."}',
            f'- Approval of entered weights and rule: {"Attested in this tool" if approved else "Not attested"}.',
            'The approval checkbox is a team entry, not independent verification of the IPS or WInS record.', '',
            '## Draft wording for the final report', report_draft, '',
            '## Proposed decision rule',
            'The conditional 75/25 rule has not demonstrated a robust advantage. A simple IPS default is '
            '100% of each approved allocation at the first eligible executable session, subject to thesis, '
            'cash, liquidity and portfolio limits. If an order cannot execute, retain the unspent amount in '
            'USD cash, document the reason, and retry at the next eligible session without changing its target.', '',
            '## Entered portfolio and evidence coverage',
            '| Ticker | Entered weight | Historical status |', '| --- | ---: | --- |',
            *[f'| {row.ticker} ({row.quote_currency}) | {row.target_weight:.1%} | '
              f'{row.evidence_status}; {row.observations} observed / {row.usable_usd_sessions} USD sessions |'
              for row in study.coverage.itertuples()], '',
            f'Comparable history covers {study.evidence_weight:.1%} of entered weight. {verdict}', '',
            '## Same-terminal historical comparison',
            f'- Completed non-overlapping starts: {int(hybrid["cases"])}.',
            f'- Mean change in terminal value, rapid hybrid versus lump sum: {advantage:+.2%}.',
            f'- Mean change in terminal value, rapid hybrid versus fixed 75/25 DCA on session 5: {advantage_fixed:+.2%}.',
            f'- Hybrid beat lump sum in {wins} of {len(result.cases)} starts.',
            f'- Mean wait for final 25%: {hybrid["average_wait_sessions"]:.1f} sessions.',
            ('The observed historical result does not support claiming an advantage over both alternatives.'
             if advantage <= 0 or advantage_fixed <= 0 else
             'The observed historical result was positive against both alternatives in this sample.'),
            'All three methods use the same terminal date, entered weights, cash rate and proportional costs. '
            'The common trigger uses a fixed-weight USD basket; terminal wealth comes from target-weighted '
            'buy-and-hold positions in each ETF. The comparison describes historical implementation results; '
            'actual purchase dates and amounts must come from trade records. '
            f'Frozen input SHA-256: {snapshot["input_sha256"]}.', '',
            '## Report table', '', _markdown_table(result.summary), '',
            '## Individual starts', '', _markdown_table(result.cases), '',
            '## Cost and cash sensitivity', '', _markdown_table(sensitivity), '',
            '## Client horizon', '',
            'This short entry study does not measure Laura Gao goal funding. The long-term model begins with '
            'USD 300,000 at the start of 2027 and USD 150,000 at the start of 2028. It must test all ten '
            'USD 50,000 payments at the starts of 2033–2042, the operating reserve, a responsible facility '
            'contribution and retained flexibility. WInS USD 500,000 is separate.', '',
            '## Attribution', '',
            'This methodology and draft text were developed with OpenAI Codex assistance. Students must '
            'independently verify and cite any AI-generated material included in their submission.', '',
            '## Entered execution reviews (unverified)',
            '| Ticker | First date | First reference | Final reference | Sessions since first | Review status |',
            '| --- | --- | --- | --- | ---: | --- |',
            *[f'| {row["ticker"]} | {row["first_execution_date"] or "—"} | '
              f'{row["first_execution_reference"] or "—"} | '
              f'{row["final_execution_reference"] or "—"} | '
              f'{row["sessions_since_first"] if row["sessions_since_first"] is not None else "—"} | '
              f'{row["review_status"]} |' for row in review_rows], '',
            '## Limits', *[f'- {warning}' for warning in study.warnings],
        ])
        st.download_button('Download portfolio entry working paper', report,
                           file_name='portfolio_entry_rationale.md', mime='text/markdown',
                           key=f'{key_prefix}_portfolio_entry_export')
        chart_html = ('<!doctype html><meta charset="utf-8"><title>Portfolio entry comparison</title>'
                      '<h1>Portfolio entry comparison</h1>' +
                      exposure_chart.to_html(full_html=False, include_plotlyjs=True) +
                      cases_chart.to_html(full_html=False, include_plotlyjs=False))
        st.download_button('Download interactive entry charts', chart_html,
                           file_name='portfolio_entry_charts.html', mime='text/html',
                           key=f'{key_prefix}_portfolio_entry_charts_export')
        st.download_button('Download historical cases (CSV)', result.cases.to_csv(index=False),
                           file_name='portfolio_entry_cases.csv', mime='text/csv',
                           key=f'{key_prefix}_portfolio_entry_cases_export')
        for warning in study.warnings:
            st.caption(warning)


def render_entry_quality(ticker: str, *, key_prefix: str, connection_factory=None,
                         scope=None, actor=None, can_edit=True):
    date = str(pd.Timestamp.now(tz="UTC").date())
    _render_portfolio_entry_framework(as_of=date, key_prefix=key_prefix)
    with st.expander("Entry Quality", expanded=False):
        st.caption("Deployment support for a strategically selected security. Not a BUY/SELL signal or price prediction.")
        benchmark = st.text_input("Deployment benchmark", value="SPY", key=f"{key_prefix}_eq_benchmark").strip().upper()
        target = st.number_input("Target portfolio weight (%)", min_value=.1, max_value=100.,
                                 value=5., step=.5, key=f"{key_prefix}_eq_target")
        if not st.checkbox("Calculate entry quality", key=f"{key_prefix}_eq_run"):
            return
        histories = []
        failures = []
        for symbol in (ticker, benchmark):
            try:
                histories.append(_entry_history(symbol, date) if symbol else None)
            except Exception:
                histories.append(None)
                failures.append(f"Market-data request failed for {symbol}.")
        result = calculate_entry_quality(ticker, *histories, benchmark=benchmark,
                                          target_weight=target / 100, as_of=date)
        research_mode = st.checkbox('Research mode — unvalidated sizing and timing', key=f'{key_prefix}_eq_research')
        if not research_mode:
            st.caption('Competition view: the score describes current conditions. Purchase amounts follow '
                       'a separately justified calendar policy; no predictive advantage is claimed.')
        if result.entry_score is None:
            st.info("Incomplete — review data before deployment.")
        else:
            st.metric("Entry Quality", f"{result.entry_score:.0f} / 100", result.rating, delta_color="off")
            if research_mode:
                st.markdown(f"**Experimental score-band deployment: {result.recommended_deployment_pct:.0%} of target allocation**")
                a, b = st.columns(2)
                a.metric("Initial portfolio weight", f"{result.recommended_initial_weight:.2%}")
                b.metric("Remaining staged allocation", f"{result.remaining_undeployed_weight:.2%}")
                st.caption(result.action)
                st.caption('Unvalidated policy illustration before existing holdings. '
                           'Use the deployment planner for a schedule and portfolio-aware proposal.')
        st.dataframe(pd.DataFrame([
            {"Component": name.replace('_', ' ').title(), "Score / 100": round(c.score) if c.score is not None else None}
            for name, c in result.components.items()]), hide_index=True, use_container_width=True)
        st.write(result.explanation)
        st.caption(f"As of {date} · Last price session: {result.data_quality['last_security_session']} · "
                   f"Data confidence: {result.data_quality['confidence']} (data sufficiency only)")
        for warning in failures + result.warnings:
            st.warning(warning)
        with st.expander("Methodology and raw metrics"):
            st.caption("Trend 25%, momentum 20%, volatility 20%, price location 15%, relative strength 20%. "
                       "Policy bands deploy 10% / 35% / 70% / 100% at scores 0 / 40 / 60 / 80. "
                       "Weights and thresholds are uncalibrated policy choices; components overlap. "
                       "Missing components suppress the composite. Daily adjusted closes; 252 sessions minimum.")
            st.json(result.to_dict())
        from ui.deployment import render_deployment_planner

        render_deployment_planner(ticker, result, *histories, key_prefix=key_prefix,
            connection_factory=connection_factory, scope=scope, actor=actor, can_edit=can_edit,
            research_mode=research_mode)
