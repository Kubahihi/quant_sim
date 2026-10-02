"""Actual observations, calendar targets and experimental timing on distinct layers."""
from html import escape

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analytics.deployment import day
from src.visualization.cockpit_charts import BASELINE_COLOR, GRID_COLOR


def build_deployment_chart(plan, prices, events, *, as_of, state=None, decision=None,
                            timing=None, signal=None, closed=False):
    today = day(as_of)
    plan.validate()
    dates = plan.dates()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.13,
                        subplot_titles=('Observed adjusted price', 'Deployment of approved budget'))
    p = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    if isinstance(prices, pd.Series) and isinstance(prices.index, pd.DatetimeIndex):
        p = prices.copy()
        p.index = p.index.tz_localize(None).normalize()
        p = p.loc[p.index <= today].sort_index()
        p = pd.to_numeric(p, errors='coerce').where(lambda s: s > 0).dropna()
        p = p[~p.index.duplicated(keep='last')]
    display_start = min(today - pd.Timedelta(days=90), dates[0])
    history = p.loc[p.index >= display_start]
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name='Observed adjusted close',
        line=dict(color='#38bdf8', width=2), hovertemplate='%{x|%d %b %Y}<br>Adjusted close: %{y:.2f}<extra></extra>'), row=1, col=1)
    reviews = {e['event_id']: e['payload'] for e in events if e.get('kind') == 'review'}
    fills = []
    for event in events:
        if event.get('kind') != 'execution':
            continue
        payload = event['payload']
        date = day(payload['execution_date'])
        if date > today:
            continue
        review = reviews.get(payload.get('review_id'), {})
        score = (review.get('entry') or {}).get('entry_score')
        reason = payload.get('override_reason') or review.get('decision', {}).get('explanation', 'No review explanation available')
        text = (f"Reported purchase: {payload['notional']:,.2f}<br>"
                f"Score at review: {score:.0f}<br>" if score is not None else
                f"Reported purchase: {payload['notional']:,.2f}<br>Score at review: unavailable<br>")
        text += escape(reason) + '<br>Reference: ' + escape(payload.get('execution_reference', ''))
        fills.append((date, float(payload['notional']), text))
    fills.sort(key=lambda item: item[0])
    actual_dates, actual_values = [min(today, dates[0])], [0.]
    cumulative = 0.
    for date, notional, text in fills:
        cumulative += notional
        actual_dates.append(date)
        actual_values.append(cumulative)
        reference = p.loc[:date]
        if len(reference) and (date - reference.index[-1]).days <= 7:
            fig.add_trace(go.Scatter(x=[date], y=[float(reference.iloc[-1])], mode='markers',
                marker=dict(symbol='triangle-up', color='#22c55e', size=12),
                name='Reported purchase', legendgroup='fills', showlegend=date == fills[0][0] and cumulative == notional,
                text=[text + '<br>Marker uses reference adjusted close, not actual fill price.'],
                hovertemplate='%{x|%d %b %Y}<br>%{text}<extra></extra>'), row=1, col=1)
    actual_dates.append(today)
    actual_values.append(cumulative)
    fig.add_trace(go.Scatter(x=actual_dates, y=actual_values, mode='lines', name='Recorded executed notional',
        line=dict(color='#22c55e', width=3, shape='hv'),
        hovertemplate='%{x|%d %b %Y}<br>Recorded executions: %{y:,.2f}<extra></extra>'), row=2, col=1)
    if fills:
        fig.add_trace(go.Scatter(x=[f[0] for f in fills], y=actual_values[1:-1], mode='markers',
            marker=dict(color='#22c55e', size=9), name='Execution details', showlegend=False,
            text=[f[2] for f in fills], hovertemplate='%{x|%d %b %Y}<br>%{text}<extra></extra>'), row=2, col=1)
    spent = state.spent if state is not None else cumulative
    if state is not None and abs(spent - cumulative) > .01:
        fig.add_trace(go.Scatter(x=[today], y=[spent], mode='markers', name='Snapshot spending (incomplete execution history)',
            marker=dict(symbol='diamond-open', color='#f59e0b', size=13)), row=2, col=1)
    fig.add_trace(go.Scatter(x=actual_dates, y=[max(0, plan.budget - v) for v in actual_values],
        name='Budget not yet recorded as executed', mode='lines',
        line=dict(color='#f59e0b', shape='hv', width=1.5),
        hovertemplate='%{x|%d %b %Y}<br>Unexecuted budget: %{y:,.2f}<br>Not portfolio cash; excludes fees.<extra></extra>'), row=2, col=1)
    if not closed:
        targets = ([plan.budget / 2, *([plan.budget / 2] * (plan.reviews - 2)), plan.budget]
                   if plan.policy == 'Signal-gated' else
                   [(i + 1) * plan.budget / plan.reviews for i in range(plan.reviews)])
        fig.add_trace(go.Scatter(x=[dates[0] - pd.Timedelta(days=1), *dates], y=[0, *targets],
            name=('Two-stage 50/50 calendar target (not executions)' if plan.is_two_stage() else
                  'Signal-gated 50% / completion ceiling (not executions)' if plan.policy == 'Signal-gated' else
                  'Fixed DCA calendar target (not executions)'),
            line=dict(color=BASELINE_COLOR, dash='dot', shape='hv')), row=2, col=1)
        committed = spent + (state.pending_plan if state is not None else 0.)
        future_dates = [d for d in dates if d > today]
        if future_dates:
            targets = [plan.budget if plan.policy == 'Immediate' else
                       (plan.budget if plan.policy == 'Signal-gated' and d == dates[-1] else
                        plan.budget / 2 if plan.policy == 'Signal-gated' else
                        (dates.index(d) + 1) * plan.budget / plan.reviews) for d in future_dates]
            fig.add_trace(go.Scatter(x=[today, *future_dates], y=[committed, *[max(committed, t) for t in targets]],
                name='Conditional calendar path (not a forecast)',
                line=dict(color='#a78bfa', dash='dash', shape='hv')), row=2, col=1)
        if decision is not None and decision.proposed_purchase > 0:
            fig.add_trace(go.Scatter(x=[today + pd.offsets.BDay(1)],
                y=[committed + decision.proposed_purchase], mode='markers', name='Current proposal · next weekday proxy',
                marker=dict(symbol='circle-open', size=13, color='#a78bfa'),
                text=[escape(decision.explanation)], hovertemplate='%{x|%d %b %Y}<br>%{text}<br>Requires eligible session and review.<extra></extra>'), row=2, col=1)
        if plan.policy == 'Signal-gated' and signal is not None:
            signal_status = getattr(signal, 'status', '')
            signal_action = getattr(signal, 'action', '')
            predicted = getattr(signal, 'predicted_excess_return', None)
            prediction_text = ('unavailable' if predicted is None else f'{predicted:.2%}')
            if signal_status == 'research_ready' and signal_action == 'Buy remaining tranche now':
                fig.add_trace(go.Scatter(x=[today + pd.offsets.BDay(1)], y=[plan.budget], mode='markers',
                    name='Model: complete remaining tranche now', marker=dict(symbol='star', size=15, color='#38bdf8'),
                    text=[f'Model action: buy remaining tranche now<br>Predicted 21-session excess return: {prediction_text}<br>'
                          'Requires current portfolio review; not an order.'],
                    hovertemplate='%{x|%d %b %Y}<br>%{text}<extra></extra>'), row=2, col=1)
                fig.add_annotation(x=today, y=.48, xref='x', yref='paper', showarrow=False,
                    text=f'Model: buy remaining tranche now · predicted excess {prediction_text}',
                    font=dict(color='#38bdf8'))
            elif signal_status == 'research_ready' and getattr(signal, 'next_review', None):
                next_review = day(signal.next_review)
                fig.add_trace(go.Scatter(x=[next_review], y=[committed], mode='markers',
                    name='Model: re-evaluate remaining tranche', marker=dict(symbol='diamond-open', size=14, color='#38bdf8'),
                    text=[f'Model action: wait and re-evaluate<br>Predicted 21-session excess return: {prediction_text}<br>'
                          f'Next model review: {next_review.date()}'],
                    hovertemplate='%{x|%d %b %Y}<br>%{text}<extra></extra>'), row=2, col=1)
                fig.add_vline(x=next_review, line_color='#38bdf8', line_dash='dash', row=2, col=1)
                fig.add_annotation(x=today, y=.48, xref='x', yref='paper', showarrow=False,
                    text=f'Model: wait until {next_review.date()} · predicted excess {prediction_text}',
                    font=dict(color='#38bdf8'))
            else:
                fig.add_annotation(x=today, y=.48, xref='x', yref='paper', showarrow=False,
                    text='Model: evidence gate not met — calendar path only', font=dict(color='#f59e0b'))
        if timing is not None and timing.status == 'estimated' and timing.as_of == str(today.date()):
            previous_cdf = 0.
            for start, end in ((1, 5), (6, 10), (11, 20)):
                cdf = timing.probabilities[str(end)]
                mass = max(0., cdf - previous_cdf)
                previous_cdf = cdf
                if mass <= 0:
                    continue
                x0, x1 = today + pd.offsets.BDay(start), today + pd.offsets.BDay(end)
                fig.add_vrect(x0=x0, x1=x1, fillcolor='#a78bfa', opacity=.08 + .45 * mass,
                              line_width=0, row=1, col=1)
                fig.add_trace(go.Scatter(x=[x0, x1], y=[1.02, 1.02], yaxis='y3', mode='lines',
                    line=dict(width=9, color=f'rgba(167,139,250,{.2 + .7 * mass})'),
                    name=f'Condition arrival in sessions {start}–{end}: {mass:.0%}',
                    text=[f'First condition arrival in sessions {start}–{end}: {mass:.0%}. '
                          f'{timing.sample_count} historical episodes; no hit by 20 sessions: {timing.no_hit_probability:.0%}. '
                          'Experimental frequency, not calibrated confidence.'] * 2,
                    hovertemplate='%{text}<extra></extra>', showlegend=False))
        fig.add_vline(x=dates[-1], line_dash='dot', line_color='#f59e0b', row=2, col=1)
    fig.add_vline(x=today, line_dash='dash', line_color=BASELINE_COLOR)
    fig.add_annotation(x=today, y=1.06, xref='x', yref='paper', text='Today / as of', showarrow=False)
    end = max(today + pd.offsets.BDay(21), dates[-1]) if not closed else today + pd.Timedelta(days=2)
    fig.update_xaxes(range=[display_start, end], gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text='Adjusted price', row=1, col=1)
    fig.update_yaxes(title_text='Notional amount', rangemode='tozero', row=2, col=1)
    fig.update_layout(height=660, margin=dict(l=20, r=20, t=65, b=100),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='closest',
        legend=dict(orientation='h', y=-.16, x=0),
        yaxis3=dict(overlaying='y', range=[0, 1.1], visible=False, fixedrange=True))
    return fig
