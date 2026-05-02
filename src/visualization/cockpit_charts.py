from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go


CHART_BACKGROUND = "rgba(15, 23, 42, 0.45)"
GRID_COLOR = "rgba(148, 163, 184, 0.18)"
BASELINE_COLOR = "#94A3B8"
STRESS_COLOR = "#F97316"
HISTORICAL_COLOR = "#F97316"
SYNTHETIC_COLOR = "#38BDF8"
HIGHLIGHT_COLOR = "#F8FAFC"
PHASE_COLORS = ["#F97316", "#FB7185", "#FACC15", "#38BDF8", "#22C55E", "#A78BFA"]


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14},
    )
    fig.update_layout(**_base_layout(title=title, height=320))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _base_layout(title: str, height: int = 500) -> Dict[str, Any]:
    return {
        "title": {"text": title, "x": 0.02},
        "template": "plotly_dark",
        "height": height,
        "margin": {"l": 18, "r": 18, "t": 56, "b": 18},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": CHART_BACKGROUND,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(0,0,0,0)",
        },
        "hoverlabel": {"bgcolor": "#0F172A"},
    }


def plot_scenario_atlas(summary_rows: pd.DataFrame, highlight_scenario: str | None = None) -> go.Figure:
    if summary_rows.empty:
        return _empty_figure("Crisis Atlas", "No scenario rows available.")

    atlas = summary_rows.copy()
    atlas["Drawdown Depth"] = (-atlas["Max Drawdown"]).clip(lower=0.0)
    atlas["Stress Loss"] = (-atlas["Total Return"]).clip(lower=0.0)
    atlas["Bubble Size"] = np.clip(atlas["Days Underwater"].astype(float) * 3.0, 12.0, 38.0)

    fig = go.Figure()
    categories = [
        ("Historical Crisis", HISTORICAL_COLOR, "diamond"),
        ("Synthetic Stress", SYNTHETIC_COLOR, "circle"),
    ]

    for category, color, symbol in categories:
        subset = atlas[atlas["Category"] == category]
        if highlight_scenario:
            subset = subset[subset["Scenario"] != highlight_scenario]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["Drawdown Depth"],
                y=subset["Stress Loss"],
                mode="markers",
                name=category,
                marker={
                    "size": subset["Bubble Size"],
                    "color": color,
                    "symbol": symbol,
                    "opacity": 0.82,
                    "line": {"color": "rgba(255,255,255,0.38)", "width": 1},
                },
                customdata=np.column_stack(
                    [
                        subset["Scenario"],
                        subset["Era"],
                        subset["Action Cue"],
                        subset["Days Underwater"],
                    ]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Era: %{customdata[1]}<br>"
                    "Stress loss: %{y:.2%}<br>"
                    "Drawdown depth: %{x:.2%}<br>"
                    "Action cue: %{customdata[2]}<br>"
                    "Underwater days: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    if highlight_scenario and highlight_scenario in atlas["Scenario"].values:
        highlighted = atlas.loc[atlas["Scenario"] == highlight_scenario].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[highlighted["Drawdown Depth"]],
                y=[highlighted["Stress Loss"]],
                mode="markers+text",
                name="Selected replay",
                text=[highlighted["Scenario"]],
                textposition="top center",
                marker={
                    "size": 26,
                    "color": HIGHLIGHT_COLOR,
                    "symbol": "star-diamond",
                    "line": {"color": STRESS_COLOR, "width": 2},
                },
                hovertemplate=(
                    f"<b>{highlighted['Scenario']}</b><br>"
                    f"Era: {highlighted['Era']}<br>"
                    "Stress loss: %{y:.2%}<br>"
                    "Drawdown depth: %{x:.2%}<extra></extra>"
                ),
            )
        )

    max_axis = float(
        max(
            float(atlas["Drawdown Depth"].max()) if not atlas.empty else 0.0,
            float(atlas["Stress Loss"].max()) if not atlas.empty else 0.0,
            0.05,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, max_axis * 1.05],
            y=[0.0, max_axis * 1.05],
            mode="lines",
            name="Loss = drawdown",
            line={"color": "rgba(255,255,255,0.18)", "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(**_base_layout(title="Crisis Atlas", height=440))
    fig.update_xaxes(
        title="Drawdown Depth",
        tickformat=".0%",
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
    )
    fig.update_yaxes(
        title="Terminal Stress Loss",
        tickformat=".0%",
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
    )
    return fig


def plot_crisis_playback(
    scenario_name: str,
    baseline_path: pd.Series,
    stressed_path: pd.Series,
    daily_phase_labels: list[str],
) -> go.Figure:
    if baseline_path.empty or stressed_path.empty:
        return _empty_figure("Crisis Playback", "Scenario path is unavailable.")

    x_values = [int(value) for value in baseline_path.index.to_list()]
    baseline_values = baseline_path.to_numpy(dtype=float)
    stressed_values = stressed_path.to_numpy(dtype=float)
    phase_labels = ["Starting point", *daily_phase_labels]
    if len(phase_labels) < len(x_values):
        phase_labels.extend([phase_labels[-1]] * (len(x_values) - len(phase_labels)))
    elif len(phase_labels) > len(x_values):
        phase_labels = phase_labels[: len(x_values)]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_values,
                y=baseline_values,
                mode="lines",
                name="Baseline shadow",
                line={"color": "rgba(148,163,184,0.20)", "width": 2, "dash": "dot"},
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=x_values,
                y=stressed_values,
                mode="lines",
                name="Stress shadow",
                line={"color": "rgba(249,115,22,0.18)", "width": 2, "dash": "dot"},
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=x_values[:1],
                y=baseline_values[:1],
                mode="lines",
                name="Baseline",
                line={"color": BASELINE_COLOR, "width": 3},
                hovertemplate="Day %{x}<br>Baseline: $%{y:,.0f}<extra></extra>",
            ),
            go.Scatter(
                x=x_values[:1],
                y=stressed_values[:1],
                mode="lines",
                name="Scenario",
                line={"color": STRESS_COLOR, "width": 4},
                hovertemplate="Day %{x}<br>Scenario: $%{y:,.0f}<extra></extra>",
            ),
            go.Scatter(
                x=[x_values[0]],
                y=[baseline_values[0]],
                mode="markers",
                name="Baseline marker",
                marker={"size": 10, "color": BASELINE_COLOR, "line": {"color": "white", "width": 1}},
                hovertemplate="Day %{x}<br>Baseline: $%{y:,.0f}<extra></extra>",
                showlegend=False,
            ),
            go.Scatter(
                x=[x_values[0]],
                y=[stressed_values[0]],
                mode="markers+text",
                name="Scenario marker",
                text=[phase_labels[0]],
                textposition="top right",
                marker={"size": 11, "color": STRESS_COLOR, "line": {"color": "white", "width": 1}},
                hovertemplate=(
                    "Day %{x}<br>"
                    "Scenario: $%{y:,.0f}<br>"
                    f"Phase: {phase_labels[0]}<extra></extra>"
                ),
                showlegend=False,
            ),
        ]
    )

    frames = []
    frame_names = [str(x_value) for x_value in x_values]
    for idx in range(len(x_values)):
        label = phase_labels[idx]
        frames.append(
            go.Frame(
                name=frame_names[idx],
                data=[
                    go.Scatter(x=x_values[: idx + 1], y=baseline_values[: idx + 1]),
                    go.Scatter(x=x_values[: idx + 1], y=stressed_values[: idx + 1]),
                    go.Scatter(x=[x_values[idx]], y=[baseline_values[idx]]),
                    go.Scatter(x=[x_values[idx]], y=[stressed_values[idx]], text=[label]),
                ],
                traces=[2, 3, 4, 5],
                layout={"title": {"text": f"Crisis Playback: {scenario_name} | Day {x_values[idx]} | {label}"}},
            )
        )

    fig.frames = frames
    steps = []
    for idx in range(len(x_values)):
        steps.append(
            {
                "label": str(x_values[idx]),
                "method": "animate",
                "args": [
                    [frame_names[idx]],
                    {"mode": "immediate", "frame": {"duration": 90, "redraw": True}, "transition": {"duration": 0}},
                ],
            }
        )

    playback_layout = _base_layout(title=f"Crisis Playback: {scenario_name}", height=520)
    playback_layout["title"] = {
        "text": f"Crisis Playback: {scenario_name}",
        "x": 0.02,
        "y": 0.96,
        "pad": {"t": 8, "b": 18},
    }
    playback_layout["margin"] = {"l": 18, "r": 18, "t": 112, "b": 30}

    fig.update_layout(
        **playback_layout,
        hovermode="x unified",
        xaxis={"title": "Scenario Day", "gridcolor": GRID_COLOR, "zerolinecolor": GRID_COLOR},
        yaxis={"title": "Portfolio Value", "tickprefix": "$", "gridcolor": GRID_COLOR, "zerolinecolor": GRID_COLOR},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.20,
                "yanchor": "top",
                "pad": {"t": 0, "r": 0},
                "bgcolor": "rgba(15,23,42,0.92)",
                "bordercolor": "rgba(148,163,184,0.28)",
                "borderwidth": 1,
                "font": {"size": 12},
                "buttons": [
                    {
                        "label": "Play replay",
                        "method": "animate",
                        "args": [
                            frame_names,
                            {
                                "frame": {"duration": 120, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": False,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                    {
                        "label": "Reset to start",
                        "method": "animate",
                        "args": [
                            [frame_names[0]],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": False,
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "x": 0.0,
                "len": 1.0,
                "y": -0.06,
                "currentvalue": {"prefix": "Replay day: "},
                "pad": {"t": 12},
                "steps": steps,
            }
        ],
    )
    return fig


def plot_phase_timeline(scenario_name: str, phase_table: pd.DataFrame) -> go.Figure:
    if phase_table.empty:
        return _empty_figure("Phase Timeline", "No phase table available.")

    fig = go.Figure()
    for idx, row in enumerate(phase_table.to_dict("records")):
        fig.add_trace(
            go.Bar(
                x=[int(row["Duration"])],
                y=[scenario_name],
                base=[int(row["Start Day"]) - 1],
                orientation="h",
                name=str(row["Phase"]),
                marker={"color": PHASE_COLORS[idx % len(PHASE_COLORS)]},
                text=[str(row["Phase"])],
                textposition="inside",
                hovertemplate=(
                    f"<b>{row['Phase']}</b><br>"
                    f"Days {int(row['Start Day'])}-{int(row['End Day'])}<br>"
                    f"Duration: {int(row['Duration'])} days<br>"
                    f"Vol multiplier: {float(row['Vol Multiplier']):.2f}x<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **_base_layout(title="Phase Timeline", height=250),
        barmode="stack",
        showlegend=False,
    )
    fig.update_xaxes(title="Scenario Day", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(showticklabels=False)
    return fig


def plot_scenario_shock_map(shock_map: pd.DataFrame, title: str = "Shock Map") -> go.Figure:
    if shock_map.empty:
        return _empty_figure(title, "Shock overlays are unavailable.")

    ordered_columns = [
        column
        for column in ["equity", "bond", "gold", "commodity", "crypto", "cash", "tech"]
        if column in shock_map.columns
    ]
    heatmap_frame = shock_map[ordered_columns].T
    z_values = heatmap_frame.to_numpy(dtype=float)
    text_values = np.vectorize(lambda value: f"{value:+.2%}")(z_values)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_values,
                x=heatmap_frame.columns.tolist(),
                y=[label.replace("_", " ").title() for label in heatmap_frame.index.tolist()],
                zmid=0.0,
                colorscale=[
                    [0.0, "#1D4ED8"],
                    [0.45, "#0F172A"],
                    [0.55, "#111827"],
                    [1.0, "#F97316"],
                ],
                text=text_values,
                texttemplate="%{text}",
                hovertemplate="Role: %{y}<br>Phase: %{x}<br>Overlay: %{z:.2%}<extra></extra>",
                colorbar={"title": "Overlay"},
            )
        ]
    )
    fig.update_layout(**_base_layout(title=title, height=360))
    fig.update_xaxes(title="Phase")
    fig.update_yaxes(title="Asset Role")
    return fig


def plot_scenario_fingerprint(
    scenario_name: str,
    stressed_stats: Dict[str, Any],
    baseline_stats: Dict[str, Any],
    horizon_days: int,
) -> go.Figure:
    labels = [
        "Stress Loss",
        "Drawdown",
        "Worst Day",
        "Underwater",
        "Recovery Risk",
        "Baseline Gap",
    ]

    baseline_raw = [
        max(0.0, -float(baseline_stats.get("total_return", 0.0))),
        max(0.0, -float(baseline_stats.get("max_drawdown", 0.0))),
        max(0.0, -float(baseline_stats.get("worst_day", 0.0))),
        float(baseline_stats.get("days_underwater", 0)) / max(float(horizon_days), 1.0),
        1.0
        if baseline_stats.get("recovery_day") is None
        else min(float(baseline_stats.get("recovery_day")) / max(float(horizon_days), 1.0), 1.0),
        0.0,
    ]
    stress_gap = max(
        0.0,
        -(float(stressed_stats.get("total_return", 0.0)) - float(baseline_stats.get("total_return", 0.0))),
    )
    stressed_raw = [
        max(0.0, -float(stressed_stats.get("total_return", 0.0))),
        max(0.0, -float(stressed_stats.get("max_drawdown", 0.0))),
        max(0.0, -float(stressed_stats.get("worst_day", 0.0))),
        float(stressed_stats.get("days_underwater", 0)) / max(float(horizon_days), 1.0),
        1.0
        if stressed_stats.get("recovery_day") is None
        else min(float(stressed_stats.get("recovery_day")) / max(float(horizon_days), 1.0), 1.0),
        stress_gap,
    ]
    caps = [0.35, 0.40, 0.12, 1.0, 1.0, 0.25]

    baseline_values = [min(raw / cap if cap else raw, 1.0) for raw, cap in zip(baseline_raw, caps, strict=False)]
    stressed_values = [min(raw / cap if cap else raw, 1.0) for raw, cap in zip(stressed_raw, caps, strict=False)]

    theta = [*labels, labels[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[*baseline_values, baseline_values[0]],
            theta=theta,
            fill="toself",
            name="Baseline path",
            line={"color": BASELINE_COLOR, "width": 2},
            fillcolor="rgba(148,163,184,0.18)",
            text=[
                f"Baseline raw: {value:.2%}" if idx < 3 or idx == 5 else f"Baseline raw: {value:.2f}"
                for idx, value in enumerate(baseline_raw)
            ]
            + [f"Baseline raw: {baseline_raw[0]:.2%}"],
            hovertemplate="<b>%{theta}</b><br>%{text}<br>Normalized: %{r:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[*stressed_values, stressed_values[0]],
            theta=theta,
            fill="toself",
            name=scenario_name,
            line={"color": STRESS_COLOR, "width": 3},
            fillcolor="rgba(249,115,22,0.28)",
            text=[
                f"Stress raw: {value:.2%}" if idx < 3 or idx == 5 else f"Stress raw: {value:.2f}"
                for idx, value in enumerate(stressed_raw)
            ]
            + [f"Stress raw: {stressed_raw[0]:.2%}"],
            hovertemplate="<b>%{theta}</b><br>%{text}<br>Normalized: %{r:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        **_base_layout(title="Crisis Fingerprint", height=450),
        polar={
            "bgcolor": CHART_BACKGROUND,
            "radialaxis": {
                "visible": True,
                "range": [0, 1],
                "gridcolor": GRID_COLOR,
                "showticklabels": False,
            },
            "angularaxis": {"gridcolor": GRID_COLOR},
        },
    )
    return fig


def plot_asset_stress_impact(asset_impact: pd.Series, title: str = "Stress Gap by Asset") -> go.Figure:
    if asset_impact.empty:
        return _empty_figure(title, "No asset stress impact is available.")

    impact = asset_impact.sort_values()
    colors = [STRESS_COLOR if value < 0 else "#22C55E" for value in impact]
    fig = go.Figure(
        data=[
            go.Bar(
                x=impact.to_numpy(dtype=float),
                y=impact.index.tolist(),
                orientation="h",
                marker={"color": colors},
                text=[f"${value:,.0f}" for value in impact.to_numpy(dtype=float)],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Stress gap: $%{x:,.0f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(**_base_layout(title=title, height=max(340, 34 * len(impact))))
    fig.update_xaxes(title="Capital Impact", tickprefix="$", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(title="")
    return fig
