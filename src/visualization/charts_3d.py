import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional


PORTFOLIO_COLORSCALE = [
    [0.0, "#17324D"],
    [0.45, "#1F8A9E"],
    [0.75, "#7BC8A4"],
    [1.0, "#F2C14E"],
]


def _base_scene(axis_titles: Dict[str, str]) -> Dict[str, Dict]:
    """Shared 3D scene styling for a polished dashboard look."""
    axis_style = {
        "showbackground": True,
        "backgroundcolor": "rgba(15,23,42,0.95)",
        "gridcolor": "rgba(148,163,184,0.25)",
        "zerolinecolor": "rgba(71,85,105,0.15)",
        "showspikes": False,
    }

    return {
        "camera": {
            "eye": {"x": 1.6, "y": 1.35, "z": 0.95},
            "up": {"x": 0, "y": 0, "z": 1},
        },
        "xaxis": {**axis_style, "title": axis_titles["x"], "tickformat": ".0%"},
        "yaxis": {**axis_style, "title": axis_titles["y"], "tickformat": ".0%"},
        "zaxis": {**axis_style, "title": axis_titles["z"], "tickformat": ".0%"},
    }


def plot_portfolio_tradeoff_3d(
    portfolio_cloud: pd.DataFrame,
    frontier_points: List[Dict],
    highlighted_portfolios: Optional[List[Dict]] = None,
    title: str = "3D Portfolio Trade-Off Space",
) -> go.Figure:
    """Render portfolio risk/return/diversification space with key portfolios highlighted."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=portfolio_cloud["volatility"],
            y=portfolio_cloud["expected_return"],
            z=portfolio_cloud["diversification_score"],
            mode="markers",
            name="Portfolio cloud",
            marker={
                "size": 4,
                "opacity": 0.5,
                "color": portfolio_cloud["sharpe_ratio"],
                "colorscale": PORTFOLIO_COLORSCALE,
                "colorbar": {
                    "title": "Sharpe",
                    "thickness": 14,
                    "len": 0.7,
                    "x": 1.02,
                },
                "line": {"color": "rgba(255,255,255,0.55)", "width": 0.4},
            },
            customdata=np.column_stack([
                portfolio_cloud["sharpe_ratio"],
                portfolio_cloud["effective_holdings"],
                portfolio_cloud["max_weight"],
                portfolio_cloud["top_holdings"],
            ]),
            hovertemplate=(
                "<b>Sampled portfolio</b><br>"
                "Volatility: %{x:.2%}<br>"
                "Expected Return: %{y:.2%}<br>"
                "Diversification: %{z:.1%}<br>"
                "Sharpe: %{customdata[0]:.2f}<br>"
                "Effective Holdings: %{customdata[1]:.1f}<br>"
                "Largest Weight: %{customdata[2]:.1%}<br>"
                "Top Holdings: %{customdata[3]}<extra></extra>"
            ),
        )
    )

    if frontier_points:
        frontier_df = pd.DataFrame(frontier_points).sort_values("volatility")
        fig.add_trace(
            go.Scatter3d(
                x=frontier_df["volatility"],
                y=frontier_df["return"],
                z=frontier_df["diversification_score"],
                mode="lines+markers",
                name="Efficient frontier",
                line={"color": "#DA4167", "width": 8},
                marker={"size": 4, "color": "#FFF7F0", "line": {"color": "#DA4167", "width": 1}},
                customdata=np.column_stack([
                    frontier_df["sharpe_ratio"],
                    frontier_df["effective_holdings"],
                    frontier_df["max_weight"],
                    frontier_df["top_holdings"],
                ]),
                hovertemplate=(
                    "<b>Efficient frontier</b><br>"
                    "Volatility: %{x:.2%}<br>"
                    "Expected Return: %{y:.2%}<br>"
                    "Diversification: %{z:.1%}<br>"
                    "Sharpe: %{customdata[0]:.2f}<br>"
                    "Effective Holdings: %{customdata[1]:.1f}<br>"
                    "Largest Weight: %{customdata[2]:.1%}<br>"
                    "Top Holdings: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    marker_symbols = ["diamond", "x", "circle"]
    highlight_colors = ["#1D4ED8", "#E11D48", "#0F766E"]

    for idx, portfolio in enumerate(highlighted_portfolios or []):
        fig.add_trace(
            go.Scatter3d(
                x=[portfolio["volatility"]],
                y=[portfolio["expected_return"]],
                z=[portfolio["diversification_score"]],
                mode="markers+text",
                name=portfolio["name"],
                text=[portfolio["name"]],
                textposition="top center",
                marker={
                    "size": 10,
                    "color": highlight_colors[idx % len(highlight_colors)],
                    "symbol": marker_symbols[idx % len(marker_symbols)],
                    "line": {"color": "white", "width": 3},
                },
                customdata=[[
                    portfolio["sharpe_ratio"],
                    portfolio["effective_holdings"],
                    portfolio["max_weight"],
                    portfolio["top_holdings"],
                ]],
                hovertemplate=(
                    f"<b>{portfolio['name']}</b><br>"
                    "Volatility: %{x:.2%}<br>"
                    "Expected Return: %{y:.2%}<br>"
                    "Diversification: %{z:.1%}<br>"
                    "Sharpe: %{customdata[0]:.2f}<br>"
                    "Effective Holdings: %{customdata[1]:.1f}<br>"
                    "Largest Weight: %{customdata[2]:.1%}<br>"
                    "Top Holdings: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title={"text": title, "x": 0.02},
        template="plotly_white",
        height=680,
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(255,255,255,0.78)",
        },
        paper_bgcolor="white",
        scene=_base_scene({
            "x": "Volatility",
            "y": "Expected Return",
            "z": "Diversification Score",
        }),
    )
    return fig


def plot_monte_carlo_percentile_surface(
    price_paths: np.ndarray,
    percentile_step: int = 5,
    title: str = "3D Monte Carlo Distribution Surface",
) -> go.Figure:
    """Render a 3D percentile surface to show how uncertainty widens through time."""
    percentiles = np.arange(percentile_step, 100, percentile_step)
    days = np.arange(price_paths.shape[0])
    surface = np.array([
        np.percentile(price_paths, percentile, axis=1)
        for percentile in percentiles
    ])

    median_path = np.percentile(price_paths, 50, axis=1)
    band_5 = np.percentile(price_paths, 5, axis=1)
    band_95 = np.percentile(price_paths, 95, axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=np.tile(days, (len(percentiles), 1)),
            y=np.tile(percentiles.reshape(-1, 1), (1, len(days))),
            z=surface,
            colorscale=PORTFOLIO_COLORSCALE,
            opacity=0.92,
            showscale=True,
            colorbar={"title": "Value", "thickness": 14, "len": 0.7, "x": 1.02},
            contours={
                "z": {
                    "show": True,
                    "usecolormap": True,
                    "highlightcolor": "rgba(23,50,77,0.35)",
                    "project_z": True,
                }
            },
            hovertemplate=(
                "Day: %{x}<br>"
                "Percentile: %{y}%<br>"
                "Portfolio Value: $%{z:,.0f}<extra></extra>"
            ),
            name="Distribution surface",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=days,
            y=np.full_like(days, 50),
            z=median_path,
            mode="lines",
            name="Median path",
            line={"color": "#DA4167", "width": 8},
            hovertemplate="Median: Day %{x}, $%{z:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=days,
            y=np.full_like(days, 5),
            z=band_5,
            mode="lines",
            name="5th percentile",
            line={"color": "#1D4ED8", "width": 5, "dash": "dot"},
            hovertemplate="5th pct: Day %{x}, $%{z:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=days,
            y=np.full_like(days, 95),
            z=band_95,
            mode="lines",
            name="95th percentile",
            line={"color": "#0F766E", "width": 5, "dash": "dot"},
            hovertemplate="95th pct: Day %{x}, $%{z:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": title, "x": 0.02},
        template="plotly_white",
        height=680,
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(255,255,255,0.78)",
        },
        paper_bgcolor="#0f172a",
        scene={
            "camera": {
                "eye": {"x": 1.5, "y": 1.45, "z": 0.88},
                "up": {"x": 0, "y": 0, "z": 1},
            },
            "xaxis": {
                "title": "Days",
                "showbackground": True,
                "backgroundcolor": "rgba(244,247,251,0.92)",
                "gridcolor": "rgba(148,163,184,0.25)",
                "zerolinecolor": "rgba(71,85,105,0.15)",
                "showspikes": False,
            },
            "yaxis": {
                "title": "Percentile",
                "showbackground": True,
                "backgroundcolor": "rgba(244,247,251,0.92)",
                "gridcolor": "rgba(148,163,184,0.25)",
                "zerolinecolor": "rgba(71,85,105,0.15)",
                "ticksuffix": "%",
                "showspikes": False,
            },
            "zaxis": {
                "title": "Portfolio Value",
                "showbackground": True,
                "backgroundcolor": "rgba(244,247,251,0.92)",
                "gridcolor": "rgba(148,163,184,0.25)",
                "zerolinecolor": "rgba(71,85,105,0.15)",
                "tickprefix": "$",
                "tickformat": ",.0f",
                "showspikes": False,
            },
        },
    )
    return fig
