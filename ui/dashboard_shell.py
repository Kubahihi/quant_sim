from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


PAGE_ORDER = [
    "overview",
    "portfolio_lab",
    "cockpit",
    "analysis",
    "workspace",
    "reports",
]

PAGE_LABELS = {
    "overview": "01  Overview",
    "portfolio_lab": "02  Portfolio lab",
    "cockpit": "03  Stress testing",
    "analysis": "04  Research",
    "workspace": "05  Workspace",
    "reports": "06  Reports",
}

PAGE_DESCRIPTIONS = {
    "overview": "Fast summary of score, regime, risk, and what needs attention.",
    "cockpit": "Stress-test the portfolio, inspect extreme scenarios, and decide what to do next.",
    "analysis": "Raw data, models, signals, news, and run-to-run comparison.",
    "portfolio_lab": "Performance charts, optimization, simulations, and asset diagnostics.",
    "workspace": "Stock picker plus portfolio and swing-tracking tools in one hub.",
    "reports": "Review summary and export actions in a focused reporting space.",
}

PRESET_PAGES = {
    "Focused": ["overview", "portfolio_lab", "reports"],
    "Research": ["overview", "portfolio_lab", "cockpit", "analysis", "reports"],
    "Workspace": ["overview", "workspace", "reports"],
    "Full": PAGE_ORDER,
}

DEFAULT_EMPTY_PRESET = "Workspace"
DEFAULT_ANALYSIS_PRESET = "Research"
THEME_MODE_KEY = "quant_manual_dark_mode"


# The Streamlit config supplies a sensible first paint.  These values are then
# applied at runtime so the in-app switch controls every surface consistently.
THEME_PALETTES = {
    False: {
        "background": "#f8fafb",
        "surface": "#ffffff",
        "input": "#ffffff",
        "text": "#17212b",
        "muted": "#526273",
        "border": "#cbd5df",
        "accent": "#0f766e",
        "accent_hover": "#0b5e58",
        "accent_text": "#ffffff",
        "accent_soft": "#d9f1ed",
        "chart_grid": "#dce4eb",
        "chart_2": "#2563eb",
        "chart_3": "#7c3aed",
        "chart_4": "#b45309",
        "chart_5": "#be123c",
        "chart_6": "#4d7c0f",
        "danger": "#b91c1c",
        "warning": "#9a5b06",
    },
    True: {
        "background": "#111827",
        "surface": "#182230",
        "input": "#202c3b",
        "text": "#f1f5f9",
        "muted": "#b4c0cf",
        "border": "#3a4a5d",
        # A darker teal keeps white labels readable in the dark theme.  The
        # previous mint accent was too bright beside white button text.
        "accent": "#167d78",
        "accent_hover": "#0f625d",
        "accent_text": "#ffffff",
        "accent_soft": "#173f3d",
        "chart_grid": "#354456",
        "chart_2": "#93c5fd",
        "chart_3": "#d8b4fe",
        "chart_4": "#fcd34d",
        "chart_5": "#fda4af",
        "chart_6": "#bef264",
        "danger": "#fca5a5",
        "warning": "#fcd34d",
    },
}


def is_dark_mode() -> bool:
    """Return the persisted appearance choice, defaulting to the light palette."""
    return bool(st.session_state.get(THEME_MODE_KEY, False))


def _theme_style() -> str:
    palette = THEME_PALETTES[is_dark_mode()]
    variables = "\n".join(
        f"    --qp-{name.replace('_', '-')}: {value} !important;"
        for name, value in palette.items()
    )
    return f"""
<style id="quant-runtime-theme">
:root, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
{variables}
    --background-color: var(--qp-background) !important;
    --secondary-background-color: var(--qp-surface) !important;
    --text-color: var(--qp-text) !important;
    --primary-color: var(--qp-accent) !important;
    --qp-ink: var(--qp-text) !important;
    --qp-line: var(--qp-border) !important;
    --qp-card: var(--qp-surface) !important;
    --qp-soft: var(--qp-input) !important;
}}
</style>
"""


def render_theme_toggle(*, disabled: bool = False) -> bool:
    """Render the shared appearance switch in the sidebar and return its state."""
    return st.toggle(
        "Dark mode",
        key=THEME_MODE_KEY,
        disabled=disabled,
        help="Use the high-contrast dark palette across the workspace, controls, and charts.",
    )


def _apply_plotly_theme(figure: Any, palette: dict[str, str]) -> None:
    """Make Plotly figures follow the selected palette even if callers set a template."""
    update_layout = getattr(figure, "update_layout", None)
    if not callable(update_layout):
        return
    update_layout(
        template="plotly_dark" if is_dark_mode() else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=palette["surface"],
        font={"color": palette["text"]},
        hoverlabel={
            "bgcolor": palette["input"],
            "bordercolor": palette["border"],
            "font": {"color": palette["text"]},
        },
        legend={"font": {"color": palette["text"]}},
    )
    update_xaxes = getattr(figure, "update_xaxes", None)
    if callable(update_xaxes):
        update_xaxes(
            showgrid=True,
            gridcolor=palette["chart_grid"],
            linecolor=palette["border"],
            tickfont={"color": palette["muted"]},
            title_font={"color": palette["text"]},
            zerolinecolor=palette["border"],
        )
    update_yaxes = getattr(figure, "update_yaxes", None)
    if callable(update_yaxes):
        update_yaxes(
            showgrid=True,
            gridcolor=palette["chart_grid"],
            linecolor=palette["border"],
            tickfont={"color": palette["muted"]},
            title_font={"color": palette["text"]},
            zerolinecolor=palette["border"],
        )


def _apply_matplotlib_theme(figure: Any, palette: dict[str, str]) -> None:
    """Bring existing Matplotlib charts into the active palette before display."""
    if figure is None or not hasattr(figure, "get_axes"):
        return
    figure.set_facecolor(palette["background"])
    for axis in figure.get_axes():
        axis.set_facecolor(palette["surface"])
        axis.tick_params(colors=palette["muted"])
        axis.xaxis.label.set_color(palette["text"])
        axis.yaxis.label.set_color(palette["text"])
        axis.title.set_color(palette["text"])
        for spine in axis.spines.values():
            spine.set_color(palette["border"])
        axis.grid(color=palette["chart_grid"], alpha=0.75)
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(palette["input"])
            legend.get_frame().set_edgecolor(palette["border"])
            for text in legend.get_texts():
                text.set_color(palette["text"])


def _render_native_chart(data: Any, mark: str, *args: Any, **kwargs: Any) -> Any:
    """Render Streamlit's convenience charts with a palette-aware Vega spec.

    Streamlit resolves ``st.line_chart`` and ``st.bar_chart`` against its static
    config at process start.  Rebuilding their simple dataframe form as Altair
    prevents a light-chart canvas from surviving after the runtime toggle.
    """
    base_renderer = getattr(st, f"_quant_base_{mark}_chart")
    if args or any(key in kwargs for key in {"x", "y", "color", "size"}):
        return base_renderer(data, *args, **kwargs)

    use_container_width = kwargs.pop("use_container_width", True)
    height = kwargs.pop("height", None)
    kwargs.pop("theme", None)
    if kwargs:
        return base_renderer(data, *args, **kwargs)

    try:
        import altair as alt
        import pandas as pd

        frame = data.to_frame() if isinstance(data, pd.Series) else pd.DataFrame(data)
        if frame.empty:
            return base_renderer(data, use_container_width=use_container_width, height=height)
        frame.columns = [str(column) or "Value" for column in frame.columns]
        index_label = str(frame.index.name or "Index")
        while index_label in frame.columns:
            index_label = f"_{index_label}"
        series_label = "__quant_series__"
        value_label = "__quant_value__"
        while series_label in frame.columns:
            series_label = f"_{series_label}"
        while value_label in frame.columns or value_label == series_label:
            value_label = f"_{value_label}"
        frame.index.name = index_label
        long_frame = frame.reset_index().melt(
            id_vars=[index_label], var_name=series_label, value_name=value_label
        )
        if pd.api.types.is_datetime64_any_dtype(long_frame[index_label]):
            x_type = "T"
        elif pd.api.types.is_numeric_dtype(long_frame[index_label]):
            x_type = "Q"
        else:
            x_type = "N"
        palette = THEME_PALETTES[is_dark_mode()]
        colors = [
            palette["accent"],
            palette["chart_2"],
            palette["chart_3"],
            palette["chart_4"],
            palette["chart_5"],
            palette["chart_6"],
        ]
        chart = (
            alt.Chart(long_frame)
            .mark_line(point=False) if mark == "line" else alt.Chart(long_frame).mark_bar()
        )
        chart = chart.encode(
            x=alt.X(f"{index_label}:{x_type}", title=None),
            y=alt.Y(f"{value_label}:Q", title=None),
            color=alt.Color(f"{series_label}:N", scale=alt.Scale(range=colors)),
            tooltip=[
                alt.Tooltip(f"{index_label}:{x_type}", title=index_label),
                alt.Tooltip(f"{series_label}:N", title="Series"),
                alt.Tooltip(f"{value_label}:Q", title="Value", format=".4~g"),
            ],
        )
        if height is not None:
            chart = chart.properties(height=height)
        chart = (
            chart
            .configure(background=palette["surface"])
            .configure_view(stroke=palette["border"])
            .configure_axis(
                domainColor=palette["border"],
                gridColor=palette["chart_grid"],
                labelColor=palette["muted"],
                tickColor=palette["border"],
                titleColor=palette["text"],
            )
            .configure_legend(labelColor=palette["text"], titleColor=palette["text"])
        )
        return st.altair_chart(chart, use_container_width=use_container_width, theme=None)
    except Exception:
        # The native renderer remains the safe fallback for uncommon data
        # formats; standard pandas Series/DataFrames take the palette-aware path.
        return base_renderer(data, use_container_width=use_container_width, height=height)


def _install_theme_aware_chart_renderers() -> None:
    """Adapt all existing Streamlit chart calls without duplicating chart code."""
    if not hasattr(st, "_quant_base_plotly_chart"):
        st._quant_base_plotly_chart = st.plotly_chart

        def plotly_chart(figure: Any, *args: Any, **kwargs: Any) -> Any:
            _apply_plotly_theme(figure, THEME_PALETTES[is_dark_mode()])
            return st._quant_base_plotly_chart(figure, *args, **kwargs)

        st.plotly_chart = plotly_chart

    if not hasattr(st, "_quant_base_pyplot"):
        st._quant_base_pyplot = st.pyplot

        def pyplot(figure: Any = None, *args: Any, **kwargs: Any) -> Any:
            _apply_matplotlib_theme(figure, THEME_PALETTES[is_dark_mode()])
            return st._quant_base_pyplot(figure, *args, **kwargs)

        st.pyplot = pyplot

    for mark in ("line", "bar"):
        attribute = f"_quant_base_{mark}_chart"
        if not hasattr(st, attribute):
            setattr(st, attribute, getattr(st, f"{mark}_chart"))

            def native_chart(data: Any, *args: Any, _mark: str = mark, **kwargs: Any) -> Any:
                return _render_native_chart(data, _mark, *args, **kwargs)

            setattr(st, f"{mark}_chart", native_chart)


@dataclass(frozen=True)
class DashboardPreferences:
    preset: str
    visible_pages: list[str]
    show_raw_tables: bool
    show_workspace_when_empty: bool


def inject_dashboard_styles() -> None:
    stylesheet = Path(__file__).with_name("dashboard.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>{_theme_style()}", unsafe_allow_html=True)
    _install_theme_aware_chart_renderers()


def metric_columns(count: int, *, key: str):
    """Keep six summary metrics readable as the available content width changes."""
    with st.container(key=f"metric_grid_{count}_{key}"):
        return st.columns(count)


def render_dashboard_preferences(has_analysis: bool) -> DashboardPreferences:
    preset_options = list(PRESET_PAGES.keys())
    preset_key = "dashboard_layout_preset"
    auto_preset_key = "dashboard_layout_preset_auto"
    applied_key = "dashboard_layout_preset_applied"
    visible_pages_key = "dashboard_visible_pages_selector"
    raw_tables_key = "dashboard_show_raw_tables"
    workspace_key = "dashboard_show_workspace_when_empty"

    desired_auto_preset = DEFAULT_ANALYSIS_PRESET if has_analysis else DEFAULT_EMPTY_PRESET
    stored_preset = str(st.session_state.get(preset_key, "") or "")
    has_auto_flag = auto_preset_key in st.session_state
    stored_auto = bool(st.session_state.get(auto_preset_key, False))

    if stored_preset not in preset_options:
        st.session_state[preset_key] = desired_auto_preset
        st.session_state[auto_preset_key] = True
    elif has_analysis and stored_preset == DEFAULT_EMPTY_PRESET and (stored_auto or not has_auto_flag):
        st.session_state[preset_key] = DEFAULT_ANALYSIS_PRESET
        st.session_state[auto_preset_key] = True

    if raw_tables_key not in st.session_state:
        st.session_state[raw_tables_key] = True
    if workspace_key not in st.session_state:
        # Keep the first Quant screen lightweight. Operational tools can load
        # portfolios, universes, and market data, so open them only when the
        # user explicitly opts in.
        st.session_state[workspace_key] = False

    with st.expander("View settings", expanded=False):
        preset = st.selectbox(
            "Workspace preset",
            options=preset_options,
            key=preset_key,
            help="Choose a simpler default layout and fine-tune visible sections below.",
        )

        if st.session_state.get(preset_key) != desired_auto_preset:
            st.session_state[auto_preset_key] = False

        applied_preset = st.session_state.get(applied_key)
        if applied_preset != preset or visible_pages_key not in st.session_state:
            st.session_state[visible_pages_key] = list(PRESET_PAGES[preset])
            st.session_state[applied_key] = preset

        visible_pages = st.multiselect(
            "Visible pages",
            options=PAGE_ORDER,
            format_func=lambda key: PAGE_LABELS.get(key, key),
            key=visible_pages_key,
            help="Hide sections you do not need right now without removing the underlying functionality.",
        )
        if not visible_pages:
            visible_pages = list(PRESET_PAGES[preset])
            st.session_state[visible_pages_key] = visible_pages

        st.caption(
            " | ".join(PAGE_DESCRIPTIONS[key] for key in PAGE_ORDER if key in visible_pages)
        )

        show_raw_tables = st.checkbox(
            "Show detailed tables",
            key=raw_tables_key,
            help="Keep raw prices, returns, and comparison tables visible inside analytical pages.",
        )
        show_workspace_when_empty = st.checkbox(
            "Show tools before first analysis",
            key=workspace_key,
            help="Useful when you want stock screening or trade tracking without running a portfolio analysis first.",
        )

    return DashboardPreferences(
        preset=preset,
        visible_pages=visible_pages,
        show_raw_tables=show_raw_tables,
        show_workspace_when_empty=show_workspace_when_empty,
    )
