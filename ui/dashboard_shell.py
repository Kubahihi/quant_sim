from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


PAGE_ORDER = [
    "overview",
    "cockpit",
    "analysis",
    "portfolio_lab",
    "workspace",
    "reports",
]

PAGE_LABELS = {
    "overview": "Overview",
    "cockpit": "Decision Cockpit",
    "analysis": "Analysis Lab",
    "portfolio_lab": "Portfolio Lab",
    "workspace": "Workspace",
    "reports": "Reports",
}

PAGE_DESCRIPTIONS = {
    "overview": "Fast summary of score, regime, risk, and what needs attention.",
    "cockpit": "Stress-test the portfolio, inspect extreme scenarios, and decide what to do next.",
    "analysis": "Raw data, models, signals, news, and run-to-run comparison.",
    "portfolio_lab": "Performance charts, optimization, simulations, and asset diagnostics.",
    "workspace": "Stock picker plus portfolio and swing-tracking tools in one hub.",
    "reports": "AI commentary and export actions in a focused reporting space.",
}

PRESET_PAGES = {
    "Focused": ["overview", "reports"],
    "Research": ["overview", "cockpit", "analysis", "portfolio_lab", "reports"],
    "Workspace": ["overview", "workspace", "reports"],
    "Full": PAGE_ORDER,
}

DEFAULT_EMPTY_PRESET = "Workspace"
DEFAULT_ANALYSIS_PRESET = "Research"


@dataclass(frozen=True)
class DashboardPreferences:
    preset: str
    visible_pages: list[str]
    show_raw_tables: bool
    show_workspace_when_empty: bool


def inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .dashboard-hero {
            background: transparent;
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 22px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }

        .dashboard-kicker {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            opacity: 0.62;
            margin-bottom: 0.45rem;
        }

        .dashboard-hero h2 {
            margin: 0 0 0.55rem 0;
            font-size: 2rem;
            line-height: 1.15;
        }

        .dashboard-hero p {
            margin: 0;
            font-size: 0.98rem;
            opacity: 0.86;
        }

        .dashboard-badge-row {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            margin-top: 0.95rem;
        }

        .dashboard-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.32rem 0.62rem;
            border-radius: 999px;
            background: transparent;
            border: 1px solid rgba(128, 128, 128, 0.18);
            font-size: 0.83rem;
        }

        .dashboard-note {
            background: transparent;
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin: 0.6rem 0 1rem 0;
        }

        .dashboard-note strong {
            display: block;
            margin-bottom: 0.25rem;
        }

        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            border-radius: 999px;
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        st.session_state[workspace_key] = True

    with st.expander("Dashboard Layout", expanded=False):
        preset = st.selectbox(
            "View preset",
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
            "Visible sections",
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
            "Show detailed data tables",
            key=raw_tables_key,
            help="Keep raw prices, returns, and comparison tables visible inside analytical pages.",
        )
        show_workspace_when_empty = st.checkbox(
            "Keep workspace visible before first run",
            key=workspace_key,
            help="Useful when you want stock screening or trade tracking without running a portfolio analysis first.",
        )

    return DashboardPreferences(
        preset=preset,
        visible_pages=visible_pages,
        show_raw_tables=show_raw_tables,
        show_workspace_when_empty=show_workspace_when_empty,
    )
