from __future__ import annotations

from dataclasses import dataclass

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
        :root {
            --qp-ink: #17202e;
            --qp-muted: #64748b;
            --qp-line: #e3e8ef;
            --qp-soft: #f5f7fa;
            --qp-card: #ffffff;
            --qp-navy: #142238;
            --qp-accent: #167d78;
            --qp-accent-soft: #eaf7f5;
            --qp-radius: 14px;
            --qp-shadow: 0 10px 30px rgba(27, 39, 54, 0.07);
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background: #f7f8fa;
            color: var(--qp-ink);
        }

        [data-testid="stAppViewContainer"] > .main {
            background:
                radial-gradient(circle at 92% 0%, rgba(22, 125, 120, 0.055), transparent 24rem),
                #f7f8fa;
        }

        .main .block-container {
            max-width: 1480px;
            padding: 2rem 2.25rem 4rem;
        }

        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 320px;
            max-width: 320px;
            background: #ffffff;
            border-right: 1px solid var(--qp-line);
        }

        [data-testid="stSidebar"][aria-expanded="false"] {
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            flex-basis: 0 !important;
            border-right: 0;
        }

        [data-testid="stSidebar"][aria-expanded="false"] ~ [data-testid="stMain"],
        [data-testid="stAppViewContainer"]:has([data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stMain"] {
            width: 100% !important;
            max-width: 100% !important;
            margin-left: 0 !important;
        }

        [data-testid="stSidebarNav"] {
            display: none;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.72rem;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--qp-ink);
            letter-spacing: -0.02em;
        }

        .qp-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.35rem 0 0.7rem;
        }

        .qp-brand-mark {
            display: grid;
            place-items: center;
            width: 2.35rem;
            height: 2.35rem;
            border-radius: 11px;
            color: #ffffff;
            background: linear-gradient(145deg, #18304d, #167d78);
            box-shadow: 0 7px 18px rgba(22, 125, 120, 0.2);
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .qp-brand-copy strong {
            display: block;
            color: var(--qp-ink);
            font-size: 0.98rem;
            line-height: 1.2;
        }

        .qp-brand-copy span {
            color: var(--qp-muted);
            font-size: 0.75rem;
        }

        .qp-eyebrow {
            color: var(--qp-accent);
            text-transform: uppercase;
            letter-spacing: 0.13em;
            font-size: 0.7rem;
            font-weight: 750;
            margin-bottom: 0.5rem;
        }

        h1, h2, h3 {
            color: var(--qp-ink);
            letter-spacing: -0.035em;
        }

        h1 { font-size: 2.05rem !important; }
        h2 { font-size: 1.55rem !important; }
        h3 { font-size: 1.12rem !important; }

        p, label, [data-testid="stCaptionContainer"] {
            color: var(--qp-muted);
        }

        hr {
            border-color: var(--qp-line) !important;
            margin: 0.7rem 0 !important;
        }

        div[data-testid="stForm"],
        div[data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stVegaLiteChart"] {
            border-color: var(--qp-line) !important;
            border-radius: var(--qp-radius) !important;
            background: rgba(255, 255, 255, 0.9);
        }

        div[data-testid="stExpander"] {
            overflow: hidden;
            box-shadow: none;
        }

        div[data-testid="stMetric"] {
            min-height: 112px;
            padding: 1rem 1.05rem;
            border: 1px solid var(--qp-line);
            border-radius: var(--qp-radius);
            background: var(--qp-card);
            box-shadow: 0 4px 18px rgba(27, 39, 54, 0.045);
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--qp-ink);
            font-size: 1.62rem;
            letter-spacing: -0.04em;
        }

        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--qp-muted);
        }

        .stButton > button,
        .stDownloadButton > button,
        button[kind="secondary"] {
            min-height: 2.65rem;
            border-radius: 10px;
            border-color: #d5dde7;
            font-weight: 650;
            transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--qp-accent);
            transform: translateY(-1px);
            box-shadow: 0 7px 16px rgba(22, 125, 120, 0.12);
        }

        .stButton > button[kind="primary"],
        button[kind="primaryFormSubmit"] {
            color: #ffffff;
            background: var(--qp-accent);
            border-color: var(--qp-accent);
        }

        input, textarea, [data-baseweb="select"] > div {
            border-radius: 10px !important;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.25rem;
            overflow-x: auto;
            border-bottom: 1px solid var(--qp-line);
        }

        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            height: 3rem;
            border-radius: 8px 8px 0 0;
            padding: 0 0.85rem;
            color: var(--qp-muted);
            font-size: 0.88rem;
            font-weight: 650;
            white-space: nowrap;
        }

        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--qp-accent);
            background: var(--qp-accent-soft);
        }

        .qp-page-nav-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            margin: 1.15rem 0 0.65rem;
        }

        .qp-page-nav-header strong {
            display: block;
            color: var(--qp-ink);
            font-size: 1rem;
        }

        .qp-page-nav-header span {
            color: var(--qp-muted);
            font-size: 0.82rem;
        }

        .st-key-dashboard_active_page [role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            padding: 0.42rem;
            border: 1px solid var(--qp-line);
            border-radius: var(--qp-radius);
            background: rgba(255, 255, 255, 0.86);
        }

        .st-key-dashboard_active_page [role="radiogroup"] label {
            min-height: 2.45rem;
            padding: 0.45rem 0.7rem;
            border-radius: 9px;
            transition: background 120ms ease, color 120ms ease;
        }

        .st-key-dashboard_active_page [role="radiogroup"] label:has(input:checked) {
            color: var(--qp-accent);
            background: var(--qp-accent-soft);
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
            border-width: 1px;
        }

        .dashboard-hero {
            position: relative;
            overflow: hidden;
            color: #f8fafc;
            background:
                radial-gradient(circle at 88% 10%, rgba(72, 193, 185, 0.24), transparent 20rem),
                linear-gradient(135deg, #142238 0%, #193450 65%, #175e61 130%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 1.75rem 1.9rem;
            margin-bottom: 1.25rem;
            box-shadow: var(--qp-shadow);
        }

        .dashboard-kicker {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            color: #80ded4;
            opacity: 1;
            font-weight: 750;
            margin-bottom: 0.45rem;
        }

        .dashboard-hero h2 {
            margin: 0 0 0.55rem 0;
            color: #ffffff;
            font-size: 2.1rem;
            line-height: 1.15;
        }

        .dashboard-hero p {
            margin: 0;
            font-size: 0.98rem;
            color: rgba(241, 245, 249, 0.76);
            opacity: 1;
            max-width: 820px;
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
            color: #e8fffc;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.15);
            font-size: 0.83rem;
        }

        .dashboard-note {
            color: #3e4c5e;
            background: linear-gradient(90deg, var(--qp-accent-soft), rgba(255, 255, 255, 0.86));
            border: 1px solid #d5ebe8;
            border-left: 4px solid var(--qp-accent);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin: 0.6rem 0 1rem 0;
        }

        .dashboard-note strong {
            color: var(--qp-ink);
            display: block;
            margin-bottom: 0.25rem;
        }

        .qp-workflow {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0 0 1.25rem;
        }

        .qp-workflow-card {
            min-height: 120px;
            padding: 1rem 1.05rem;
            border: 1px solid var(--qp-line);
            border-radius: var(--qp-radius);
            background: #ffffff;
            box-shadow: 0 4px 16px rgba(27, 39, 54, 0.04);
        }

        .qp-workflow-card span {
            display: grid;
            place-items: center;
            width: 1.7rem;
            height: 1.7rem;
            margin-bottom: 0.65rem;
            border-radius: 8px;
            color: var(--qp-accent);
            background: var(--qp-accent-soft);
            font-size: 0.76rem;
            font-weight: 800;
        }

        .qp-workflow-card strong {
            display: block;
            margin-bottom: 0.25rem;
            color: var(--qp-ink);
        }

        .qp-workflow-card p {
            margin: 0;
            font-size: 0.86rem;
            line-height: 1.45;
        }

        @media (max-width: 900px) {
            .main .block-container { padding: 1.15rem 1rem 3rem; }
            .dashboard-hero { padding: 1.35rem 1.2rem; border-radius: 16px; }
            .dashboard-hero h2 { font-size: 1.65rem; }
            .qp-workflow { grid-template-columns: 1fr; }
            div[data-testid="stMetric"] { min-height: 96px; }
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
