from __future__ import annotations

from pathlib import Path
from runpy import run_module
import sys
import time


_APP_RUN_STARTED_AT = time.perf_counter()

import streamlit as st


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Guard against accidentally reusing a non-local `src` package from
# site-packages after a rolling deployment.
for module_name, module_obj in list(sys.modules.items()):
    if module_name != "src" and not module_name.startswith("src."):
        continue
    module_file = getattr(module_obj, "__file__", None)
    if not module_file:
        continue
    resolved = str(Path(module_file).resolve())
    if not resolved.startswith(PROJECT_ROOT):
        sys.modules.pop(module_name, None)


def _run_quant_platform() -> None:
    """Execute the large Quant workspace outside Streamlit's AST rewrite.

    A fresh namespace mirrors Streamlit's normal per-session script execution
    and avoids sharing top-level widget state through Python's module cache.
    Dependencies and bytecode remain cached by Python.
    """
    run_module("ui.quant_platform", run_name="ui.__quant_platform_streamlit_run__")


# The analytical workspace deliberately remains a normal Python module.  When
# it is selected, delegate before configuring or drawing this lightweight
# launcher so there is only one page shell in the current run.
if st.session_state.get("quant_sim_workspace_route") == "Quant Platform":
    _run_quant_platform()
    st.stop()


from ui.dashboard_shell import inject_dashboard_styles
from ui.runtime_diagnostics import (
    PerformanceTrace,
    append_trace_history,
    resolve_build_identity,
    summarize_trace_history,
)


_RUNTIME_TRACE = PerformanceTrace(started_at=_APP_RUN_STARTED_AT)
_BUILD_IDENTITY = resolve_build_identity(PROJECT_ROOT)
_RUNTIME_HISTORY_KEY = "quant_workspace_runtime_history_v1"


st.set_page_config(
    page_title="Quant Workspace",
    layout="wide",
    page_icon=":material/bar_chart:",
    initial_sidebar_state="expanded",
)
inject_dashboard_styles()

SIDEBAR_HIDDEN_KEY = "quant_workspace_sidebar_hidden"
if SIDEBAR_HIDDEN_KEY not in st.session_state:
    st.session_state[SIDEBAR_HIDDEN_KEY] = False


def _set_sidebar_hidden(hidden: bool) -> None:
    st.session_state[SIDEBAR_HIDDEN_KEY] = hidden


if st.session_state[SIDEBAR_HIDDEN_KEY]:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
        }
        [data-testid="stMain"] {
            width: 100vw !important;
            max-width: 100vw !important;
            margin-left: 0 !important;
        }
        [data-testid="stMainBlockContainer"],
        .main .block-container {
            width: 100% !important;
            max-width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    restore_col, _ = st.columns([0.45, 9.55])
    with restore_col:
        st.button(
            "»",
            key="quant_show_sidebar",
            help="Open navigation",
            use_container_width=True,
            on_click=_set_sidebar_hidden,
            args=(False,),
        )

with st.sidebar:
    st.markdown(
        """
        <div class="qp-brand">
            <div class="qp-brand-mark">QS</div>
            <div class="qp-brand-copy">
                <strong>Quant Workspace</strong>
                <span>Portfolio intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="qp-eyebrow">Workspace</div>', unsafe_allow_html=True)
    app_route = st.radio(
        "Choose workspace",
        options=["Wharton Cockpit", "Quant Platform"],
        key="quant_sim_workspace_route",
        label_visibility="collapsed",
    )
    st.button(
        "Hide navigation",
        key="quant_hide_sidebar",
        use_container_width=True,
        on_click=_set_sidebar_hidden,
        args=(True,),
    )
    st.markdown("---")

    is_dark = st.toggle("Dark Mode", key="quant_manual_dark_mode")
    if is_dark:
        st.markdown(
            """
            <style>
            [data-testid="stAppViewContainer"], [data-testid="stSidebar"], :root {
                --background-color: #0f172a !important;
                --text-color: #e2e8f0 !important;
                --secondary-background-color: #1e293b !important;
                --qp-ink: #f8fafc !important;
                --qp-line: #334155 !important;
                --qp-card: #1e293b !important;
                --qp-muted: #94a3b8 !important;
                --qp-soft: #334155 !important;
                --qp-navy: #020617 !important;
                --qp-accent: #0f766e !important;
                --qp-accent-text: #2dd4bf !important;
                --qp-accent-soft: rgba(45, 212, 191, 0.15) !important;
                --qp-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            [data-testid="stAppViewContainer"], [data-testid="stSidebar"], :root {
                --background-color: #f8fafc !important;
                --text-color: #334155 !important;
                --secondary-background-color: #ffffff !important;
                --qp-ink: #1e293b !important;
                --qp-line: #e2e8f0 !important;
                --qp-card: #ffffff !important;
                --qp-muted: #64748b !important;
                --qp-soft: #f1f5f9 !important;
                --qp-navy: #0f172a !important;
                --qp-accent: #167d78 !important;
                --qp-accent-text: #167d78 !important;
                --qp-accent-soft: #eaf7f5 !important;
                --qp-shadow: 0 10px 30px rgba(15, 23, 42, 0.05) !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")


def _render_runtime_diagnostics(*, route: str, stage: str) -> None:
    """Show privacy-safe server timings for the current and recent reruns."""
    _RUNTIME_TRACE.mark(stage)
    snapshot = _RUNTIME_TRACE.snapshot(route=route, stage=stage)
    history = st.session_state.setdefault(_RUNTIME_HISTORY_KEY, [])
    append_trace_history(history, snapshot, limit=20)
    summary = summarize_trace_history(history)

    with st.sidebar.expander("Runtime & build", expanded=False):
        st.caption(f"Build `{_BUILD_IDENTITY.label}`")
        st.write(f"Current server run: **{float(snapshot['total_ms']) / 1000.0:.2f} s**")
        st.caption(
            f"Recent median {summary['median_ms'] / 1000.0:.2f} s · "
            f"p95 {summary['p95_ms'] / 1000.0:.2f} s · "
            f"{int(summary['count'])} run(s)"
        )
        st.json({
            "route": route,
            "stage": stage,
            "server_phase_ms": {
                name: round(float(duration), 1)
                for name, duration in dict(snapshot["phases_ms"]).items()
            },
            "scope": "Server-side Python only; browser and container wake-up are excluded.",
        })


_RUNTIME_TRACE.mark("launcher")

if app_route == "Wharton Cockpit":
    from ui.pages.wharton_dash import render_wharton_cockpit

    _RUNTIME_TRACE.mark("wharton_import")
    render_wharton_cockpit()
    _render_runtime_diagnostics(route=app_route, stage="wharton_ready")
    st.stop()

# A route change is reflected in Session State before the next script run. This
# fallback keeps programmatic invocations deterministic as well.
_run_quant_platform()
st.stop()
