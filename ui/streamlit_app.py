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


def _judge_session_active() -> bool:
    profile = st.session_state.get("wharton_user_profile_v2")
    return bool(
        isinstance(profile, dict)
        and str(profile.get("username") or "").strip() == "judge"
    )


# The analytical workspace deliberately remains a normal Python module.  When
# it is selected, delegate before configuring or drawing this lightweight
# launcher so there is only one page shell in the current run.
_JUDGE_SESSION_ACTIVE = _judge_session_active()
if _JUDGE_SESSION_ACTIVE:
    st.session_state["quant_sim_workspace_route"] = "Wharton Cockpit"

if (
    not _JUDGE_SESSION_ACTIVE
    and st.session_state.get("quant_sim_workspace_route") == "Quant Platform"
):
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

with st.sidebar:
    st.markdown(
        """
        <div class="qp-brand">
            <div class="qp-brand-mark">QS</div>
            <div class="qp-brand-copy">
                <strong>Quant Workspace</strong>
                <span>Research &amp; portfolio</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if _JUDGE_SESSION_ACTIVE:
        app_route = "Wharton Cockpit"
        st.caption("Judge View · Read-only competition record")
    else:
        app_route = st.selectbox(
            "Choose workspace",
            options=["Wharton Cockpit", "Quant Platform"],
            key="quant_sim_workspace_route",
            label_visibility="collapsed",
        )


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
    if not _JUDGE_SESSION_ACTIVE:
        _render_runtime_diagnostics(route=app_route, stage="wharton_ready")
    st.stop()

# A route change is reflected in Session State before the next script run. This
# fallback keeps programmatic invocations deterministic as well.
_run_quant_platform()
st.stop()
