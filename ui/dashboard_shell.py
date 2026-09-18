from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import tomllib
import re
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


# Native widgets, canvas tables and default charts are themed by Streamlit,
# not CSS. Keep custom semantic tokens alongside the native configuration.
_CONFIG = tomllib.loads(
    (Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml").read_text(encoding="utf-8")
)
_TOKENS = tomllib.loads(Path(__file__).with_name("theme_tokens.toml").read_text(encoding="utf-8"))
THEME_PALETTES = {
    dark: {
        **_TOKENS[mode],
        **{name: _CONFIG["theme"][mode][native] for name, native in {
            "background": "backgroundColor", "surface": "secondaryBackgroundColor",
            "text": "textColor", "border": "borderColor", "accent": "primaryColor",
        }.items()},
    }
    for dark, mode in ((False, "light"), (True, "dark"))
}


def is_dark_mode() -> bool:
    """Latest browser-reported native theme, scoped to this user's session."""
    mode = st.session_state.get("_quant_native_theme", {}).get("mode")
    if mode in ("light", "dark"):
        return mode == "dark"
    try:
        return st.context.theme.type == "dark"
    except (AttributeError, KeyError):
        return False


def _theme_style() -> str:
    """Custom HTML only. Never override Streamlit or Glide theme variables."""
    rules = []
    for dark, mode in ((False, "light"), (True, "dark")):
        variables = "\n".join(
            f"  --qp-{name.replace('_', '-')}: {value};"
            for name, value in THEME_PALETTES[dark].items()
        )
        selector = ':root:not([data-quant-theme="dark"])' if not dark else ':root[data-quant-theme="dark"]'
        rules.append(f"{selector} {{\n{variables}\n}}")
    return '<style id="quant-runtime-theme">' + "\n".join(rules) + "</style>"


def _THEME_SWITCH(**kwargs: Any) -> Any:
    # Register against the current runtime, not the runtime present at import.
    # This also supports AppTest and processes with multiple app runtimes.
    component = st.components.v2.component(
        "quant_native_theme_switch",
        html='<button type="button" role="switch" aria-label="Dark mode" aria-checked="false">'
             '<span class="track" aria-hidden="true"><span class="thumb"></span></span>'
             '<span>Dark mode</span></button><div role="status" aria-live="polite"></div>',
        css=Path(__file__).with_name("theme_switch.css").read_text(encoding="utf-8"),
        js=Path(__file__).with_name("theme_switch.js").read_text(encoding="utf-8"),
    )
    return component(**kwargs)


def render_theme_toggle(*, disabled: bool = False) -> bool:
    """Select native Light/Dark without reloading or losing edited form data.

    v2 components can access the document. The small frontend adapter clicks
    the native theme menu, which also persists the choice in this browser.
    It reports back for server-rendered charts, and observes native menu and
    system-theme changes. There is no separate Python appearance preference.
    """
    _THEME_SWITCH(
        data={"mode": "dark" if is_dark_mode() else "light", "disabled": disabled},
        default={"mode": "dark" if is_dark_mode() else "light"},
        key="_quant_native_theme",
        on_mode_change=lambda: None,
    )
    return is_dark_mode()


def _apply_plotly_theme(figure: Any, palette: dict[str, str]) -> None:
    """Apply the shared palette without changing data or chart geometry."""
    figure.update_layout(
        template="plotly_dark" if palette is THEME_PALETTES[True] else "plotly_white",
        paper_bgcolor=palette["background"], plot_bgcolor=palette["surface"],
        font={"color": palette["text"]},
        title_font_color=palette["text"],
        colorway=[palette[f"chart_{i}"] for i in range(1, 7)],
        hoverlabel={"bgcolor": palette["surface"], "bordercolor": palette["border"],
                    "font": {"color": palette["text"]}},
        legend={"font": {"color": palette["text"]}, "bgcolor": palette["surface"],
                "title_font_color": palette["text"]},
    )
    axes = dict(gridcolor=palette["chart_grid"], linecolor=palette["border"],
                tickfont={"color": palette["muted"]}, title_font={"color": palette["text"]},
                zerolinecolor=palette["border"])
    figure.update_xaxes(**axes)
    figure.update_yaxes(**axes)
    figure.update_annotations(font_color=palette["text"])
    # Explicit scene/polar settings in an old figure override its template too.
    for key in figure.layout:
        if key.startswith("scene"):
            figure.layout[key].update(
                bgcolor=palette["surface"],
                **{f"{axis}axis": {**axes, "backgroundcolor": palette["surface"]}
                   for axis in "xyz"})
        elif key.startswith("polar"):
            figure.layout[key].update(bgcolor=palette["surface"],
                angularaxis={"color": palette["text"], "gridcolor": palette["chart_grid"]},
                radialaxis={"color": palette["text"], "gridcolor": palette["chart_grid"]})
    for trace in figure.data:
        if "textfont" in trace:
            trace.textfont.color = palette["text"]
        if "colorbar" in trace:
            trace.colorbar.update(tickfont_color=palette["text"], title_font_color=palette["text"])
        if ("marker" in trace and "colorbar" in trace.marker
                and ("colorbar" in trace.marker.to_plotly_json() or trace.marker.showscale)):
            trace.marker.colorbar.update(tickfont_color=palette["text"], title_font_color=palette["text"])


def render_plotly_chart(figure: Any, *args: Any, **kwargs: Any) -> Any:
    """Theme a copy: cached/shared analytical figures must remain unchanged."""
    import plotly.graph_objects as go
    spec = go.Figure(figure).to_dict()
    def recolor(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {name: recolor(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [recolor(item, key) for item in value]
        if isinstance(value, str) and key in {"color", "colors", "fillcolor"}:
            return theme_color(value)
        return value
    themed = go.Figure(recolor(spec))
    _apply_plotly_theme(themed, THEME_PALETTES[is_dark_mode()])
    kwargs["theme"] = None
    return st.plotly_chart(themed, *args, **kwargs)


def theme_color(value: str) -> str:
    """Map legacy analytical series colors to semantic colors in both modes."""
    palette = THEME_PALETTES[is_dark_mode()]
    roles = {
        "#6366f1": "chart_3", "#8b5cf6": "chart_3", "#7c3aed": "chart_3",
        "#4f46e5": "chart_3", "#9333ea": "chart_3",
        "#2563eb": "chart_2", "#3b82f6": "chart_2",
        "#ef4444": "danger", "#dc2626": "danger",
        "#22c55e": "success", "#10b981": "success", "#059669": "success",
        "#f59e0b": "warning", "#d97706": "warning", "#f97316": "warning",
        "#64748b": "muted", "#94a3b8": "muted", "#475569": "muted",
        "#e2e8f0": "text", "#f8fafc": "text", "#ffffff": "text",
        "#000000": "text", "white": "text", "black": "text",
        "#0891b2": "chart_1", "#14b8a6": "chart_1",
        "#ec4899": "chart_5", "#db2777": "chart_5", "#84cc16": "chart_6",
    }
    lowered = value.lower().replace(" ", "")
    rgba = re.fullmatch(r"rgba?\((\d+),(\d+),(\d+)(?:,([\d.]+))?\)", lowered)
    if rgba:
        lowered = "#" + "".join(f"{int(rgba[i]):02x}" for i in (1, 2, 3))
    color = palette.get(roles.get(lowered, ""), value)
    if rgba and rgba[4] is not None and color.startswith("#"):
        rgb = [int(color[i:i+2], 16) for i in (1, 3, 5)]
        return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{rgba[4]})"
    return color


def semantic_cell_style(role: str) -> str:
    palette = THEME_PALETTES[is_dark_mode()]
    return f"background-color: {palette[role + '_soft']}; color: {palette[role]};"


def _apply_matplotlib_theme(figure: Any, palette: dict[str, str]) -> None:
    from matplotlib.text import Text
    figure.set_facecolor(palette["background"])
    for text in figure.findobj(Text):
        text.set_color(palette["text"])
    for axis in figure.get_axes():
        axis.set_facecolor(palette["surface"])
        axis.tick_params(colors=palette["muted"])
        for spine in axis.spines.values():
            spine.set_color(palette["border"])
        # Keep existing grid visibility, instead of enabling grids on heatmaps.
        for line in [*axis.get_xgridlines(), *axis.get_ygridlines()]:
            line.set_color(palette["chart_grid"])
        cycle = {color: palette[f"chart_{i % 6 + 1}"] for i, color in enumerate(
            ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"))}
        for line in axis.lines:
            color = line.get_color()
            if isinstance(color, str) and color in cycle:
                line.set_color(cycle[color])
        # Heatmap annotations sit on the colormap, not the axes background.
        # Choose black/white from each underlying cell instead of theme text.
        for mesh in axis.collections:
            from matplotlib.collections import QuadMesh
            if not isinstance(mesh, QuadMesh):
                continue
            values = mesh.get_array().reshape(mesh.get_coordinates().shape[0] - 1, -1)
            for text in axis.texts:
                x, y = text.get_position()
                if 0 <= int(y) < values.shape[0] and 0 <= int(x) < values.shape[1]:
                    rgb = mesh.cmap(mesh.norm(values[int(y), int(x)]))[:3]
                    linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb]
                    luminance = sum(a*b for a, b in zip(linear, (.2126, .7152, .0722)))
                    text.set_color("#000000" if luminance > .179 else "#ffffff")
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(palette["surface"])
            legend.get_frame().set_edgecolor(palette["border"])


def render_matplotlib_chart(figure: Any, *args: Any, **kwargs: Any) -> Any:
    import matplotlib.pyplot as plt
    themed = deepcopy(figure)
    try:
        _apply_matplotlib_theme(themed, THEME_PALETTES[is_dark_mode()])
        return st.pyplot(themed, *args, **kwargs)
    finally:
        plt.close(themed)


@dataclass(frozen=True)
class DashboardPreferences:
    preset: str
    visible_pages: list[str]
    show_raw_tables: bool
    show_workspace_when_empty: bool


def inject_dashboard_styles() -> None:
    stylesheet = Path(__file__).with_name("dashboard.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>{_theme_style()}", unsafe_allow_html=True)


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
