from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from ui import dashboard_shell


APP_PATH = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py"


def test_runtime_dark_style_exposes_a_complete_palette(monkeypatch):
    monkeypatch.setattr(dashboard_shell, "is_dark_mode", lambda: True)

    style = dashboard_shell._theme_style()

    for color in dashboard_shell.THEME_PALETTES[True].values():
        assert color in style
    assert "--background-color: var(--qp-background)" in style
    assert "--text-color: var(--qp-text)" in style


def test_plotly_palette_overrides_a_chart_specific_template(monkeypatch):
    palette = dashboard_shell.THEME_PALETTES[True]
    monkeypatch.setattr(dashboard_shell, "is_dark_mode", lambda: True)
    figure = go.Figure().update_layout(template="plotly_white")

    dashboard_shell._apply_plotly_theme(figure, palette)

    assert figure.layout.plot_bgcolor == palette["surface"]
    assert figure.layout.font.color == palette["text"]
    assert figure.layout.xaxis.gridcolor == palette["chart_grid"]
    assert figure.layout.yaxis.tickfont.color == palette["muted"]


def test_matplotlib_palette_colours_the_figure_and_legend():
    palette = dashboard_shell.THEME_PALETTES[True]
    figure, axis = plt.subplots()
    axis.plot([1, 2], label="Portfolio")
    axis.legend()

    dashboard_shell._apply_matplotlib_theme(figure, palette)

    assert axis.get_facecolor()[:3] != (1.0, 1.0, 1.0)
    assert axis.title.get_color() == palette["text"]
    assert axis.get_legend().get_texts()[0].get_color() == palette["text"]
    plt.close(figure)


def test_theme_toggle_uses_the_persisted_preference_key(monkeypatch):
    rendered = {}
    monkeypatch.setattr(
        dashboard_shell.st,
        "toggle",
        lambda label, **kwargs: rendered.update(label=label, **kwargs) or True,
    )

    assert dashboard_shell.render_theme_toggle() is True
    assert rendered["label"] == "Dark mode"
    assert rendered["key"] == dashboard_shell.THEME_MODE_KEY


def test_native_dataframe_charts_use_a_palette_aware_vega_spec(monkeypatch):
    rendered = {}
    monkeypatch.setattr(dashboard_shell, "is_dark_mode", lambda: True)
    monkeypatch.setattr(
        dashboard_shell.st,
        "_quant_base_line_chart",
        lambda *args, **kwargs: pytest.fail("native fallback should not be used"),
        raising=False,
    )
    monkeypatch.setattr(
        dashboard_shell.st,
        "altair_chart",
        lambda chart, **kwargs: rendered.update(chart=chart, **kwargs),
    )

    dashboard_shell._render_native_chart(pd.DataFrame({"Value": [1.0, 2.0]}), "line")

    assert rendered["theme"] is None
    assert rendered["use_container_width"] is True
    assert rendered["chart"].to_dict()["config"]["background"] == dashboard_shell.THEME_PALETTES[True]["surface"]


def test_workspace_theme_toggle_persists_across_a_streamlit_rerun(monkeypatch):
    monkeypatch.setenv("QUANT_SIM_ENV", "development")
    app = AppTest.from_file(str(APP_PATH))

    app.run(timeout=30)
    toggle = next(item for item in app.sidebar.toggle if item.label == "Dark mode")
    toggle.set_value(True).run(timeout=30)

    assert app.session_state[dashboard_shell.THEME_MODE_KEY] is True
    assert next(item for item in app.sidebar.toggle if item.label == "Dark mode").value is True
    assert not app.exception
