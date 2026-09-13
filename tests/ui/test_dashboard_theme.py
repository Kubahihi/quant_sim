from __future__ import annotations

from pathlib import Path
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from ui import dashboard_shell as theme

ROOT = Path(__file__).resolve().parents[2]


def contrast(foreground, background):
    def luminance(color):
        values = [int(color[i:i+2], 16) / 255 for i in (1, 3, 5)]
        values = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in values]
        return sum(a*b for a, b in zip(values, (.2126, .7152, .0722)))
    a, b = sorted((luminance(foreground), luminance(background)))
    return (b + .05) / (a + .05)


@pytest.mark.parametrize("dark", [False, True])
def test_palettes_match_native_config_and_readable_text(dark):
    p = theme.THEME_PALETTES[dark]
    config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
    native = config["theme"]["dark" if dark else "light"]
    assert p["background"] == native["backgroundColor"]
    assert p["text"] == native["textColor"]
    for bg in ("background", "surface"):
        for fg in ("text", "muted", "accent_ink", "danger", "warning", "success"):
            assert contrast(p[fg], p[bg]) >= 4.5, (dark, fg, bg)
    assert contrast(p["accent_text"], p["accent"]) >= 4.5
    for role in ("danger", "warning", "success"):
        assert contrast(p[role], p[role + "_soft"]) >= 4.5
    for color in config["theme"]["chartCategoricalColors"]:
        assert contrast(color, p["surface"]) >= 3


def test_judge_cards_share_the_palette():
    source = (ROOT / "ui/judge_view.py").read_text(encoding="utf-8")
    css = source.split("def _inject_styles()", 1)[1].split("</style>", 1)[0]
    assert "--judge-ink:var(--qp-text)" in css
    assert "background:#fff" not in css
    assert "color:#fff" not in css


def test_css_does_not_fake_native_or_canvas_theme():
    style = theme._theme_style()
    css = (ROOT / "ui/dashboard.css").read_text()
    assert 'data-quant-theme="dark"' in style
    for forbidden in ("--background-color:", "--text-color:", "--gdg-bg-cell:", "canvas {"):
        assert forbidden not in style + css
    assert "p, label," not in css
    assert "_quant_base_" not in (ROOT / "ui/dashboard_shell.py").read_text()


@pytest.mark.parametrize("dark", [False, True])
def test_plotly_copy_themes_axes_annotations_and_3d_without_changing_data(monkeypatch, dark):
    monkeypatch.setattr(theme, "is_dark_mode", lambda: dark)
    rendered = {}
    monkeypatch.setattr(theme.st, "plotly_chart", lambda fig, **kw: rendered.update(fig=fig, kw=kw))
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4], line_color="#e2e8f0"))
    fig.add_annotation(x=1, y=3, text="Peak", font_color="white")
    fig.update_layout(template="plotly_dark", xaxis_showgrid=False)
    original = fig.to_json()
    theme.render_plotly_chart(fig)
    result = rendered["fig"]
    p = theme.THEME_PALETTES[dark]
    assert fig.to_json() == original
    assert list(result.data[0].y) == [3, 4]
    assert result.data[0].line.color == p["text"]
    assert result.layout.font.color == p["text"]
    assert result.layout.annotations[0].font.color == p["text"]
    assert result.layout.xaxis.showgrid is False
    assert "colorbar" not in result.data[0].marker.to_plotly_json()
    assert rendered["kw"]["theme"] is None
    surface = go.Figure(go.Surface(z=[[1, 2], [3, 4]]))
    theme.render_plotly_chart(surface)
    assert rendered["fig"].layout.scene.xaxis.tickfont.color == p["muted"]
    assert rendered["fig"].data[0].colorbar.tickfont.color == p["text"]


@pytest.mark.parametrize("dark", [False, True])
def test_matplotlib_copy_preserves_original_and_grid_visibility(monkeypatch, dark):
    monkeypatch.setattr(theme, "is_dark_mode", lambda: dark)
    rendered = {}
    monkeypatch.setattr(theme.st, "pyplot", lambda fig, **kw: rendered.update(fig=fig))
    fig, ax = plt.subplots()
    ax.plot([1, 2], label="Portfolio")
    ax.legend()
    ax.grid(False)
    original_bg = ax.get_facecolor()
    theme.render_matplotlib_chart(fig)
    result = rendered["fig"].axes[0]
    assert ax.get_facecolor() == original_bg
    assert result.title.get_color() == theme.THEME_PALETTES[dark]["text"]
    assert not any(line.get_visible() for line in result.get_xgridlines())
    assert list(result.lines[0].get_ydata()) == [1, 2]
    plt.close(fig)


def test_switch_mount_is_native_component_not_independent_toggle(monkeypatch):
    calls = {}
    monkeypatch.setattr(theme, "_THEME_SWITCH", lambda **kw: calls.update(kw))
    monkeypatch.setattr(theme, "is_dark_mode", lambda: False)
    theme.render_theme_toggle()
    assert calls["key"] == "_quant_native_theme"
    assert calls["default"]["mode"] == "light"
    script = (ROOT / "ui/theme_switch.js").read_text()
    assert "stMainMenuItem-theme-" in script
    assert "setStateValue('mode',mode)" in script.replace(" ", "")
    assert "location.reload" not in script
    assert "localStorage" not in script


def test_native_chart_calls_are_unchanged(monkeypatch):
    line, bar = theme.st.line_chart, theme.st.bar_chart
    monkeypatch.setattr(theme.st, "markdown", lambda *a, **kw: None)
    theme.inject_dashboard_styles()
    assert theme.st.line_chart is line
    assert theme.st.bar_chart is bar


def test_login_and_gallery_render_without_exceptions(monkeypatch):
    monkeypatch.setenv("QUANT_SIM_ENV", "development")
    for path in ("ui/streamlit_app.py", "tests/ui/theme_gallery.py"):
        app = AppTest.from_file(str(ROOT / path)).run(timeout=45)
        assert not app.exception
