"""Local browser acceptance fixture; synthetic data, no authentication bypass.

Run from repository root: python -m streamlit run tests/ui/theme_gallery.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from ui.dashboard_shell import (
    inject_dashboard_styles, render_theme_toggle, is_dark_mode,
    render_plotly_chart, render_matplotlib_chart,
)

st.set_page_config(page_title="Theme acceptance", layout="wide")
inject_dashboard_styles()
with st.sidebar:
    st.markdown('<div class="qp-brand-copy"><strong>Quant Workspace</strong><span>Theme acceptance</span></div>', unsafe_allow_html=True)
    st.selectbox("Workspace", ["Wharton Cockpit", "Quant Platform"])
    render_theme_toggle()
    st.caption(f"Server theme: {'dark' if is_dark_mode() else 'light'} · Streamlit {st.__version__}")
    page = st.radio("Page", ["Controls and tables", "Charts"])

st.title(page)
st.caption("Readable native controls, canvas tables and custom charts in both modes.")
if page == "Controls and tables":
    left, right = st.columns(2)
    with left:
        st.number_input("Minimum holdings", min_value=0, value=2)
        st.checkbox("Require every holding in approved universe")
        st.multiselect("Allowed security types", ["Stock", "ETF", "Bond"], default=["Stock", "ETF"])
        with st.expander("Screener Configuration", expanded=True):
            st.text_area("Tickers", "AAPL\nMSFT")
    with right:
        with st.form("example"):
            st.selectbox("Account", ["Jakub", "Test account"])
            st.text_input("Password", type="password")
            st.text_input("Unsaved note", key="acceptance_note")
            st.form_submit_button("Enter workspace", type="primary")
            st.form_submit_button("Save draft", type="secondary")
        st.button("Analyze Companies", type="primary")
        st.button("Disabled action", disabled=True)
    st.subheader("Goal and capital buckets")
    edited = st.data_editor(pd.DataFrame({"Goal": ["Retirement", "Reserve"], "Capital allocation %": [70.0, 30.0], "Target amount": [500000, 20000], "Years": [10, 2]}), key="goals", num_rows="dynamic", width="stretch")
    st.caption(f"First goal: {edited.iloc[0, 0]}")
    st.subheader("Field-level provenance")
    st.dataframe(pd.DataFrame({"Mandate field": ["Risk tolerance", "Risk capacity"], "Origin": ["Temporary assumption", "Client statement"]}), width="stretch")
    st.info("Information is readable.")
    st.warning("Warning is readable.")
    st.error("Error is readable.")
    st.success("Success is readable.")
    st.link_button("Documentation", "https://docs.streamlit.io")
    st.code("print('native syntax colors')")
else:
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[2, 4, 3], name="Portfolio"))
        fig.update_layout(title="Plotly", template="plotly_dark")
        fig.add_annotation(x=2, y=4, text="Peak", font_color="white")
        render_plotly_chart(fig, width="stretch")
        st.line_chart(pd.DataFrame({"Portfolio": [2, 4, 3], "Benchmark": [1, 3, 2]}))
    with b:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [2, 4, 3], label="Portfolio")
        ax.set_title("Matplotlib")
        ax.set_xlabel("Year")
        ax.legend()
        render_matplotlib_chart(fig)
        plt.close(fig)
        fig = go.Figure(go.Surface(z=[[1, 2], [3, 4]]))
        fig.update_layout(title="3D surface", height=300)
        render_plotly_chart(fig, width="stretch")
