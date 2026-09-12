from __future__ import annotations

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ui.pages.wharton_dash import _format_company_metric


@pytest.mark.parametrize("value", [28_400_000_000, 28_400_000_000.0, np.int64(28_400_000_000)])
def test_market_cap_uses_the_same_units_for_integer_and_float_data(value):
    assert _format_company_metric("Market cap", value) == "28.40B"


@pytest.mark.parametrize("key", ["ROE", "returnOnEquity", "Operating margin"])
def test_percentage_metrics_keep_their_units_in_summary_and_detail(key):
    assert _format_company_metric(key, 0.224) == "22.40%"
    assert _format_company_metric(key, 0) == "0.00%"
    assert _format_company_metric(key, -0.052) == "-5.20%"


def test_price_keeps_cents_and_missing_values_are_not_zero():
    assert _format_company_metric("Price", 142.35) == "142.35"
    assert _format_company_metric("Price", 14) == "14.00"
    assert _format_company_metric("ROE", None) == "—"
    assert _format_company_metric("Market cap", float("nan")) == "—"


def test_existing_company_tab_selection_opens_its_new_section():
    def app():
        import streamlit as st
        from ui.pages.wharton_dash import _render_company_detail_navigation

        view, _ = _render_company_detail_navigation("DEMO")
        st.write(view)

    at = AppTest.from_function(app)
    at.session_state["company_detail_tab_DEMO"] = "DCF"
    at.run()

    assert not at.exception
    assert at.session_state["company_section_tab_DEMO"] == "Valuation"
    assert any(item.value == "DCF" for item in at.markdown)
