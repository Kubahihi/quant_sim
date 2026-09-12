"""Offline company-loading benchmark; never calls live data or AI providers."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--endpoint-delay", type=float, default=0.05)
    args = parser.parse_args()
    if args.repeats < 1 or args.endpoint_delay < 0:
        parser.error("repeats must be positive and endpoint-delay must be non-negative")
    sys.path.insert(0, str(args.project_root.resolve()))

    import pandas as pd
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import yfinance as yf
    from src.analytics import company_analysis
    from ui.pages import wharton_dash

    info = {
        "longName": "Benchmark Company", "currency": "USD", "financialCurrency": "USD",
        "currentPrice": 100.0, "marketCap": 100_000_000_000,
        "freeCashflow": 10_000_000_000, "sharesOutstanding": 1_000_000_000,
        "totalCash": 5_000_000_000, "totalDebt": 1_000_000_000,
        "beta": 1.0, "revenueGrowth": 0.1, "operatingMargins": 0.3,
    }
    history = pd.DataFrame(
        {"Close": [90.0, 100.0]}, index=pd.date_range("2025-01-01", periods=2),
    )

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            time.sleep(args.endpoint_delay)
            return history.copy()

        def __getattr__(self, field):
            time.sleep(args.endpoint_delay)
            if field == "info":
                return dict(info)
            if field in {"news", "sec_filings"}:
                return []
            return pd.DataFrame()

    def geography(*args_, **kwargs):
        time.sleep(args.endpoint_delay)
        return {"available": False}

    provider_times = []
    with patch.object(yf, "Ticker", Ticker), patch.object(
        company_analysis, "fetch_geographic_revenue", geography,
    ):
        for _ in range(args.repeats):
            started = time.perf_counter()
            snapshot = company_analysis.fetch_company_data("TEST")
            provider_times.append((time.perf_counter() - started) * 1000)
            assert snapshot["provider_status"] == "live"
            assert snapshot["info"] == info
            pd.testing.assert_frame_equal(snapshot["history"], history)

    hidden_calls = Counter()

    def render_hidden(name):
        def render(*args_, **kwargs):
            hidden_calls[name] += 1
            st.caption(f"Offline {name}")
        return render

    def ai_defaults(*args_, **kwargs):
        hidden_calls["ai_dcf"] += 1
        return {"available": False}

    ui_times = []
    with ExitStack() as patches:
        patches.enter_context(patch.object(st, "secrets", {}))
        patches.enter_context(patch.object(wharton_dash, "_fetch_competition_positions", lambda: []))
        for name in ("geographic_revenue", "industry_peer_analysis", "research_evidence"):
            patches.enter_context(patch.object(wharton_dash, f"_render_{name}", render_hidden(name)))
        patches.enter_context(patch.object(wharton_dash, "_fetch_ai_dcf_assumptions_cached", ai_defaults))
        app = AppTest.from_string(
            "from ui.pages import wharton_dash\n"
            "wharton_dash._render_company_analysis({'username': 'benchmark'})\n"
        )
        app.session_state[wharton_dash.COMPANY_ANALYSIS_KEY] = {"TEST": snapshot}
        # Warm imports and Streamlit's test runner, then measure rerenders.
        app.run(timeout=30)
        assert not app.exception, app.exception
        hidden_calls.clear()
        for _ in range(args.repeats):
            started = time.perf_counter()
            app.run(timeout=30)
            ui_times.append((time.perf_counter() - started) * 1000)
            assert not app.exception, app.exception
        rendered_elements = sum(1 for _ in app)

    result = {
        "project_root": str(args.project_root.resolve()),
        "python": sys.version.split()[0], "streamlit": st.__version__,
        "repeats": args.repeats, "endpoint_delay_ms": args.endpoint_delay * 1000,
        "company_snapshot_ms": provider_times,
        "company_snapshot_median_ms": statistics.median(provider_times),
        "company_overview_rerender_ms": ui_times,
        "company_overview_rerender_median_ms": statistics.median(ui_times),
        "hidden_calls_total": dict(hidden_calls),
        "rendered_elements": rendered_elements,
        "scope": "Synthetic endpoint delays; UI server rendering with external enrichment stubbed. No live-network measurement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
