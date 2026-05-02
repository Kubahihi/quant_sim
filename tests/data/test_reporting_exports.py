from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.reporting import (
    export_full_report_json,
    export_portfolio_data_csv,
    generate_pdf_report,
)


def _sample_report_payload() -> dict:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return {
        "inputs": {
            "tickers": ["AAPL", "BND"],
            "weights_pct": [60.0, 40.0],
            "horizon_days": 252,
            "risk_profile": "balanced",
            "risk_free_rate": 0.03,
        },
        "metrics": {
            "daily_return_mean": 0.0005,
            "annualized_return": 0.11,
            "volatility": 0.14,
            "sharpe_ratio": 0.78,
            "max_drawdown": -0.12,
            "total_return": 0.09,
            "hhi": 0.52,
            "effective_holdings": 1.92,
            "max_weight": 0.60,
            "avg_correlation": 0.21,
        },
        "score": {"score": 78, "rating": "good", "breakdown": []},
        "flags": [],
        "correlation_matrix": pd.DataFrame(
            [[1.0, 0.2], [0.2, 1.0]],
            index=["AAPL", "BND"],
            columns=["AAPL", "BND"],
        ),
        "portfolio_timeseries": pd.DataFrame(
            {"value": [100_000, 101_500, 101_000, 102_200, 103_400]},
            index=dates,
        ),
        "simulation": {
            "mean": 108_000,
            "median": 107_500,
            "percentile_5": 92_000,
            "percentile_95": 121_000,
        },
        "simulation_percentiles": pd.DataFrame(
            {"p5": [95_000, 96_200], "p50": [100_000, 103_000], "p95": [106_000, 110_000]}
        ),
        "ai_review": {
            "summary": "Healthy risk/reward balance.",
            "risks": "Equity drawdown sensitivity.",
            "improvements": "Add more diversifiers.",
            "verdict": "Acceptable with monitoring.",
        },
        "recommendation": "Acceptable with monitoring.",
    }


def test_reporting_exports_generate_pdf_csv_and_json():
    payload = _sample_report_payload()
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [100_000, 101_000, 102_500])
    figures = {"cumulative": fig}

    try:
        pdf_buffer = generate_pdf_report(payload, figures)
    finally:
        plt.close(fig)

    csv_bytes = export_portfolio_data_csv(payload)
    json_bytes = export_full_report_json(payload)

    assert len(pdf_buffer.getvalue()) > 1000
    assert b"value" in csv_bytes
    loaded = json.loads(json_bytes.decode("utf-8"))
    assert loaded["inputs"]["tickers"] == ["AAPL", "BND"]
