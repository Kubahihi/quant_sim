from __future__ import annotations

import json
import textwrap
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _safe_text(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="split")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.02, 0.98, title, fontsize=18, fontweight="bold", va="top")

    cursor_y = 0.93
    for line in lines:
        wrapped = textwrap.fill(_safe_text(line), width=105)
        ax.text(0.02, cursor_y, wrapped, fontsize=10.5, va="top")
        cursor_y -= 0.035 * (wrapped.count("\n") + 1)
        if cursor_y < 0.05:
            break

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_dataframe_page(
    pdf: PdfPages,
    title: str,
    dataframe: pd.DataFrame,
    max_rows: int = 24,
) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", loc="left", pad=16)

    if dataframe.empty:
        ax.text(0.02, 0.95, "No data available.", fontsize=11, va="top")
    else:
        df_to_show = dataframe.head(max_rows).copy()
        df_to_show.columns = [str(col) for col in df_to_show.columns]
        table = ax.table(
            cellText=df_to_show.values,
            colLabels=df_to_show.columns,
            loc="upper left",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.2)

        if len(dataframe) > max_rows:
            ax.text(
                0.02,
                0.04,
                f"Showing first {max_rows} rows out of {len(dataframe)}.",
                fontsize=9,
                color="#666666",
            )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf_report(
    report_payload: Dict[str, Any],
    figures: Mapping[str, Any],
) -> BytesIO:
    """Generate a multi-page PDF report with metrics, charts, simulations and AI review."""
    output = BytesIO()

    with PdfPages(output) as pdf:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        inputs = report_payload.get("inputs", {})
        metrics = report_payload.get("metrics", {})
        score = report_payload.get("score", {})
        ai_review = report_payload.get("ai_review", {})
        flags = report_payload.get("flags", [])
        simulation = report_payload.get("simulation", {})
        recommendation = report_payload.get("recommendation", "")

        _add_text_page(
            pdf,
            "Portfolio Evaluation Report",
            [
                f"Generated at: {generated_at}",
                f"Tickers: {', '.join(inputs.get('tickers', []))}",
                f"Weights (%): {inputs.get('weights_pct', [])}",
                f"Investment horizon (days): {inputs.get('horizon_days', '-')}",
                f"Risk profile: {inputs.get('risk_profile', '-')}",
                f"Risk-free rate: {inputs.get('risk_free_rate', 0.0):.2%}",
                "",
                f"Deterministic score: {score.get('score', '-')} / 100",
                f"Rating: {score.get('rating', '-')}",
                "",
                "Final recommendation:",
                recommendation or "-",
            ],
        )

        metrics_lines = [
            f"Daily return mean: {metrics.get('daily_return_mean', 0.0):.4%}",
            f"Annualized return: {metrics.get('annualized_return', 0.0):.2%}",
            f"Volatility: {metrics.get('volatility', 0.0):.2%}",
            f"Sharpe ratio: {metrics.get('sharpe_ratio', 0.0):.3f}",
            f"Max drawdown: {metrics.get('max_drawdown', 0.0):.2%}",
            f"Total return: {metrics.get('total_return', 0.0):.2%}",
            "",
            f"Concentration (HHI): {metrics.get('hhi', 0.0):.3f}",
            f"Effective holdings: {metrics.get('effective_holdings', 0.0):.2f}",
            f"Largest weight: {metrics.get('max_weight', 0.0):.2%}",
            f"Average pairwise correlation: {metrics.get('avg_correlation', 0.0):.3f}",
            "",
            "Flags:",
            *([f"- {flag}" for flag in flags] if flags else ["- No critical flags."]),
            "",
            "AI review summary:",
            ai_review.get("summary", "AI review unavailable."),
            "AI risks:",
            ai_review.get("risks", "-"),
            "AI improvements:",
            ai_review.get("improvements", "-"),
            "AI verdict:",
            ai_review.get("verdict", "-"),
        ]
        _add_text_page(pdf, "Metrics, Flags, and AI Review", metrics_lines)

        allocation_rows = [
            {"Ticker": ticker, "Weight (%)": weight}
            for ticker, weight in zip(inputs.get("tickers", []), inputs.get("weights_pct", []))
        ]
        allocation_df = pd.DataFrame(allocation_rows)
        _add_dataframe_page(pdf, "Input Allocation", allocation_df, max_rows=30)

        corr_df = report_payload.get("correlation_matrix", pd.DataFrame())
        if isinstance(corr_df, pd.DataFrame):
            _add_dataframe_page(pdf, "Correlation Matrix", corr_df.round(3), max_rows=20)

        simulation_df = report_payload.get("simulation_percentiles", pd.DataFrame())
        if isinstance(simulation_df, pd.DataFrame) and not simulation_df.empty:
            _add_dataframe_page(pdf, "Simulation Percentiles", simulation_df.round(2), max_rows=40)

        simulation_lines = [
            f"Mean final value: {simulation.get('mean', 0.0):,.0f}",
            f"Median final value: {simulation.get('median', 0.0):,.0f}",
            f"5th percentile: {simulation.get('percentile_5', 0.0):,.0f}",
            f"95th percentile: {simulation.get('percentile_95', 0.0):,.0f}",
        ]
        _add_text_page(pdf, "Simulation Summary", simulation_lines)

        figure_titles = {
            "cumulative": "Portfolio and Asset Cumulative Returns",
            "assets": "Asset Cumulative Returns",
            "drawdown": "Portfolio Drawdown",
            "correlation": "Correlation Heatmap",
            "frontier": "Efficient Frontier",
            "monte_carlo": "Monte Carlo Fan Chart",
        }
        for key, title in figure_titles.items():
            fig = figures.get(key)
            if fig is None:
                continue
            fig.suptitle(title, fontsize=14, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight")

        info = pdf.infodict()
        info["Title"] = "Portfolio Evaluation Report"
        info["Author"] = "Quant Platform"
        info["CreationDate"] = datetime.now()

    output.seek(0)
    return output


def export_portfolio_data_csv(report_payload: Dict[str, Any]) -> bytes:
    """Export computed portfolio time series and key metrics to CSV."""
    timeseries = report_payload.get("portfolio_timeseries", pd.DataFrame())
    if not isinstance(timeseries, pd.DataFrame) or timeseries.empty:
        summary_df = pd.DataFrame([report_payload.get("metrics", {})])
        return summary_df.to_csv(index=False).encode("utf-8")

    export_df = timeseries.copy()
    export_df = export_df.reset_index().rename(columns={"index": "date"})
    return export_df.to_csv(index=False).encode("utf-8")


def export_full_report_json(report_payload: Dict[str, Any]) -> bytes:
    """Export all computed report artifacts to JSON."""
    json_text = json.dumps(
        report_payload,
        ensure_ascii=False,
        default=_json_default,
        indent=2,
    )
    return json_text.encode("utf-8")
