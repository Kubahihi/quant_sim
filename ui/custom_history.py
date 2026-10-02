"""Custom history workspace with optional, explicit market-data downloads."""
from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pandas as pd
import streamlit as st

from src.data.custom_history import (
    FREQUENCIES, ImportedHistory, prepare_history, read_history_table, workbook_sheets,
)
from src.data.mixed_history import combine_histories, download_market_history, parse_market_tickers


def _download_csv(frame: pd.DataFrame) -> bytes:
    # Escape spreadsheet formula prefixes in user-controlled column labels.
    safe = frame.copy()
    safe.columns = ["'" + str(c) if str(c).startswith(("=", "+", "-", "@")) else c for c in safe.columns]
    return safe.to_csv(index=True).encode("utf-8-sig")


def render_custom_history(*, embedded: bool = False) -> None:
    if embedded:
        st.subheader("Custom data analysis")
    else:
        st.title("Custom data analysis")
    st.caption("Import your own history, optionally add market tickers, and evaluate the combined portfolio at the original data frequency.")
    uploaded = st.file_uploader("Historical data", type=["csv", "xlsx"], key="custom_history_upload", help="Up to 20 MB. One date column and one column per asset. XLSX formulas must have saved calculated values.")
    if uploaded is None:
        st.info("Upload a CSV or Excel workbook to begin. Both periodic simple returns and price or index levels are supported.")
        st.download_button("Download CSV example", "Date,Asset A,Asset B\n2024-01-31,0.02,0.01\n2024-02-29,-0.01,0.005\n2024-03-31,0.015,-0.002\n", "custom_history_example.csv", "text/csv")
        st.caption("The example contains monthly decimal returns: 0.02 means 2%. Use adjusted prices or total-return indices when dividends should be included.")
        return

    content = uploaded.getvalue()
    file_id = sha256(content).hexdigest()
    try:
        sheets = workbook_sheets(content, uploaded.name)
        st.subheader("1. Data layout")
        layout = st.columns(3)
        sheet = layout[0].selectbox("Worksheet", sheets, index=sheets.index("Mesicni data") if "Mesicni data" in sheets else 0, key=f"sheet_{file_id}") if sheets else None
        known_layout = sheet == "Mesicni data"
        header = int(layout[1].number_input("Header row", min_value=1, max_value=100, value=5 if known_layout else 1, key=f"header_{file_id}_{sheet}"))
        separator, decimal = ",", "."
        if not sheets:
            separator_label = layout[0].selectbox("CSV separator", ["Comma", "Semicolon", "Tab"])
            separator = {"Comma": ",", "Semicolon": ";", "Tab": "\t"}[separator_label]
            decimal = layout[2].selectbox("Decimal separator", [".", ","])
        table = read_history_table(content, uploaded.name, sheet=sheet, header_row=header, separator=separator, decimal=decimal)
    except Exception as exc:
        st.error(f"Unable to read the table: {exc}")
        return

    st.dataframe(table.head(8), hide_index=True, width="stretch")
    mapping_key = sha256(f"{file_id}:{sheet}:{header}:{separator}:{decimal}".encode()).hexdigest()[:16]
    columns = table.columns.tolist()
    controls = st.columns(3)
    date_column = controls[0].selectbox("Date column", columns, key=f"date_{mapping_key}")
    candidates = [c for c in columns if c != date_column]
    model_columns = [c for c in ["XUSE model výnos", "EXCS model výnos"] if c in candidates]
    default_assets = model_columns or [c for c in candidates if pd.to_numeric(table[c], errors="coerce").notna().any()]
    assets = controls[1].multiselect("Asset columns", candidates, default=default_assets, key=f"assets_{mapping_key}_{date_column}")
    frequency = controls[2].selectbox("Observation frequency", list(FREQUENCIES), index=2 if model_columns else 0, key=f"freq_{mapping_key}")
    controls = st.columns(3)
    value_type = controls[0].selectbox("Values represent", ["Returns", "Prices / index levels"], help="Returns must be simple periodic returns, not logarithmic returns. Price-based returns include dividends only if the supplied levels include them.", key=f"kind_{mapping_key}")
    return_unit = controls[1].selectbox("Return units", ["Decimal", "Percent"], disabled=value_type != "Returns", help="Decimal: 0.01 = 1%. Percent: 1 = 1%. Excel percentage cells normally contain decimals.", key=f"units_{mapping_key}")
    date_format = controls[2].selectbox("Date format", ["ISO / Excel dates", "Day / month / year", "Month / day / year"], key=f"dateformat_{mapping_key}")
    details = st.columns(2)
    currency = details[0].text_input("Data currency", value="USD" if model_columns else "", placeholder="For example USD or EUR", help="Currency of the uploaded data and the combined portfolio. Downloaded assets in other currencies are converted using historical exchange rates.", key=f"currency_{mapping_key}").strip().upper()
    source_kind = details[1].selectbox("History type", ["Observed history", "Synthetic / index proxy"], index=1 if model_columns else 0, key=f"source_{mapping_key}")
    st.caption("All uploaded assets must use the selected data currency. Downloaded assets are converted to this portfolio currency using historical FX rates.")
    if not assets:
        st.info("Select at least one asset column.")
        return

    st.subheader("2. Portfolio assumptions")
    add_market = st.checkbox("Add assets by ticker", key="custom_add_market")
    tickers = []
    if add_market:
        ticker_text = st.text_area(
            "Market tickers", key="custom_market_tickers", placeholder="SPY, AGG, GLD",
            help="Yahoo Finance symbols, separated by commas or new lines. Up to 20 assets.",
        )
        st.caption("Adjusted prices and any required historical FX rates will be downloaded. Combined returns use the uploaded observation intervals and data currency.")
        try:
            tickers = parse_market_tickers(ticker_text)
        except ValueError as exc:
            st.error(str(exc))
            return
        if not tickers:
            st.info("Enter the market tickers to add to the uploaded assets.")
            return
    default_names = [c.removesuffix(" model výnos") for c in assets] + tickers
    source_columns = assets + [f"Yahoo Finance: {ticker}" for ticker in tickers]
    asset_count = len(default_names)
    allocation = st.data_editor(
        pd.DataFrame({"Source column": source_columns, "Asset": default_names, "Weight (%)": np.full(asset_count, 100.0 / asset_count)}),
        disabled=["Source column"], hide_index=True, width="stretch",
        column_config={"Weight (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, format="%.4f")},
        key=f"allocation_{mapping_key}_{','.join(assets)}_{','.join(tickers)}",
    )
    try:
        history = prepare_history(table, date_column=date_column, asset_columns=assets,
                                  frequency=frequency, value_type=value_type, date_format=date_format,
                                  return_unit=return_unit, names=allocation["Asset"].iloc[:len(assets)].tolist())
    except ValueError as exc:
        st.error(str(exc))
        return
    market_metadata = None
    market_digest = None
    if tickers:
        boundary = history.baseline_date if history.baseline_date is not None else history.returns.index[0]
        download_start = pd.Timestamp(boundary).tz_localize(None).normalize() - pd.Timedelta(days=7)
        download_end = history.returns.index[-1].tz_localize(None).normalize()
        request = {"tickers": tickers, "start": str(download_start.date()), "end": str(download_end.date()), "base_currency": currency}
        request_key = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if st.button("Download market data", key="custom_download_market", disabled=not currency):
            st.session_state.pop("custom_market_snapshot", None)
            st.session_state.pop("custom_history_result", None)
            try:
                with st.spinner("Downloading adjusted market history..."):
                    market = download_market_history(tickers, download_start, download_end, base_currency=currency)
                digest = sha256(pd.util.hash_pandas_object(market.prices, index=True).values.tobytes() + pd.util.hash_pandas_object(market.fx_prices, index=True).values.tobytes()).hexdigest()
                st.session_state["custom_market_snapshot"] = (request_key, market, digest)
            except Exception as exc:
                st.error(f"Market data download failed: {exc}")
                return
        if not currency:
            st.info("Specify the common data currency to download market data.")
            return
        snapshot = st.session_state.get("custom_market_snapshot")
        if snapshot is None or snapshot[0] != request_key:
            st.info("Download market data to check coverage and prepare the combined portfolio.")
            return
        try:
            history, market_metadata = combine_histories(
                history, snapshot[1], tickers=tickers, currency=currency,
                market_names=allocation["Asset"].iloc[len(assets):].tolist(),
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        market_digest = snapshot[2]
        st.info(f"Common history: {history.returns.index[0].date()} to {history.returns.index[-1].date()} · {len(history.returns):,} {frequency.lower()} returns · {asset_count} assets.")
        with st.expander("Market coverage and sources"):
            st.dataframe(pd.DataFrame([
                {"Ticker": ticker, "Quote currency": detail["currency"], "Portfolio currency": detail["reporting_currency"], "First return": detail["first_return"],
                 "Last return": detail["last_return"], "Available returns": detail["available_returns"]}
                for ticker, detail in market_metadata["coverage"].items()
            ]), hide_index=True, width="stretch")
            st.caption(f"Yahoo Finance adjusted closes · downloaded {market_metadata['downloaded_at']}. Only common observations are used. Refresh with Download market data.")
            fx = market_metadata["fx_conversion"]
            if fx["symbols"]:
                st.caption(f"Historical FX conversion to {currency}: {', '.join(fx['symbols'])}. Rates use the latest available close on or before each observation, at most seven days earlier.")
    for warning in history.warnings:
        st.caption(warning)
    controls = st.columns(3)
    dates = controls[0].date_input("Return observation range", value=(history.returns.index.min().date(), history.returns.index.max().date()), key=f"range_{mapping_key}_{frequency}_{value_type}_{date_column}_{','.join(tickers)}_{history.returns.index[0]}_{history.returns.index[-1]}")
    risk_free_pct = controls[1].number_input("Annual risk-free rate (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.25, help="Use a risk-free rate in the portfolio's data currency.")
    max_weight = controls[2].number_input("Maximum optimized position (%)", min_value=float(100.0 / asset_count), max_value=100.0, value=100.0, key=f"maxweight_{asset_count}")
    if len(dates) != 2:
        st.info("Select both dates for the analysis range.")
        return
    returns = history.returns.loc[str(dates[0]):str(dates[1])]
    if len(returns) < 2:
        st.error("The selected range must contain at least two return observations.")
        return
    first_position = history.returns.index.get_loc(returns.index[0])
    baseline_date = history.returns.index[first_position - 1] if first_position else history.baseline_date
    st.caption(f"Effective analysis range: {returns.index[0].date()} to {returns.index[-1].date()}.")
    history = ImportedHistory(returns, frequency, history.warnings, baseline_date)
    st.caption(f"{len(returns):,} return observations · {asset_count} assets · {history.periods_per_year} periods per year. Weights are reset at each observation period.")
    st.caption(f"Normalized asset returns range from {returns.min().min():.2%} to {returns.max().max():.2%}. Check these percentages against your source before running the analysis.")
    if len(returns) < 3 * history.periods_per_year:
        st.warning("This selection contains less than three years of observations. Estimates and resampled scenarios may be unstable.")
    with st.expander("Simulation assumptions"):
        controls = st.columns(3)
        years = int(controls[0].number_input("Horizon (years)", min_value=1, max_value=30, value=5))
        paths = int(controls[1].number_input("Simulation paths", min_value=100, max_value=10000, value=2000, step=100))
        block = int(controls[2].number_input("Consecutive observations per block", min_value=1, max_value=len(returns), value=min(3, len(returns))))
        st.caption("Historical block bootstrap with seed 42. Assets are sampled together to retain their relationships; blocks retain short-run dependence. No additional taxes or trading costs are deducted.")
    metadata = {
        "filename": uploaded.name, "sha256": file_id, "worksheet": sheet, "header_row": header,
        "date_column": date_column, "asset_columns": dict(zip(assets, allocation["Asset"].iloc[:len(assets)].tolist())),
        "value_type": value_type, "return_unit": return_unit, "date_format": date_format,
        "csv_separator": separator if not sheets else None, "decimal_separator": decimal if not sheets else None,
        "frequency": frequency, "periods_per_year": history.periods_per_year,
        "currency": currency, "history_type": source_kind,
        "market_data": market_metadata, "market_prices_sha256": market_digest,
        "asset_order": allocation["Asset"].tolist(),
        "start": str(returns.index.min().date()), "end": str(returns.index.max().date()),
        "baseline_date": str(baseline_date.date()) if baseline_date is not None else None,
        "observations": len(returns), "weights_percent": allocation["Weight (%)"].tolist(),
        "annual_risk_free_rate": risk_free_pct / 100, "maximum_optimized_weight": max_weight / 100,
        "simulation": {"method": "joint_circular_block_bootstrap", "years": years, "paths": paths, "block_length": block, "seed": 42},
        "rebalancing": "constant weights reset each observation period", "warnings": history.warnings,
    }
    fingerprint = sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    if st.button("Analyze imported data", type="primary"):
        st.session_state.pop("custom_history_result", None)
        try:
            if not currency:
                raise ValueError("Specify the common data currency before running the analysis.")
            from src.analytics.custom_history import analyze_history, bootstrap_history
            from src.optimization import estimate_portfolio_inputs, optimize_maximum_sharpe, optimize_minimum_variance

            weights = allocation["Weight (%)"].to_numpy(dtype=float) / 100
            with st.spinner("Analyzing imported history..."):
                result = analyze_history(history, weights, risk_free_pct / 100)
                result["simulation"] = bootstrap_history(history, weights, years=years, simulations=paths, block_length=block)
                result["optimizations"] = {}
                try:
                    estimates = estimate_portfolio_inputs(returns, trading_days=history.periods_per_year)
                    for name, optimizer in [("Minimum variance", optimize_minimum_variance), ("Maximum Sharpe", optimize_maximum_sharpe)]:
                        result["optimizations"][name] = optimizer(returns, risk_free_rate=risk_free_pct / 100, max_weight=max_weight / 100, portfolio_estimates=estimates)
                except ValueError as exc:
                    result["optimization_error"] = str(exc)
                st.session_state["custom_history_result"] = (fingerprint, result)
        except ValueError as exc:
            st.error(str(exc))
            return
    saved = st.session_state.get("custom_history_result")
    if saved is None:
        return
    if saved[0] != fingerprint:
        st.info("Inputs have changed. Run the analysis to refresh the results.")
        return
    _render_results(saved[1], returns, metadata)


def _render_results(result: dict, returns: pd.DataFrame, metadata: dict) -> None:
    st.divider()
    st.subheader("Portfolio results")
    st.caption(f"{metadata['filename']} · {metadata['history_type']} · {metadata['frequency']} · {metadata['currency']} · {metadata['start']} to {metadata['end']}")
    if metadata.get("market_data"):
        st.caption("Additional market assets: " + ", ".join(metadata["market_data"]["tickers"]) + " · Yahoo Finance adjusted prices")
    for column, (label, value) in zip(st.columns(5), result["metrics"].items()):
        column.metric(label, f"{value:.2f}" if label == "Sharpe ratio" else f"{value:.2%}")
    overview, allocation, simulation, data = st.tabs(["Performance & risk", "Allocation comparison", "Simulation", "Data & export"])
    with overview:
        st.caption("Cumulative return at observation dates. Drawdown includes losses from initial capital; intra-period losses are not observable.")
        growth = (1 + returns).cumprod() - 1
        growth["Portfolio (weighted)"] = result["timeseries"]["cumulative_return"]
        st.line_chart(growth, width="stretch")
        st.dataframe(result["assets"].style.format({c: "{:.2f}" if c == "Sharpe ratio" else "{:.2%}" for c in result["assets"]}), width="stretch")
        st.markdown("**Asset correlations**")
        st.dataframe(result["correlation"].style.format("{:.2f}", na_rep="Undefined"), width="stretch")
    with allocation:
        st.caption("Exploratory allocations fitted to the same history, using the existing shrunk return and covariance estimates. Expected returns here are annual arithmetic estimates.")
        if result.get("optimization_error"):
            st.warning(f"Optimization unavailable: {result['optimization_error']}")
        comparison = pd.DataFrame({"Current weight": np.asarray(metadata["weights_percent"]) / 100}, index=returns.columns)
        for name, optimized in result["optimizations"].items():
            if optimized.get("success"):
                comparison[name] = optimized["weights"]
                st.write(f"**{name}** — expected return {optimized['expected_return']:.2%}, volatility {optimized['volatility']:.2%}")
            else:
                st.warning(f"{name}: {optimized.get('message', 'No feasible solution.')}")
        st.dataframe(comparison.style.format("{:.2%}"), width="stretch")
    with simulation:
        st.caption("Projected wealth multiples from joint historical resampling. Starting wealth = 1. Percentiles reflect this historical model, not guaranteed outcomes.")
        st.line_chart(result["simulation"], width="stretch")
        st.dataframe(result["simulation"].tail(1).style.format("{:.2f}×"), width="stretch")
    with data:
        st.caption("Normalized simple returns in decimal units. Source settings are available separately for reproducibility. Uploaded data stays in this session and is not added to saved market history.")
        st.dataframe(returns, width="stretch")
        st.download_button("Download normalized returns", _download_csv(returns), "custom_returns.csv", "text/csv")
        st.download_button("Download import settings", json.dumps(metadata, indent=2, ensure_ascii=False), "custom_import_settings.json", "application/json")
