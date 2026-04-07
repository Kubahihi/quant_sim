import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.fetchers.yahoo_fetcher import YahooFetcher
from src.data.cache_manager import CacheManager
from src.portfolio.portfolio import Portfolio, Asset
from src.optimization import (
    optimize_minimum_variance,
    optimize_maximum_sharpe,
    calculate_efficient_frontier,
    calculate_portfolio_statistics,
    sample_portfolio_cloud,
)
from src.simulation import run_monte_carlo_simulation
from src.analytics.correlation import calculate_correlation_matrix
from src.visualization.charts_2d import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_correlation_heatmap,
    plot_efficient_frontier,
    plot_monte_carlo_fan,
)
from src.visualization.charts_3d import (
    plot_portfolio_tradeoff_3d,
    plot_monte_carlo_percentile_surface,
)

st.set_page_config(page_title="Quant Platform", layout="wide", page_icon="📊")

st.title("Quant Platform v0.1")
st.markdown("---")

with st.sidebar:
    st.header("Portfolio Configuration")
    
    symbols_input = st.text_area(
        "Asset Symbols (one per line)",value="AAPL\nELIL\nGLD\nAMZN\nVTI\nQQQ\nASML\nNEE\nJNJ\nKO\nBRK-B\nIEI\nBND\nVEA\nSHY\nBIL\nTIP\nPG\nTSM\nV\nNVDA\nMSFT\nBTC",
        height=150,
    )
    
    symbols = [s.strip().upper() for s in symbols_input.split("\n") if s.strip()]
    
    st.subheader("Date Range")
    end_date = datetime.now()
    start_date = st.date_input(
        "Start Date",
        value=end_date - timedelta(days=365 * 2),
    )
    
    risk_free_rate = st.slider(
        "Risk-Free Rate",
        min_value=0.0,
        max_value=0.10,
        value=0.03,
        step=0.005,
        format="%.3f",
    )
    
    st.subheader("Simulation Settings")
    n_simulations = st.slider(
        "Monte Carlo Simulations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
    )
    
    simulation_days = st.slider(
        "Simulation Horizon (days)",
        min_value=30,
        max_value=252 * 3,
        value=252,
        step=30,
    )

    st.subheader("3D Visuals")
    portfolio_samples = st.slider(
        "Sampled Portfolios",
        min_value=500,
        max_value=6000,
        value=2500,
        step=250,
        help="More samples create a richer 3D trade-off cloud at the cost of a slightly longer render time.",
    )

if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
    with st.spinner("Fetching data..."):
        fetcher = YahooFetcher()
        cache = CacheManager()
        
        prices = fetcher.fetch_close_prices(symbols, start_date, end_date)
        
        if prices.empty:
            st.error("No data fetched. Please check symbols and date range.")
            st.stop()
        
        st.success(f"Fetched data for {len(prices.columns)} assets")
    
    returns = prices.pct_change().dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cumulative Returns")
        fig = plot_cumulative_returns(returns)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Correlation Heatmap")
        corr = calculate_correlation_matrix(returns)
        fig = plot_correlation_heatmap(corr)
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.header("Portfolio Optimization")
    
    tab1, tab2, tab3 = st.tabs(["Minimum Variance", "Maximum Sharpe", "Efficient Frontier"])
    
    with tab1:
        with st.spinner("Optimizing..."):
            min_var_result = optimize_minimum_variance(returns)
        
        if min_var_result["success"]:
            st.success("Optimization successful")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Return", f"{min_var_result['expected_return']:.2%}")
            col2.metric("Volatility", f"{min_var_result['volatility']:.2%}")
            col3.metric("Sharpe Ratio", f"{min_var_result['sharpe_ratio']:.3f}")
            
            weights_df = pd.DataFrame({
                "Symbol": min_var_result["symbols"],
                "Weight": min_var_result["weights"],
            })
            weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(weights_df, use_container_width=True)
    
    with tab2:
        with st.spinner("Optimizing..."):
            max_sharpe_result = optimize_maximum_sharpe(returns, risk_free_rate)
        
        if max_sharpe_result["success"]:
            st.success("Optimization successful")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Return", f"{max_sharpe_result['expected_return']:.2%}")
            col2.metric("Volatility", f"{max_sharpe_result['volatility']:.2%}")
            col3.metric("Sharpe Ratio", f"{max_sharpe_result['sharpe_ratio']:.3f}")
            
            weights_df = pd.DataFrame({
                "Symbol": max_sharpe_result["symbols"],
                "Weight": max_sharpe_result["weights"],
            })
            weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(weights_df, use_container_width=True)
    
    with tab3:
        with st.spinner("Calculating efficient frontier..."):
            frontier = calculate_efficient_frontier(returns, n_points=30)
        
        st.success(f"Calculated {len(frontier)} frontier points")
        
        fig = plot_efficient_frontier(frontier)
        st.pyplot(fig)
    
    st.markdown("---")

    st.header("3D Visualizations")
    st.caption(
        "Explore the portfolio space in three dimensions: risk, return, and diversification. "
        "This makes concentration trade-offs much easier to spot than a flat frontier alone."
    )

    mean_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252
    equal_weights = np.array([1.0 / len(returns.columns)] * len(returns.columns))
    portfolio_return = (returns * equal_weights).sum(axis=1)
    ann_return = portfolio_return.mean() * 252
    ann_vol = portfolio_return.std() * np.sqrt(252)
    current_value = 100000

    with st.spinner("Running simulation..."):
        price_paths, stats = run_monte_carlo_simulation(
            current_value=current_value,
            expected_return=ann_return,
            volatility=ann_vol,
            time_horizon=simulation_days,
            n_simulations=n_simulations,
        )

    def build_portfolio_marker(name: str, weights: np.ndarray, expected_return: float, volatility: float, sharpe_ratio: float):
        metrics = calculate_portfolio_statistics(
            weights=weights,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
            symbols=returns.columns.tolist(),
        )
        return {
            "name": name,
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "diversification_score": metrics["diversification_score"],
            "effective_holdings": metrics["effective_holdings"],
            "max_weight": metrics["max_weight"],
            "top_holdings": metrics["top_holdings"],
        }

    equal_weight_metrics = calculate_portfolio_statistics(
        weights=equal_weights,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate,
        symbols=returns.columns.tolist(),
    )

    highlighted_portfolios = [
        build_portfolio_marker(
            "Equal Weight",
            equal_weights,
            expected_return=float(equal_weights @ mean_returns),
            volatility=float(np.sqrt(equal_weights.T @ cov_matrix @ equal_weights)),
            sharpe_ratio=equal_weight_metrics["sharpe_ratio"],
        )
    ]

    if min_var_result["success"]:
        highlighted_portfolios.append(
            build_portfolio_marker(
                "Min Variance",
                np.array(min_var_result["weights"]),
                expected_return=min_var_result["expected_return"],
                volatility=min_var_result["volatility"],
                sharpe_ratio=min_var_result["sharpe_ratio"],
            )
        )

    if max_sharpe_result["success"]:
        highlighted_portfolios.append(
            build_portfolio_marker(
                "Max Sharpe",
                np.array(max_sharpe_result["weights"]),
                expected_return=max_sharpe_result["expected_return"],
                volatility=max_sharpe_result["volatility"],
                sharpe_ratio=max_sharpe_result["sharpe_ratio"],
            )
        )

    with st.spinner("Rendering 3D analytics..."):
        portfolio_cloud = sample_portfolio_cloud(
            returns,
            n_samples=portfolio_samples,
            risk_free_rate=risk_free_rate,
        )

    viz_tab1, viz_tab2 = st.tabs(["Portfolio Lab", "Scenario Surface"])

    with viz_tab1:
        fig = plot_portfolio_tradeoff_3d(
            portfolio_cloud=portfolio_cloud,
            frontier_points=frontier,
            highlighted_portfolios=highlighted_portfolios,
        )
        st.plotly_chart(fig, use_container_width=True)

    with viz_tab2:
        st.caption(
            "The surface shows how the full range of possible outcomes expands over time, "
            "with median and tail paths called out directly in 3D."
        )
        fig = plot_monte_carlo_percentile_surface(price_paths)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.header("Monte Carlo Simulation")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Final Value", f"${stats['mean']:,.0f}")
    col2.metric("Median Final Value", f"${stats['median']:,.0f}")
    col3.metric("5th Percentile", f"${stats['percentile_5']:,.0f}")
    col4.metric("95th Percentile", f"${stats['percentile_95']:,.0f}")
    
    fig = plot_monte_carlo_fan(price_paths)
    st.pyplot(fig)
    
    st.markdown("---")
    
    st.header("Individual Asset Metrics")
    
    metrics_data = []
    
    for symbol in symbols:
        if symbol in returns.columns:
            asset_returns = returns[symbol]
            
            from src.analytics.returns import calculate_annualized_return
            from src.analytics.risk_metrics import (
                calculate_volatility,
                calculate_sharpe_ratio,
                calculate_max_drawdown,
            )
            
            metrics_data.append({
                "Symbol": symbol,
                "Ann. Return": calculate_annualized_return(asset_returns),
                "Volatility": calculate_volatility(asset_returns),
                "Sharpe": calculate_sharpe_ratio(asset_returns, risk_free_rate),
                "Max DD": calculate_max_drawdown(asset_returns),
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df["Ann. Return"] = metrics_df["Ann. Return"].apply(lambda x: f"{x:.2%}")
    metrics_df["Volatility"] = metrics_df["Volatility"].apply(lambda x: f"{x:.2%}")
    metrics_df["Sharpe"] = metrics_df["Sharpe"].apply(lambda x: f"{x:.3f}")
    metrics_df["Max DD"] = metrics_df["Max DD"].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(metrics_df, use_container_width=True)

else:
    st.info("<--- Configure portfolio settings in the sidebar and click 'Run Analysis'")
