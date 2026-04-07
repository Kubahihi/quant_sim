#!/usr/bin/env python3
"""
Example usage of Quant Platform
"""
from datetime import datetime, timedelta
import numpy as np

from src.data.fetchers.yahoo_fetcher import YahooFetcher
from src.data.cache_manager import CacheManager
from src.portfolio.portfolio import Portfolio, Asset
from src.optimization import optimize_minimum_variance, optimize_maximum_sharpe
from src.simulation import run_monte_carlo_simulation
from src.analytics.correlation import calculate_correlation_matrix


def main():
    print("=== Quant Platform - Example Usage ===\n")
    
    symbols = ["AAPL", "ELIL", "GLD", "AMZN", "VTI", "QQQ", "ASML", "NEE", "JNJ", "KO", "BRK-B", "IEI", "BND", "VEA", "SHY", "BIL", "TIP", "PG", "TSM", "V", "NVDA", "MSFT", "BTC"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    
    print(f"Fetching data for {symbols}")
    print(f"Period: {start_date.date()} to {end_date.date()}\n")
    
    fetcher = YahooFetcher()
    cache = CacheManager()
    
    prices = fetcher.fetch_close_prices(symbols, start_date, end_date)
    
    if prices.empty:
        print("Error: No data fetched")
        return
    
    print(f"✓ Fetched data for {len(prices.columns)} assets\n")
    
    returns = prices.pct_change().dropna()
    
    print("=== Correlation Matrix ===")
    corr = calculate_correlation_matrix(returns)
    print(corr.round(2))
    print()
    
    print("=== Minimum Variance Optimization ===")
    min_var = optimize_minimum_variance(returns)
    
    if min_var["success"]:
        print(f"Expected Return: {min_var['expected_return']:.2%}")
        print(f"Volatility: {min_var['volatility']:.2%}")
        print(f"Sharpe Ratio: {min_var['sharpe_ratio']:.3f}")
        print("\nWeights:")
        for symbol, weight in zip(min_var["symbols"], min_var["weights"]):
            print(f"  {symbol}: {weight:.2%}")
    print()
    
    print("=== Maximum Sharpe Optimization ===")
    max_sharpe = optimize_maximum_sharpe(returns)
    
    if max_sharpe["success"]:
        print(f"Expected Return: {max_sharpe['expected_return']:.2%}")
        print(f"Volatility: {max_sharpe['volatility']:.2%}")
        print(f"Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")
        print("\nWeights:")
        for symbol, weight in zip(max_sharpe["symbols"], max_sharpe["weights"]):
            print(f"  {symbol}: {weight:.2%}")
    print()
    
    print("=== Monte Carlo Simulation ===")
    equal_weights = np.ones(returns.shape[1]) / returns.shape[1]
    portfolio_return = (returns * equal_weights).sum(axis=1)
    
    ann_return = portfolio_return.mean() * 252
    ann_vol = portfolio_return.std() * np.sqrt(252)
    
    price_paths, stats = run_monte_carlo_simulation(
        current_value=100000,
        expected_return=ann_return,
        volatility=ann_vol,
        time_horizon=252,
        n_simulations=1000,
    )
    
    print(f"Initial Value: $100,000")
    print(f"Mean Final Value: ${stats['mean']:,.0f}")
    print(f"Median Final Value: ${stats['median']:,.0f}")
    print(f"5th Percentile: ${stats['percentile_5']:,.0f}")
    print(f"95th Percentile: ${stats['percentile_95']:,.0f}")
    print()
    
    print("=== Analysis Complete ===")
    print("Run Streamlit app: streamlit run ui/streamlit_app.py")


if __name__ == "__main__":
    main()
