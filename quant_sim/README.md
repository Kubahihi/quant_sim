# Quant Investment Platform

Professional-grade investment analytics platform for portfolio analysis, optimization, and risk management.

## Features

- **Multi-asset data ingestion** (Yahoo Finance)
- **Portfolio analytics** (returns, volatility, Sharpe, drawdown)
- **Portfolio optimization** (minimum variance, maximum Sharpe)
- **Efficient frontier** calculation
- **Monte Carlo simulation** (GBM)
- **Correlation analysis**
- **Interactive visualizations** (2D charts, heatmaps)
- **Streamlit UI** for easy interaction

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line

```bash
python main.py
```

### Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

## Project Structure

```
quant_platform/
├── config/              # Configuration files
├── data/                # Data storage (cache, exports)
├── src/
│   ├── data/            # Data fetching and caching
│   ├── analytics/       # Returns, risk metrics, correlation
│   ├── portfolio/       # Portfolio class and analytics
│   ├── optimization/    # Portfolio optimization
│   ├── simulation/      # Monte Carlo simulation
│   └── visualization/   # Charts and plots
├── ui/                  # Streamlit interface
├── main.py              # Example usage
└── requirements.txt
```

## Usage Examples

### Fetch Data

```python
from src.data.fetchers.yahoo_fetcher import YahooFetcher
from datetime import datetime, timedelta

fetcher = YahooFetcher()
symbols = ["AAPL", "MSFT", "GOOGL"]
prices = fetcher.fetch_close_prices(symbols, start_date, end_date)
```

### Calculate Metrics

```python
from src.analytics.risk_metrics import calculate_sharpe_ratio, calculate_max_drawdown

returns = prices.pct_change().dropna()
sharpe = calculate_sharpe_ratio(returns["AAPL"])
max_dd = calculate_max_drawdown(returns["AAPL"])
```

### Optimize Portfolio

```python
from src.optimization import optimize_minimum_variance, optimize_maximum_sharpe

min_var_result = optimize_minimum_variance(returns)
max_sharpe_result = optimize_maximum_sharpe(returns)
```

### Run Simulation

```python
from src.simulation import run_monte_carlo_simulation

price_paths, stats = run_monte_carlo_simulation(
    current_value=100000,
    expected_return=0.10,
    volatility=0.20,
    time_horizon=252,
    n_simulations=1000,
)
```

## Configuration

Edit `config/settings.yaml` to customize:
- Cache settings
- Risk-free rate
- Optimization parameters
- Simulation defaults

## MVP Roadmap

**v0.1 (Current)**: Basic portfolio analytics and optimization
**v0.2**: Advanced optimizations, backtesting, more data sources
**v0.3**: 3D visualizations, ML layer, regime detection
**v1.0**: Production release with API, live trading integration

## License

MIT License

## Contributing

Contributions welcome! Please open issues for bugs or feature requests.
