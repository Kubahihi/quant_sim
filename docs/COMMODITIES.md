# Commodity analysis

The **Research → Commodity Analysis** panel provides a starter universe of exchange-traded commodity vehicles and Yahoo Finance continuous-futures proxies. It supports custom Yahoo tickers as well.

## Market monitor

The monitor downloads adjusted daily price history through the existing Yahoo data source and reports:

- 1-month, 3-month, and 12-month total price returns;
- annualized geometric return and daily-return volatility;
- Sharpe ratio using the entered annual risk-free rate;
- maximum drawdown;
- rebased price performance; and
- pairwise daily-return correlation.

Metrics are calculated independently for each available series. A missing ticker does not remove the histories that were fetched successfully. The snapshot can be exported to CSV.

The same commodity ETF and futures-proxy tickers can be entered as **Market Tickers** in Quant Engine. They then participate in portfolio metrics, optimization, Monte Carlo, and scenario analysis. The scenario engine classifies catalogued commodity instruments as commodity exposure; precious-metal proxies retain their separate gold role where applicable.

## Position stress

The position stress table applies explicit percentage price shocks to:

`current price × units or contracts × contract multiplier × FX to USD`

For ETF shares, the multiplier is normally 1. For futures, the multiplier, quoted currency, expiry, and exchange specification must be verified before relying on the result. The output is direct mark-to-market sensitivity only. It excludes margin requirements, transaction costs, liquidity, carry, collateral income, and tax.

## Important limitations

- A continuous-futures ticker is a research proxy, not an executable contract. Its data provider can roll expiries using a methodology that differs from the team's intended implementation.
- Commodity ETF returns can diverge materially from spot price changes because of futures-curve roll yield, collateral, fees, and tracking.
- Historical correlations are regime dependent and can change during inflation, growth, supply, geopolitical, and liquidity shocks.
- The panel does not provide a live futures curve, delivery calendar, exchange margin calculation, or verified contract specification.
- Outputs are research diagnostics, not forecasts or investment advice.
