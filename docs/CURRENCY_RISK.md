# Currency Risk & Hedging

The **Risk & Quant -> Currency Risk & Hedging** panel measures the part of a
portfolio's reporting-currency value that can change solely because exchange
rates move. It is designed for portfolios that contain assets, cash, short
positions, or liabilities in more than one currency.

## Input convention

Each row contains an asset or sleeve, its three-letter currency code, and its
current market value in that local currency. Liabilities and short positions
are entered as negative values. The reporting currency is selected separately.

All fetched exchange rates are normalized to:

> units of reporting currency per one unit of foreign currency

Under that convention, a positive FX return means that the foreign currency
strengthened against the reporting currency. The value of an unhedged long
foreign asset therefore rises in the reporting currency. Cross rates are built
through consistent USD values, so the same convention also applies when the
reporting currency is EUR, CZK, GBP, or another supported currency.

The supported currencies are USD, EUR, GBP, AUD, NZD, JPY, CHF, CAD, CZK, PLN,
SEK, NOK, DKK, CNY, HKD, and SGD. Historical daily rates are sourced through
the project's Yahoo Finance market-data adapter.

## Exposure measures

- **Net exposure** offsets assets and liabilities in the same currency.
- **Gross exposure** preserves the absolute size of every position before
  offsetting and is used for concentration shares and optimizer scaling.
- **Foreign share** is gross non-reporting-currency exposure divided by total
  gross exposure.

The FX analysis does not replace the portfolio's full market-risk model. It
isolates currency translation risk from movements in the local price of the
underlying asset.

## Risk estimates

Daily FX P/L is approximated as the vector of current net currency exposures
multiplied by daily exchange-rate returns. The panel reports:

- annualized FX volatility in the reporting currency;
- parametric one-day VaR using a normal quantile;
- historical one-day VaR at the selected confidence level;
- historical expected shortfall below the VaR threshold;
- covariance-based component contributions to annualized FX volatility.

Historical VaR is a threshold exceeded at the selected frequency, not a
maximum possible loss. Close-to-close data can miss intraday gaps, illiquidity,
and discontinuous market moves.

## Hedge optimizer

For each foreign currency, the optimizer chooses a hedge ratio between zero and
the user-defined maximum. It minimizes:

1. annualized residual FX variance as a fraction of gross exposure, multiplied
   by the selected risk-aversion parameter; plus
2. estimated annual hedge cost as a fraction of gross exposure.

The hedge cost input can represent indicative forward carry, bid/ask spread,
rollover, and operational cost. It is not a live executable quote. A zero-cost,
positive-risk-aversion solution will generally fully hedge every permitted
currency; a positive cost can create a partial hedge.

The generated direction is expressed as a forward-style action. A positive
foreign asset exposure produces **Sell foreign / Buy reporting currency**. A
negative foreign exposure produces the opposite direction. The optimizer never
hedges more than the configured limit and never reverses an exposure into a
speculative currency position.

Before implementation, verify mandate permissions, counterparty and collateral
limits, instrument tenor, forward points, minimum lot size, settlement dates,
liquidity, accounting treatment, and tax consequences.

## Stress test

The deterministic stress editor accepts one percentage shock per currency. A
positive value means the currency strengthens against the reporting currency;
a negative value means it weakens. The table compares P/L on the original net
exposure with P/L on the residual exposure after the optimized hedge.

The stress result excludes underlying asset returns, changing correlations,
option convexity, margin calls, transaction costs, and counterparty default.

## Operating checklist

1. Refresh current local market values after material portfolio trades.
2. Confirm liabilities and short positions use negative signs.
3. Review the history window and missing-data alignment.
4. Compare risk contributions, not only currency notional sizes.
5. Validate hedge cost and forward points with an executable market source.
6. Run both appreciation and depreciation stresses.
7. Re-run after cash flows, valuation changes, or material FX moves.
