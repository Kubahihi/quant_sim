# Portfolio optimization methodology

## Shared estimation contract

All portfolio optimizers use the same annualized input bundle from
`src.optimization.estimate_portfolio_inputs`:

- complete-case finite daily returns;
- sample expected returns shrunk toward their cross-sectional mean;
- sample covariance shrunk toward a diagonal equal-variance target;
- a positive-semidefinite numerical repair;
- recorded observation count, annualization and shrinkage parameters.

This makes minimum variance, maximum Sharpe, the efficient frontier, sampled
portfolios and cost-aware rebalancing directly comparable. The metadata is
included in every optimization result.

The dashboard constructs this bundle once per return matrix and passes the
same immutable object to minimum variance, maximum Sharpe, the efficient
frontier, portfolio sampling and cost-aware rebalancing. Reuse is accepted
only when asset names and their order match exactly; otherwise the run fails
rather than risking a silent weight/data misalignment.

## Streamlit execution model

The Streamlit launcher keeps SciPy and CVXPY lazy: importing
`src.optimization` does not load either numerical backend until the relevant
optimizer is requested. Wharton's full simulation paths, models, news retrieval
and causal backtest are also computed on demand when their module is opened.
The initial portfolio and optimizer results therefore do not wait for external
research providers or retain unused path matrices in each user session.

This is deferred execution, not reduced precision. Requested modules retain
their full data windows, solver tolerances, simulation counts and validation
rules. Production source-file watching is disabled because deployed containers
are immutable; a deployment restart is required to pick up code changes.

The sidebar's **Runtime & build** panel identifies the branch and commit without
spawning Git and records privacy-safe server-side timings for the latest 20
Streamlit reruns. The Wharton Quant Engine separately records market-data,
analytics, and optimization/validation phases. These measurements intentionally
exclude browser rendering and a sleeping Community Cloud container's wake-up
time, which must be measured externally.

## Constraints and failure behavior

Current core constraints are full investment, long-only or bounded shorting,
maximum position weight and, for rebalancing, maximum turnover. An infeasible
position cap raises a clear input error. Solver output is checked for finite
weights, full investment and bound residuals before it can be returned as a
recommendation. A failed or invalid solution contains no target weights.

The dashboards may raise a position cap to the minimum feasible value before
calling the optimizer. When this happens, the requested and effective limits
are retained and the UI displays a warning.

## Efficient frontier

The frontier begins at the global minimum-variance portfolio and ends at the
maximum feasible expected-return portfolio. It therefore excludes the
inefficient lower branch. Every point uses the same expected-return estimate,
covariance, risk-free rate and position cap as the highlighted portfolios.

The 3D portfolio cloud is projected onto the same capped long-only simplex.

## Cost-aware rebalance

The current single-period objective maximizes:

`expected return - risk aversion * variance - execution cost`

Turnover is the L1 change in weights. Execution cost can contain:

- asset-specific commissions/fees;
- estimated half-spread;
- convex square-root market impact, proportional to
  `trade weight * sqrt(trade notional / average daily dollar volume)`.

When portfolio value and 30-day average daily dollar volume (ADV) are supplied,
the optimizer can also cap every trade's share of ADV. The Wharton engine derives
rolling ADV from the same OHLCV download as prices, so historical validation uses
only volume information available at each rebalance date.

## Executable trade plan

The continuous optimum and the executable plan are deliberately kept separate.
`build_execution_plan` converts target weights into a deterministic list of
trades while enforcing:

- whole or configured lot sizes;
- available cash after estimated execution costs;
- minimum trade notional;
- maximum ADV participation;
- minimum/maximum executed holding counts;
- optional tax-lot selection.

Tax lots are sold in ascending estimated tax per share. This harvests the most
valuable losses first and then chooses the lowest estimated-tax gains using the
configured short- and long-term rates. The output retains each selected lot,
realized gain, estimated tax, residual cash, execution cost and L1 difference
from the continuous target. It is an estimate, not tax advice.

Tax-lot CSV input is long-form with `Ticker`, `Shares`,
`Cost Basis Per Share`, and `Acquired At` columns.

## Mandate-aware convex engine

`src.optimization.optimize_portfolio` supports minimum variance, maximum
risk-adjusted utility, target volatility, minimum historical CVaR and minimum
tracking error. The engine can enforce the normalized Strategy Rulebook's:

- maximum asset and sector weights;
- sector minimum/maximum bands;
- cash floor and ceiling;
- prohibited tickers and sectors;
- allowed asset types and required tags;
- approved-universe requirement;
- portfolio beta range;
- turnover limit and asset-specific proportional costs.

Every accepted result contains a constraint report with actual values, limits,
binding flags and pass/fail status. Exact holding counts are not forced into the
continuous convex target because the installed solvers do not provide a robust
mixed-integer quadratic path. They are enforced and audited in the executable
lot-level plan, while both target versions remain visible.

Black-Litterman expected returns are available with explicit absolute views and
per-view confidence. The Wharton UI labels current portfolio weights as a
neutral reference when it uses them; they are not presented as market-cap
weights.

## Rolling out-of-sample validation

`run_optimization_walk_forward` performs a causal rolling evaluation for the
selected construction objective. The convex objectives retain Strategy
Rulebook constraints in every estimation window; Black-Litterman inputs are
also recomputed from that window rather than reused from the full sample:

1. estimate inputs using only the preceding training observations;
2. calculate a new target allocation;
3. apply it only to the following test window;
4. allow weights to drift with realized asset returns;
5. deduct commission, spread and square-root impact costs on rebalance dates
   when causal liquidity inputs are available;
6. compare after-cost results with an equal-weight baseline.

The output includes target-weight history, active symbols by window, gross and
net returns, turnover, cost breakdown, return, volatility, Sharpe ratio and
maximum drawdown. This validates the allocation process; it does not guarantee
future performance. The equal-weight portfolio is a neutral comparator and is
not asserted to satisfy the mandate.

An optional point-in-time membership table removes the specific current-universe
shortcut. Membership is lagged by one observation, forward-filled, and never
back-filled. The return matrix must contain the union of historical constituents.
Missing returns for a held asset fail closed: the caller must supply a delisting
return or shorten the holding window instead of silently treating the loss as
zero. Without a membership table, the result is explicitly marked as still
exposed to survivorship bias.

Membership CSV input supports either:

- long form: `Date`, `Ticker`, `Is Member`; or
- wide form: `Date` plus one boolean column per ticker.

## Remaining limitations

- Yahoo Finance volume and prices are estimates for research, not execution feeds.
- Spread and impact coefficients are assumptions and should be calibrated to a
  broker or venue before trading.
- Manual bonds need instrument-specific denomination, accrued-interest and
  liquidity inputs before they can receive a generic equity-style trade plan.
- Point-in-time membership controls survivorship bias only when the supplied
  historical constituent union and delisting returns are complete.
- Tax estimates omit jurisdiction-specific wash-sale, currency, account and
  investor rules.
