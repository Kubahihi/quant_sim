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

`expected return - risk aversion * variance - proportional trading cost`

Turnover is the L1 change in weights. Trading cost is turnover multiplied by
the configured basis-point rate. Asset-specific spreads, nonlinear market
impact, tax lots and integer share quantities are not yet modeled.

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
binding flags and pass/fail status. Holding-count constraints are reported as
unsupported until a mixed-integer solver is deliberately introduced.

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
5. deduct proportional trading costs on rebalance dates;
6. compare after-cost results with an equal-weight baseline.

The output includes target-weight history, gross and net returns, turnover,
cost drag, return, volatility, Sharpe ratio and maximum drawdown. This validates
the allocation process; it does not guarantee future performance. The
equal-weight portfolio is a neutral comparator and is not asserted to satisfy
the mandate. The evaluation currently uses the selected current universe, not
a point-in-time historical universe, and therefore does not remove
universe-selection or survivorship bias.

## Next extensions

The next implementation layer should add liquidity/market-impact estimates,
integer trade sizing, tax-aware lots and point-in-time universe data.
