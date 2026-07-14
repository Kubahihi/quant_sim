# QuantSim model validation standard

## Bottom line

QuantSim is a strong educational decision-support platform, but no honest
single percentage can describe its "accuracy". Portfolio return, risk,
simulation and directional forecasts are different estimands and must be
validated separately. The Wharton Cockpit therefore reports a transparent
methodology score and uncertainty evidence, not an invented hit-rate claim.

The score is internal. It is not an official Wharton standard, affiliation or
endorsement.

## What the score measures

The 100-point methodology score is the sum of six visible gates:

| Gate | Maximum | Evidence |
|---|---:|---|
| Historical depth | 20 | Number of aligned daily observations |
| Parameter uncertainty | 20 | Moving-block bootstrap intervals |
| Simulation convergence | 15 | Relative standard error of terminal mean |
| Distribution/model risk | 15 | GBM assumptions versus sample diagnostics |
| Out-of-sample process | 20 | Causal walk-forward evidence and its scope |
| Reproducibility | 10 | Recorded deterministic simulation seed |

Bands are interpreted as follows:

- 85–100: research-ready workflow, still requiring full-ensemble validation;
- 70–84: strong decision support, not a validated forecasting system;
- 55–69: exploratory decision support with material gaps;
- below 55: prototype evidence only.

## Statistical methods

### Parameter uncertainty

QuantSim uses a seeded moving-block bootstrap. Sampling contiguous blocks
retains short-run dependence that an ordinary i.i.d. bootstrap would erase.
The 2.5th and 97.5th percentiles form intervals for annualized return,
volatility, Sharpe ratio, historical VaR and historical CVaR.

These are sampling intervals conditional on the observed regime. They do not
cover structural breaks, data-source errors or future distribution shifts.

### Monte Carlo

The GBM engine treats its expected-return input as annualized arithmetic drift,
uses a local random generator, and reports:

- analytic and simulated terminal means;
- standard error and 95% interval for the simulated mean;
- relative Monte Carlo error;
- loss probability, terminal VaR and expected shortfall;
- the model assumptions and seed.

The engine models Monte Carlo sampling error. It does not remove parameter or
model risk. A single GBM model cannot represent jumps, volatility clustering,
liquidity shocks or changing correlations.

### Backtest integrity

The old dashboard replayed one full-sample model score across the same history.
Lagging that constant exposure by one day did not make the score genuinely
out-of-sample. The production pipeline now uses a causal walk-forward baseline:
each position uses data available strictly before that return, turnover is
measured, and transaction costs are deducted.

This baseline validates the process plumbing and provides an honest comparator.
It does **not** validate the full model ensemble. That limitation is displayed
in the UI and caps the out-of-sample gate at 15 of 20 points.

## What is still needed before claiming predictive accuracy

1. Freeze model definitions, features, hyperparameters and rebalance rules.
2. Refit the complete ensemble inside each rolling or expanding training fold.
3. Calibrate all confidence scores using only earlier data.
4. Reserve a final untouched holdout period.
5. Compare after-cost results with investable benchmarks and simple baselines.
6. Report confidence intervals, turnover, drawdown and performance by regime.
7. Correct for multiple testing before selecting among many models.
8. Record data snapshots and every configuration needed to reproduce a run.

Until those steps are complete, QuantSim outputs should support a documented
investment thesis and risk discussion, not be presented as guaranteed forecasts
or investment advice.
