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
| Out-of-sample process | 20 | Causal rolling evidence, costs, comparator, and universe scope |
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
out-of-sample. The production pipeline now keeps two existing evidence layers
in one report rather than creating a second validation path:

- the causal trend/risk baseline verifies signal timing and transaction-cost
  plumbing;
- the rolling portfolio optimizer re-estimates inputs and target weights inside
  each training window, charges turnover costs, and reports an equal-weight
  after-cost comparator. When lagged point-in-time membership is supplied, it
  also controls the principal survivorship-bias path.

Point-in-time rolling re-optimization can earn 18 of 20 points. It still does
**not** validate the full current-state model and signal bundle, reserve an
untouched final holdout, or correct selection across many tried specifications.
Those limitations remain explicit rather than being hidden inside the score.

### Accuracy evidence checklist

The report displays one non-scoring checklist for causal OOS evidence, costs,
simple comparator, point-in-time membership, frozen input snapshot, full-bundle
refitting, untouched holdout, and multiple-testing control. This checklist does
not create another readiness score. It separates numerical reproducibility from
evidence about future performance and identifies the next missing proof.

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
