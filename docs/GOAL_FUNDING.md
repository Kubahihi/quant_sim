# Client Goal Funding

The Client Goal Outlook answers a planning question that return/risk metrics do
not answer directly:

> Given this client goal, cash-flow plan and portfolio, how often is the target
> reached across the modeled scenarios?

## Required mandate inputs

Each evaluated goal needs:

- target wealth;
- horizon in whole years;
- capital allocation from current investable capital;
- annual net cash flow (positive contribution, negative withdrawal);
- nominal or real wealth basis; and
- inflation when nominal returns must be converted to real terms.

Incomplete goal buckets remain valid for allocation analysis but are not shown
in the funding outlook.

## Scenario method

The UI compounds only near-complete calendar years (at least 227 aligned daily
observations, roughly 90% of a 252-session year), samples
historical rows with replacement, and preserves each row's cross-asset
relationship. Every candidate portfolio receives the same sampled row indices.
This paired design prevents independent Monte Carlo noise from looking like a
portfolio advantage.

The default UI run uses 10,000 scenarios and seed `2027`. The engine itself
accepts a caller-supplied seed and scenario count, and exposes the sampled row
indices for audit.

Current strategy weights and portfolios optimized on that same return history
have different evidence scope. Same-history optimized candidates are labelled
in-sample exploratory; their displayed success frequencies are not an
out-of-sample selection result. The UI assumes constant weights with annual
rebalancing and does not deduct taxes or transaction costs.

## Reported metrics

- probability of reaching the goal and a Wilson interval for the simulated
  scenario proportion;
- expected and median terminal wealth;
- 10th-percentile terminal wealth;
- shortfall probability;
- expected deficit conditional on failure; and
- ruin probability when complete wealth paths are evaluated.

Expected shortfall versus the goal is the average target deficit among failed
scenarios. It is zero when no modeled scenario fails.

## Interpretation limits

Historical bootstrap is not a forecast and does not model structural breaks,
taxes, changing allocations, dynamic spending, fees not already reflected in
returns, or estimation uncertainty outside the sampled history. The confidence
interval is labelled an MC sampling interval and measures only finite-scenario
proportion uncertainty; it is not a confidence interval for the true future
success probability.

Use the output to compare decisions under a common model, document assumptions
and identify fragile goals. Do not present the displayed probability as a
guarantee.
