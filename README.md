# QuantSim

## Portfolio analytics, investment research and disciplined decision-making

QuantSim connects security research, portfolio construction and risk analysis in a single investment workspace. It measures how a portfolio has behaved, evaluates alternative allocations and models whether an investment strategy can meet a client's financial goals.

The platform combines quantitative analysis with a documented investment process: define the mandate, evaluate the evidence, approve decisions, reconcile holdings and explain the results. Its competition workspace supports the team's Wharton Global High School Investment Competition workflow.

[Portfolio analytics](#portfolio-analytics) · [Portfolio construction](#portfolio-construction) · [Scenarios and client goals](#scenarios-and-client-goals) · [Investment process](#investment-process) · [Methodology](#methodology-and-validation)

## Portfolio analytics

### Return, risk and diversification

The core analysis aligns asset returns with portfolio weights and builds a daily portfolio return series:

$$
r_{p,t}=\sum_{i=1}^{n}w_i r_{i,t}
$$

For a fixed-weight analysis, the target allocation is applied to each day's returns, equivalent to a daily-rebalanced constant-weight portfolio. This differs from buy-and-hold; rolling rebalancing analysis separately allows holdings to drift between scheduled trades.

| Measure | Calculation and interpretation |
|---|---|
| Total and annualized return | Compounds daily returns into cumulative wealth and an effective annual growth rate. |
| Volatility | Scales the sample standard deviation of daily returns by the square root of 252 trading days. |
| Sharpe ratio | Annualizes average daily excess return divided by its standard deviation. The annual risk-free rate is converted to a daily equivalent first. |
| Sortino ratio | Uses downside deviation relative to the risk-free target, retaining the frequency of shortfalls. |
| Maximum drawdown | Measures the largest decline from initial wealth or a subsequent portfolio peak. |
| Historical VaR and CVaR | Estimate a loss threshold at the selected confidence level and the average loss in the corresponding tail. |
| Concentration | Reports the largest position, the sum of squared weights (HHI) and effective holdings, calculated as 1 / HHI. |
| Correlation | Measures how asset returns move together, including average pairwise correlation. |

Historical growth and estimated expected return serve different purposes. Performance reporting uses compounded returns; optimization uses annualized arithmetic return estimates.

### Benchmark and attribution

Benchmark analysis evaluates active return, tracking error, information ratio, beta and alpha. Tracking error measures the variability of daily portfolio-minus-benchmark returns, helping distinguish excess performance from the amount of active risk taken.

Single-period **Brinson–Fachler attribution** explains active performance through three effects:

- **Allocation:** overweights and underweights in sectors that outperformed or underperformed the overall benchmark.
- **Selection:** differences between portfolio and benchmark returns within each sector.
- **Interaction:** the combined effect of active sector weights and within-sector performance differences.

Attribution is calculated from portfolio and benchmark weights and returns. Competition reporting requires reconciled inputs so the effects can be checked against the reported performance.

### Portfolio score

The portfolio diagnostic starts at **100** and applies explicit penalties for weak return, excessive position size, insufficient effective diversification, high volatility, low Sharpe ratio, deep drawdowns and high correlation. Thresholds change with the selected conservative, balanced or aggressive risk profile.

When supplied, linear-trend, ARIMA and GARCH signals can add heuristic adjustments. Each adjustment appears in the score breakdown. The final score is bounded between 0 and 100 and assigned a descriptive rating.

This is a rule-based diagnostic: its value describes the selected inputs against internal thresholds, not the probability of future investment success.

## Portfolio construction

### Comparable estimates

Alternative allocations share the same return history and estimation assumptions. Expected returns are shrunk toward their cross-sectional average; covariance estimates are shrunk toward an equal-variance diagonal target. This reduces reliance on unstable sample estimates. Constant cash series retain their own return and zero covariance.

**Black–Litterman** adds explicit return views with a confidence level for each view. A reference allocation supplies the prior; when current holdings are used, they are identified as the reference rather than assumed to represent market capitalization.

### Objectives and constraints

| Objective | Portfolio decision |
|---|---|
| Minimum variance | Find the lowest estimated volatility within the permitted allocation. |
| Maximum Sharpe | Seek the highest estimated excess return per unit of volatility. |
| Risk-adjusted utility | Balance expected return against risk aversion and applicable trading costs. |
| Target volatility | Construct an allocation subject to a selected volatility ceiling. |
| Minimum historical CVaR | Reduce average loss in the historical tail. |
| Minimum tracking error | Minimize estimated deviation from a specified benchmark allocation. |

The efficient frontier displays the efficient range from minimum variance to maximum feasible expected return.

Depending on the selected engine, constraints cover position and sector limits, cash bounds, eligible securities, permitted asset types, beta and turnover. Accepted allocations include checks against the applicable limits. Infeasible or failed solutions are identified explicitly.

### From target weights to trades

Cost-aware rebalancing evaluates expected return, variance and execution cost together. Cost assumptions can include commissions, half-spread and square-root market impact linked to trade size relative to average daily dollar volume.

The execution plan then converts continuous weights into trades, applying lot sizes, available cash, minimum order values, liquidity participation limits and holding-count requirements. Optional tax-lot selection ranks sales by estimated tax per share. Actual post-cost weights are checked again against the mandate.

The result distinguishes the mathematical target from the feasible trade plan, including residual cash, costs and any remaining allocation difference.

## Scenarios and client goals

### Monte Carlo and stress analysis

**Geometric Brownian Motion (GBM)** simulates wealth under constant return and volatility assumptions. **Merton jump diffusion** adds discrete jumps as a stress overlay. Both interpret their return input as an effective annual simple return.

Outputs include wealth distributions, percentile paths, loss probability, terminal VaR and expected shortfall. Simulated terminal means are compared with analytic expectations to assess sampling error.

Deterministic stress analysis answers a separate question: how would the portfolio or position respond to specified price, yield, spread or currency shocks? These are conditional sensitivities, with results dependent on the entered shock and model assumptions.

### Client Goal Outlook

Goal analysis combines starting capital, target wealth, horizon, contributions or withdrawals, and a nominal or inflation-adjusted wealth basis.

It compounds sufficiently complete calendar years from aligned historical returns, then resamples annual cross-asset observations. Every candidate portfolio receives the **same sampled scenarios**, making comparisons less sensitive to random simulation noise. The outlook assumes constant weights with annual rebalancing.

Reported outcomes include:

- the modeled frequency of reaching the target;
- mean, median and 10th-percentile terminal wealth;
- the probability of falling short and average deficit among failed scenarios;
- ruin probability when complete wealth paths are evaluated.

The probability interval measures finite-simulation uncertainty. It does not capture all uncertainty about future markets. Candidates optimized on the same history remain exploratory, and this outlook does not deduct taxes or transaction costs.

## Analysis across asset classes

### Fixed income

Individual-bond analysis starts with contractual cash flows and entered valuation data. It calculates clean and dirty value, current yield, yield to maturity, yield to first call and yield to worst, alongside duration, convexity, DV01 and future coupon and principal payments.

Yield to maturity is solved from price and cash flows when no override is supplied. Yield to worst is the lower available yield to maturity or first call. Rate and spread scenarios combine repricing with carry and a simple expected credit loss estimate based on default probability and recovery.

For portfolio-wide analysis, a manually entered bond can use an explicitly selected bond ETF as a risk proxy. The proxy retains its historical correlation pattern, is rescaled to assumed volatility and has its mean adjusted to yield to worst less expected credit loss. This approximation is disclosed separately from the bond's contractual valuation.

### Currency exposure and hedging

Currency analysis converts assets and liabilities into the reporting currency and distinguishes net exposure from gross exposure. It estimates FX volatility, historical and parametric VaR, expected shortfall and currency contributions to volatility.

Hedge ratios balance residual currency variance against estimated hedge cost, within the configured limits. Stress results compare unhedged and hedged exposure. This isolates currency translation risk from changes in the underlying assets' local prices.

### Commodities

Commodity analysis compares exchange-traded vehicles and continuous-futures research proxies using return, volatility, drawdown and correlation. Position stresses apply explicit price shocks to position value, including contract multipliers and currency conversion where relevant.

Proxy performance can differ from spot commodities or an executable futures position because of rolling contracts, collateral, fees and tracking.

## Investment process

| Stage | Decision logic |
|---|---|
| **Client & Policy** | Establish capital, financial goals, risk tolerance and portfolio constraints. |
| **Research** | Screen opportunities and assemble a Security Dossier with evidence, valuation, catalysts, risks and exit criteria. |
| **Decisions** | Record independent initial views, discussion, final votes, authorization and position sizing through the Investment Committee. |
| **Portfolio** | Reconcile WInS holdings and evaluate allocation, performance, risks and client-goal outcomes. |
| **Deliverables** | Build reports and pitch material from reconciled portfolio data and preserved research evidence. |

Research evidence and investment approval have distinct roles. The Security Dossier holds the thesis; the Investment Committee records the decision. Verified source records support reports, and signed WInS reconciliation establishes the portfolio snapshot used in competition reporting.

The client behavioral profile adds a transparent questionnaire and responses to hypothetical drawdowns. It flags inconsistencies with declared risk tolerance and proposes process controls such as cooling-off periods, independent challenge and pre-agreed responses to losses.

Optional AI commentary summarizes calculated metrics and flags. Core calculations and rule-based explanations remain available independently of that commentary.

## Methodology and validation

QuantSim evaluates the strength of its evidence alongside its analytical outputs:

- **Parameter uncertainty:** moving-block bootstrap intervals for return, volatility, Sharpe ratio, VaR and CVaR preserve short-run dependence.
- **Simulation quality:** analytic cross-checks, sampling-error estimates and recorded seeds assess numerical convergence and reproducibility.
- **Rolling validation:** portfolio inputs are re-estimated on preceding observations, allocations are evaluated on subsequent periods and after-cost results are compared with an equal-weight baseline.
- **Historical universe controls:** optional lagged membership data and explicit delisting returns reduce survivorship bias when the supplied history is complete.

The separate methodology score summarizes evidence quality. Neither it nor the portfolio score is an official Wharton rating or a forecast-accuracy percentage. Full-model refitting, an untouched holdout and controls for repeated strategy selection remain necessary before claiming predictive validity.

Detailed assumptions and interpretation rules:

[Portfolio construction](docs/PORTFOLIO_OPTIMIZATION.md) · [Model validation](docs/MODEL_VALIDATION.md) · [Goal funding](docs/GOAL_FUNDING.md) · [Fixed income](docs/FIXED_INCOME.md) · [Currency risk](docs/CURRENCY_RISK.md) · [Commodities](docs/COMMODITIES.md) · [Behavioral profile](docs/BEHAVIORAL_PROFILE.md)

## Quick start

Requires Python 3.12.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:QUANT_SIM_ENV="development"
streamlit run ui/streamlit_app.py
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export QUANT_SIM_ENV=development
streamlit run ui/streamlit_app.py
```

AI commentary is optional. To enable it, set `GROQ_API_KEY` either in the environment or in Streamlit secrets. For deployment and production storage configuration, see [production readiness](docs/PRODUCTION_READINESS.md) and [storage setup](docs/STORAGE_PRODUCTION_SETUP.md).
