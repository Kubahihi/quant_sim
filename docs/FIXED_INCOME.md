# Fixed-income support

The competition portfolio supports both bond ETFs and individual bonds.

The standalone **Research → Bond Analysis** panel can analyze a proposed instrument without saving it to the portfolio. It includes exact price/yield sensitivity for individual bonds and duration-convexity sensitivity for bond ETFs. Saved holdings are available in a separate tab of the same panel.

## Valuation conventions

- Bond ETFs use the same convention as equities: shares multiplied by market price.
- Individual-bond clean prices and accrued interest are quote points per 100 of face value.
- Individual-bond USD value is `quantity × face value × dirty quote / 100 × FX to USD`.
- Dirty quote is clean price plus accrued interest.
- Recorded coupon income is cumulative cash received in USD and is included in total return and portfolio cash.
- Entry, current, and exit FX rates are stored separately so currency P/L is retained.

## Required individual-bond data

Every individual bond needs an ISIN, face value, maturity date, currency, USD FX rate, and coupon frequency. Coupon rate may be zero. Store the source/reference and observation date with every manual valuation.

## Analytics

The fixed-income dashboard provides:

- clean and dirty valuation;
- current yield, yield to maturity, yield to first call, and yield to worst;
- spread to a user-supplied maturity-matched benchmark;
- Macaulay and modified duration, convexity, and DV01;
- annual carry, carry breakeven yield move, and probability-of-default/recovery expected loss;
- future coupon and principal cash flows;
- maturity, currency, issuer, and credit-rating concentrations;
- parallel yield-curve and credit-spread sensitivity using duration and convexity.
- a single-instrument price/yield curve with downloadable sensitivity results.
- a two-dimensional curve/spread scenario grid combining repricing, carry, and expected credit loss;
- an input-quality score with stale-price, missing-source, call-term, benchmark, and credit-data checks.

YTM is solved from entered price and contractual cash flows when no override is supplied. For a callable bond, YTC uses the first entered call date and price; YTW is the lower available YTM or YTC. Sensitivity is repriced to the YTW redemption path. Bond ETFs need provider or manual YTM/duration overrides because their holdings and cash flows change over time.

Benchmark spread is only meaningful when the entered benchmark has a comparable currency and maturity/duration. Expected credit loss is a simple one-period estimate (`PD × (1 − recovery)`) and is not a structural credit model. The scenario grid keeps carry, PD, and recovery constant over the chosen horizon.

## Market data policy

Ticker-priced bond ETFs use the existing market-price history source. Individual bonds are deliberately excluded from ticker downloads: use the current WInS statement or another identified bond-price source and record its reference. This avoids silently treating an ISIN as an equity ticker.

Stress outputs are deterministic sensitivities, not forecasts. Historical risk for an individual bond remains unmodeled unless a compatible return series is explicitly supplied; duration/DV01 analytics remain available independently.

## Wharton competition workflow

The **Competition Case** tab converts the analysis into an auditable decision worksheet:

- client goal, portfolio role, thesis, why-now rationale, counter-thesis, and sell discipline;
- proposed weight versus the team's own position limit;
- a hard eligibility gate that requires a current official rule/WInS-list reference and verification date;
- evidence, risk, and execution completeness checks that do not use past performance;
- bond-specific pitch-defense questions and a downloadable Markdown working memo;
- a relative-value shortlist that keeps yield, spread, duration, expected loss, and evidence readiness separate instead of hiding them in a buy/sell score.

Competition eligibility is intentionally not inferred from instrument type. When current trading rules are unpublished or the instrument is not confirmed in WInS, keep the status at **Pending verification**. A verified-ineligible instrument and an oversized proposed position are both explicit do-not-trade blockers.

The generated memo is a working evidence draft. Every number and source must be verified, and the student team remains responsible for writing and citing its final competition submission in its own voice.
