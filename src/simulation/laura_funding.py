"""Laura Gao's dated, nominal-USD liability plan.

Scenario rows stay paired across accumulation, reserve valuation and drawdown.
The 2033 reserve includes the immediate first payment. It is a transfer within
the portfolio, never a second withdrawal. No borrowing or external rescue is
permitted. These are model results, not investment recommendations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.simulation.goal_funding import _wilson_interval


ACCUMULATION_YEARS = tuple(range(2027, 2033))
PAYMENT_YEARS = tuple(range(2033, 2043))
DEPOSITS = {2027: 300_000.0, 2028: 150_000.0}
ANNUAL_PAYMENT = 50_000.0
TOLERANCE = 0.01


def _number(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


@dataclass(frozen=True)
class LauraPolicy:
    """Team assumptions; none of these numerical choices is a case requirement.

    The facility receives a share of the surplus AFTER a minimum flexibility
    amount. The remainder of that surplus also stays available for flexibility.
    Annual fees are deducted multiplicatively after each year's asset return.
    Use zero fees when the supplied returns are already net of all modeled costs.
    """

    reserve_discount_rate: float = 0.03
    reserve_buffer: float = 0.0
    facility_surplus_share: float = 0.8
    minimum_flexibility: float = 0.0
    accumulation_fee: float = 0.0
    reserve_fee: float = 0.0
    confidence_target: float = 0.95

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            object.__setattr__(self, name, _number(name, value))
        if self.reserve_discount_rate <= -1:
            raise ValueError("Reserve discount rate must be greater than -100%.")
        if self.reserve_buffer < 0 or self.minimum_flexibility < 0:
            raise ValueError("Reserve buffer and minimum flexibility cannot be negative.")
        if not 0 <= self.facility_surplus_share <= 1:
            raise ValueError("Facility surplus share must be between 0 and 100%.")
        if not 0 <= self.accumulation_fee < 1 or not 0 <= self.reserve_fee < 1:
            raise ValueError("Annual fees must be between 0% (inclusive) and 100% (exclusive).")
        if not 0 < self.confidence_target < 1:
            raise ValueError("The team confidence target must be between 0 and 100%.")


def _matrix(name: str, values: ArrayLike, periods: int) -> NDArray[np.float64]:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 2 or not array.shape[0] or array.shape[1] != periods:
        raise ValueError(f"{name} must have scenario rows and exactly {periods} annual columns.")
    if not np.isfinite(array).all() or (array < -1).any():
        raise ValueError(f"{name} must contain finite returns of at least -100%.")
    return array


def reserve_value(discount_rates: ArrayLike, buffer: float = 0.0) -> NDArray[np.float64]:
    """Annuity due at the beginning of 2033; first payment has exponent zero."""
    rates = np.atleast_1d(np.asarray(discount_rates, dtype=float))
    if rates.ndim != 1 or not np.isfinite(rates).all() or (rates <= -1).any():
        raise ValueError("Discount rates must be a finite one-dimensional vector above -100%.")
    buffer = _number("reserve_buffer", buffer)
    if buffer < 0:
        raise ValueError("Reserve buffer cannot be negative.")
    with np.errstate(over="ignore", divide="ignore"):
        values = ANNUAL_PAYMENT * np.sum((1 + rates[:, None]) ** -np.arange(10), axis=1)
        values *= 1 + buffer
    if not np.isfinite(values).all():
        raise ValueError("Discount assumptions produce an unrepresentable reserve value.")
    return values


@dataclass(frozen=True)
class LauraProjection:
    policy: LauraPolicy
    valuation_year: int
    accumulation_years: tuple[int, ...]
    accumulation_opening: NDArray[np.float64]
    accumulation_closing: NDArray[np.float64]
    portfolio_2033: NDArray[np.float64]
    required_reserve: NDArray[np.float64]
    allocated_reserve: NDArray[np.float64]
    reserve_gap: NDArray[np.float64]
    flexibility_gap: NDArray[np.float64]
    facility: NDArray[np.float64]
    flexibility: NDArray[np.float64]
    reserve_before_payment: NDArray[np.float64]
    reserve_after_payment: NDArray[np.float64]
    paid: NDArray[np.float64]
    unpaid: NDArray[np.float64]
    first_shortfall_year: NDArray[np.int64]
    all_payments_met: NDArray[np.bool_]
    plan_met: NDArray[np.bool_]

    def summary(self, *, probabilistic: bool = True) -> dict[str, Any]:
        """Joint success is evaluated on complete rows, never by averaging years."""
        n = len(self.facility)
        funded = self.reserve_gap <= TOLERANCE
        deficits = self.unpaid.sum(axis=1)
        failed = ~self.all_payments_met
        result: dict[str, Any] = {
            "scenario_count": n,
            "median_portfolio_2033": float(np.median(self.portfolio_2033)),
            "median_required_reserve": float(np.median(self.required_reserve)),
            "median_facility": float(np.median(self.facility)),
            "median_flexibility": float(np.median(self.flexibility)),
            "mean_unpaid": float(np.mean(deficits)),
            "conditional_mean_unpaid": float(np.mean(deficits[failed])) if failed.any() else 0.0,
            "maximum_unpaid": float(np.max(deficits)),
            "first_shortfall_counts": {str(year): int(np.sum(self.first_shortfall_year == year)) for year in PAYMENT_YEARS},
            "confidence_target": self.policy.confidence_target,
        }
        if probabilistic:
            if n < 2:
                raise ValueError("A deterministic path cannot establish a probability.")
            joint = float(np.mean(self.all_payments_met))
            result.update({
                "reserve_funding_probability": float(np.mean(funded)),
                "reserve_shortfall_probability": float(np.mean(~funded)),
                "all_payments_probability": joint,
                "plan_success_probability": float(np.mean(self.plan_met)),
                "payments_success_given_funded_reserve": float(np.mean(self.all_payments_met[funded])) if funded.any() else None,
                "mc_sampling_interval_95": list(_wilson_interval(int(self.all_payments_met.sum()), n, 0.95)),
                "meets_team_confidence_target": float(np.mean(self.plan_met)) >= self.policy.confidence_target,
            })
        else:
            result.update({"all_payments_met": bool(self.all_payments_met.all()), "plan_met": bool(self.plan_met.all())})
        return result

    def scenario_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "scenario": i + 1,
                "portfolio_2033": float(self.portfolio_2033[i]),
                "required_reserve": float(self.required_reserve[i]),
                "allocated_reserve": float(self.allocated_reserve[i]),
                "reserve_gap": float(self.reserve_gap[i]),
                "flexibility_gap": float(self.flexibility_gap[i]),
                "facility": float(self.facility[i]),
                "flexibility": float(self.flexibility[i]),
                "total_unpaid": float(self.unpaid[i].sum()),
                "first_shortfall_year": int(self.first_shortfall_year[i]) or None,
                "all_payments_met": bool(self.all_payments_met[i]),
                "plan_met": bool(self.plan_met[i]),
            }
            for i in range(len(self.facility))
        ]

    def payment_rows(self, scenario: int = 0) -> list[dict[str, Any]]:
        return [
            {"year": year, "due": ANNUAL_PAYMENT,
             "before_payment": float(self.reserve_before_payment[scenario, j]),
             "paid": float(self.paid[scenario, j]), "unpaid": float(self.unpaid[scenario, j]),
             "after_payment": float(self.reserve_after_payment[scenario, j])}
            for j, year in enumerate(PAYMENT_YEARS)
        ]


def project_laura_plan(
    accumulation_returns: ArrayLike,
    reserve_returns: ArrayLike,
    policy: LauraPolicy | None = None,
    *,
    valuation_year: int = 2027,
    opening_wealth_2031: float | None = None,
    reserve_discount_rates: ArrayLike | None = None,
) -> LauraProjection:
    """Project 2027 deposits, or a separate beginning-of-2031 conditional state.

    2027 uses six growth periods; 2031 uses two, with no new deposits. Reserve
    returns contain nine periods (2033..2041). No return after the final payment
    is counted. Discount-rate scenarios, when supplied, must refer to information
    available at 2033, not to realized future reserve returns.
    """
    policy = policy or LauraPolicy()
    if valuation_year not in (2027, 2031):
        raise ValueError("Valuation must be at the beginning of 2027 or 2031.")
    years = tuple(range(valuation_year, 2033))
    growth = _matrix("Accumulation returns", accumulation_returns, len(years))
    reserve = _matrix("Reserve returns", reserve_returns, 9)
    n = len(growth)
    if len(reserve) != n:
        raise ValueError("Accumulation and reserve scenario rows must be paired.")
    if valuation_year == 2031:
        initial = _number("Opening wealth at 2031", opening_wealth_2031)
        if initial < 0:
            raise ValueError("Opening wealth cannot be negative.")
    else:
        if opening_wealth_2031 is not None:
            raise ValueError("The 2027 plan must use only the two fixed case deposits.")
        initial = 0.0
    balance = np.full(n, initial)
    opening = np.zeros_like(growth)
    closing = np.zeros_like(growth)
    for j, year in enumerate(years):
        opening[:, j] = balance
        balance = (balance + DEPOSITS.get(year, 0.0)) * (1 + growth[:, j]) * (1 - policy.accumulation_fee)
        closing[:, j] = balance
    portfolio = balance.copy()
    rates = np.asarray(policy.reserve_discount_rate if reserve_discount_rates is None else reserve_discount_rates, dtype=float)
    if rates.ndim == 0:
        rates = np.full(n, float(rates))
    if rates.shape != (n,):
        raise ValueError("Reserve discount-rate scenarios must match the scenario rows.")
    required = reserve_value(rates, policy.reserve_buffer)
    allocated = np.minimum(portfolio, required)
    gap = np.maximum(required - portfolio, 0)
    surplus = np.maximum(portfolio - required, 0)
    flexibility_gap = np.maximum(policy.minimum_flexibility - surplus, 0)
    facility = np.maximum(surplus - policy.minimum_flexibility, 0) * policy.facility_surplus_share
    flexibility = surplus - facility
    before = np.zeros((n, 10))
    after = np.zeros_like(before)
    paid = np.zeros_like(before)
    unpaid = np.zeros_like(before)
    first_shortfall = np.zeros(n, dtype=np.int64)
    balance = allocated.copy()
    for j, year in enumerate(PAYMENT_YEARS):
        before[:, j] = balance
        paid[:, j] = np.minimum(balance, ANNUAL_PAYMENT)
        unpaid[:, j] = ANNUAL_PAYMENT - paid[:, j]
        first_shortfall[(unpaid[:, j] > TOLERANCE) & (first_shortfall == 0)] = year
        balance = np.maximum(balance - paid[:, j], 0)
        after[:, j] = balance
        if j < 9:
            balance *= (1 + reserve[:, j]) * (1 - policy.reserve_fee)
    payments_met = unpaid.sum(axis=1) <= TOLERANCE
    plan_met = payments_met & (gap <= TOLERANCE) & (flexibility_gap <= TOLERANCE)
    arrays = [opening, closing, portfolio, required, allocated, gap, flexibility_gap,
              facility, flexibility, before, after, paid, unpaid, first_shortfall, payments_met, plan_met]
    for values in arrays:
        if not np.isfinite(values).all():
            raise ValueError("Return assumptions produce unrepresentable wealth values.")
        values.setflags(write=False)
    return LauraProjection(policy, valuation_year, years, *arrays)


def illustrative_cases() -> dict[str, LauraProjection]:
    """Reproduce the workbook's four unweighted, deterministic illustrations."""
    cases = {
        "Zero return": ([0.0] * 6, [0.0] * 9, 0.0),
        "Base illustration": ([0.06] * 6, [0.03] * 9, 0.03),
        "Growth illustration": ([0.09] * 6, [0.03] * 9, 0.03),
        "2032 drawdown": ([0.06] * 5 + [-0.25], [0.0] * 9, 0.03),
    }
    return {name: project_laura_plan([g], [r], LauraPolicy(reserve_discount_rate=y))
            for name, (g, r, y) in cases.items()}


def _weights(values: ArrayLike, periods: int, assets: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.shape == (assets,):
        array = np.tile(array, (periods, 1))
    if array.shape != (periods, assets) or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Weights must be finite, long-only asset weights, optionally by year.")
    if not np.allclose(array.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Weights must sum to 100% in every year; include cash explicitly.")
    return array


def bootstrap_laura_plan(
    annual_asset_returns: ArrayLike,
    accumulation_weights: ArrayLike,
    reserve_weights: ArrayLike,
    policy: LauraPolicy,
    *,
    n_scenarios: int = 10_000,
    random_seed: int = 2027,
    valuation_year: int = 2027,
    opening_wealth_2031: float | None = None,
) -> tuple[LauraProjection, NDArray[np.int64]]:
    """Shared annual row shocks, with explicit annual rebalancing/glide paths.

    Each row of history is one complete year of aligned nominal USD total
    returns. Asset correlations are preserved within that row. Independent year
    sampling does not preserve serial dependence or predict new market regimes.
    """
    history = np.asarray(annual_asset_returns, dtype=float)
    if history.ndim != 2 or history.shape[0] < 2 or not history.shape[1]:
        raise ValueError("At least two complete, aligned annual observations are required.")
    _matrix("Annual asset history", history, history.shape[1])
    if isinstance(n_scenarios, bool) or not isinstance(n_scenarios, Integral) or not 2 <= n_scenarios <= 100_000:
        raise ValueError("Use between 2 and 100,000 scenarios.")
    if valuation_year not in (2027, 2031):
        raise ValueError("Valuation must be at the beginning of 2027 or 2031.")
    periods = 2033 - valuation_year
    growth_weights = _weights(accumulation_weights, periods, history.shape[1])
    payment_weights = _weights(reserve_weights, 9, history.shape[1])
    indices = np.random.default_rng(random_seed).integers(0, len(history), (n_scenarios, periods + 9))
    growth = np.einsum("sta,ta->st", history[indices[:, :periods]], growth_weights)
    reserve = np.einsum("sta,ta->st", history[indices[:, periods:]], payment_weights)
    projection = project_laura_plan(growth, reserve, policy, valuation_year=valuation_year,
                                    opening_wealth_2031=opening_wealth_2031)
    indices.setflags(write=False)
    return projection, indices


def partner_interval(projection: LauraProjection, lower_quantile: float = 0.05,
                     upper_quantile: float = 0.95) -> dict[str, Any]:
    """A two-sided interval conditional on a declared beginning-of-2031 state.

    Failed reserve/flexibility plans contribute zero to the responsible amount;
    they stay in the denominator. This avoids survivorship-biased promises.
    """
    if projection.valuation_year != 2031 or len(projection.facility) < 2:
        raise ValueError("Partner ranges require a separate probabilistic 2031 valuation.")
    low_q = _number("Lower quantile", lower_quantile)
    high_q = _number("Upper quantile", upper_quantile)
    if not 0 <= low_q < high_q <= 1:
        raise ValueError("Interval quantiles must satisfy 0 <= lower < upper <= 1.")
    # Payment outcomes remain separate: they are unknown at the 2033 decision.
    eligible = (projection.reserve_gap <= TOLERANCE) & (projection.flexibility_gap <= TOLERANCE)
    contribution = np.where(eligible, projection.facility, 0)
    low, high = np.quantile(contribution, [low_q, high_q])
    inside = (contribution >= low) & (contribution <= high)
    return {
        "valuation_date": "2031-01-01",
        "contribution_date": "2033-01-01",
        "lower_usd": float(low), "upper_usd": float(high),
        "nominal_quantile_coverage": high_q - low_q,
        "empirical_interval_coverage": float(np.mean(inside)),
        "below_lower_probability": float(np.mean(contribution < low)),
        "minimum_attainment_probability": float(np.mean(contribution >= low)),
        "zero_contribution_probability": float(np.mean(contribution <= TOLERANCE)),
        "interval_and_all_payments_probability": float(np.mean(inside & projection.all_payments_met)),
        "reserve_shortfall_probability": float(np.mean(projection.reserve_gap > TOLERANCE)),
        "plan_success_probability": float(np.mean(projection.plan_met)),
    }
