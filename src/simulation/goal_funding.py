"""Client-goal funding projections and portfolio comparisons.

The engine deliberately answers one narrow question: given a set of return
scenarios and the client's cash-flow plan, how often is the terminal goal met?
It is not a return forecaster.  Bootstrap simulations therefore expose their
sampled observation indices so a result can be reproduced and audited.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from statistics import NormalDist
from typing import Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


WealthBasis = Literal["nominal", "real"]
CashflowTiming = Literal["start", "end"]


def _finite_float(name: str, value: Real) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _positive_integer(name: str, value: Integral, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return normalized


def _validate_basis(name: str, value: str) -> WealthBasis:
    if value not in {"nominal", "real"}:
        raise ValueError(f"{name} must be either 'nominal' or 'real'.")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class GoalFundingConfig:
    """Inputs that are shared by every portfolio evaluated for a client goal.

    ``target_wealth``, ``initial_capital`` and ``annual_net_cashflow`` must all
    use ``wealth_basis``.  A real-basis cash flow is consequently constant in
    purchasing-power terms; a nominal-basis cash flow is constant in currency
    units.  Return inputs declare their own basis when a projection is run and
    are converted using ``inflation_rate`` when necessary.

    Cash flow is applied once per year.  Positive values are contributions and
    negative values are withdrawals.  Wealth is floored at zero, so the model
    does not silently assume that the client can borrow to fund a withdrawal.
    """

    target_wealth: float
    horizon_years: int
    initial_capital: float
    annual_net_cashflow: float = 0.0
    inflation_rate: float = 0.0
    wealth_basis: WealthBasis = "nominal"
    cashflow_timing: CashflowTiming = "end"
    ruin_threshold: float = 0.0
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        target = _finite_float("target_wealth", self.target_wealth)
        initial = _finite_float("initial_capital", self.initial_capital)
        cashflow = _finite_float("annual_net_cashflow", self.annual_net_cashflow)
        inflation = _finite_float("inflation_rate", self.inflation_rate)
        ruin_threshold = _finite_float("ruin_threshold", self.ruin_threshold)
        confidence_level = _finite_float("confidence_level", self.confidence_level)
        horizon = _positive_integer("horizon_years", self.horizon_years)

        if target <= 0.0:
            raise ValueError("target_wealth must be positive.")
        if initial < 0.0:
            raise ValueError("initial_capital must be non-negative.")
        if inflation <= -1.0:
            raise ValueError("inflation_rate must be greater than -1.")
        if ruin_threshold < 0.0:
            raise ValueError("ruin_threshold must be non-negative.")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1.")
        basis = _validate_basis("wealth_basis", self.wealth_basis)
        if self.cashflow_timing not in {"start", "end"}:
            raise ValueError("cashflow_timing must be either 'start' or 'end'.")

        # Normalize NumPy scalar inputs so dataclass values serialize cleanly.
        object.__setattr__(self, "target_wealth", target)
        object.__setattr__(self, "horizon_years", horizon)
        object.__setattr__(self, "initial_capital", initial)
        object.__setattr__(self, "annual_net_cashflow", cashflow)
        object.__setattr__(self, "inflation_rate", inflation)
        object.__setattr__(self, "wealth_basis", basis)
        object.__setattr__(self, "ruin_threshold", ruin_threshold)
        object.__setattr__(self, "confidence_level", confidence_level)


@dataclass(frozen=True, slots=True)
class GoalFundingMetrics:
    """Decision-facing summary of a set of equally weighted scenarios.

    Wilson intervals describe finite-scenario sampling uncertainty only.  They
    do not cover uncertainty in return history, inflation, cash flows, or the
    scenario-generation model.
    """

    probability_goal_achieved: float
    goal_achievement_confidence_interval: tuple[float, float]
    expected_terminal_wealth: float
    median_terminal_wealth: float
    percentile_10_terminal_wealth: float
    shortfall_probability: float
    expected_shortfall_vs_goal: float
    ruin_probability: float | None
    ruin_probability_confidence_interval: tuple[float, float] | None
    target_wealth: float
    n_scenarios: int
    confidence_level: float
    wealth_basis: WealthBasis


@dataclass(frozen=True, slots=True)
class GoalFundingProjection:
    """Scenario paths plus the client-goal metrics derived from them.

    Arrays use the scenario-first convention.  ``wealth_paths`` has shape
    ``(n_scenarios, horizon_years + 1)`` and ``portfolio_return_scenarios`` has
    shape ``(n_scenarios, horizon_years)``.  Bootstrap simulations also carry
    ``sampled_observation_indices`` with the latter shape.
    """

    metrics: GoalFundingMetrics
    wealth_paths: NDArray[np.float64]
    portfolio_return_scenarios: NDArray[np.float64]
    sampled_observation_indices: NDArray[np.int64] | None = None


def _one_dimensional_scenarios(name: str, values: ArrayLike) -> NDArray[np.float64]:
    scenarios = np.asarray(values, dtype=float)
    if scenarios.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if scenarios.size < 2:
        raise ValueError(f"{name} must contain at least two scenarios.")
    if not np.isfinite(scenarios).all():
        raise ValueError(f"{name} must contain only finite values.")
    return scenarios


def _scenario_matrix(
    name: str,
    values: ArrayLike,
    *,
    expected_periods: int,
    include_initial_period: bool,
) -> NDArray[np.float64]:
    scenarios = np.asarray(values, dtype=float)
    required_columns = expected_periods + int(include_initial_period)
    if scenarios.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional, scenario-first matrix."
        )
    if scenarios.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two scenarios.")
    if scenarios.shape[1] != required_columns:
        raise ValueError(
            f"{name} must have {required_columns} columns for the configured horizon."
        )
    if not np.isfinite(scenarios).all():
        raise ValueError(f"{name} must contain only finite values.")
    return scenarios


def _wilson_interval(successes: int, observations: int, confidence_level: float) -> tuple[float, float]:
    """Wilson score interval for a binomial scenario proportion."""

    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    probability = successes / observations
    z_squared = z_score**2
    denominator = 1.0 + z_squared / observations
    center = (probability + z_squared / (2.0 * observations)) / denominator
    margin = (
        z_score
        * np.sqrt(
            probability * (1.0 - probability) / observations
            + z_squared / (4.0 * observations**2)
        )
        / denominator
    )
    return max(0.0, float(center - margin)), min(1.0, float(center + margin))


def evaluate_goal_funding(
    config: GoalFundingConfig,
    *,
    terminal_wealth: ArrayLike | None = None,
    wealth_paths: ArrayLike | None = None,
) -> GoalFundingMetrics:
    """Evaluate terminal values or complete wealth paths against the goal.

    Exactly one input must be provided.  Supplied values are assumed to use
    ``config.wealth_basis``.  Complete paths enable the optional ruin metric;
    terminal-only scenarios do not contain enough information, so their ruin
    probability is ``None``.
    """

    if not isinstance(config, GoalFundingConfig):
        raise TypeError("config must be a GoalFundingConfig instance.")
    if (terminal_wealth is None) == (wealth_paths is None):
        raise ValueError("Provide exactly one of terminal_wealth or wealth_paths.")

    paths: NDArray[np.float64] | None = None
    if wealth_paths is not None:
        paths = _scenario_matrix(
            "wealth_paths",
            wealth_paths,
            expected_periods=config.horizon_years,
            include_initial_period=True,
        )
        if not np.allclose(paths[:, 0], config.initial_capital, rtol=1e-10, atol=1e-8):
            raise ValueError(
                "The first wealth_paths column must equal config.initial_capital."
            )
        terminal_values = paths[:, -1]
    else:
        terminal_values = _one_dimensional_scenarios("terminal_wealth", terminal_wealth)

    achieved = terminal_values >= config.target_wealth
    successes = int(np.count_nonzero(achieved))
    n_scenarios = int(terminal_values.size)
    probability = successes / n_scenarios
    shortfall_probability = 1.0 - probability
    deficits = config.target_wealth - terminal_values[~achieved]
    expected_shortfall = float(np.mean(deficits)) if deficits.size else 0.0

    ruin_probability: float | None = None
    ruin_interval: tuple[float, float] | None = None
    if paths is not None:
        ruined = np.any(paths[:, 1:] <= config.ruin_threshold, axis=1)
        ruin_count = int(np.count_nonzero(ruined))
        ruin_probability = ruin_count / n_scenarios
        ruin_interval = _wilson_interval(
            ruin_count,
            n_scenarios,
            config.confidence_level,
        )

    return GoalFundingMetrics(
        probability_goal_achieved=float(probability),
        goal_achievement_confidence_interval=_wilson_interval(
            successes,
            n_scenarios,
            config.confidence_level,
        ),
        expected_terminal_wealth=float(np.mean(terminal_values)),
        median_terminal_wealth=float(np.median(terminal_values)),
        percentile_10_terminal_wealth=float(np.percentile(terminal_values, 10.0)),
        shortfall_probability=float(shortfall_probability),
        expected_shortfall_vs_goal=expected_shortfall,
        ruin_probability=ruin_probability,
        ruin_probability_confidence_interval=ruin_interval,
        target_wealth=config.target_wealth,
        n_scenarios=n_scenarios,
        confidence_level=config.confidence_level,
        wealth_basis=config.wealth_basis,
    )


def _convert_return_basis(
    returns: NDArray[np.float64],
    *,
    source_basis: WealthBasis,
    target_basis: WealthBasis,
    inflation_rate: float,
) -> NDArray[np.float64]:
    converted = np.array(returns, dtype=float, copy=True)
    if source_basis == target_basis:
        return converted
    if source_basis == "nominal" and target_basis == "real":
        return (1.0 + converted) / (1.0 + inflation_rate) - 1.0
    return (1.0 + converted) * (1.0 + inflation_rate) - 1.0


def _read_only(array: NDArray) -> NDArray:
    array.setflags(write=False)
    return array


def project_goal_funding(
    config: GoalFundingConfig,
    portfolio_return_scenarios: ArrayLike,
    *,
    returns_basis: WealthBasis = "nominal",
    sampled_observation_indices: ArrayLike | None = None,
) -> GoalFundingProjection:
    """Build annual wealth paths from aligned portfolio return scenarios.

    Simple return scenarios must have shape ``(n_scenarios, horizon_years)``.
    Returns below -100% are rejected.  Cash flow is applied according to
    ``config.cashflow_timing`` and wealth is floored at zero after each period.
    """

    if not isinstance(config, GoalFundingConfig):
        raise TypeError("config must be a GoalFundingConfig instance.")
    basis = _validate_basis("returns_basis", returns_basis)
    raw_returns = _scenario_matrix(
        "portfolio_return_scenarios",
        portfolio_return_scenarios,
        expected_periods=config.horizon_years,
        include_initial_period=False,
    )
    if np.any(raw_returns < -1.0):
        raise ValueError("Simple portfolio returns cannot be below -100%.")
    returns = _convert_return_basis(
        raw_returns,
        source_basis=basis,
        target_basis=config.wealth_basis,
        inflation_rate=config.inflation_rate,
    )
    if np.any(returns < -1.0 - 1e-12):
        raise ValueError("Converted simple portfolio returns cannot be below -100%.")

    n_scenarios = returns.shape[0]
    wealth = np.empty((n_scenarios, config.horizon_years + 1), dtype=float)
    wealth[:, 0] = config.initial_capital
    cashflow = config.annual_net_cashflow
    for year in range(config.horizon_years):
        opening = wealth[:, year]
        if config.cashflow_timing == "start":
            investable = np.maximum(opening + cashflow, 0.0)
            closing = investable * (1.0 + returns[:, year])
        else:
            closing = opening * (1.0 + returns[:, year]) + cashflow
        wealth[:, year + 1] = np.maximum(closing, 0.0)

    indices: NDArray[np.int64] | None = None
    if sampled_observation_indices is not None:
        raw_indices = np.asarray(sampled_observation_indices)
        if raw_indices.shape != returns.shape:
            raise ValueError(
                "sampled_observation_indices must match portfolio_return_scenarios."
            )
        if not np.issubdtype(raw_indices.dtype, np.integer):
            raise TypeError("sampled_observation_indices must contain integers.")
        if np.any(raw_indices < 0):
            raise ValueError("sampled_observation_indices cannot contain negatives.")
        indices = np.array(raw_indices, dtype=np.int64, copy=True)

    metrics = evaluate_goal_funding(config, wealth_paths=wealth)
    return GoalFundingProjection(
        metrics=metrics,
        wealth_paths=_read_only(wealth),
        portfolio_return_scenarios=_read_only(returns),
        sampled_observation_indices=_read_only(indices) if indices is not None else None,
    )


def _historical_asset_returns(
    annual_asset_returns: ArrayLike,
) -> NDArray[np.float64]:
    returns = np.asarray(annual_asset_returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError("annual_asset_returns must be a two-dimensional matrix.")
    if returns.shape[0] < 2:
        raise ValueError("annual_asset_returns must contain at least two observations.")
    if returns.shape[1] < 1:
        raise ValueError("annual_asset_returns must contain at least one asset.")
    if not np.isfinite(returns).all():
        raise ValueError("annual_asset_returns must contain only finite values.")
    if np.any(returns < -1.0):
        raise ValueError("Simple asset returns cannot be below -100%.")
    return returns


def _portfolio_weights(
    weights: ArrayLike,
    *,
    n_assets: int,
    name: str = "weights",
) -> NDArray[np.float64]:
    normalized = np.asarray(weights, dtype=float)
    if normalized.ndim != 1 or normalized.size != n_assets:
        raise ValueError(f"{name} must contain one value per asset.")
    if not np.isfinite(normalized).all():
        raise ValueError(f"{name} must contain only finite values.")
    if not np.isclose(np.sum(normalized), 1.0, rtol=1e-9, atol=1e-9):
        raise ValueError(f"{name} must sum to 1.")
    return normalized


def _simulation_count(value: Integral) -> int:
    return _positive_integer("n_scenarios", value, minimum=2)


def _random_seed(value: Integral | None) -> int | None:
    if value is None:
        return None
    seed = _positive_integer("random_seed", value, minimum=0)
    return seed


def simulate_goal_funding(
    config: GoalFundingConfig,
    annual_asset_returns: ArrayLike,
    weights: ArrayLike,
    *,
    n_scenarios: int = 10_000,
    random_seed: int | None = None,
    returns_basis: WealthBasis = "nominal",
) -> GoalFundingProjection:
    """Bootstrap annual asset-return rows and project a weighted portfolio.

    Sampling whole rows preserves the cross-sectional relationship between
    assets in each historical observation.  Years are sampled independently;
    this is a transparent historical bootstrap, not a claim that returns are
    normally distributed or that estimated parameters are known with certainty.
    A local random generator makes seeded runs deterministic without mutating
    NumPy's process-wide random state.
    """

    if not isinstance(config, GoalFundingConfig):
        raise TypeError("config must be a GoalFundingConfig instance.")
    asset_returns = _historical_asset_returns(annual_asset_returns)
    portfolio_weights = _portfolio_weights(weights, n_assets=asset_returns.shape[1])
    simulations = _simulation_count(n_scenarios)
    seed = _random_seed(random_seed)
    _validate_basis("returns_basis", returns_basis)

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        asset_returns.shape[0],
        size=(simulations, config.horizon_years),
        dtype=np.int64,
    )
    historical_portfolio_returns = asset_returns @ portfolio_weights
    return_scenarios = historical_portfolio_returns[indices]
    return project_goal_funding(
        config,
        return_scenarios,
        returns_basis=returns_basis,
        sampled_observation_indices=indices,
    )


def _validated_portfolio_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Every portfolio name must be a non-empty string.")
    return name


def compare_goal_funding_scenarios(
    config: GoalFundingConfig,
    named_portfolio_return_scenarios: Mapping[str, ArrayLike],
    *,
    returns_basis: WealthBasis = "nominal",
) -> dict[str, GoalFundingMetrics]:
    """Compare named portfolios whose rows represent the same scenarios.

    Every matrix must have the same shape and row order.  The helper does not
    resample, which keeps portfolio comparisons paired under common scenarios.
    """

    if not isinstance(named_portfolio_return_scenarios, Mapping):
        raise TypeError("named_portfolio_return_scenarios must be a mapping.")
    if not named_portfolio_return_scenarios:
        raise ValueError("At least one named portfolio is required.")

    results: dict[str, GoalFundingMetrics] = {}
    expected_shape: tuple[int, int] | None = None
    for raw_name, scenarios in named_portfolio_return_scenarios.items():
        name = _validated_portfolio_name(raw_name)
        matrix = _scenario_matrix(
            f"named_portfolio_return_scenarios[{name!r}]",
            scenarios,
            expected_periods=config.horizon_years,
            include_initial_period=False,
        )
        if expected_shape is None:
            expected_shape = matrix.shape
        elif matrix.shape != expected_shape:
            raise ValueError("All named portfolio scenario matrices must share one shape.")
        results[name] = project_goal_funding(
            config,
            matrix,
            returns_basis=returns_basis,
        ).metrics
    return results


def compare_portfolios_from_asset_returns(
    config: GoalFundingConfig,
    annual_asset_returns: ArrayLike,
    named_weights: Mapping[str, ArrayLike],
    *,
    n_scenarios: int = 10_000,
    random_seed: int | None = None,
    returns_basis: WealthBasis = "nominal",
) -> dict[str, GoalFundingProjection]:
    """Bootstrap once, then evaluate named weights under the common shocks.

    The same sampled observation index is used for every candidate, avoiding
    Monte Carlo noise from being mistaken for a portfolio difference.
    """

    if not isinstance(named_weights, Mapping):
        raise TypeError("named_weights must be a mapping.")
    if not named_weights:
        raise ValueError("At least one named portfolio is required.")
    asset_returns = _historical_asset_returns(annual_asset_returns)
    simulations = _simulation_count(n_scenarios)
    seed = _random_seed(random_seed)
    _validate_basis("returns_basis", returns_basis)

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        asset_returns.shape[0],
        size=(simulations, config.horizon_years),
        dtype=np.int64,
    )

    results: dict[str, GoalFundingProjection] = {}
    for raw_name, raw_weights in named_weights.items():
        name = _validated_portfolio_name(raw_name)
        weights = _portfolio_weights(
            raw_weights,
            n_assets=asset_returns.shape[1],
            name=f"named_weights[{name!r}]",
        )
        portfolio_history = asset_returns @ weights
        results[name] = project_goal_funding(
            config,
            portfolio_history[indices],
            returns_basis=returns_basis,
            sampled_observation_indices=indices,
        )
    return results


__all__ = [
    "CashflowTiming",
    "GoalFundingConfig",
    "GoalFundingMetrics",
    "GoalFundingProjection",
    "WealthBasis",
    "compare_goal_funding_scenarios",
    "compare_portfolios_from_asset_returns",
    "evaluate_goal_funding",
    "project_goal_funding",
    "simulate_goal_funding",
]
