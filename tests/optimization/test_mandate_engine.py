from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    build_constraint_set,
    estimate_black_litterman_inputs,
    optimize_portfolio,
)


SYMBOLS = ["TECH1", "TECH2", "HEALTH", "CASH", "BAD"]


def _returns(periods: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(20260804)
    market = rng.normal(0.00030, 0.0070, size=(periods, 1))
    residual = rng.normal(0.00010, 0.0080, size=(periods, 4))
    risky = market + residual
    cash = rng.normal(0.00010, 0.00015, size=(periods, 1))
    values = np.column_stack([
        risky[:, 0],
        risky[:, 1],
        risky[:, 2],
        cash[:, 0],
        risky[:, 3] + 0.0010,
    ])
    return pd.DataFrame(values, columns=SYMBOLS)


def _metadata() -> dict[str, dict[str, object]]:
    return {
        "TECH1": {"sector": "Technology", "asset_type": "Stock", "beta": 1.20, "approved": True, "tags": ["liquid"]},
        "TECH2": {"sector": "Technology", "asset_type": "Stock", "beta": 1.05, "approved": True, "tags": ["liquid"]},
        "HEALTH": {"sector": "Health Care", "asset_type": "Stock", "beta": 0.80, "approved": True, "tags": ["liquid"]},
        "CASH": {"sector": "", "asset_type": "Cash", "beta": 0.0, "approved": True, "tags": ["liquid"]},
        "BAD": {"sector": "Tobacco", "asset_type": "Stock", "beta": 1.50, "approved": False, "tags": []},
    }


def _strategy() -> dict[str, object]:
    return {
        "long_only": True,
        "max_position_weight": 0.45,
        "max_sector_weight": 0.60,
        "min_cash_weight": 0.05,
        "max_cash_weight": 0.20,
        "max_turnover": 0.80,
        "min_beta": 0.60,
        "max_beta": 1.00,
        "prohibited_tickers": ["BAD"],
        "excluded_sectors": ["Tobacco"],
        "allowed_asset_types": ["stock", "cash"],
        "required_tags": ["liquid"],
        "require_approved": True,
        "min_holdings": 3,
        "sector_targets": [
            {"sector": "Technology", "min_weight": 0.30, "max_weight": 0.55},
            {"sector": "Health Care", "min_weight": 0.25, "max_weight": 0.50},
        ],
    }


def _current_weights() -> np.ndarray:
    return np.asarray([0.25, 0.20, 0.30, 0.15, 0.10], dtype=float)


def test_strategy_rulebook_translates_to_auditable_constraints():
    constraints = build_constraint_set(
        SYMBOLS,
        strategy=_strategy(),
        asset_metadata=_metadata(),
        current_weights=_current_weights(),
    )

    assert constraints.upper_bounds[SYMBOLS.index("BAD")] == 0.0
    assert constraints.turnover_limit == pytest.approx(0.80)
    assert constraints.minimum_beta == pytest.approx(0.60)
    assert constraints.maximum_beta == pytest.approx(1.00)
    assert {group.name for group in constraints.groups} >= {
        "cash",
        "sector_target:technology",
        "sector_target:health care",
    }
    assert any("mixed-integer" in warning for warning in constraints.warnings)


def test_shorting_requires_explicit_permission_and_cash_is_not_treated_as_a_security():
    unconstrained = build_constraint_set(["A", "B"], allow_short=True)
    long_only = build_constraint_set(
        ["A", "B"],
        allow_short=True,
        strategy={"long_only": True},
    )
    cash_exempt = build_constraint_set(
        ["A", "CASH"],
        strategy={
            "required_tags": ["approved-tag"],
            "require_approved": True,
            "min_cash_weight": 0.10,
        },
        asset_metadata={
            "A": {"tags": ["approved-tag"], "approved": True},
            "CASH": {"asset_type": "cash"},
        },
    )

    assert np.all(unconstrained.lower_bounds < 0)
    assert np.all(long_only.lower_bounds == 0)
    assert cash_exempt.upper_bounds[1] > 0


@pytest.mark.parametrize(
    "objective",
    ["minimum_variance", "maximum_utility", "minimum_cvar"],
)
def test_mandate_aware_objectives_respect_every_supported_constraint(objective):
    result = optimize_portfolio(
        _returns(),
        objective=objective,
        strategy=_strategy(),
        asset_metadata=_metadata(),
        current_weights=_current_weights(),
        transaction_cost_bps={symbol: 8.0 + index for index, symbol in enumerate(SYMBOLS)},
        risk_aversion=2.0,
    )

    assert result["success"] is True, result.get("message")
    weights = dict(zip(result["symbols"], result["weights"], strict=False))
    assert weights["BAD"] == pytest.approx(0.0, abs=2e-5)
    assert 0.05 - 2e-5 <= weights["CASH"] <= 0.20 + 2e-5
    assert 0.30 - 2e-5 <= weights["TECH1"] + weights["TECH2"] <= 0.55 + 2e-5
    assert 0.25 - 2e-5 <= weights["HEALTH"] <= 0.50 + 2e-5
    assert result["turnover"] <= 0.80 + 2e-5
    assert all(row["passed"] for row in result["constraint_report"])


def test_target_volatility_and_tracking_error_objectives_are_available():
    returns = _returns()
    constraints = build_constraint_set(
        SYMBOLS,
        strategy=_strategy(),
        asset_metadata=_metadata(),
        current_weights=_current_weights(),
    )
    minimum_variance = optimize_portfolio(
        returns,
        objective="minimum_variance",
        constraint_set=constraints,
    )
    target = float(minimum_variance["volatility"]) * 1.20
    target_result = optimize_portfolio(
        returns,
        objective="target_volatility",
        constraint_set=constraints,
        target_volatility=target,
    )
    tracking_result = optimize_portfolio(
        returns,
        objective="minimum_tracking_error",
        constraint_set=constraints,
        benchmark_weights=[0.25, 0.20, 0.35, 0.20, 0.0],
    )

    assert target_result["success"] is True, target_result.get("message")
    assert target_result["volatility"] <= target + 2e-5
    assert tracking_result["success"] is True, tracking_result.get("message")
    assert tracking_result["tracking_error"] is not None


def test_required_cash_and_missing_beta_fail_before_solver():
    no_cash_symbols = ["TECH1", "TECH2", "HEALTH"]
    with pytest.raises(ValueError, match="no cash asset"):
        build_constraint_set(
            no_cash_symbols,
            strategy={"min_cash_weight": 0.10},
            asset_metadata=_metadata(),
        )
    incomplete_metadata = _metadata()
    incomplete_metadata["HEALTH"] = {"sector": "Health Care", "asset_type": "Stock"}
    with pytest.raises(ValueError, match="beta is missing"):
        build_constraint_set(
            SYMBOLS,
            strategy={"min_beta": 0.5, "max_beta": 1.2},
            asset_metadata=incomplete_metadata,
        )


def test_sector_cap_fails_closed_on_missing_metadata_and_exempts_cash():
    with pytest.raises(ValueError, match="sector metadata for: UNKNOWN"):
        build_constraint_set(
            ["KNOWN", "UNKNOWN"],
            strategy={"max_sector_weight": 0.75},
            asset_metadata={"KNOWN": {"sector": "Technology"}},
        )

    constraints = build_constraint_set(
        ["STOCK", "CASH"],
        strategy={"max_sector_weight": 0.75},
        asset_metadata={
            "STOCK": {"sector": "Technology", "asset_type": "Stock"},
            "CASH": {"sector": "Cash", "asset_type": "Cash"},
        },
    )
    sector_groups = [
        group
        for group in constraints.groups
        if group.name.startswith("sector:")
    ]
    assert len(sector_groups) == 1
    assert sector_groups[0].indices == (0,)


def test_black_litterman_confidence_controls_view_impact_and_integrates_with_engine():
    returns = _returns()[["TECH1", "TECH2", "HEALTH"]]
    market_weights = {"TECH1": 0.40, "TECH2": 0.35, "HEALTH": 0.25}
    low_confidence = estimate_black_litterman_inputs(
        returns,
        market_weights=market_weights,
        views={"TECH1": 0.20},
        view_confidences={"TECH1": 0.20},
    )
    high_confidence = estimate_black_litterman_inputs(
        returns,
        market_weights=market_weights,
        views={"TECH1": 0.20},
        view_confidences={"TECH1": 0.90},
    )
    equilibrium = high_confidence.expected_return_details["equilibrium_returns"]["TECH1"]
    tech_index = list(high_confidence.symbols).index("TECH1")

    assert abs(high_confidence.mean_returns[tech_index] - equilibrium) > abs(
        low_confidence.mean_returns[tech_index] - equilibrium
    )
    result = optimize_portfolio(
        returns,
        objective="maximum_utility",
        portfolio_estimates=high_confidence,
        max_weight=0.60,
        risk_aversion=3.0,
    )
    assert result["success"] is True
    assert result["expected_return_model"] == "black_litterman"
    assert result["estimation"]["method"] == "black_litterman_shrunk_covariance"
