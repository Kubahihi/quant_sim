from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.efficient_frontier import calculate_efficient_frontier


def _sample_returns(periods: int = 180, assets: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    return pd.DataFrame(
        rng.normal(0.0005, 0.012, size=(periods, assets)),
        columns=[f"A{index}" for index in range(assets)],
    )


def test_efficient_frontier_satisfies_target_and_weight_constraints():
    returns = _sample_returns()
    points = calculate_efficient_frontier(returns, n_points=20)

    assert len(points) == 20
    point_returns = []
    for point in points:
        weights = np.asarray(point["weights"], dtype=float)
        assert np.isclose(weights.sum(), 1.0, atol=1e-7)
        assert np.all(weights >= -1e-8)
        assert np.isfinite(point["volatility"])
        point_returns.append(float(point["return"]))

    assert np.all(np.diff(point_returns) >= -1e-7)


@pytest.mark.parametrize("n_points", [0, 1, 2.5])
def test_efficient_frontier_rejects_invalid_point_count(n_points):
    with pytest.raises(ValueError, match="n_points"):
        calculate_efficient_frontier(_sample_returns(), n_points=n_points)


def test_efficient_frontier_rejects_empty_returns():
    with pytest.raises(ValueError, match="returns"):
        calculate_efficient_frontier(pd.DataFrame())


def test_single_asset_has_one_feasible_frontier_point():
    returns = _sample_returns(assets=1)

    points = calculate_efficient_frontier(returns, n_points=10)

    assert len(points) == 1
    assert np.array_equal(points[0]["weights"], np.array([1.0]))
    assert points[0]["return"] == pytest.approx(float(returns.mean().iloc[0] * 252))


@pytest.mark.parametrize(
    ("allow_short", "max_weight"),
    [(False, None), (False, 0.30), (True, 0.40)],
)
def test_reusable_solver_matches_slsqp_fallback(
    monkeypatch: pytest.MonkeyPatch,
    allow_short: bool,
    max_weight: float | None,
):
    returns = _sample_returns(periods=320, assets=6)
    accelerated = calculate_efficient_frontier(
        returns,
        n_points=12,
        allow_short=allow_short,
        max_weight=max_weight,
    )

    import osqp

    class UnavailableSolver:
        def setup(self, **_kwargs):
            raise RuntimeError("solver unavailable")

    monkeypatch.setattr(osqp, "OSQP", UnavailableSolver)
    fallback = calculate_efficient_frontier(
        returns,
        n_points=12,
        allow_short=allow_short,
        max_weight=max_weight,
    )

    assert len(accelerated) == len(fallback) == 12
    np.testing.assert_allclose(
        [point["return"] for point in accelerated],
        [point["return"] for point in fallback],
        atol=1e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        [point["volatility"] for point in accelerated],
        [point["volatility"] for point in fallback],
        atol=1e-7,
        rtol=0.0,
    )
