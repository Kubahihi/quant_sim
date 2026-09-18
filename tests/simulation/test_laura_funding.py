from dataclasses import replace

import numpy as np
import pytest

from src.simulation.laura_funding import (
    LauraPolicy, bootstrap_laura_plan, illustrative_cases, partner_interval,
    project_laura_plan, reserve_value,
)


def test_zero_returns_have_fifty_thousand_unfunded_and_no_borrowing():
    p = illustrative_cases()["Zero return"]
    assert p.portfolio_2033[0] == 450_000
    assert p.required_reserve[0] == 500_000
    assert p.reserve_gap[0] == 50_000
    assert p.facility[0] == p.flexibility[0] == 0
    assert p.unpaid[0].sum() == 50_000
    assert p.first_shortfall_year[0] == 2042
    assert not p.all_payments_met[0]
    assert np.all(p.reserve_after_payment >= 0)
    with pytest.raises(ValueError, match="deterministic"):
        p.summary()


def test_first_deposit_has_six_returns_second_has_five():
    p = project_laura_plan([[.05] * 6], [[.03] * 9])
    assert p.portfolio_2033[0] == pytest.approx(300_000 * 1.05**6 + 150_000 * 1.05**5)
    np.testing.assert_allclose(p.accumulation_opening[0, 1], 315_000)


def test_workbook_growth_case_reconciles_to_cached_source_cells():
    p = illustrative_cases()["Growth illustration"]
    # Akumulace!F20,F21,F26,F27 and Rezerva!F31,F32 in the supplied XLSX.
    assert p.portfolio_2033[0] == pytest.approx(733_923.6264872999)
    assert p.required_reserve[0] == pytest.approx(439_305.4460939553)
    assert p.facility[0] == pytest.approx(235_694.5443146757)
    assert p.flexibility[0] == pytest.approx(58_923.636078668904)
    assert p.unpaid.sum() < .01
    assert abs(p.reserve_after_payment[0, -1]) < .01
    assert p.all_payments_met[0]


def test_reserve_is_annuity_due_including_the_first_undiscounted_payment():
    assert reserve_value(0)[0] == 500_000
    assert reserve_value(.03)[0] == pytest.approx(sum(50_000 / 1.03**k for k in range(10)))
    assert reserve_value(.03, .1)[0] == pytest.approx(reserve_value(.03)[0] * 1.1)


def test_reserve_is_internal_transfer_not_a_second_withdrawal():
    p = illustrative_cases()["Base illustration"]
    np.testing.assert_allclose(p.allocated_reserve + p.facility + p.flexibility, p.portfolio_2033)
    assert p.reserve_before_payment[0, 0] == p.allocated_reserve[0]
    assert p.reserve_after_payment[0, 0] == p.allocated_reserve[0] - 50_000


def test_no_2033_return_can_rescue_an_immediate_shortfall():
    p = project_laura_plan([[-1, -1, 0, 0, 0, 0]], [[100.0] * 9])
    assert p.portfolio_2033[0] == 0
    assert p.first_shortfall_year[0] == 2033
    assert p.unpaid.sum() == 500_000


def test_last_payment_cannot_use_an_extra_year_of_returns():
    with pytest.raises(ValueError, match="9 annual"):
        project_laura_plan([[0] * 6], [[0] * 10])


def test_discount_rate_does_not_force_realized_reserve_returns():
    p = project_laura_plan([[.06] * 6], [[0] * 9])
    assert p.reserve_gap[0] == 0
    assert p.unpaid.sum() == pytest.approx(500_000 - reserve_value(.03)[0])
    assert not p.all_payments_met[0]
    assert p.first_shortfall_year[0] == 2041


def test_operating_success_does_not_hide_failure_to_establish_reserve():
    p = project_laura_plan([[0] * 6], [[.15] * 9], LauraPolicy(reserve_discount_rate=0))
    assert p.all_payments_met[0]
    assert p.reserve_gap[0] == 50_000
    assert not p.plan_met[0]


def test_insufficient_flexibility_is_not_presented_as_a_success():
    p = project_laura_plan([[.06] * 6], [[.03] * 9], LauraPolicy(minimum_flexibility=1_000_000))
    assert p.all_payments_met[0]
    assert p.facility[0] == 0
    assert p.flexibility_gap[0] > 0
    assert not p.plan_met[0]


def test_first_failure_year_and_joint_success_are_path_based():
    p = project_laura_plan([[.1] * 6] * 3,
                           [[-1] + [0] * 8, [0] * 8 + [-1], [0] * 9],
                           LauraPolicy(reserve_discount_rate=0))
    assert p.first_shortfall_year.tolist() == [2034, 2042, 0]
    summary = p.summary()
    assert summary["reserve_funding_probability"] == 1
    assert summary["all_payments_probability"] == pytest.approx(1/3)
    assert summary["payments_success_given_funded_reserve"] == pytest.approx(1/3)
    assert summary["conditional_mean_unpaid"] == 250_000
    assert summary["first_shortfall_counts"]["2034"] == 1


def test_annual_costs_compound_and_can_break_a_fully_priced_reserve():
    policy = LauraPolicy(accumulation_fee=.01, reserve_fee=.01)
    p = project_laura_plan([[.06] * 6], [[.03] * 9], policy)
    factor = 1.06 * .99
    assert p.portfolio_2033[0] == pytest.approx(300_000 * factor**6 + 150_000 * factor**5)
    assert not p.all_payments_met[0]


def test_scenario_specific_reserve_prices_stay_paired_with_portfolio():
    p = project_laura_plan([[.05] * 6, [.1] * 6], [[0] * 9] * 2,
                           reserve_discount_rates=[0, .05])
    np.testing.assert_allclose(p.required_reserve, reserve_value([0, .05]))
    np.testing.assert_allclose(p.allocated_reserve + p.facility + p.flexibility, p.portfolio_2033)


def test_2031_projection_has_two_periods_no_new_deposits():
    p = project_laura_plan([[.05, .05]] * 2, [[0] * 9] * 2,
                           valuation_year=2031, opening_wealth_2031=550_000)
    np.testing.assert_allclose(p.portfolio_2033, 550_000 * 1.05**2)
    assert p.accumulation_years == (2031, 2032)
    assert partner_interval(p)["valuation_date"] == "2031-01-01"
    with pytest.raises(ValueError, match="2031 valuation"):
        partner_interval(project_laura_plan([[.05] * 6] * 2, [[0] * 9] * 2))


def test_partner_interval_distinguishes_two_sided_coverage_and_minimum_attainment():
    growth = np.column_stack([np.linspace(.05, .25, 1001), np.zeros(1001)])
    p = project_laura_plan(growth, np.zeros((1001, 9)), LauraPolicy(reserve_discount_rate=0),
                           valuation_year=2031, opening_wealth_2031=600_000)
    result = partner_interval(p)
    assert result["nominal_quantile_coverage"] == pytest.approx(.9)
    assert result["empirical_interval_coverage"] == pytest.approx(901 / 1001)
    assert result["minimum_attainment_probability"] == pytest.approx(951 / 1001)
    assert result["below_lower_probability"] == pytest.approx(50 / 1001)


def test_failed_partner_paths_remain_in_interval_denominator():
    p = project_laura_plan([[-1, 0], [.1, .1]], [[0] * 9] * 2,
                           LauraPolicy(reserve_discount_rate=0), valuation_year=2031,
                           opening_wealth_2031=600_000)
    result = partner_interval(p, 0, 1)
    assert result["lower_usd"] == 0
    assert result["reserve_shortfall_probability"] == .5
    assert result["empirical_interval_coverage"] == 1
    assert result["interval_and_all_payments_probability"] == .5


def test_bootstrap_preserves_asset_rows_glide_path_and_seed():
    history = np.array([[.2, -.1], [-.1, .2], [.05, .03]])
    weights = np.array([[1, 0]] * 4 + [[0, 1]] * 2)
    p, indices = bootstrap_laura_plan(history, weights, [0, 1], LauraPolicy(), n_scenarios=100, random_seed=31)
    other, other_indices = bootstrap_laura_plan(history, weights, [0, 1], LauraPolicy(), n_scenarios=100, random_seed=31)
    np.testing.assert_array_equal(indices, other_indices)
    np.testing.assert_array_equal(p.facility, other.facility)
    expected = project_laura_plan(np.einsum("sta,ta->st", history[indices[:, :6]], weights),
                                  history[indices[:, 6:], 1], LauraPolicy())
    np.testing.assert_array_equal(p.portfolio_2033, expected.portfolio_2033)
    assert not indices.flags.writeable
    assert not p.portfolio_2033.flags.writeable


@pytest.mark.parametrize("changes", [
    {"confidence_target": 1}, {"confidence_target": float("nan")},
    {"facility_surplus_share": 1.1}, {"reserve_discount_rate": -1},
    {"minimum_flexibility": -1}, {"reserve_fee": 1}, {"accumulation_fee": True},
])
def test_invalid_team_assumptions_are_rejected(changes):
    with pytest.raises(ValueError):
        LauraPolicy(**changes)


def test_invalid_history_and_unfunded_asset_weights_are_rejected():
    with pytest.raises(ValueError, match="100%"):
        bootstrap_laura_plan([[.1, .02], [.02, .03]], [.8, 0], [0, 1], LauraPolicy())
    with pytest.raises(ValueError, match="finite"):
        project_laura_plan([[np.nan] * 6], [[0] * 9])
    with pytest.raises(ValueError, match="paired"):
        project_laura_plan([[0] * 6], [[0] * 9] * 2)
    with pytest.raises(ValueError, match="fixed case deposits"):
        project_laura_plan([[0] * 6], [[0] * 9], opening_wealth_2031=500_000)
