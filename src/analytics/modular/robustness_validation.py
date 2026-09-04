from __future__ import annotations

from numbers import Integral, Real
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from src.analytics.returns import calculate_annualized_return

TRADING_DAYS = 252


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def _validated_index(index: pd.Index) -> pd.Index:
    index = pd.Index(index)
    if not index.is_unique:
        raise ValueError("returns index must be unique.")
    if not index.is_monotonic_increasing:
        raise ValueError("returns index must be monotonically increasing.")
    return index


def _validated_simple_returns(portfolio_returns: pd.Series) -> pd.Series:
    try:
        returns = pd.Series(portfolio_returns).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("portfolio_returns must contain numeric simple returns.") from exc

    _validated_index(returns.index)
    values = returns.to_numpy(dtype=float, copy=False)
    if not bool(np.isfinite(values).all()):
        raise ValueError("portfolio_returns must contain only finite simple returns.")
    if bool((values <= -1.0).any()):
        raise ValueError("Simple returns must be greater than -100%.")
    return returns


def _annual_to_periodic_rate(risk_free_rate: float) -> float:
    annual_rate = _finite_real(risk_free_rate, "risk_free_rate")
    if annual_rate <= -1.0:
        raise ValueError("risk_free_rate must be greater than -100%.")
    return float(np.expm1(np.log1p(annual_rate) / TRADING_DAYS))


def _annualized_sharpe(returns: pd.Series, risk_free_rate: float) -> float:
    if len(returns) < 2:
        return 0.0
    std_dev = float(returns.std())
    if not np.isfinite(std_dev) or std_dev <= 0.0:
        return 0.0
    periodic_rf = _annual_to_periodic_rate(risk_free_rate)
    return float((float(returns.mean()) - periodic_rf) / std_dev * np.sqrt(TRADING_DAYS))


def _display_index_value(value: object) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def create_walk_forward_splits(
    index: pd.Index,
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[Tuple[pd.Index, pd.Index]]:
    """Split an ordered index into rolling windows by observation count.

    The ``*_days`` names are retained for API compatibility, but each value is
    a number of observations, not elapsed calendar days.
    """
    train_observations = _positive_integer(train_days, "train_days")
    test_observations = _positive_integer(test_days, "test_days")
    step_observations = _positive_integer(step_days, "step_days")
    index = _validated_index(index)

    splits: List[Tuple[pd.Index, pd.Index]] = []
    train_start = 0
    while train_start + train_observations + test_observations <= len(index):
        train_end = train_start + train_observations
        test_end = train_end + test_observations
        splits.append((index[train_start:train_end], index[train_end:test_end]))
        train_start += step_observations
    return splits


def calculate_psr(
    returns: pd.Series,
    benchmark_sr: float = 0.0,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate the Probabilistic Sharpe Ratio.

    ``risk_free_rate`` is an effective annual rate and is converted to the
    periodic frequency of the daily returns. ``benchmark_sr`` remains a
    periodic Sharpe ratio, matching the PSR formula.
    """
    returns = _validated_simple_returns(returns)
    benchmark_sr = _finite_real(benchmark_sr, "benchmark_sr")
    periodic_rf = _annual_to_periodic_rate(risk_free_rate)
    if len(returns) < 3:
        return 0.0

    std_dev = float(returns.std())
    if not np.isfinite(std_dev) or std_dev <= 0.0:
        return 0.0

    sr_periodic = (float(returns.mean()) - periodic_rf) / std_dev
    skewness = float(returns.skew())
    excess_kurtosis = float(returns.kurtosis())
    if not np.isfinite(skewness) or not np.isfinite(excess_kurtosis):
        return 0.0

    pearson_kurtosis = excess_kurtosis + 3.0
    variance_term = (
        1.0
        - skewness * sr_periodic
        + ((pearson_kurtosis - 1.0) / 4.0) * (sr_periodic**2)
    )
    denominator = np.sqrt(max(1e-10, variance_term))
    psr_stat = ((sr_periodic - benchmark_sr) * np.sqrt(len(returns) - 1)) / denominator
    return float(stats.norm.cdf(psr_stat))


def calculate_dsr(
    returns: pd.Series,
    num_trials: int,
    variance_trials: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate the Deflated Sharpe Ratio from genuine trial-level inputs.

    ``variance_trials`` is the variance of periodic Sharpe ratios across all
    strategy trials. Merely splitting one fixed return series does not create
    such trials.
    """
    num_trials = _positive_integer(num_trials, "num_trials")
    variance_trials = _finite_real(variance_trials, "variance_trials")
    if variance_trials < 0.0:
        raise ValueError("variance_trials must be non-negative.")
    if num_trials == 1:
        return calculate_psr(returns, benchmark_sr=0.0, risk_free_rate=risk_free_rate)

    euler_gamma = np.euler_gamma
    z1 = float(stats.norm.ppf(1.0 - 1.0 / num_trials))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (num_trials * np.e)))
    expected_max_sr = np.sqrt(variance_trials) * (
        (1.0 - euler_gamma) * z1 + euler_gamma * z2
    )
    return calculate_psr(
        returns,
        benchmark_sr=float(expected_max_sr),
        risk_free_rate=risk_free_rate,
    )


def run_walk_forward_validation(
    portfolio_returns: pd.Series,
    train_days: int = 1095,
    test_days: int = 180,
    step_days: int = 90,
    num_trials: int = 1,
    risk_free_rate: float = 0.0,
) -> Dict[str, object]:
    """Describe rolling segments of one fixed, already-realized return series.

    Despite the legacy function name, this routine does not fit or refit a
    strategy. Its evaluation segments therefore are not evidence of a genuine
    walk-forward strategy process and are not labelled as such in canonical
    result fields.
    """
    portfolio_returns = _validated_simple_returns(portfolio_returns)
    train_days = _positive_integer(train_days, "train_days")
    test_days = _positive_integer(test_days, "test_days")
    step_days = _positive_integer(step_days, "step_days")
    num_trials = _positive_integer(num_trials, "num_trials")
    _annual_to_periodic_rate(risk_free_rate)
    splits = create_walk_forward_splits(
        portfolio_returns.index,
        train_days,
        test_days,
        step_days,
    )

    windows = []
    evaluation_returns_list = []
    for i, (reference_idx, evaluation_idx) in enumerate(splits):
        reference_returns = portfolio_returns.loc[reference_idx]
        evaluation_returns = portfolio_returns.loc[evaluation_idx]

        reference_sharpe = _annualized_sharpe(reference_returns, risk_free_rate)
        evaluation_sharpe = _annualized_sharpe(evaluation_returns, risk_free_rate)
        reference_annualized_return = calculate_annualized_return(reference_returns, TRADING_DAYS)
        evaluation_annualized_return = calculate_annualized_return(
            evaluation_returns,
            TRADING_DAYS,
        )

        windows.append(
            {
                "window_id": i + 1,
                "reference_start": _display_index_value(reference_idx[0]),
                "reference_end": _display_index_value(reference_idx[-1]),
                "evaluation_start": _display_index_value(evaluation_idx[0]),
                "evaluation_end": _display_index_value(evaluation_idx[-1]),
                "reference_sharpe": reference_sharpe,
                "evaluation_sharpe": evaluation_sharpe,
                "reference_return": reference_annualized_return,
                "evaluation_return": evaluation_annualized_return,
            }
        )
        evaluation_returns_list.append(evaluation_returns)

    scope = (
        "Rolling segmentation diagnostics for one fixed return series; no strategy fitting "
        "or refitting is performed, so this is not a strategy OOS validation."
    )
    dsr_text = (
        "Not calculated: DSR requires Sharpe ratios from the complete set of genuine strategy "
        "trials; rolling segments of one fixed return series are not trials."
    )
    if not evaluation_returns_list:
        aggregate_evaluation_returns = pd.Series(dtype=float)
        metrics = {
            "psr": 0.0,
            "dsr": None,
            "evaluation_sharpe": 0.0,
            "evaluation_annualized_return": 0.0,
            "psr_interpretation": "Not enough observations for rolling evaluation segments.",
            "dsr_interpretation": dsr_text,
        }
        return {
            "validation_type": "rolling_fixed_returns_segmentation",
            "strategy_refit_performed": False,
            "supports_strategy_oos_claim": False,
            "scope": scope,
            "requested_num_trials": num_trials,
            "windows": [],
            "aggregate_evaluation_returns": aggregate_evaluation_returns,
            "metrics": metrics,
        }

    aggregate_evaluation_returns = pd.concat(evaluation_returns_list)
    aggregate_evaluation_returns = aggregate_evaluation_returns[
        ~aggregate_evaluation_returns.index.duplicated(keep="first")
    ].sort_index()

    evaluation_sharpe = _annualized_sharpe(aggregate_evaluation_returns, risk_free_rate)
    evaluation_annualized_return = calculate_annualized_return(
        aggregate_evaluation_returns,
        TRADING_DAYS,
    )
    psr = calculate_psr(aggregate_evaluation_returns, risk_free_rate=risk_free_rate)

    psr_pct = psr * 100.0
    if psr > 0.95:
        psr_text = f"Strong confidence ({psr_pct:.1f}%) that the segmented-return Sharpe is > 0."
    elif psr > 0.8:
        psr_text = (
            f"Moderate confidence ({psr_pct:.1f}%) that the segmented-return Sharpe is > 0."
        )
    else:
        psr_text = f"Low confidence ({psr_pct:.1f}%) that the segmented-return Sharpe is > 0."

    metrics = {
        "psr": psr,
        "dsr": None,
        "evaluation_sharpe": evaluation_sharpe,
        "evaluation_annualized_return": evaluation_annualized_return,
        "psr_interpretation": psr_text,
        "dsr_interpretation": dsr_text,
    }
    return {
        "validation_type": "rolling_fixed_returns_segmentation",
        "strategy_refit_performed": False,
        "supports_strategy_oos_claim": False,
        "scope": scope,
        "requested_num_trials": num_trials,
        "windows": windows,
        "aggregate_evaluation_returns": aggregate_evaluation_returns,
        "metrics": metrics,
    }
