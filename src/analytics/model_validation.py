"""Evidence and uncertainty diagnostics for QuantSim outputs.

The score in this module measures methodology readiness, not future predictive
accuracy and not affiliation with or endorsement by Wharton.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _clean_returns(returns: pd.Series) -> pd.Series:
    clean = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if bool((clean <= -1.0).any()):
        raise ValueError("Simple returns must be greater than -100%.")
    return clean


def _point_metrics(values: np.ndarray, risk_free_rate: float) -> Dict[str, float]:
    n = int(values.size)
    if n == 0:
        return {name: 0.0 for name in ("annualized_return", "volatility", "sharpe_ratio", "var_95", "cvar_95")}
    log_growth = float(np.log1p(values).sum())
    annualized_return = float(np.expm1(log_growth * TRADING_DAYS / n))
    volatility = float(np.std(values, ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    daily_rf = float(np.expm1(np.log1p(risk_free_rate) / TRADING_DAYS)) if risk_free_rate > -1 else 0.0
    excess = values - daily_rf
    excess_std = float(np.std(excess, ddof=1)) if n > 1 else 0.0
    sharpe = float(np.mean(excess) / excess_std * np.sqrt(TRADING_DAYS)) if excess_std > 0 else 0.0
    cutoff = float(np.percentile(values, 5))
    tail = values[values <= cutoff]
    return {
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "var_95": float(-cutoff),
        "cvar_95": float(-np.mean(tail)) if tail.size else float(-cutoff),
    }


def moving_block_bootstrap_intervals(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    n_bootstrap: int = 600,
    block_size: int | None = None,
    random_seed: int = 1729,
) -> Dict[str, Dict[str, float | str]]:
    """Estimate 95% intervals while retaining short-run return dependence."""
    clean = _clean_returns(returns)
    values = clean.to_numpy(dtype=float)
    n = int(values.size)
    if n < 20:
        return {}
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100.")

    block = int(block_size or max(2, round(n ** (1.0 / 3.0))))
    block = min(block, n)
    blocks_per_sample = int(np.ceil(n / block))
    rng = np.random.default_rng(random_seed)
    starts = rng.integers(0, n - block + 1, size=(n_bootstrap, blocks_per_sample))
    offsets = np.arange(block, dtype=int)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_bootstrap, -1)[:, :n]
    samples = values[indices]

    log_growth = np.log1p(samples).sum(axis=1)
    annualized_return = np.expm1(log_growth * TRADING_DAYS / n)
    volatility = np.std(samples, axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
    daily_rf = float(np.expm1(np.log1p(risk_free_rate) / TRADING_DAYS)) if risk_free_rate > -1 else 0.0
    excess = samples - daily_rf
    excess_std = np.std(excess, axis=1, ddof=1)
    sharpe = np.divide(
        np.mean(excess, axis=1) * np.sqrt(TRADING_DAYS),
        excess_std,
        out=np.zeros(n_bootstrap, dtype=float),
        where=excess_std > 0,
    )
    cutoffs = np.percentile(samples, 5, axis=1)
    cvar = np.array(
        [-float(np.mean(row[row <= cutoff])) for row, cutoff in zip(samples, cutoffs, strict=True)],
        dtype=float,
    )
    distributions = {
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "var_95": -cutoffs,
        "cvar_95": cvar,
    }
    estimates = _point_metrics(values, risk_free_rate)
    output: Dict[str, Dict[str, float | str]] = {}
    for name, distribution in distributions.items():
        finite = np.asarray(distribution, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        output[name] = {
            "estimate": float(estimates[name]),
            "ci_low": float(np.percentile(finite, 2.5)),
            "ci_high": float(np.percentile(finite, 97.5)),
            "method": f"95% moving-block bootstrap ({n_bootstrap} resamples, block={block})",
        }
    return output


def distribution_diagnostics(returns: pd.Series) -> Dict[str, float | bool]:
    """Return transparent normality and dependence diagnostics."""
    values = _clean_returns(returns).to_numpy(dtype=float)
    n = int(values.size)
    if n < 3:
        return {
            "observations": float(n),
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "jarque_bera": 0.0,
            "normality_p_value": 1.0,
            "lag1_autocorrelation": 0.0,
            "normality_rejected_5pct": False,
        }
    centered = values - float(np.mean(values))
    scale = float(np.sqrt(np.mean(centered ** 2)))
    if scale <= 0:
        skewness = excess_kurtosis = 0.0
    else:
        standardized = centered / scale
        skewness = float(np.mean(standardized ** 3))
        excess_kurtosis = float(np.mean(standardized ** 4) - 3.0)
    jarque_bera = float(n / 6.0 * (skewness ** 2 + 0.25 * excess_kurtosis ** 2))
    # For a chi-square distribution with two degrees of freedom, SF(x)=exp(-x/2).
    p_value = float(np.exp(-0.5 * jarque_bera))
    lag1 = float(np.corrcoef(values[:-1], values[1:])[0, 1]) if n > 3 else 0.0
    if not np.isfinite(lag1):
        lag1 = 0.0
    return {
        "observations": float(n),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "jarque_bera": jarque_bera,
        "normality_p_value": p_value,
        "lag1_autocorrelation": lag1,
        "normality_rejected_5pct": bool(p_value < 0.05),
    }


def _gate(name: str, status: str, points: float, maximum: float, evidence: str) -> Dict[str, Any]:
    return {
        "gate": name,
        "status": status,
        "points": float(points),
        "maximum": float(maximum),
        "evidence": evidence,
    }


def build_model_validation_report(
    portfolio_returns: pd.Series,
    simulation_stats: Dict[str, Any] | None = None,
    backtest: Dict[str, Any] | None = None,
    risk_free_rate: float = 0.03,
    n_bootstrap: int = 600,
    random_seed: int = 1729,
) -> Dict[str, Any]:
    """Build an internal, evidence-based QuantSim methodology report."""
    clean = _clean_returns(portfolio_returns)
    n = int(len(clean))
    intervals = moving_block_bootstrap_intervals(
        clean,
        risk_free_rate=risk_free_rate,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    distribution = distribution_diagnostics(clean)
    simulation = dict(simulation_stats or {})
    backtest_data = dict(backtest or {})
    gates: list[Dict[str, Any]] = []

    if n >= 756:
        data_points, data_status = 20.0, "pass"
    elif n >= 504:
        data_points, data_status = 16.0, "pass"
    elif n >= 252:
        data_points, data_status = 11.0, "warning"
    elif n >= 126:
        data_points, data_status = 6.0, "warning"
    else:
        data_points, data_status = 2.0, "fail"
    gates.append(_gate("Historical depth", data_status, data_points, 20, f"{n} daily observations ({n / TRADING_DAYS:.1f} years)"))

    interval_points = 20.0 if len(intervals) == 5 else 0.0
    gates.append(_gate(
        "Parameter uncertainty",
        "pass" if interval_points else "fail",
        interval_points,
        20,
        "Moving-block bootstrap intervals available for return, risk, Sharpe, VaR and CVaR." if interval_points else "At least 20 observations are required.",
    ))

    relative_error = simulation.get("relative_standard_error_mean")
    if relative_error is None or not np.isfinite(relative_error):
        mc_points, mc_status, mc_evidence = 0.0, "fail", "No Monte Carlo convergence diagnostic."
    else:
        relative_error = float(relative_error)
        if relative_error <= 0.005:
            mc_points, mc_status = 15.0, "pass"
        elif relative_error <= 0.01:
            mc_points, mc_status = 12.0, "pass"
        elif relative_error <= 0.02:
            mc_points, mc_status = 8.0, "warning"
        else:
            mc_points, mc_status = 3.0, "fail"
        mc_evidence = f"Terminal-mean Monte Carlo relative standard error: {relative_error:.2%}."
    gates.append(_gate("Simulation convergence", mc_status, mc_points, 15, mc_evidence))

    rejects_normality = bool(distribution["normality_rejected_5pct"])
    model_name = str(simulation.get("model", "unknown"))
    if model_name == "geometric_brownian_motion":
        distribution_points = 5.0 if rejects_normality else 9.0
        distribution_status = "warning"
        distribution_evidence = (
            "GBM is a single-model scenario and historical returns reject normality; fat-tail model risk remains."
            if rejects_normality
            else "GBM remains a single-model scenario; normality is not rejected in this sample."
        )
    else:
        distribution_points, distribution_status = 0.0, "fail"
        distribution_evidence = "Simulation model metadata is unavailable."
    gates.append(_gate("Distribution/model risk", distribution_status, distribution_points, 15, distribution_evidence))

    validation_type = str(backtest_data.get("validation_type", "unknown"))
    is_causal = bool(backtest_data.get("lookahead_safe", False))
    if is_causal and validation_type == "walk_forward_causal_baseline":
        backtest_points, backtest_status = 15.0, "warning"
        backtest_evidence = "Causal walk-forward baseline is available, but it does not validate the full ensemble."
    elif is_causal:
        backtest_points, backtest_status = 10.0, "warning"
        backtest_evidence = f"Causal flag present for {validation_type}; full-ensemble walk-forward validation is absent."
    else:
        backtest_points, backtest_status = 0.0, "fail"
        backtest_evidence = "No causal walk-forward evidence."
    gates.append(_gate("Out-of-sample process", backtest_status, backtest_points, 20, backtest_evidence))

    reproducible = simulation.get("random_seed") is not None
    gates.append(_gate(
        "Reproducibility",
        "pass" if reproducible else "warning",
        10.0 if reproducible else 4.0,
        10,
        f"Seeded simulation ({simulation.get('random_seed')})." if reproducible else "Simulation seed was not recorded.",
    ))

    score = float(sum(float(item["points"]) for item in gates))
    if score >= 85:
        band = "research_ready"
        verdict = "Strong research workflow; complete full-ensemble walk-forward validation before claiming predictive accuracy."
    elif score >= 70:
        band = "strong_decision_support"
        verdict = "Strong decision support, but not yet a validated forecasting system."
    elif score >= 55:
        band = "exploratory_decision_support"
        verdict = "Useful for structured exploration; material validation gaps remain."
    else:
        band = "prototype"
        verdict = "Prototype evidence only; do not rely on point estimates for investment decisions."

    limitations = [
        "The methodology score is not predictive hit rate and is not a Wharton endorsement.",
        "Historical estimates are sensitive to the selected sample and market regime.",
        "GBM does not model jumps, stochastic volatility, liquidity, taxes, or transaction costs.",
        "The causal backtest is a baseline; the full model ensemble still needs nested walk-forward validation.",
    ]
    if rejects_normality:
        limitations.append("Historical returns reject normality at 5%; Gaussian tail estimates may be optimistic.")
    if n < 504:
        limitations.append("Fewer than two years of aligned observations weakens parameter stability.")

    return {
        "methodology_score": score,
        "score_maximum": 100.0,
        "band": band,
        "verdict": verdict,
        "predictive_accuracy_measured": False,
        "gates": gates,
        "metric_intervals": intervals,
        "distribution": distribution,
        "limitations": limitations,
        "generated_with": {
            "bootstrap_resamples": int(n_bootstrap),
            "bootstrap_seed": int(random_seed),
            "simulation_model": model_name,
            "backtest_validation_type": validation_type,
        },
    }
