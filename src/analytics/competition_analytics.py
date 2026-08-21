"""Auditable quantitative building blocks for competition analysis."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd


def calculate_policy_benchmark(
    component_returns: pd.DataFrame,
    component_weights: Mapping[str, float],
) -> pd.Series:
    """Return a daily rebalanced policy benchmark from explicit components."""
    if component_returns.empty:
        raise ValueError("Component returns must not be empty.")
    missing = sorted(set(component_weights) - set(component_returns.columns))
    if missing:
        raise ValueError(f"Missing benchmark components: {', '.join(missing)}")
    weights = pd.Series(component_weights, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Benchmark weights must be finite, non-negative, and have a positive sum.")
    weights = weights / weights.sum()
    selected = component_returns.loc[:, weights.index].astype(float)
    if not np.isfinite(selected.to_numpy()).all():
        raise ValueError("Component returns must contain only finite values.")
    result = selected.mul(weights, axis=1).sum(axis=1)
    result.name = "policy_benchmark_return"
    return result


def calculate_brinson_attribution(
    portfolio: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    sector_col: str = "sector",
    weight_col: str = "weight",
    return_col: str = "return",
) -> pd.DataFrame:
    """Single-period Brinson-Fachler allocation, selection, and interaction."""
    required = {sector_col, weight_col, return_col}
    for name, frame in (("Portfolio", portfolio), ("Benchmark", benchmark)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")
    p = portfolio.groupby(sector_col, dropna=False).apply(
        lambda x: pd.Series({"wp": x[weight_col].sum(), "rp": np.average(x[return_col], weights=x[weight_col]) if x[weight_col].sum() else 0.0}),
        include_groups=False,
    )
    b = benchmark.groupby(sector_col, dropna=False).apply(
        lambda x: pd.Series({"wb": x[weight_col].sum(), "rb": np.average(x[return_col], weights=x[weight_col]) if x[weight_col].sum() else 0.0}),
        include_groups=False,
    )
    result = p.join(b, how="outer").fillna(0.0)
    for column in ("wp", "rp", "wb", "rb"):
        if not np.isfinite(result[column]).all():
            raise ValueError("Weights and returns must contain only finite values.")
    if result["wp"].sum() <= 0 or result["wb"].sum() <= 0:
        raise ValueError("Portfolio and benchmark weights must each have a positive sum.")
    result["wp"] /= result["wp"].sum()
    result["wb"] /= result["wb"].sum()
    benchmark_total = float((result["wb"] * result["rb"]).sum())
    result["allocation"] = (result["wp"] - result["wb"]) * (result["rb"] - benchmark_total)
    result["selection"] = result["wb"] * (result["rp"] - result["rb"])
    result["interaction"] = (result["wp"] - result["wb"]) * (result["rp"] - result["rb"])
    result["total_effect"] = result[["allocation", "selection", "interaction"]].sum(axis=1)
    return result.reset_index().rename(columns={sector_col: "sector"})


def run_walk_forward_rank_backtest(
    returns: pd.DataFrame,
    point_in_time_scores: pd.DataFrame,
    *,
    top_n: int = 5,
    rebalance_every: int = 21,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Rank on information available at t and apply weights only from t+1.

    This deliberately accepts precomputed point-in-time scores. Model fitting and
    score calibration therefore remain outside the function and must be frozen in
    the accompanying manifest.
    """
    if top_n <= 0 or rebalance_every <= 0 or transaction_cost_bps < 0:
        raise ValueError("top_n and rebalance_every must be positive; costs cannot be negative.")
    common_index = returns.index.intersection(point_in_time_scores.index).sort_values()
    common_columns = returns.columns.intersection(point_in_time_scores.columns)
    if len(common_index) < 2 or common_columns.empty:
        raise ValueError("Returns and scores need at least two shared dates and one shared asset.")
    r = returns.loc[common_index, common_columns].astype(float)
    scores = point_in_time_scores.loc[common_index, common_columns].astype(float)
    n_periods, n_assets = r.shape
    return_values = r.to_numpy(dtype=float, copy=False)
    gross = np.zeros(n_periods, dtype=float)
    turnover = np.zeros(n_periods, dtype=float)
    current = np.zeros(n_assets, dtype=float)

    # Only visit actual rebalance dates and broadcast each target over its
    # holding period.  The original row loop repeated the same N-asset write
    # on every date, which dominated larger point-in-time backtests.
    for position in range(0, n_periods - 1, rebalance_every):
        available = scores.iloc[position].dropna().sort_values(ascending=False)
        selected = available.head(min(top_n, len(available))).index
        target = np.zeros(n_assets, dtype=float)
        if len(selected):
            selected_positions = common_columns.get_indexer(selected)
            target[selected_positions] = 1.0 / len(selected)
        turnover[position + 1] = float(np.abs(target - current).sum())
        current = target
        holding_period_end = min(position + rebalance_every + 1, n_periods)
        if len(selected):
            gross[position + 1:holding_period_end] = np.nansum(
                return_values[
                    position + 1:holding_period_end,
                    selected_positions,
                ] * current[selected_positions],
                axis=1,
            )

    # np.nansum above preserves DataFrame.sum's handling of missing returns
    # without materializing a full date-by-asset weight matrix.
    costs = turnover * float(transaction_cost_bps) / 10_000.0
    return pd.DataFrame({
        "gross_return": gross,
        "turnover": turnover,
        "transaction_cost": costs,
        "net_return": gross - costs,
    }, index=common_index)


def build_reproducibility_manifest(
    data: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    source: str,
    as_of: str,
) -> dict[str, Any]:
    """Hash a canonical data snapshot and configuration for repeatable research."""
    canonical_data = data.sort_index().sort_index(axis=1).to_csv(index=True, float_format="%.12g", lineterminator="\n")
    canonical_config = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return {
        "source": str(source).strip(),
        "as_of": str(as_of).strip(),
        "rows": int(len(data)),
        "columns": [str(column) for column in data.columns],
        "data_sha256": hashlib.sha256(canonical_data.encode("utf-8")).hexdigest(),
        "config_sha256": hashlib.sha256(canonical_config.encode("utf-8")).hexdigest(),
        "config": json.loads(canonical_config),
    }


__all__ = [
    "build_reproducibility_manifest", "calculate_brinson_attribution",
    "calculate_policy_benchmark", "run_walk_forward_rank_backtest",
]
