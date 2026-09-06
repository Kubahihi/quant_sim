from __future__ import annotations

from datetime import date, datetime
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from .constraint_sets import build_constraint_set, build_constraint_report


def _aligned_values(
    values: float | Sequence[float] | Mapping[str, float],
    symbols: Sequence[str],
    *,
    name: str,
    minimum: float = 0.0,
) -> np.ndarray:
    if isinstance(values, Mapping):
        array = np.asarray([values.get(symbol, np.nan) for symbol in symbols], dtype=float)
    elif np.isscalar(values):
        array = np.full(len(symbols), float(values), dtype=float)
    else:
        array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size != len(symbols):
        raise ValueError(f"{name} length must match symbols.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must provide a finite value for every symbol.")
    if np.any(array < minimum):
        raise ValueError(f"{name} must be at least {minimum}.")
    return array


def _optional_aligned_mapping(
    values: Optional[Mapping[str, float]],
    symbols: Sequence[str],
) -> np.ndarray:
    if values is None:
        return np.full(len(symbols), np.nan, dtype=float)
    return np.asarray([float(values.get(symbol, np.nan)) for symbol in symbols], dtype=float)


def estimate_trade_costs(
    trade_weights: Sequence[float] | np.ndarray,
    symbols: Sequence[str],
    *,
    portfolio_value: float,
    transaction_cost_bps: float | Sequence[float] | Mapping[str, float] = 0.0,
    half_spread_bps: float | Sequence[float] | Mapping[str, float] = 0.0,
    market_impact_bps: float | Sequence[float] | Mapping[str, float] = 0.0,
    average_daily_dollar_volume: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Estimate one-way commission, spread, and square-root market impact."""
    names = [str(symbol) for symbol in symbols]
    trades = np.asarray(trade_weights, dtype=float)
    if trades.ndim != 1 or trades.size != len(names) or not np.all(np.isfinite(trades)):
        raise ValueError("trade_weights must be a finite vector matching symbols.")
    capital = float(portfolio_value)
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("portfolio_value must be positive.")

    commission_bps = _aligned_values(
        transaction_cost_bps, names, name="transaction_cost_bps"
    )
    spread_bps = _aligned_values(
        half_spread_bps, names, name="half_spread_bps"
    )
    impact_bps = _aligned_values(
        market_impact_bps, names, name="market_impact_bps"
    )
    adv = _optional_aligned_mapping(average_daily_dollar_volume, names)
    invalid_adv = np.isfinite(adv) & (adv <= 0)
    if np.any(invalid_adv):
        raise ValueError("average_daily_dollar_volume values must be positive.")

    absolute_weights = np.abs(trades)
    missing_impact = (absolute_weights > 1e-12) & (impact_bps > 0) & (~np.isfinite(adv) | (adv <= 0))
    if np.any(missing_impact):
        missing_symbols = [names[index] for index in np.flatnonzero(missing_impact)]
        raise ValueError("Market impact cannot be estimated without positive ADV for traded symbols: " + ", ".join(missing_symbols))
    notionals = absolute_weights * capital
    participation = np.divide(
        notionals,
        adv,
        out=np.full(len(names), np.nan, dtype=float),
        where=np.isfinite(adv) & (adv > 0),
    )
    impact_rate = np.zeros(len(names), dtype=float)
    modeled = np.isfinite(participation)
    impact_rate[modeled] = (
        impact_bps[modeled] / 10_000.0 * np.sqrt(participation[modeled])
    )

    commission_drag_by_asset = absolute_weights * commission_bps / 10_000.0
    spread_drag_by_asset = absolute_weights * spread_bps / 10_000.0
    impact_drag_by_asset = absolute_weights * impact_rate
    total_by_asset = (
        commission_drag_by_asset + spread_drag_by_asset + impact_drag_by_asset
    )
    return {
        "symbols": names,
        "notional_by_asset": notionals,
        "participation_by_asset": participation,
        "commission_drag_by_asset": commission_drag_by_asset,
        "spread_drag_by_asset": spread_drag_by_asset,
        "market_impact_drag_by_asset": impact_drag_by_asset,
        "total_drag_by_asset": total_by_asset,
        "commission_drag": float(commission_drag_by_asset.sum()),
        "spread_drag": float(spread_drag_by_asset.sum()),
        "market_impact_drag": float(impact_drag_by_asset.sum()),
        "total_drag": float(total_by_asset.sum()),
        "unmodeled_impact_symbols": [
            symbol
            for symbol, rate, is_modeled, weight in zip(
                names, impact_bps, modeled, absolute_weights, strict=False
            )
            if weight > 1e-12 and rate > 0 and not is_modeled
        ],
    }


def parse_tax_lots(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Parse a long-form tax-lot table with case-insensitive column names."""
    if frame is None or frame.empty:
        return {}
    normalized = {
        str(column).strip().casefold().replace(" ", "_"): column
        for column in frame.columns
    }

    def required(*aliases: str) -> Any:
        for alias in aliases:
            if alias in normalized:
                return normalized[alias]
        raise ValueError(f"tax lots are missing column: {aliases[0]}.")

    ticker_column = required("ticker", "symbol")
    shares_column = required("shares", "quantity")
    basis_column = required(
        "cost_basis_per_share", "cost_basis", "unit_cost", "purchase_price"
    )
    acquired_column = required("acquired_at", "acquisition_date", "purchase_date")

    lots: dict[str, list[dict[str, Any]]] = {}
    for _, row in frame.iterrows():
        symbol = str(row[ticker_column] or "").strip().upper()
        if not symbol:
            continue
        shares = float(row[shares_column])
        basis = float(row[basis_column])
        acquired = pd.Timestamp(row[acquired_column]).date().isoformat()
        if not np.isfinite(shares) or shares <= 0:
            raise ValueError(f"tax-lot shares for {symbol} must be positive.")
        if not np.isfinite(basis) or basis < 0:
            raise ValueError(f"tax-lot cost basis for {symbol} must be non-negative.")
        lots.setdefault(symbol, []).append({
            "shares": shares,
            "cost_basis_per_share": basis,
            "acquired_at": acquired,
        })
    return lots


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _allocate_tax_lots(
    lots: Sequence[Mapping[str, Any]],
    *,
    shares_to_sell: float,
    sale_price: float,
    as_of: date,
    short_term_tax_rate: float,
    long_term_tax_rate: float,
) -> dict[str, Any]:
    remaining = float(shares_to_sell)
    allocations: list[dict[str, Any]] = []
    def estimated_tax_per_share(item: Mapping[str, Any]) -> tuple[float, float]:
        basis = float(item.get("cost_basis_per_share", 0.0))
        acquired_at = _as_date(item.get("acquired_at"))
        long_term = (as_of - acquired_at).days > 365
        rate = long_term_tax_rate if long_term else short_term_tax_rate
        return ((sale_price - basis) * rate, -basis)

    # Lowest estimated tax per sold share first. This naturally harvests the
    # most valuable losses before choosing low-tax gains.
    ordered = sorted(
        lots,
        key=estimated_tax_per_share,
    )
    realized_gain = 0.0
    estimated_tax = 0.0
    for lot in ordered:
        available = max(0.0, float(lot.get("shares", 0.0)))
        quantity = min(available, remaining)
        if quantity <= 1e-12:
            continue
        basis = float(lot.get("cost_basis_per_share", 0.0))
        acquired_at = _as_date(lot.get("acquired_at"))
        long_term = (as_of - acquired_at).days > 365
        gain = (sale_price - basis) * quantity
        rate = long_term_tax_rate if long_term else short_term_tax_rate
        tax = gain * rate
        allocations.append({
            "shares": quantity,
            "cost_basis_per_share": basis,
            "acquired_at": acquired_at.isoformat(),
            "holding_period": "long_term" if long_term else "short_term",
            "realized_gain": gain,
            "estimated_tax": tax,
        })
        realized_gain += gain
        estimated_tax += tax
        remaining -= quantity
        if remaining <= 1e-12:
            break
    return {
        "allocations": allocations,
        "covered_shares": float(shares_to_sell - remaining),
        "uncovered_shares": float(max(0.0, remaining)),
        "realized_gain": float(realized_gain),
        "estimated_tax": float(estimated_tax),
    }


def build_execution_plan(
    symbols: Sequence[str],
    target_weights: Sequence[float] | np.ndarray,
    *,
    prices: Mapping[str, float],
    portfolio_value: float,
    current_weights: Optional[Sequence[float] | np.ndarray] = None,
    current_shares: Optional[Mapping[str, float]] = None,
    lot_sizes: float | Sequence[float] | Mapping[str, float] = 1.0,
    minimum_trade_value: float = 0.0,
    maximum_holdings: Optional[int] = None,
    minimum_holdings: Optional[int] = None,
    average_daily_dollar_volume: Optional[Mapping[str, float]] = None,
    maximum_adv_participation: Optional[float] = None,
    transaction_cost_bps: float | Sequence[float] | Mapping[str, float] = 0.0,
    half_spread_bps: float | Sequence[float] | Mapping[str, float] = 0.0,
    market_impact_bps: float | Sequence[float] | Mapping[str, float] = 0.0,
    tax_lots: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    short_term_tax_rate: float = 0.0,
    long_term_tax_rate: float = 0.0,
    as_of: Optional[date] = None,
    strategy: Optional[Mapping[str, Any]] = None,
    asset_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    max_weight: Optional[float] = None,
    turnover_limit: Optional[float] = None,
    target_volatility: Optional[float] = None,
    annualized_covariance: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Translate continuous weights into a cash- and liquidity-feasible trade list."""
    names = [str(symbol).strip() for symbol in symbols]
    if not names or len(set(names)) != len(names):
        raise ValueError("symbols must be non-empty and unique.")
    target = np.asarray(target_weights, dtype=float)
    if target.ndim != 1 or target.size != len(names):
        raise ValueError("target_weights length must match symbols.")
    if not np.all(np.isfinite(target)) or np.any(target < -1e-10):
        raise ValueError("target_weights must be finite and non-negative.")
    if not np.isclose(float(target.sum()), 1.0, atol=1e-6):
        raise ValueError("target_weights must sum to one.")

    capital = float(portfolio_value)
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("portfolio_value must be positive.")
    price = _aligned_values(prices, names, name="prices", minimum=np.finfo(float).tiny)
    lots = _aligned_values(lot_sizes, names, name="lot_sizes", minimum=np.finfo(float).tiny)
    minimum_notional = float(minimum_trade_value)
    if not np.isfinite(minimum_notional) or minimum_notional < 0:
        raise ValueError("minimum_trade_value must be non-negative.")
    if maximum_adv_participation is not None:
        maximum_adv_participation = float(maximum_adv_participation)
        if not np.isfinite(maximum_adv_participation) or not 0 < maximum_adv_participation <= 1:
            raise ValueError("maximum_adv_participation must be in (0, 1].")
    for rate_name, rate in (
        ("short_term_tax_rate", short_term_tax_rate),
        ("long_term_tax_rate", long_term_tax_rate),
    ):
        if not np.isfinite(float(rate)) or not 0 <= float(rate) <= 1:
            raise ValueError(f"{rate_name} must be between 0 and 1.")

    warnings: list[str] = []
    if current_shares is not None:
        current = np.asarray(
            [float(current_shares.get(symbol, 0.0)) for symbol in names],
            dtype=float,
        )
        if not np.all(np.isfinite(current)) or np.any(current < 0):
            raise ValueError("current_shares must be finite and non-negative.")
    elif current_weights is not None:
        weights = np.asarray(current_weights, dtype=float)
        if weights.ndim != 1 or weights.size != len(names):
            raise ValueError("current_weights length must match symbols.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("current_weights must be finite and non-negative.")
        total_weight = float(weights.sum())
        if total_weight <= 0:
            current = np.zeros(len(names), dtype=float)
        else:
            if total_weight > 1.0 + 1e-6:
                raise ValueError("current_weights cannot exceed one including residual cash.")
            current = np.floor((weights * capital / price) / lots) * lots
            warnings.append(
                "Current shares were inferred from weights and rounded down to lot sizes."
            )
    else:
        current = np.zeros(len(names), dtype=float)

    current_market_value = float(current @ price)
    if current_market_value > capital + 1e-6:
        raise ValueError("current shares exceed portfolio_value at supplied prices.")
    cash = capital - current_market_value

    rules_for_holdings = dict(strategy or {})
    maximum_candidates = [int(value) for value in (maximum_holdings, rules_for_holdings.get("max_holdings")) if value is not None]
    minimum_candidates = [int(value) for value in (minimum_holdings, rules_for_holdings.get("min_holdings")) if value is not None]
    maximum_holdings = min(maximum_candidates) if maximum_candidates else None
    minimum_holdings = max(minimum_candidates) if minimum_candidates else None
    positive_indices = [index for index, weight in enumerate(target) if weight > 1e-10]
    max_count = len(positive_indices) if maximum_holdings is None else int(maximum_holdings)
    if max_count < 1:
        raise ValueError("maximum_holdings must be positive.")
    if minimum_holdings is not None:
        minimum_count = int(minimum_holdings)
        if minimum_count < 1:
            raise ValueError("minimum_holdings must be positive.")
        if minimum_count > max_count:
            raise ValueError("minimum_holdings cannot exceed maximum_holdings.")
        if len(positive_indices) < minimum_count:
            raise ValueError("target portfolio has fewer assets than minimum_holdings.")
    selected = set(
        sorted(positive_indices, key=lambda index: (-target[index], names[index]))[:max_count]
    )
    executable_target = target.copy()
    for index in range(len(names)):
        if index not in selected:
            executable_target[index] = 0.0
    selected_total = float(executable_target.sum())
    if selected_total <= 0:
        raise ValueError("holding-count constraint removed every target asset.")
    executable_target /= selected_total
    if len(selected) < len(positive_indices):
        warnings.append(
            f"Continuous target was restricted to the {len(selected)} largest positions for execution."
        )

    desired = np.floor((executable_target * capital / price) / lots) * lots
    adv = _optional_aligned_mapping(average_daily_dollar_volume, names)
    liquidity_violations: list[str] = []

    def liquidity_cap(index: int, requested_shares: float) -> float:
        if maximum_adv_participation is None:
            return requested_shares
        if not np.isfinite(adv[index]) or adv[index] <= 0:
            if abs(requested_shares) > 1e-12:
                violation = f"{names[index]} has no ADV estimate; the requested liquidity limit cannot be verified."
                warnings.append(violation)
                liquidity_violations.append(violation)
            return 0.0
        maximum_shares = math.floor(
            (adv[index] * maximum_adv_participation / price[index]) / lots[index]
        ) * lots[index]
        return math.copysign(
            min(abs(requested_shares), max(0.0, maximum_shares)),
            requested_shares,
        )

    planned = current.copy()
    trades: list[dict[str, Any]] = []
    tax_lot_map = {str(key).upper(): value for key, value in (tax_lots or {}).items()}
    tax_date = as_of or date.today()

    def trade_cost(index: int, trade_shares: float) -> dict[str, Any]:
        trade_weight = np.zeros(len(names), dtype=float)
        trade_weight[index] = trade_shares * price[index] / capital
        return estimate_trade_costs(
            trade_weight,
            names,
            portfolio_value=capital,
            transaction_cost_bps=transaction_cost_bps,
            half_spread_bps=half_spread_bps,
            market_impact_bps=market_impact_bps,
            average_daily_dollar_volume=average_daily_dollar_volume,
        )

    def append_trade(index: int, shares: float, costs: dict[str, Any]) -> None:
        nonlocal cash
        notional = abs(shares) * price[index]
        cost_dollars = float(costs["total_drag"]) * capital
        tax_result = {
            "allocations": [],
            "covered_shares": 0.0,
            "uncovered_shares": 0.0,
            "realized_gain": 0.0,
            "estimated_tax": 0.0,
        }
        if shares < 0:
            symbol_lots = tax_lot_map.get(names[index].upper(), [])
            if symbol_lots:
                tax_result = _allocate_tax_lots(
                    symbol_lots,
                    shares_to_sell=abs(shares),
                    sale_price=price[index],
                    as_of=tax_date,
                    short_term_tax_rate=float(short_term_tax_rate),
                    long_term_tax_rate=float(long_term_tax_rate),
                )
                if tax_result["uncovered_shares"] > 1e-8:
                    warnings.append(
                        f"Tax lots do not cover the full planned sale of {names[index]}."
                    )
            elif tax_lots is not None:
                warnings.append(f"No tax lots were supplied for the sale of {names[index]}.")
            cash += notional - cost_dollars
        else:
            cash -= notional + cost_dollars
        participation = float(costs["participation_by_asset"][index])
        trades.append({
            "symbol": names[index],
            "side": "BUY" if shares > 0 else "SELL",
            "shares": float(abs(shares)),
            "signed_shares": float(shares),
            "price": float(price[index]),
            "notional": float(notional),
            "adv_dollars": float(adv[index]) if np.isfinite(adv[index]) else None,
            "adv_participation": participation if np.isfinite(participation) else None,
            "commission_cost": float(costs["commission_drag"]) * capital,
            "spread_cost": float(costs["spread_drag"]) * capital,
            "market_impact_cost": float(costs["market_impact_drag"]) * capital,
            "total_execution_cost": cost_dollars,
            "realized_gain": float(tax_result["realized_gain"]),
            "estimated_tax": float(tax_result["estimated_tax"]),
            "tax_lot_allocations": tax_result["allocations"],
        })

    # Sales create the cash that can fund buys.
    for index in range(len(names)):
        requested = min(0.0, desired[index] - current[index])
        requested = liquidity_cap(index, requested)
        requested = math.ceil(requested / lots[index]) * lots[index]
        if abs(requested) * price[index] < minimum_notional - 1e-9:
            requested = 0.0
        if requested < -1e-12:
            costs = trade_cost(index, requested)
            planned[index] += requested
            append_trade(index, requested, costs)

    # Buy the largest target weights first and never spend cash that is not available.
    for index in sorted(selected, key=lambda item: (-executable_target[item], names[item])):
        requested = max(0.0, desired[index] - planned[index])
        requested = liquidity_cap(index, requested)
        requested = math.floor(requested / lots[index]) * lots[index]
        if requested * price[index] < minimum_notional - 1e-9:
            continue
        maximum_lots = max(0, int(math.floor(requested / lots[index] + 1e-10)))
        affordable_lots = 0
        lower, upper = 0, maximum_lots
        while lower <= upper:
            midpoint = (lower + upper) // 2
            candidate = midpoint * lots[index]
            costs = trade_cost(index, candidate)
            required_cash = candidate * price[index] + float(costs["total_drag"]) * capital
            if required_cash <= cash + 1e-8:
                affordable_lots = midpoint
                lower = midpoint + 1
            else:
                upper = midpoint - 1
        requested = affordable_lots * lots[index]
        if requested * price[index] < minimum_notional - 1e-9:
            continue
        if requested > 1e-12:
            costs = trade_cost(index, requested)
            planned[index] += requested
            append_trade(index, requested, costs)

    ending_asset_values = planned * price
    ending_total = float(ending_asset_values.sum() + max(cash, 0.0))
    if ending_total <= 0:
        raise ValueError("execution plan produced a non-positive ending value.")
    final_weights = ending_asset_values / ending_total
    cash_weight = max(cash, 0.0) / ending_total
    holding_count = int(np.count_nonzero(planned > 1e-10))
    holding_constraint_violations: list[str] = []
    if holding_count > max_count:
        violation = (
            "Liquidity or minimum-trade limits prevented the requested maximum holding count."
        )
        warnings.append(violation)
        holding_constraint_violations.append(violation)
    if minimum_holdings is not None and holding_count < int(minimum_holdings):
        violation = (
            "Lot sizes or available cash prevented the requested minimum holding count."
        )
        warnings.append(violation)
        holding_constraint_violations.append(violation)

    total_execution_cost = float(sum(row["total_execution_cost"] for row in trades))
    total_tax = float(sum(row["estimated_tax"] for row in trades))
    total_realized_gain = float(sum(row["realized_gain"] for row in trades))
    mandate_report: list[dict[str, Any]] = []
    mandate_errors: list[str] = []
    if strategy is not None or max_weight is not None or turnover_limit is not None:
        # An explicit residual cash asset keeps sector/beta/cash checks on the
        # actual post-cost NAV, without renormalizing away uninvested money.
        residual_symbol = "__EXECUTION_RESIDUAL_CASH__"
        metadata = {**dict(asset_metadata or {}), residual_symbol: {"is_cash": True, "asset_type": "cash", "beta": 0.0}}
        rules = dict(strategy or {})
        effective_turnover = min([float(v) for v in (turnover_limit, rules.pop("max_turnover", None)) if v is not None], default=None)
        try:
            spec = build_constraint_set([*names, residual_symbol], strategy=rules,
                                        asset_metadata=metadata, max_weight=max_weight)
            mandate_report = build_constraint_report(np.append(final_weights, cash_weight), spec, binding_tolerance=2e-5)
            if effective_turnover is not None:
                actual_turnover = sum(row["notional"] for row in trades) / capital
                mandate_report.append({"name": "turnover", "actual": actual_turnover,
                    "maximum": effective_turnover, "minimum": None,
                    "passed": actual_turnover <= effective_turnover + 2e-5})
            mandate_errors = [f"Execution violates {row['name']}." for row in mandate_report if not row["passed"]]
        except ValueError as exc:
            mandate_errors = [str(exc)]
    if target_volatility is not None:
        limit = float(target_volatility)
        covariance = np.asarray(annualized_covariance, dtype=float)
        if not np.isfinite(limit) or limit <= 0:
            mandate_errors.append("Target volatility must be finite and positive.")
        elif covariance.shape != (len(names), len(names)) or not np.isfinite(covariance).all():
            mandate_errors.append("Target volatility requires aligned, finite annualized covariance.")
        elif not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() < -1e-10:
            mandate_errors.append("Annualized covariance must be symmetric and positive semidefinite.")
        else:
            actual_volatility = float(np.sqrt(max(float(final_weights @ covariance @ final_weights), 0.0)))
            passed = actual_volatility <= limit + 2e-5
            mandate_report.append({"name": "portfolio_volatility", "actual": actual_volatility,
                                   "maximum": limit, "minimum": None, "passed": passed})
            if not passed:
                mandate_errors.append("Execution violates target volatility.")
    violations = list(dict.fromkeys(holding_constraint_violations + liquidity_violations + mandate_errors))
    return {
        "success": not violations,
        "message": (
            "ok"
            if not violations
            else " ".join(violations)
        ),
        "symbols": names,
        "continuous_target_weights": target,
        "execution_target_weights": executable_target,
        "final_shares": planned,
        "final_weights": final_weights,
        "cash": float(max(cash, 0.0)),
        "cash_weight": float(cash_weight),
        "holding_count": holding_count,
        "holding_constraints_satisfied": not holding_constraint_violations,
        "mandate_constraints_satisfied": not mandate_errors,
        "liquidity_constraints_satisfied": not liquidity_violations,
        "constraint_report": mandate_report,
        "trades": trades,
        "turnover": float(sum(row["notional"] for row in trades) / capital),
        "total_execution_cost": total_execution_cost,
        "total_execution_cost_drag": total_execution_cost / capital,
        "realized_gain": total_realized_gain,
        "estimated_tax": total_tax,
        "tracking_difference_l1": float(np.sum(np.abs(final_weights - target)) + cash_weight),
        "warnings": list(dict.fromkeys(warnings)),
        "assumptions": {
            "market_impact_model": "square_root_participation",
            "tax_lot_selection": "minimum_estimated_tax_per_share_first",
            "whole_lot_execution": True,
            "portfolio_value": capital,
            "minimum_trade_value": minimum_notional,
            "maximum_adv_participation": maximum_adv_participation,
            "maximum_holdings": maximum_holdings,
            "minimum_holdings": minimum_holdings,
        },
    }
