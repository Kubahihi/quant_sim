from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .constraints import WEIGHT_TOLERANCE, build_weight_bounds, validate_weight_solution


_CASH_SYMBOLS = {"CASH", "USD", "EUR", "GBP", "CZK", "JPY", "CHF"}
_CASH_ASSET_TYPES = {"cash", "currency", "money market", "money-market"}


@dataclass(frozen=True)
class GroupConstraint:
    name: str
    indices: tuple[int, ...]
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass(frozen=True)
class PortfolioConstraintSet:
    symbols: tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    groups: tuple[GroupConstraint, ...] = ()
    beta: Optional[np.ndarray] = None
    minimum_beta: Optional[float] = None
    maximum_beta: Optional[float] = None
    current_weights: Optional[np.ndarray] = None
    turnover_limit: Optional[float] = None
    warnings: tuple[str, ...] = ()


def _normalise_text(value: Any) -> str:
    return str(value or "").strip().casefold()


def _metadata_for_symbol(
    asset_metadata: Mapping[str, Mapping[str, Any]],
    symbol: str,
) -> Mapping[str, Any]:
    direct = asset_metadata.get(symbol)
    if isinstance(direct, Mapping):
        return direct
    target = symbol.casefold()
    for key, value in asset_metadata.items():
        if str(key).casefold() == target and isinstance(value, Mapping):
            return value
    return {}


def _normalise_optional_weight(value: Any, name: str) -> Optional[float]:
    if value is None or value == "":
        return None
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return result


def _normalise_current_weights(
    current_weights: Optional[Sequence[float] | np.ndarray],
    n_assets: int,
) -> Optional[np.ndarray]:
    if current_weights is None:
        return None
    values = np.asarray(current_weights, dtype=float)
    if values.ndim != 1 or values.size != n_assets:
        raise ValueError("current_weights length must match symbols.")
    if not np.all(np.isfinite(values)):
        raise ValueError("current_weights must be finite.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("current_weights must have a positive sum.")
    return values / total


def build_constraint_set(
    symbols: Sequence[str],
    *,
    strategy: Optional[Mapping[str, Any]] = None,
    asset_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
    current_weights: Optional[Sequence[float] | np.ndarray] = None,
    turnover_limit: Optional[float] = None,
) -> PortfolioConstraintSet:
    """Translate a normalized Strategy Rulebook into optimizer constraints."""
    names = tuple(str(symbol).strip() for symbol in symbols)
    if not names or any(not symbol for symbol in names):
        raise ValueError("symbols must contain at least one non-empty identifier.")
    if len(set(names)) != len(names):
        raise ValueError("symbols must be unique.")

    rules = dict(strategy or {})
    metadata = asset_metadata or {}
    strategy_long_only = bool(rules.get("long_only", not allow_short))
    effective_allow_short = bool(allow_short and not strategy_long_only)
    strategy_max_weight = _normalise_optional_weight(
        rules.get("max_position_weight"), "max_position_weight"
    )
    requested_caps = [value for value in (max_weight, strategy_max_weight) if value is not None]
    effective_max_weight = min(float(value) for value in requested_caps) if requested_caps else None
    bounds = build_weight_bounds(
        len(names),
        allow_short=effective_allow_short,
        max_weight=effective_max_weight,
    )
    lower_bounds = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper_bounds = np.asarray([bound[1] for bound in bounds], dtype=float)

    prohibited = {
        _normalise_text(value) for value in rules.get("prohibited_tickers", [])
    }
    excluded_sectors = {
        _normalise_text(value) for value in rules.get("excluded_sectors", [])
    }
    allowed_sectors = {
        _normalise_text(value) for value in rules.get("allowed_sectors", [])
    }
    allowed_asset_types = {
        _normalise_text(value) for value in rules.get("allowed_asset_types", [])
    }
    required_tags = {
        _normalise_text(value) for value in rules.get("required_tags", [])
    }
    require_approved = bool(rules.get("require_approved", False))
    warnings: list[str] = []
    sectors: list[str] = []
    cash_indices: list[int] = []

    for index, symbol in enumerate(names):
        item = _metadata_for_symbol(metadata, symbol)
        sector = _normalise_text(item.get("sector"))
        asset_type = _normalise_text(item.get("asset_type", item.get("security_type")))
        tags = {
            _normalise_text(tag)
            for tag in item.get("tags", [])
            if _normalise_text(tag)
        }
        is_cash = (
            bool(item.get("is_cash", False))
            or symbol.upper() in _CASH_SYMBOLS
            or asset_type in _CASH_ASSET_TYPES
        )
        sectors.append(sector)
        if is_cash:
            cash_indices.append(index)

        excluded = _normalise_text(symbol) in prohibited
        excluded = excluded or (bool(sector) and sector in excluded_sectors)
        excluded = excluded or (bool(allowed_sectors) and sector not in allowed_sectors and not is_cash)
        excluded = excluded or (
            bool(allowed_asset_types) and asset_type not in allowed_asset_types and not is_cash
        )
        excluded = excluded or (
            bool(required_tags) and not is_cash and not required_tags.issubset(tags)
        )
        excluded = excluded or (
            require_approved and not is_cash and not bool(item.get("approved", False))
        )
        if excluded:
            lower_bounds[index] = 0.0
            upper_bounds[index] = 0.0

    groups: list[GroupConstraint] = []
    maximum_sector_weight = _normalise_optional_weight(
        rules.get("max_sector_weight"), "max_sector_weight"
    )
    if maximum_sector_weight is not None and maximum_sector_weight < 1.0:
        missing_sector = [
            names[index]
            for index, sector in enumerate(sectors)
            if not sector
            and index not in cash_indices
            and upper_bounds[index] > WEIGHT_TOLERANCE
        ]
        if missing_sector:
            raise ValueError(
                "max_sector_weight requires sector metadata for: "
                + ", ".join(missing_sector)
                + "."
            )
        investable_sectors = {
            sector
            for index, sector in enumerate(sectors)
            if sector and index not in cash_indices
        }
        for sector in sorted(investable_sectors):
            indices = tuple(
                index
                for index, value in enumerate(sectors)
                if value == sector and index not in cash_indices
            )
            groups.append(GroupConstraint(
                name=f"sector:{sector}",
                indices=indices,
                maximum=maximum_sector_weight,
            ))

    for raw_target in rules.get("sector_targets", []):
        if not isinstance(raw_target, Mapping):
            continue
        sector_name = _normalise_text(
            raw_target.get("sector", raw_target.get("name"))
        )
        if not sector_name:
            continue
        indices = tuple(
            index for index, value in enumerate(sectors) if value == sector_name
        )
        minimum = _normalise_optional_weight(
            raw_target.get("min_weight"), f"{sector_name} min_weight"
        )
        maximum = _normalise_optional_weight(
            raw_target.get("max_weight"), f"{sector_name} max_weight"
        )
        if not indices and (minimum or 0.0) > 0:
            raise ValueError(
                f"strategy requires sector {sector_name!r}, but the universe has no matching asset."
            )
        if indices:
            groups.append(GroupConstraint(
                name=f"sector_target:{sector_name}",
                indices=indices,
                minimum=minimum,
                maximum=maximum,
            ))

    minimum_cash = _normalise_optional_weight(
        rules.get("min_cash_weight"), "min_cash_weight"
    )
    maximum_cash = _normalise_optional_weight(
        rules.get("max_cash_weight"), "max_cash_weight"
    )
    if (minimum_cash or 0.0) > 0 and not cash_indices:
        raise ValueError("strategy requires cash, but the universe has no cash asset.")
    if cash_indices and (minimum_cash is not None or maximum_cash is not None):
        groups.append(GroupConstraint(
            name="cash",
            indices=tuple(cash_indices),
            minimum=minimum_cash,
            maximum=maximum_cash,
        ))

    minimum_beta = rules.get("min_beta")
    maximum_beta = rules.get("max_beta")
    beta_values: Optional[np.ndarray] = None
    if minimum_beta is not None or maximum_beta is not None:
        beta_list: list[float] = []
        for index, symbol in enumerate(names):
            item = _metadata_for_symbol(metadata, symbol)
            if index in cash_indices:
                beta_list.append(0.0)
                continue
            raw_beta = item.get("beta")
            if raw_beta is None or not np.isfinite(float(raw_beta)):
                raise ValueError(
                    f"strategy configures beta limits, but beta is missing for {symbol}."
                )
            beta_list.append(float(raw_beta))
        beta_values = np.asarray(beta_list, dtype=float)
        minimum_beta = float(minimum_beta) if minimum_beta is not None else None
        maximum_beta = float(maximum_beta) if maximum_beta is not None else None
        if (
            minimum_beta is not None
            and maximum_beta is not None
            and minimum_beta > maximum_beta
        ):
            raise ValueError("minimum beta cannot exceed maximum beta.")

    strategy_turnover = rules.get("max_turnover")
    limits = [value for value in (turnover_limit, strategy_turnover) if value is not None]
    effective_turnover = min(float(value) for value in limits) if limits else None
    if effective_turnover is not None and (
        not np.isfinite(effective_turnover) or effective_turnover < 0
    ):
        raise ValueError("turnover_limit must be non-negative.")
    normalized_current = _normalise_current_weights(current_weights, len(names))
    if effective_turnover is not None and normalized_current is None:
        raise ValueError("turnover_limit requires current_weights.")

    if rules.get("min_holdings") is not None or rules.get("max_holdings") is not None:
        warnings.append(
            "Exact mixed-integer holding counts are not imposed on the continuous target; "
            "they are enforced in the executable lot-level trade plan."
        )
    if float(np.sum(upper_bounds)) < 1.0 - WEIGHT_TOLERANCE:
        raise ValueError("asset exclusions and position caps make full investment infeasible.")
    if float(np.sum(lower_bounds)) > 1.0 + WEIGHT_TOLERANCE:
        raise ValueError("minimum asset weights exceed full investment.")

    return PortfolioConstraintSet(
        symbols=names,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        groups=tuple(groups),
        beta=beta_values,
        minimum_beta=minimum_beta,
        maximum_beta=maximum_beta,
        current_weights=normalized_current,
        turnover_limit=effective_turnover,
        warnings=tuple(warnings),
    )


def validate_constraint_solution(
    weights: Sequence[float] | np.ndarray,
    constraints: PortfolioConstraintSet,
    *,
    tolerance: float = 1e-5,
) -> np.ndarray:
    bounds = list(zip(
        constraints.lower_bounds.tolist(),
        constraints.upper_bounds.tolist(),
        strict=False,
    ))
    values = validate_weight_solution(weights, bounds, tolerance=tolerance)
    for group in constraints.groups:
        actual = float(np.sum(values[list(group.indices)]))
        if group.minimum is not None and actual < group.minimum - tolerance:
            raise ValueError(f"weights violate minimum for {group.name}.")
        if group.maximum is not None and actual > group.maximum + tolerance:
            raise ValueError(f"weights violate maximum for {group.name}.")
    if constraints.beta is not None:
        beta = float(values @ constraints.beta)
        if constraints.minimum_beta is not None and beta < constraints.minimum_beta - tolerance:
            raise ValueError("weights violate minimum portfolio beta.")
        if constraints.maximum_beta is not None and beta > constraints.maximum_beta + tolerance:
            raise ValueError("weights violate maximum portfolio beta.")
    if constraints.turnover_limit is not None and constraints.current_weights is not None:
        turnover = float(np.sum(np.abs(values - constraints.current_weights)))
        if turnover > constraints.turnover_limit + tolerance:
            raise ValueError("weights violate maximum turnover.")
    return values


def build_constraint_report(
    weights: Sequence[float] | np.ndarray,
    constraints: PortfolioConstraintSet,
    *,
    binding_tolerance: float = 1e-4,
) -> list[dict[str, Any]]:
    values = np.asarray(weights, dtype=float)
    report: list[dict[str, Any]] = []
    for index, symbol in enumerate(constraints.symbols):
        actual = float(values[index])
        lower = float(constraints.lower_bounds[index])
        upper = float(constraints.upper_bounds[index])
        report.append({
            "name": f"asset:{symbol}",
            "actual": actual,
            "minimum": lower,
            "maximum": upper,
            "binding": bool(
                abs(actual - lower) <= binding_tolerance
                or abs(actual - upper) <= binding_tolerance
            ),
            "passed": bool(lower - binding_tolerance <= actual <= upper + binding_tolerance),
        })
    for group in constraints.groups:
        actual = float(np.sum(values[list(group.indices)]))
        report.append({
            "name": group.name,
            "actual": actual,
            "minimum": group.minimum,
            "maximum": group.maximum,
            "binding": bool(
                (group.minimum is not None and abs(actual - group.minimum) <= binding_tolerance)
                or (group.maximum is not None and abs(actual - group.maximum) <= binding_tolerance)
            ),
            "passed": bool(
                (group.minimum is None or actual >= group.minimum - binding_tolerance)
                and (group.maximum is None or actual <= group.maximum + binding_tolerance)
            ),
        })
    if constraints.beta is not None:
        actual_beta = float(values @ constraints.beta)
        report.append({
            "name": "portfolio_beta",
            "actual": actual_beta,
            "minimum": constraints.minimum_beta,
            "maximum": constraints.maximum_beta,
            "binding": bool(
                (
                    constraints.minimum_beta is not None
                    and abs(actual_beta - constraints.minimum_beta) <= binding_tolerance
                )
                or (
                    constraints.maximum_beta is not None
                    and abs(actual_beta - constraints.maximum_beta) <= binding_tolerance
                )
            ),
            "passed": bool(
                (constraints.minimum_beta is None or actual_beta >= constraints.minimum_beta - binding_tolerance)
                and (constraints.maximum_beta is None or actual_beta <= constraints.maximum_beta + binding_tolerance)
            ),
        })
    if constraints.turnover_limit is not None and constraints.current_weights is not None:
        turnover = float(np.sum(np.abs(values - constraints.current_weights)))
        report.append({
            "name": "turnover",
            "actual": turnover,
            "minimum": None,
            "maximum": constraints.turnover_limit,
            "binding": abs(turnover - constraints.turnover_limit) <= binding_tolerance,
            "passed": turnover <= constraints.turnover_limit + binding_tolerance,
        })
    return report
