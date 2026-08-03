"""Deterministic industry and peer-comparison analytics.

The module is deliberately independent of data vendors and UI frameworks.  Callers
provide company and peer metrics, plus any qualitative Porter/SWOT assessments.
Nothing in the qualitative output is inferred from a ticker or an industry name.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
import re
from typing import Any, Mapping, Sequence


CATEGORY_LABELS: dict[str, str] = {
    "valuation": "Valuation",
    "growth": "Growth",
    "profitability": "Profitability",
    "balance_sheet_risk": "Balance sheet & risk",
}

DEFAULT_CATEGORY_WEIGHTS: dict[str, float] = {
    "valuation": 1.0,
    "growth": 1.0,
    "profitability": 1.0,
    "balance_sheet_risk": 1.0,
}


# Peer selection deliberately uses a small, provider-agnostic set of fields.  The
# aliases cover both the snake_case records used by our analytics layer and the
# camelCase names commonly returned by market-data providers.
_PEER_MARKET_CAP_ALIASES: tuple[str, ...] = (
    "market_cap",
    "marketCap",
    "market_capitalization",
    "capitalization",
    "equity_value",
)

_PEER_PROFITABILITY_FIELDS: tuple[tuple[str, tuple[str, ...], float], ...] = (
    ("gross_margin", ("grossMargins", "gross_profit_margin"), 0.20),
    ("operating_margin", ("operatingMargins", "ebit_margin"), 0.15),
    ("net_margin", ("profitMargins", "profit_margin", "net_profit_margin"), 0.15),
    ("return_on_assets", ("returnOnAssets", "roa"), 0.10),
    ("return_on_equity", ("returnOnEquity", "roe"), 0.25),
    (
        "return_on_invested_capital",
        ("returnOnInvestedCapital", "roic"),
        0.15,
    ),
)

_PEER_GROWTH_FIELDS: tuple[tuple[str, tuple[str, ...], float], ...] = (
    ("revenue_growth", ("revenueGrowth", "sales_growth", "salesGrowth"), 0.25),
    ("earnings_growth", ("earningsGrowth", "eps_growth", "net_income_growth"), 0.35),
    ("ebitda_growth", ("ebitdaGrowth",), 0.30),
    (
        "free_cash_flow_growth",
        ("freeCashFlowGrowth", "fcf_growth"),
        0.40,
    ),
)

_SECTOR_CANONICAL_ALIASES: dict[str, str] = {
    "information_technology": "technology",
    "technology": "technology",
    "financial_services": "financials",
    "financial": "financials",
    "financials": "financials",
    "health_care": "healthcare",
    "healthcare": "healthcare",
    "consumer_cyclical": "consumer_discretionary",
    "consumer_discretionary": "consumer_discretionary",
    "consumer_defensive": "consumer_staples",
    "consumer_staples": "consumer_staples",
    "basic_materials": "materials",
    "materials": "materials",
}


@dataclass(frozen=True)
class PeerMetricSpec:
    """Describe how one company metric should be compared with peers."""

    key: str
    label: str
    category: str
    higher_is_better: bool
    weight: float = 1.0
    aliases: tuple[str, ...] = ()
    minimum_exclusive: float | None = None
    minimum_inclusive: float | None = None
    maximum_inclusive: float | None = None


DEFAULT_METRIC_SPECS: tuple[PeerMetricSpec, ...] = (
    PeerMetricSpec(
        "pe_ratio",
        "P/E ratio",
        "valuation",
        False,
        aliases=("pe", "trailing_pe", "trailingPE", "price_earnings_ratio"),
        minimum_exclusive=0.0,
    ),
    PeerMetricSpec(
        "forward_pe",
        "Forward P/E",
        "valuation",
        False,
        aliases=("forwardPE", "forward_price_earnings"),
        minimum_exclusive=0.0,
    ),
    PeerMetricSpec(
        "price_to_sales",
        "Price / sales",
        "valuation",
        False,
        aliases=("priceToSalesTrailing12Months", "ps_ratio", "p_s"),
        minimum_exclusive=0.0,
    ),
    PeerMetricSpec(
        "price_to_book",
        "Price / book",
        "valuation",
        False,
        aliases=("priceToBook", "pb_ratio", "p_b"),
        minimum_exclusive=0.0,
    ),
    PeerMetricSpec(
        "ev_to_ebitda",
        "EV / EBITDA",
        "valuation",
        False,
        aliases=("enterpriseToEbitda", "enterprise_value_to_ebitda", "ev_ebitda"),
        minimum_exclusive=0.0,
    ),
    PeerMetricSpec(
        "revenue_growth",
        "Revenue growth",
        "growth",
        True,
        aliases=("revenueGrowth", "sales_growth", "salesGrowth"),
    ),
    PeerMetricSpec(
        "earnings_growth",
        "Earnings growth",
        "growth",
        True,
        aliases=("earningsGrowth", "eps_growth", "net_income_growth"),
    ),
    PeerMetricSpec(
        "ebitda_growth",
        "EBITDA growth",
        "growth",
        True,
        aliases=("ebitdaGrowth",),
    ),
    PeerMetricSpec(
        "free_cash_flow_growth",
        "Free-cash-flow growth",
        "growth",
        True,
        aliases=("freeCashFlowGrowth", "fcf_growth"),
    ),
    PeerMetricSpec(
        "gross_margin",
        "Gross margin",
        "profitability",
        True,
        aliases=("grossMargins", "gross_profit_margin"),
    ),
    PeerMetricSpec(
        "operating_margin",
        "Operating margin",
        "profitability",
        True,
        aliases=("operatingMargins", "ebit_margin"),
    ),
    PeerMetricSpec(
        "net_margin",
        "Net margin",
        "profitability",
        True,
        aliases=("profitMargins", "profit_margin", "net_profit_margin"),
    ),
    PeerMetricSpec(
        "return_on_equity",
        "Return on equity",
        "profitability",
        True,
        aliases=("returnOnEquity", "roe"),
    ),
    PeerMetricSpec(
        "return_on_assets",
        "Return on assets",
        "profitability",
        True,
        aliases=("returnOnAssets", "roa"),
    ),
    PeerMetricSpec(
        "return_on_invested_capital",
        "Return on invested capital",
        "profitability",
        True,
        aliases=("returnOnInvestedCapital", "roic"),
    ),
    PeerMetricSpec(
        "current_ratio",
        "Current ratio",
        "balance_sheet_risk",
        True,
        aliases=("currentRatio",),
        minimum_inclusive=0.0,
    ),
    PeerMetricSpec(
        "debt_to_equity",
        "Debt / equity",
        "balance_sheet_risk",
        False,
        aliases=("debtToEquity", "d_e"),
        minimum_inclusive=0.0,
    ),
    PeerMetricSpec(
        "net_debt_to_ebitda",
        "Net debt / EBITDA",
        "balance_sheet_risk",
        False,
        aliases=("netDebtToEbitda", "net_debt_ebitda"),
    ),
    PeerMetricSpec(
        "interest_coverage",
        "Interest coverage",
        "balance_sheet_risk",
        True,
        aliases=("interestCoverage", "times_interest_earned"),
    ),
    PeerMetricSpec(
        "beta",
        "Equity beta",
        "balance_sheet_risk",
        False,
        aliases=("beta_5y", "five_year_beta"),
    ),
)


PORTER_FORCES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("competitive_rivalry", "Competitive rivalry", ("rivalry", "industry_rivalry")),
    (
        "threat_of_new_entrants",
        "Threat of new entrants",
        ("new_entrants", "entry_threat", "barriers_to_entry"),
    ),
    ("supplier_power", "Supplier bargaining power", ("suppliers", "supplier_bargaining_power")),
    ("buyer_power", "Buyer bargaining power", ("buyers", "customer_power", "buyer_bargaining_power")),
    ("threat_of_substitutes", "Threat of substitutes", ("substitutes", "substitution_threat")),
)

SWOT_QUADRANTS: tuple[tuple[str, str, str, str], ...] = (
    ("strengths", "Strengths", "internal", "positive"),
    ("weaknesses", "Weaknesses", "internal", "negative"),
    ("opportunities", "Opportunities", "external", "positive"),
    ("threats", "Threats", "external", "negative"),
)


def _normalise_key(value: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value or ""))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _coerce_number(value: Any) -> float | None:
    """Convert common UI/data-provider numeric values; reject booleans and NaN."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        result = float(value)
        return result if math.isfinite(result) else None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text or text.lower() in {"n/a", "na", "none", "null", "nan", "-", "--"}:
        return None
    is_percent = text.endswith("%")
    is_parenthesised = text.startswith("(") and text.endswith(")")
    if is_parenthesised:
        text = text[1:-1]
    text = text.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    text = re.sub(r"(?i)(x|times)$", "", text.strip())
    if is_percent:
        text = text[:-1].strip()
    try:
        result = float(text)
    except ValueError:
        return None
    if is_parenthesised:
        result = -result
    if is_percent:
        result /= 100.0
    return result if math.isfinite(result) else None


def _is_valid_metric_value(value: float | None, spec: PeerMetricSpec) -> bool:
    if value is None:
        return False
    if spec.minimum_exclusive is not None and value <= spec.minimum_exclusive:
        return False
    if spec.minimum_inclusive is not None and value < spec.minimum_inclusive:
        return False
    if spec.maximum_inclusive is not None and value > spec.maximum_inclusive:
        return False
    return True


def _normalised_metric_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        return {}
    nested = values.get("metrics")
    payload = nested if isinstance(nested, Mapping) else values
    return {_normalise_key(key): value for key, value in payload.items()}


def _metric_value(values: Mapping[str, Any], spec: PeerMetricSpec) -> float | None:
    for key in (spec.key, *spec.aliases):
        normalised = _normalise_key(key)
        if normalised not in values:
            continue
        candidate = _coerce_number(values[normalised])
        if _is_valid_metric_value(candidate, spec):
            return candidate
    return None


def _normalise_peers(
    peers: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    if isinstance(peers, Mapping):
        iterator = peers.items()
    elif isinstance(peers, Sequence) and not isinstance(peers, (str, bytes)):
        iterator = enumerate(peers, start=1)
    else:
        return rows

    for fallback_name, raw_values in iterator:
        if not isinstance(raw_values, Mapping):
            continue
        if isinstance(fallback_name, int):
            supplied_name = raw_values.get("ticker") or raw_values.get("symbol") or raw_values.get("name")
            name = str(supplied_name or f"Peer {fallback_name}").strip()
        else:
            name = str(fallback_name).strip() or f"Peer {len(rows) + 1}"
        rows.append((name, _normalised_metric_mapping(raw_values)))
    return rows


def _normalised_company_profile(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten common company wrappers into one normalised field mapping.

    The Company Analysis UI stores vendor data below ``info`` while standalone
    analytics callers frequently pass the info mapping directly.  Supporting both
    shapes here keeps peer selection independent of either caller.
    """

    if not isinstance(values, Mapping):
        return {}

    profile: dict[str, Any] = {}
    nested_keys = {"info", "metrics", "profile", "fundamentals"}
    for key, value in values.items():
        if _normalise_key(key) in nested_keys and isinstance(value, Mapping):
            profile.update({_normalise_key(item): content for item, content in value.items()})
    for key, value in values.items():
        if _normalise_key(key) not in nested_keys:
            profile[_normalise_key(key)] = value
    return profile


def _first_profile_value(profile: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        value = profile.get(_normalise_key(alias))
        if value is not None and (not isinstance(value, str) or value.strip()):
            return value
    return None


def _profile_number(profile: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    return _coerce_number(_first_profile_value(profile, aliases))


def _taxonomy_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if not cleaned or cleaned.casefold() in {"n/a", "na", "none", "null", "unknown", "-", "--"}:
        return ""
    return cleaned


def _canonical_sector(value: Any) -> str:
    normalised = _normalise_key(_taxonomy_text(value))
    return _SECTOR_CANONICAL_ALIASES.get(normalised, normalised)


def _taxonomy_tokens(value: Any) -> set[str]:
    stop_words = {"and", "the", "of", "other", "miscellaneous", "general", "diversified"}
    tokens: set[str] = set()
    for token in _normalise_key(_taxonomy_text(value)).split("_"):
        if not token or token in stop_words:
            continue
        # Minimal plural handling is enough to align common labels such as
        # "Banks" / "Regional Bank" and "Semiconductors" / "Semiconductor".
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 4:
            token = token[:-1]
        tokens.add(token)
    return tokens


def _industry_similarity(first: Any, second: Any) -> float:
    first_key = _normalise_key(_taxonomy_text(first))
    second_key = _normalise_key(_taxonomy_text(second))
    if not first_key or not second_key:
        return 0.0
    if first_key == second_key:
        return 1.0
    first_tokens = _taxonomy_tokens(first)
    second_tokens = _taxonomy_tokens(second)
    if not first_tokens or not second_tokens:
        return 0.0
    # The overlap coefficient recognises a specific sub-industry as related to a
    # shorter industry label without allowing unrelated sectors to outrank an
    # exact sector match.
    return len(first_tokens & second_tokens) / min(len(first_tokens), len(second_tokens))


def _rate_group_similarity(
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
    fields: Sequence[tuple[str, tuple[str, ...], float]],
) -> tuple[float | None, float, list[str]]:
    target_values: list[tuple[str, float, tuple[str, ...], float]] = []
    for key, aliases, scale in fields:
        value = _profile_number(target, (key, *aliases))
        if value is not None:
            target_values.append((key, value, aliases, scale))
    if not target_values:
        return None, 0.0, []

    scores: list[float] = []
    matched: list[str] = []
    for key, target_value, aliases, scale in target_values:
        candidate_value = _profile_number(candidate, (key, *aliases))
        if candidate_value is None:
            continue
        difference = abs(candidate_value - target_value)
        scores.append(100.0 / (1.0 + difference / scale))
        matched.append(key)
    coverage = len(scores) / len(target_values)
    return (sum(scores) / len(scores) if scores else None), coverage, matched


def _market_cap_similarity(target: Mapping[str, Any], candidate: Mapping[str, Any]) -> float | None:
    target_cap = _profile_number(target, _PEER_MARKET_CAP_ALIASES)
    candidate_cap = _profile_number(candidate, _PEER_MARKET_CAP_ALIASES)
    if target_cap is None or candidate_cap is None or target_cap <= 0 or candidate_cap <= 0:
        return None
    log_distance = abs(math.log10(candidate_cap / target_cap))
    # A 2x size difference scores about 85, a 10x difference scores 50, and a
    # 100x difference scores zero.  Log distance makes the measure symmetric.
    return max(0.0, 100.0 * (1.0 - log_distance / 2.0))


def _financial_similarity(
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    market_cap = _market_cap_similarity(target, candidate)
    target_market_cap = _profile_number(target, _PEER_MARKET_CAP_ALIASES)
    profitability, profitability_coverage, profitability_metrics = _rate_group_similarity(
        target,
        candidate,
        _PEER_PROFITABILITY_FIELDS,
    )
    growth, growth_coverage, growth_metrics = _rate_group_similarity(
        target,
        candidate,
        _PEER_GROWTH_FIELDS,
    )

    target_has_market_cap = target_market_cap is not None and target_market_cap > 0
    target_has_profitability = any(
        _profile_number(target, (key, *aliases)) is not None
        for key, aliases, _ in _PEER_PROFITABILITY_FIELDS
    )
    target_has_growth = any(
        _profile_number(target, (key, *aliases)) is not None
        for key, aliases, _ in _PEER_GROWTH_FIELDS
    )
    possible_weight = (
        (0.50 if target_has_market_cap else 0.0)
        + (0.30 if target_has_profitability else 0.0)
        + (0.20 if target_has_growth else 0.0)
    )
    observed_weight = 0.0
    weighted_score = 0.0
    components = (
        (market_cap, 0.50, 1.0 if target_has_market_cap else 0.0),
        (profitability, 0.30, profitability_coverage),
        (growth, 0.20, growth_coverage),
    )
    for score, weight, coverage in components:
        if score is None or coverage <= 0:
            continue
        effective_weight = weight * coverage
        observed_weight += effective_weight
        weighted_score += score * effective_weight

    coverage = observed_weight / possible_weight if possible_weight else 0.0
    raw_score = weighted_score / observed_weight if observed_weight else 50.0
    adjusted_score = 50.0 + (raw_score - 50.0) * coverage
    return {
        "score": max(0.0, min(100.0, adjusted_score)),
        "coverage": max(0.0, min(1.0, coverage)),
        "market_cap_similarity": market_cap,
        "profitability_similarity": profitability,
        "growth_similarity": growth,
        "matched_profitability_metrics": profitability_metrics,
        "matched_growth_metrics": growth_metrics,
    }


def _peer_classification(
    target_sector: str,
    target_industry: str,
    candidate_sector: str,
    candidate_industry: str,
) -> tuple[str, int, float]:
    industry_similarity = _industry_similarity(target_industry, candidate_industry)
    target_industry_tokens = _taxonomy_tokens(target_industry)
    candidate_industry_tokens = _taxonomy_tokens(candidate_industry)
    same_industry = (
        bool(target_industry and candidate_industry)
        and (
            _normalise_key(target_industry) == _normalise_key(candidate_industry)
            or (
                bool(target_industry_tokens)
                and target_industry_tokens == candidate_industry_tokens
            )
        )
    )
    if same_industry:
        return "same_industry", 0, industry_similarity

    target_sector_key = _canonical_sector(target_sector)
    candidate_sector_key = _canonical_sector(candidate_sector)
    if target_sector_key and candidate_sector_key and target_sector_key == candidate_sector_key:
        return "same_sector", 1, industry_similarity

    sectors_conflict = (
        bool(target_sector_key)
        and bool(candidate_sector_key)
        and target_sector_key != candidate_sector_key
    )
    if industry_similarity >= 0.5 and not sectors_conflict:
        return "related_industry", 2, industry_similarity
    if not target_sector and not target_industry:
        return "metrics_only", 3, industry_similarity
    if not candidate_sector and not candidate_industry:
        return "classification_missing", 4, industry_similarity
    return "different_sector", 5, industry_similarity


def select_similar_peers(
    target_company: Mapping[str, Any] | None,
    candidate_universe: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
    *,
    target_ticker: str | None = None,
    max_peers: int | None = 8,
    include_out_of_sector: bool = False,
) -> list[dict[str, Any]]:
    """Select and rank relevant peer companies without fetching external data.

    Classification is intentionally hierarchical: an exact industry match always
    ranks before a same-sector match, which always ranks before a textually related
    industry.  Market capitalisation (50%), profitability (30%), and growth (20%)
    rank candidates *within* those tiers.  Missing financial fields are not imputed;
    their score is shrunk toward neutral and ``financial_coverage_pct`` makes that
    uncertainty visible.

    By default, candidates outside the target's industry/sector are excluded.  If
    the target has no classification, the function falls back to financial
    similarity.  Both direct provider ``info`` mappings and ``{"info": ...}``
    wrappers are accepted.
    """

    if max_peers is not None and (
        isinstance(max_peers, bool) or not isinstance(max_peers, int) or max_peers < 0
    ):
        raise ValueError("max_peers must be a non-negative integer or None.")
    if max_peers == 0 or not isinstance(target_company, Mapping):
        return []

    target = _normalised_company_profile(target_company)
    target_identifier = str(
        target_ticker
        or _first_profile_value(target, ("ticker", "symbol"))
        or ""
    ).strip().upper()
    target_name = str(
        _first_profile_value(target, ("short_name", "long_name", "company_name", "name"))
        or ""
    ).strip()
    target_sector = _taxonomy_text(
        _first_profile_value(target, ("sector", "sector_name", "gics_sector", "sector_key", "sectorDisp"))
    )
    target_industry = _taxonomy_text(
        _first_profile_value(
            target,
            ("industry", "industry_name", "gics_industry", "gics_sub_industry", "industry_key", "industryDisp"),
        )
    )

    has_target_financials = any(
        _profile_number(target, aliases) is not None
        for aliases in (
            _PEER_MARKET_CAP_ALIASES,
            *((key, *aliases) for key, aliases, _ in _PEER_PROFITABILITY_FIELDS),
            *((key, *aliases) for key, aliases, _ in _PEER_GROWTH_FIELDS),
        )
    )
    if not target_sector and not target_industry and not has_target_financials:
        return []

    if isinstance(candidate_universe, Mapping):
        iterator: Sequence[tuple[Any, Any]] = list(candidate_universe.items())
        mapping_input = True
    elif isinstance(candidate_universe, Sequence) and not isinstance(candidate_universe, (str, bytes)):
        iterator = list(enumerate(candidate_universe, start=1))
        mapping_input = False
    else:
        return []

    rows: list[dict[str, Any]] = []
    target_has_classification = bool(target_sector or target_industry)
    for fallback_identifier, raw_candidate in iterator:
        if not isinstance(raw_candidate, Mapping) or raw_candidate is target_company:
            continue
        candidate = _normalised_company_profile(raw_candidate)
        supplied_identifier = _first_profile_value(candidate, ("ticker", "symbol"))
        identifier = str(
            fallback_identifier if mapping_input else supplied_identifier or ""
        ).strip().upper()
        if not identifier:
            continue
        candidate_name = str(
            _first_profile_value(candidate, ("short_name", "long_name", "company_name", "name"))
            or identifier
        ).strip()
        if target_identifier and identifier.casefold() == target_identifier.casefold():
            continue
        if (
            not target_identifier
            and target_name
            and candidate_name.casefold() == target_name.casefold()
        ):
            continue

        candidate_sector = _taxonomy_text(
            _first_profile_value(candidate, ("sector", "sector_name", "gics_sector", "sector_key", "sectorDisp"))
        )
        candidate_industry = _taxonomy_text(
            _first_profile_value(
                candidate,
                ("industry", "industry_name", "gics_industry", "gics_sub_industry", "industry_key", "industryDisp"),
            )
        )
        classification, classification_rank, industry_similarity = _peer_classification(
            target_sector,
            target_industry,
            candidate_sector,
            candidate_industry,
        )
        if (
            target_has_classification
            and not include_out_of_sector
            and classification_rank > 2
        ):
            continue

        financial = _financial_similarity(target, candidate)
        band_base, band_width = {
            "same_industry": (80.0, 20.0),
            "same_sector": (55.0, 20.0),
            "related_industry": (35.0, 15.0),
            "metrics_only": (15.0, 15.0),
            "classification_missing": (10.0, 10.0),
            "different_sector": (0.0, 10.0),
        }[classification]
        similarity_score = band_base + band_width * float(financial["score"]) / 100.0
        rows.append(
            {
                "ticker": identifier,
                "name": candidate_name,
                "sector": candidate_sector or None,
                "industry": candidate_industry or None,
                "classification_match": classification,
                "score": round(similarity_score, 2),
                "financial_similarity": round(float(financial["score"]), 2),
                "financial_coverage_pct": round(float(financial["coverage"]) * 100.0, 1),
                "market_cap_similarity": _round_optional(financial["market_cap_similarity"], 2),
                "profitability_similarity": _round_optional(financial["profitability_similarity"], 2),
                "growth_similarity": _round_optional(financial["growth_similarity"], 2),
                "industry_similarity": round(industry_similarity * 100.0, 1),
                "matched_profitability_metrics": financial["matched_profitability_metrics"],
                "matched_growth_metrics": financial["matched_growth_metrics"],
                "_classification_rank": classification_rank,
            }
        )

    rows.sort(
        key=lambda row: (
            int(row["_classification_rank"]),
            -float(row["financial_similarity"]),
            -float(row["financial_coverage_pct"]),
            str(row["ticker"]).casefold(),
        )
    )
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        identifier_key = str(row["ticker"]).casefold()
        if identifier_key in seen:
            continue
        seen.add(identifier_key)
        row.pop("_classification_rank", None)
        deduplicated.append(row)
        if max_peers is not None and len(deduplicated) >= max_peers:
            break
    return deduplicated


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _midrank_percentile(value: float, peers: Sequence[float]) -> float:
    lower = sum(peer < value for peer in peers)
    equal = sum(peer == value for peer in peers)
    return 100.0 * (lower + 0.5 * equal) / len(peers)


def _round_optional(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


def _resolve_metric_specs(
    metric_specs: Sequence[PeerMetricSpec] | Mapping[str, PeerMetricSpec | Mapping[str, Any]] | None,
) -> tuple[PeerMetricSpec, ...]:
    if metric_specs is None:
        return DEFAULT_METRIC_SPECS
    if isinstance(metric_specs, Mapping):
        resolved: list[PeerMetricSpec] = []
        for key, raw in metric_specs.items():
            if isinstance(raw, PeerMetricSpec):
                resolved.append(raw)
                continue
            if not isinstance(raw, Mapping):
                raise TypeError("Each custom metric specification must be a mapping or PeerMetricSpec.")
            resolved.append(
                PeerMetricSpec(
                    key=str(raw.get("key") or key),
                    label=str(raw.get("label") or key),
                    category=str(raw.get("category") or "other"),
                    higher_is_better=bool(raw.get("higher_is_better", True)),
                    weight=float(raw.get("weight", 1.0)),
                    aliases=tuple(str(alias) for alias in raw.get("aliases", ())),
                    minimum_exclusive=_coerce_number(raw.get("minimum_exclusive")),
                    minimum_inclusive=_coerce_number(raw.get("minimum_inclusive")),
                    maximum_inclusive=_coerce_number(raw.get("maximum_inclusive")),
                )
            )
        specs = tuple(resolved)
    elif isinstance(metric_specs, Sequence) and not isinstance(metric_specs, (str, bytes)):
        specs = tuple(metric_specs)
        if not all(isinstance(spec, PeerMetricSpec) for spec in specs):
            raise TypeError("metric_specs sequences may contain only PeerMetricSpec objects.")
    else:
        raise TypeError("metric_specs must be a mapping or sequence of PeerMetricSpec objects.")

    seen: set[str] = set()
    for spec in specs:
        key = _normalise_key(spec.key)
        if not key or key in seen:
            raise ValueError("Metric specification keys must be non-empty and unique.")
        if not math.isfinite(float(spec.weight)) or spec.weight <= 0:
            raise ValueError("Metric weights must be finite and greater than zero.")
        seen.add(key)
    return specs


def _rating_from_score(score: float | None) -> str:
    if score is None:
        return "Insufficient data"
    if score >= 70:
        return "Strong versus peers"
    if score >= 58:
        return "Above peers"
    if score >= 42:
        return "In line with peers"
    if score >= 30:
        return "Below peers"
    return "Weak versus peers"


def _confidence_label(
    coverage: float,
    category_coverage: float,
    median_peer_count: float,
    min_peer_count: int,
) -> str:
    if coverage <= 0:
        return "Insufficient"
    if coverage >= 80 and category_coverage >= 75 and median_peer_count >= max(5, min_peer_count):
        return "High"
    if coverage >= 60 and category_coverage >= 50 and median_peer_count >= min_peer_count:
        return "Moderate"
    return "Low"


def analyze_peer_comparison(
    company_metrics: Mapping[str, Any] | None,
    peer_metrics: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
    *,
    company_name: str = "Company",
    metric_specs: Sequence[PeerMetricSpec] | Mapping[str, PeerMetricSpec | Mapping[str, Any]] | None = None,
    category_weights: Mapping[str, float] | None = None,
    min_peer_count: int = 3,
) -> dict[str, Any]:
    """Compare one company with peers using robust medians and empirical percentiles.

    ``score`` is direction-adjusted and coverage-adjusted toward a neutral 50.
    ``raw_score`` is the observed-metric score before that coverage adjustment.
    This separation prevents a sparse peer table from looking more certain than it is.
    """

    if min_peer_count < 1:
        raise ValueError("min_peer_count must be at least 1.")
    specs = _resolve_metric_specs(metric_specs)
    company = _normalised_metric_mapping(company_metrics)
    peers = _normalise_peers(peer_metrics)

    spec_categories = list(dict.fromkeys(spec.category for spec in specs))
    weights = {category: DEFAULT_CATEGORY_WEIGHTS.get(category, 1.0) for category in spec_categories}
    if category_weights is not None:
        for category, value in category_weights.items():
            number = _coerce_number(value)
            if number is None or number < 0:
                raise ValueError("Category weights must be finite and non-negative.")
            if str(category) in weights:
                weights[str(category)] = number

    metric_rows: list[dict[str, Any]] = []
    for spec in specs:
        company_value = _metric_value(company, spec)
        peer_values = [
            (peer_name, value)
            for peer_name, values in peers
            if (value := _metric_value(values, spec)) is not None
        ]
        in_scope = company_value is not None or bool(peer_values)
        if not in_scope:
            continue

        values = [value for _, value in peer_values]
        comparable = company_value is not None and len(values) >= min_peer_count
        median = _quantile(values, 0.5) if values else None
        q1 = _quantile(values, 0.25) if values else None
        q3 = _quantile(values, 0.75) if values else None
        raw_percentile: float | None = None
        desirability: float | None = None
        relative_difference: float | None = None
        median_position = "unavailable"
        relative_flag = "insufficient_data"

        if company_value is not None and median is not None:
            if company_value > median:
                median_position = "above"
            elif company_value < median:
                median_position = "below"
            else:
                median_position = "equal"
            if median != 0:
                relative_difference = (company_value - median) / abs(median)

        if comparable:
            raw_percentile = _midrank_percentile(company_value, values)
            desirability = raw_percentile if spec.higher_is_better else 100.0 - raw_percentile
            if desirability >= 67:
                relative_flag = "favorable"
            elif desirability <= 33:
                relative_flag = "unfavorable"
            else:
                relative_flag = "in_line"

        metric_rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "category": spec.category,
                "direction": "higher_is_better" if spec.higher_is_better else "lower_is_better",
                "weight": float(spec.weight),
                "company_value": _round_optional(company_value, 6),
                "peer_count": len(values),
                "peer_median": _round_optional(median, 6),
                "peer_q1": _round_optional(q1, 6),
                "peer_q3": _round_optional(q3, 6),
                "raw_percentile": _round_optional(raw_percentile),
                "desirability_percentile": _round_optional(desirability),
                "relative_difference_pct": _round_optional(
                    relative_difference * 100.0 if relative_difference is not None else None
                ),
                "median_position": median_position,
                "relative_flag": relative_flag,
                "comparable": comparable,
                "missing_reason": (
                    None
                    if comparable
                    else "company_value_missing"
                    if company_value is None
                    else "insufficient_peer_values"
                ),
                "peer_values": [
                    {"name": name, "value": _round_optional(value, 6)} for name, value in peer_values
                ],
            }
        )

    categories_in_order = spec_categories
    categories: dict[str, dict[str, Any]] = {}
    for category in categories_in_order:
        rows = [row for row in metric_rows if row["category"] == category]
        if not rows:
            categories[category] = {
                "label": CATEGORY_LABELS.get(category, category.replace("_", " ").title()),
                "score": None,
                "raw_score": None,
                "coverage_pct": 0.0,
                "metrics_analyzed": 0,
                "metrics_in_scope": 0,
                "weight": float(weights.get(category, 1.0)),
            }
            continue
        expected_weight = sum(float(row["weight"]) for row in rows)
        analyzed = [row for row in rows if row["comparable"]]
        analyzed_weight = sum(float(row["weight"]) for row in analyzed)
        coverage = 100.0 * analyzed_weight / expected_weight if expected_weight else 0.0
        raw_score = (
            sum(float(row["desirability_percentile"]) * float(row["weight"]) for row in analyzed)
            / analyzed_weight
            if analyzed_weight
            else None
        )
        adjusted = (
            50.0 + (raw_score - 50.0) * (coverage / 100.0) if raw_score is not None else 50.0
        )
        categories[category] = {
            "label": CATEGORY_LABELS.get(category, category.replace("_", " ").title()),
            "score": round(adjusted, 1),
            "raw_score": _round_optional(raw_score, 1),
            "coverage_pct": round(coverage, 1),
            "metrics_analyzed": len(analyzed),
            "metrics_in_scope": len(rows),
            "weight": float(weights.get(category, 1.0)),
        }

    scoped_categories = [
        (key, value)
        for key, value in categories.items()
        if value["metrics_in_scope"] > 0 and float(value["weight"]) > 0
    ]
    score_weight = sum(float(row["weight"]) for _, row in scoped_categories)
    score = (
        sum(float(row["score"]) * float(row["weight"]) for _, row in scoped_categories) / score_weight
        if score_weight
        else None
    )
    raw_categories = [(key, row) for key, row in scoped_categories if row["raw_score"] is not None]
    raw_weight = sum(float(row["weight"]) for _, row in raw_categories)
    raw_score = (
        sum(float(row["raw_score"]) * float(row["weight"]) for _, row in raw_categories) / raw_weight
        if raw_weight
        else None
    )
    coverage = (
        sum(float(row["coverage_pct"]) * float(row["weight"]) for _, row in scoped_categories)
        / score_weight
        if score_weight
        else 0.0
    )
    analyzed_categories = sum(row["metrics_analyzed"] > 0 for row in categories.values())
    category_coverage = 100.0 * analyzed_categories / len(categories) if categories else 0.0
    analyzed_rows = [row for row in metric_rows if row["comparable"]]
    peer_counts = sorted(float(row["peer_count"]) for row in analyzed_rows)
    median_peer_count = _quantile(peer_counts, 0.5) if peer_counts else 0.0

    flags: list[dict[str, Any]] = []
    for row in analyzed_rows:
        status = row["relative_flag"]
        severity = "positive" if status == "favorable" else "warning" if status == "unfavorable" else "neutral"
        flags.append(
            {
                "severity": severity,
                "category": row["category"],
                "metric": row["key"],
                "status": status,
                "message": (
                    f"{row['label']}: {status.replace('_', ' ')} versus the supplied peer set "
                    f"({row['desirability_percentile']:.0f}th direction-adjusted percentile)."
                ),
            }
        )
    for row in metric_rows:
        if row["comparable"]:
            continue
        flags.append(
            {
                "severity": "data_gap",
                "category": row["category"],
                "metric": row["key"],
                "status": "insufficient_data",
                "message": (
                    f"{row['label']}: company value is missing."
                    if row["missing_reason"] == "company_value_missing"
                    else f"{row['label']}: requires at least {min_peer_count} valid peer values."
                ),
            }
        )

    final_score = _round_optional(score, 1) if analyzed_rows else None
    return {
        "available": bool(analyzed_rows),
        "company_name": str(company_name or "Company"),
        "peer_count": len(peers),
        "score": final_score,
        "raw_score": _round_optional(raw_score, 1),
        "rating": _rating_from_score(final_score),
        "coverage_pct": round(coverage, 1),
        "category_coverage_pct": round(category_coverage, 1),
        "confidence": _confidence_label(coverage, category_coverage, median_peer_count, min_peer_count),
        "metrics_analyzed": len(analyzed_rows),
        "metrics_in_scope": len(metric_rows),
        "categories": categories,
        "metrics": metric_rows,
        "flags": flags,
        "methodology": {
            "score_range": "0-100; 50 is neutral",
            "percentile_method": "Empirical midrank percentile against valid supplied peers",
            "direction_adjustment": "Lower-is-better metrics invert the raw percentile",
            "central_tendency": "Peer median with interpolated 25th and 75th percentiles",
            "coverage_adjustment": "Category scores are shrunk toward 50 in proportion to missing in-scope metrics",
            "minimum_peer_count": min_peer_count,
            "qualitative_inference": False,
        },
    }


def compare_company_to_peers(
    company_metrics: Mapping[str, Any] | None,
    peer_metrics: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Readable alias for :func:`analyze_peer_comparison`."""

    return analyze_peer_comparison(company_metrics, peer_metrics, **kwargs)


def _lookup_assessment(values: Mapping[str, Any], key: str, aliases: Sequence[str]) -> Any:
    normalised = {_normalise_key(candidate): value for candidate, value in values.items()}
    for candidate in (key, *aliases):
        if _normalise_key(candidate) in normalised:
            return normalised[_normalise_key(candidate)]
    return None


def _porter_rating(value: Any) -> float | None:
    numeric = _coerce_number(value)
    if numeric is not None:
        return numeric if 1.0 <= numeric <= 5.0 else None
    if not isinstance(value, str):
        return None
    key = _normalise_key(value)
    labels = {
        "very_low": 1.0,
        "low": 1.0,
        "low_to_moderate": 2.0,
        "moderately_low": 2.0,
        "medium": 3.0,
        "moderate": 3.0,
        "moderate_to_high": 4.0,
        "moderately_high": 4.0,
        "high": 5.0,
        "very_high": 5.0,
    }
    return labels.get(key)


def _porter_label(rating: float) -> str:
    if rating <= 1.5:
        return "Low"
    if rating <= 2.5:
        return "Low–moderate"
    if rating <= 3.5:
        return "Moderate"
    if rating <= 4.5:
        return "Moderate–high"
    return "High"


def _text_items(value: Any) -> list[str]:
    if isinstance(value, str):
        candidates = value.splitlines()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        candidates = [item for item in value if isinstance(item, str)]
    else:
        candidates = []
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = re.sub(r"^\s*[-*•]\s*", "", candidate).strip()
        dedupe_key = cleaned.casefold()
        if cleaned and dedupe_key not in seen:
            result.append(cleaned)
            seen.add(dedupe_key)
    return result


def synthesize_porter_five_forces(assessments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalise user-entered Porter ratings without supplying missing judgments."""

    supplied = assessments if isinstance(assessments, Mapping) else {}
    force_rows: list[dict[str, Any]] = []
    recognised_keys: set[str] = set()
    for key, label, aliases in PORTER_FORCES:
        recognised_keys.update(_normalise_key(item) for item in (key, *aliases))
        raw = _lookup_assessment(supplied, key, aliases)
        evidence: list[str] = []
        raw_rating = raw
        if isinstance(raw, Mapping):
            raw_rating = raw.get("rating", raw.get("score", raw.get("level")))
            evidence_value = raw.get("evidence", raw.get("notes", raw.get("rationale")))
            evidence = _text_items(evidence_value)
        rating = _porter_rating(raw_rating)
        force_rows.append(
            {
                "key": key,
                "label": label,
                "rating": _round_optional(rating, 1),
                "rating_label": _porter_label(rating) if rating is not None else "Not assessed",
                "evidence": evidence,
                "assessed": rating is not None,
                "input_valid": raw is None or rating is not None,
            }
        )

    assessed = [row for row in force_rows if row["assessed"]]
    coverage = 100.0 * len(assessed) / len(PORTER_FORCES)
    average_pressure = (
        sum(float(row["rating"]) for row in assessed) / len(assessed) if assessed else None
    )
    raw_attractiveness = (
        100.0 - (average_pressure - 1.0) * 25.0 if average_pressure is not None else None
    )
    adjusted_attractiveness = (
        50.0 + (raw_attractiveness - 50.0) * coverage / 100.0
        if raw_attractiveness is not None
        else None
    )
    evidence_count = sum(bool(row["evidence"]) for row in assessed)
    input_keys = {_normalise_key(key) for key in supplied}
    unknown_keys = sorted(input_keys - recognised_keys)
    gaps = [row["label"] for row in force_rows if not row["assessed"]]
    summary = (
        f"{len(assessed)} of {len(PORTER_FORCES)} forces assessed; average user-entered "
        f"competitive pressure is {average_pressure:.1f}/5 ({_porter_label(average_pressure).lower()})."
        if average_pressure is not None
        else "No valid Porter Five Forces ratings were supplied."
    )
    return {
        "available": bool(assessed),
        "forces": force_rows,
        "completed_forces": len(assessed),
        "coverage_pct": round(coverage, 1),
        "evidence_coverage_pct": round(100.0 * evidence_count / len(assessed), 1) if assessed else 0.0,
        "average_pressure": _round_optional(average_pressure, 1),
        "pressure_label": _porter_label(average_pressure) if average_pressure is not None else "Not assessed",
        "industry_attractiveness_score": _round_optional(adjusted_attractiveness, 1),
        "raw_industry_attractiveness_score": _round_optional(raw_attractiveness, 1),
        "gaps": gaps,
        "unknown_input_keys": unknown_keys,
        "summary": summary,
        "methodology": {
            "rating_scale": "1 (low competitive pressure) to 5 (high competitive pressure)",
            "attractiveness": "Competitive pressure is inverted; incomplete coverage shrinks the score toward 50",
            "input_policy": "Only user-entered ratings and evidence are included; missing forces are not inferred",
        },
    }


def synthesize_swot(assessments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Clean and count user-entered SWOT items without generating new claims."""

    supplied = assessments if isinstance(assessments, Mapping) else {}
    normalised = {_normalise_key(key): value for key, value in supplied.items()}
    singular_aliases = {
        "strengths": "strength",
        "weaknesses": "weakness",
        "opportunities": "opportunity",
        "threats": "threat",
    }
    quadrant_rows: dict[str, dict[str, Any]] = {}
    for key, label, scope, orientation in SWOT_QUADRANTS:
        singular = singular_aliases[key]
        raw = normalised.get(key, normalised.get(singular))
        items = _text_items(raw)
        quadrant_rows[key] = {
            "label": label,
            "scope": scope,
            "orientation": orientation,
            "items": items,
            "item_count": len(items),
        }

    completed = sum(bool(row["items"]) for row in quadrant_rows.values())
    total_items = sum(int(row["item_count"]) for row in quadrant_rows.values())
    positive_items = sum(
        int(row["item_count"]) for row in quadrant_rows.values() if row["orientation"] == "positive"
    )
    risk_items = total_items - positive_items
    gaps = [row["label"] for row in quadrant_rows.values() if not row["items"]]
    recognised = {
        _normalise_key(candidate)
        for key, _, _, _ in SWOT_QUADRANTS
        for candidate in (key, singular_aliases[key])
    }
    unknown_keys = sorted(set(normalised) - recognised)
    return {
        "available": total_items > 0,
        "quadrants": quadrant_rows,
        "coverage_pct": round(100.0 * completed / len(SWOT_QUADRANTS), 1),
        "completed_quadrants": completed,
        "item_count": total_items,
        "positive_item_count": positive_items,
        "risk_item_count": risk_items,
        "internal_item_count": sum(
            int(row["item_count"]) for row in quadrant_rows.values() if row["scope"] == "internal"
        ),
        "external_item_count": sum(
            int(row["item_count"]) for row in quadrant_rows.values() if row["scope"] == "external"
        ),
        "gaps": gaps,
        "unknown_input_keys": unknown_keys,
        "summary": (
            f"{total_items} user-entered SWOT items across {completed} of 4 quadrants "
            f"({positive_items} upside-oriented and {risk_items} risk-oriented)."
            if total_items
            else "No SWOT items were supplied."
        ),
        "methodology": {
            "input_policy": "Items are cleaned and deduplicated only; no strengths, weaknesses, opportunities, or threats are generated",
            "coverage": "Share of the four SWOT quadrants containing at least one user-entered item",
        },
    }


def build_industry_analysis(
    company_metrics: Mapping[str, Any] | None,
    peer_metrics: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
    *,
    porter_assessments: Mapping[str, Any] | None = None,
    swot_assessments: Mapping[str, Any] | None = None,
    **peer_options: Any,
) -> dict[str, Any]:
    """Build one UI-ready payload from quantitative and user-entered inputs."""

    peers = analyze_peer_comparison(company_metrics, peer_metrics, **peer_options)
    porter = synthesize_porter_five_forces(porter_assessments)
    swot = synthesize_swot(swot_assessments)
    return {
        "available": peers["available"] or porter["available"] or swot["available"],
        "peer_comparison": peers,
        "porter_five_forces": porter,
        "swot": swot,
        "input_policy": "Quantitative outputs use supplied metrics; qualitative outputs use only user-entered assessments.",
    }


__all__ = [
    "CATEGORY_LABELS",
    "DEFAULT_CATEGORY_WEIGHTS",
    "DEFAULT_METRIC_SPECS",
    "PORTER_FORCES",
    "SWOT_QUADRANTS",
    "PeerMetricSpec",
    "analyze_peer_comparison",
    "build_industry_analysis",
    "compare_company_to_peers",
    "select_similar_peers",
    "synthesize_porter_five_forces",
    "synthesize_swot",
]
