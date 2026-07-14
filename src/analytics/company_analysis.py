"""Evidence-oriented company deep dive and discounted cash-flow helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from html import unescape
from html.parser import HTMLParser
import json
import os
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
SEC_ANNUAL_FORMS = {"10-K", "20-F", "40-F"}


def _sec_user_agent() -> str:
    return os.environ.get(
        "SEC_USER_AGENT",
        "QuantSim/1.0 educational-research (configure SEC_USER_AGENT with contact email)",
    )


def _sec_request(url: str, timeout: int = 15) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": _sec_user_agent(),
            "Accept-Encoding": "identity",
            "Accept": "application/json,text/html,application/xhtml+xml",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _sec_request_json(url: str, timeout: int = 15) -> dict[str, Any]:
    return json.loads(_sec_request(url, timeout=timeout).decode("utf-8"))


@lru_cache(maxsize=1)
def _sec_ticker_map() -> dict[str, dict[str, Any]]:
    payload = _sec_request_json(SEC_TICKER_MAP_URL)
    return {
        str(item.get("ticker") or "").upper(): dict(item)
        for item in payload.values()
        if isinstance(item, Mapping) and item.get("ticker")
    }


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "table", "sup"}:
            self._ignored_depth += 1
        elif tag in {"p", "li", "br", "h2", "h3"} and not self._ignored_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "table", "sup"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag in {"p", "li"} and not self._ignored_depth:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def _html_to_text(value: str) -> str:
    parser = _PlainTextParser()
    parser.feed(value or "")
    text = unescape("".join(parser.parts))
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def classify_biography_sections(sections: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Select MediaWiki section indexes that contain education or career facts."""
    result = {"education": [], "career": []}
    for section in sections:
        title = str(section.get("line") or "").lower()
        index = str(section.get("index") or "")
        if not index:
            continue
        if any(token in title for token in ("education", "early life", "childhood")):
            result["education"].append(index)
        if any(token in title for token in ("career", "professional life", "business life")):
            result["career"].append(index)
    return result


def _wikipedia_request(params: Mapping[str, Any], timeout: int = 12) -> dict[str, Any]:
    query = urlencode({**params, "format": "json", "formatversion": 2})
    request = Request(
        f"{WIKIPEDIA_API_URL}?{query}",
        headers={"User-Agent": "QuantSim-CompanyResearch/1.0 (educational dashboard)"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_management_biography(name: str, company_name: str) -> dict[str, Any]:
    """Fetch source-linked education and career sections from English Wikipedia."""
    clean_name = str(name or "").strip()
    if not clean_name:
        return {"available": False, "error": "Manager name is missing."}

    search = _wikipedia_request(
        {
            "action": "query",
            "list": "search",
            "srsearch": f'"{clean_name}" "{company_name}"',
            "srnamespace": 0,
            "srlimit": 5,
        }
    )
    candidates = search.get("query", {}).get("search", [])
    if not candidates:
        return {"available": False, "error": "No matching public biography was found."}

    last_name = clean_name.split()[-1].lower()
    company_tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9]+", company_name) if len(token) >= 4]
    ranked: list[tuple[int, Mapping[str, Any]]] = []
    for candidate in candidates:
        haystack = f"{candidate.get('title', '')} {_html_to_text(candidate.get('snippet', ''))}".lower()
        score = (3 if last_name in str(candidate.get("title", "")).lower() else 0)
        score += sum(1 for token in company_tokens[:3] if token in haystack)
        ranked.append((score, candidate))
    score, candidate = max(ranked, key=lambda item: item[0])
    if score < 3:
        return {"available": False, "error": "Search results could not be matched confidently to this manager."}

    title = str(candidate.get("title"))
    page = _wikipedia_request({"action": "parse", "page": title, "prop": "sections"})
    sections = page.get("parse", {}).get("sections", [])
    selected = classify_biography_sections(sections)

    def load_group(indexes: list[str]) -> str:
        chunks: list[str] = []
        for section_index in indexes[:3]:
            parsed = _wikipedia_request(
                {"action": "parse", "page": title, "section": section_index, "prop": "text"}
            )
            html_value = parsed.get("parse", {}).get("text", "")
            if isinstance(html_value, Mapping):
                html_value = html_value.get("*", "")
            plain = _html_to_text(html_value)
            if plain:
                chunks.append(plain)
        return "\n\n".join(chunks)

    intro_data = _wikipedia_request(
        {"action": "query", "prop": "extracts", "titles": title, "exintro": 1, "explaintext": 1}
    )
    pages = intro_data.get("query", {}).get("pages", [])
    if isinstance(pages, Mapping):
        pages = list(pages.values())
    intro = str(pages[0].get("extract") or "") if pages else ""
    education = load_group(selected["education"])
    career = load_group(selected["career"])
    if not career:
        career = intro
    if not education and not career:
        return {"available": False, "error": "The matched biography contains no usable education or career text."}
    return {
        "available": True,
        "name": clean_name,
        "matched_title": title,
        "education": education or "Education is not documented in the matched public biography.",
        "career": career or "Career history is not documented in the matched public biography.",
        "source_url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
        "source_name": "English Wikipedia",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "verification_note": "Public-biography match; verify material facts against company filings or official biographies.",
    }


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _statement_row(frame: pd.DataFrame, *names: str) -> list[float]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    normalized = {str(index).lower().replace(" ", ""): index for index in frame.index}
    for name in names:
        key = name.lower().replace(" ", "")
        if key in normalized:
            return [
                _number(value, np.nan)
                for value in frame.loc[normalized[key]].tolist()
                if pd.notna(value)
            ]
    return []


def calculate_dcf(
    free_cash_flow: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    years: int,
    cash: float,
    debt: float,
    shares_outstanding: float,
    current_price: float = 0.0,
) -> dict[str, Any]:
    """Calculate an unlevered five-to-ten-year DCF from normalized inputs."""
    if free_cash_flow <= 0:
        return {"available": False, "error": "DCF requires positive normalized free cash flow."}
    if shares_outstanding <= 0:
        return {"available": False, "error": "Shares outstanding are unavailable or non-positive."}
    if years < 1 or years > 20:
        return {"available": False, "error": "Projection horizon must be between 1 and 20 years."}
    if discount_rate <= terminal_growth_rate:
        return {"available": False, "error": "Discount rate must be higher than terminal growth."}

    projected: list[dict[str, float]] = []
    present_value_sum = 0.0
    annual_fcf = float(free_cash_flow)
    for year in range(1, years + 1):
        annual_fcf *= 1.0 + growth_rate
        present_value = annual_fcf / ((1.0 + discount_rate) ** year)
        present_value_sum += present_value
        projected.append({"year": year, "free_cash_flow": annual_fcf, "present_value": present_value})

    terminal_value = annual_fcf * (1.0 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_present_value = terminal_value / ((1.0 + discount_rate) ** years)
    enterprise_value = present_value_sum + terminal_present_value
    equity_value = enterprise_value + cash - debt
    fair_value_per_share = equity_value / shares_outstanding
    upside = ((fair_value_per_share / current_price) - 1.0) if current_price > 0 else None
    return {
        "available": True,
        "projected": projected,
        "terminal_value": terminal_value,
        "terminal_present_value": terminal_present_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "fair_value_per_share": fair_value_per_share,
        "current_price": current_price,
        "upside_pct": upside,
        "terminal_value_share": terminal_present_value / enterprise_value if enterprise_value else 0.0,
        "assumptions": {
            "free_cash_flow": free_cash_flow,
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_growth_rate": terminal_growth_rate,
            "years": years,
            "cash": cash,
            "debt": debt,
            "shares_outstanding": shares_outstanding,
        },
    }


def default_dcf_assumptions(info: Mapping[str, Any]) -> dict[str, float | int]:
    growth_candidates = [
        _number(info.get("revenueGrowth"), np.nan),
        _number(info.get("earningsGrowth"), np.nan),
        _number(info.get("earningsQuarterlyGrowth"), np.nan),
    ]
    finite_growth = [value for value in growth_candidates if np.isfinite(value)]
    growth_rate = float(np.median(finite_growth)) if finite_growth else 0.05
    growth_rate = min(max(growth_rate, -0.05), 0.15)
    return {
        "free_cash_flow": _number(info.get("freeCashflow")),
        "growth_rate": growth_rate,
        "discount_rate": 0.10,
        "terminal_growth_rate": 0.025,
        "years": 5,
        "cash": _number(info.get("totalCash")),
        "debt": _number(info.get("totalDebt")),
        "shares_outstanding": _number(info.get("sharesOutstanding")),
        "current_price": _number(info.get("currentPrice") or info.get("regularMarketPrice")),
    }


def build_dcf_scenarios(info: Mapping[str, Any], assumptions: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    base = {**default_dcf_assumptions(info), **dict(assumptions or {})}
    base_growth = _number(base["growth_rate"])
    base_discount = _number(base["discount_rate"])
    base_terminal = _number(base["terminal_growth_rate"])
    variants = {
        "Bear": {"growth_rate": max(-0.10, base_growth - 0.04), "discount_rate": base_discount + 0.02, "terminal_growth_rate": max(0.0, base_terminal - 0.01)},
        "Base": {},
        "Bull": {"growth_rate": min(0.30, base_growth + 0.04), "discount_rate": max(0.055, base_discount - 0.015), "terminal_growth_rate": min(0.04, base_terminal + 0.01)},
    }
    return {
        name: calculate_dcf(**{**base, **changes})
        for name, changes in variants.items()
    }


def analyze_moat(info: Mapping[str, Any]) -> dict[str, Any]:
    """Score observable moat signals without claiming qualitative certainty."""
    signals = [
        ("Pricing power / gross margin", _number(info.get("grossMargins")) >= 0.40, f"Gross margin {_number(info.get('grossMargins')):.1%}"),
        ("Operating efficiency", _number(info.get("operatingMargins")) >= 0.15, f"Operating margin {_number(info.get('operatingMargins')):.1%}"),
        ("Returns on shareholder capital", _number(info.get("returnOnEquity")) >= 0.15, f"ROE {_number(info.get('returnOnEquity')):.1%}"),
        ("Organic growth", _number(info.get("revenueGrowth")) >= 0.05, f"Revenue growth {_number(info.get('revenueGrowth')):.1%}"),
        ("Cash generation", _number(info.get("freeCashflow")) > 0, f"FCF ${_number(info.get('freeCashflow')) / 1e9:,.2f}B"),
        ("Scale", _number(info.get("marketCap")) >= 10e9, f"Market cap ${_number(info.get('marketCap')) / 1e9:,.1f}B"),
    ]
    score = sum(passed for _, passed, _ in signals)
    label = "Wide moat signal" if score >= 5 else "Narrow moat signal" if score >= 3 else "Moat not demonstrated"
    return {
        "score": score,
        "max_score": len(signals),
        "label": label,
        "signals": [{"name": name, "passed": passed, "evidence": evidence} for name, passed, evidence in signals],
        "warning": "This is a quantitative moat screen, not proof of a durable competitive advantage.",
    }


def analyze_track_record(info: Mapping[str, Any], history: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Build factual operating successes and setbacks from observable metrics."""
    successes: list[str] = []
    failures: list[str] = []
    revenue_growth = _number(info.get("revenueGrowth"), np.nan)
    earnings_growth = _number(info.get("earningsGrowth"), np.nan)
    free_cash_flow = _number(info.get("freeCashflow"), np.nan)
    operating_margin = _number(info.get("operatingMargins"), np.nan)

    if np.isfinite(revenue_growth):
        target = successes if revenue_growth > 0 else failures
        target.append(f"Latest reported revenue growth: {revenue_growth:+.1%}.")
    if np.isfinite(earnings_growth):
        target = successes if earnings_growth > 0 else failures
        target.append(f"Latest reported earnings growth: {earnings_growth:+.1%}.")
    if np.isfinite(free_cash_flow):
        target = successes if free_cash_flow > 0 else failures
        target.append(f"Trailing free cash flow: ${free_cash_flow / 1e9:+,.2f}B.")
    if np.isfinite(operating_margin):
        target = successes if operating_margin >= 0.15 else failures
        target.append(f"Operating margin: {operating_margin:.1%}.")

    if isinstance(history, pd.DataFrame) and not history.empty and "Close" in history:
        close = pd.Series(history["Close"]).dropna().astype(float)
        if len(close) > 1 and close.iloc[0] > 0:
            total_return = close.iloc[-1] / close.iloc[0] - 1.0
            rolling_high = close.cummax()
            max_drawdown = float((close / rolling_high - 1.0).min())
            (successes if total_return >= 0 else failures).append(f"Five-year share-price return: {total_return:+.1%}.")
            if max_drawdown <= -0.30:
                failures.append(f"Five-year maximum share-price drawdown: {max_drawdown:.1%}.")

    return {"successes": successes, "failures": failures}


def format_statement(frame: pd.DataFrame) -> pd.DataFrame:
    """Transpose a yfinance statement into a compact, display-ready table."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    display = frame.copy()
    display.columns = [getattr(column, "date", lambda: column)().isoformat() if hasattr(column, "date") else str(column)[:10] for column in display.columns]
    for column in display.columns:
        display[column] = pd.to_numeric(display[column], errors="coerce") / 1e6
    display.index = display.index.astype(str)
    return display


def _xml_local_name(value: str) -> str:
    return str(value).rsplit("}", 1)[-1].lower()


def _xml_attr(element: ET.Element, name: str, default: str = "") -> str:
    target = name.lower()
    for key, value in element.attrib.items():
        if _xml_local_name(key) == target:
            return str(value)
    return default


def _humanize_xbrl_member(value: str) -> str:
    label = str(value).split(":")[-1]
    label = re.sub(r"Member$", "", label, flags=re.IGNORECASE)
    label = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", label)
    label = re.sub(r"[_-]+", " ", label)
    label = re.sub(r"\s+", " ", label).strip() or "Unknown region"
    compact = re.sub(r"[^a-z]", "", label.lower())
    if compact in {"us", "usa", "unitedstates", "unitedstatesofamerica"}:
        return "United States"
    if compact in {"nonus", "nonusa", "nonunitedstates", "outsideunitedstates", "international", "foreign"}:
        return "Non-US / International"
    return label


def _is_geographic_axis(axis: str) -> bool:
    normalized = str(axis).lower()
    return any(token in normalized for token in ("geograph", "country", "region", "domicile"))


def _is_revenue_concept(concept: str) -> bool:
    local = str(concept).split(":")[-1].lower()
    excluded = ("cost", "growth", "percentage", "percent", "per share", "pershare", "remaining")
    if any(token in local for token in excluded):
        return False
    return "revenue" in local or "netsales" in local or local.endswith("sales")


def _parse_inline_number(element: ET.Element) -> float | None:
    if _xml_attr(element, "nil").lower() == "true":
        return None
    raw = "".join(element.itertext()).strip().replace("\u2212", "-").replace("\u2014", "")
    if not raw or raw in {"-", "—", "N/A"}:
        return None
    negative_parentheses = raw.startswith("(") and raw.endswith(")")
    transform = _xml_attr(element, "format").lower()
    if "numcommadot" in transform:
        normalized = raw.replace(".", "").replace(",", ".")
    else:
        normalized = raw.replace(",", "")
    normalized = re.sub(r"[^0-9eE+\-.]", "", normalized)
    try:
        value = float(normalized)
        scale = int(_xml_attr(element, "scale", "0") or 0)
        value *= 10.0 ** scale
        if negative_parentheses or _xml_attr(element, "sign") == "-":
            value = -abs(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _extract_geographic_revenue_from_ixbrl(
    document: bytes | str,
    report_date: str = "",
) -> dict[str, Any]:
    """Extract a non-overlapping annual geographic revenue table from Inline XBRL."""
    payload = document if isinstance(document, bytes) else document.encode("utf-8")
    # Annual reports are XHTML but occasionally contain HTML named entities
    # that XML parsers do not know. Preserve the five XML entities and expand
    # only the remaining named HTML entities to Unicode.
    text = payload.decode("utf-8", errors="replace")
    xml_entities = {"amp", "lt", "gt", "quot", "apos"}
    text = re.sub(
        r"&([A-Za-z][A-Za-z0-9]+);",
        lambda match: match.group(0) if match.group(1) in xml_entities else unescape(match.group(0)),
        text,
    )
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        return {"available": False, "error": f"Annual filing is not parseable Inline XBRL: {exc}."}

    contexts: dict[str, dict[str, Any]] = {}
    units: dict[str, str] = {}
    for element in root.iter():
        local = _xml_local_name(element.tag)
        if local == "context":
            context_id = _xml_attr(element, "id")
            if not context_id:
                continue
            members: list[tuple[str, str]] = []
            start_date = end_date = instant = ""
            for child in element.iter():
                child_local = _xml_local_name(child.tag)
                if child_local == "explicitmember":
                    members.append((_xml_attr(child, "dimension"), "".join(child.itertext()).strip()))
                elif child_local == "startdate":
                    start_date = "".join(child.itertext()).strip()
                elif child_local == "enddate":
                    end_date = "".join(child.itertext()).strip()
                elif child_local == "instant":
                    instant = "".join(child.itertext()).strip()
            contexts[context_id] = {
                "members": members,
                "start_date": start_date,
                "end_date": end_date or instant,
            }
        elif local == "unit":
            unit_id = _xml_attr(element, "id")
            measures = ["".join(child.itertext()).strip().split(":")[-1] for child in element.iter() if _xml_local_name(child.tag) == "measure"]
            if unit_id and measures:
                units[unit_id] = " per ".join(measures)

    dimensional_facts: list[dict[str, Any]] = []
    consolidated_facts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()
    for element in root.iter():
        if _xml_local_name(element.tag) != "nonfraction":
            continue
        concept = _xml_attr(element, "name")
        context_ref = _xml_attr(element, "contextref")
        if not _is_revenue_concept(concept) or context_ref not in contexts:
            continue
        value = _parse_inline_number(element)
        if value is None or value <= 0:
            continue
        fingerprint = (context_ref, concept, float(value))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        context = contexts[context_ref]
        members = list(context["members"])
        geo_members = [(axis, member) for axis, member in members if _is_geographic_axis(axis)]
        fact = {
            "concept": concept,
            "value": float(value),
            "start_date": context["start_date"],
            "end_date": context["end_date"],
            "currency": units.get(_xml_attr(element, "unitref"), _xml_attr(element, "unitref") or "reporting currency"),
            "member_count": len(members),
        }
        if geo_members:
            fact["region"] = _humanize_xbrl_member(geo_members[0][1])
            fact["only_geography_dimension"] = len(members) == 1
            dimensional_facts.append(fact)
        elif not members:
            consolidated_facts.append(fact)

    excluded_regions = ("total", "consolidated", "elimination")
    groups: dict[tuple[str, str, str, str, bool], dict[str, float]] = {}
    for fact in dimensional_facts:
        region = str(fact["region"])
        if any(token in region.lower() for token in excluded_regions):
            continue
        key = (
            str(fact["concept"]),
            str(fact["start_date"]),
            str(fact["end_date"]),
            str(fact["currency"]),
            bool(fact["only_geography_dimension"]),
        )
        groups.setdefault(key, {})[region] = max(
            float(fact["value"]),
            groups.get(key, {}).get(region, 0.0),
        )

    candidates: list[dict[str, Any]] = []
    for key, region_values in groups.items():
        concept, start_date, end_date, currency, only_geo = key
        if len(region_values) < 2:
            continue
        total_candidates = [
            float(fact["value"])
            for fact in consolidated_facts
            if fact["concept"] == concept
            and fact["start_date"] == start_date
            and fact["end_date"] == end_date
            and fact["currency"] == currency
        ]
        total_revenue = max(total_candidates) if total_candidates else None
        disclosed_sum = float(sum(region_values.values()))
        coverage = disclosed_sum / total_revenue if total_revenue and total_revenue > 0 else None
        if coverage is not None and coverage > 1.15:
            continue
        date_match = 1 if report_date and end_date == report_date else 0
        latest_key = end_date or ""
        coverage_quality = 1.0 - abs(1.0 - coverage) if coverage is not None else 0.25
        candidates.append({
            "concept": concept,
            "start_date": start_date,
            "end_date": end_date,
            "currency": currency,
            "only_geo": only_geo,
            "regions": region_values,
            "total_revenue": total_revenue,
            "coverage": coverage,
            "rank": (date_match, latest_key, int(only_geo), coverage_quality, len(region_values)),
        })

    if not candidates:
        return {
            "available": False,
            "error": "The latest annual filing did not expose a reliable non-overlapping geographic revenue table in Inline XBRL.",
        }
    selected = max(candidates, key=lambda item: item["rank"])
    rows = [
        {"region": region, "revenue": float(value)}
        for region, value in selected["regions"].items()
    ]
    total_revenue = selected["total_revenue"]
    disclosed_sum = float(sum(row["revenue"] for row in rows))
    if total_revenue and total_revenue > disclosed_sum * 1.01:
        rows.append({"region": "Not separately disclosed", "revenue": float(total_revenue - disclosed_sum)})
    rows.sort(key=lambda item: item["revenue"], reverse=True)
    return {
        "available": True,
        "rows": rows,
        "currency": selected["currency"],
        "fiscal_start": selected["start_date"],
        "fiscal_end": selected["end_date"],
        "concept": selected["concept"],
        "consolidated_revenue": total_revenue,
        "coverage_ratio": selected["coverage"],
        "extraction_note": "Only contexts with a geographic dimension were used; overlapping multi-axis tables are deprioritized.",
    }


def analyze_geographic_revenue(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score geographic revenue concentration without inferring regional quality."""
    combined: dict[str, float] = {}
    for row in rows:
        region = str(row.get("region") or row.get("Region") or "").strip()
        revenue = _number(row.get("revenue", row.get("Revenue")), np.nan)
        if region and np.isfinite(revenue) and revenue > 0:
            combined[region] = combined.get(region, 0.0) + float(revenue)
    total = float(sum(combined.values()))
    if len(combined) < 2 or total <= 0:
        return {"available": False, "error": "At least two regions with positive revenue are required."}

    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    analyzed_rows: list[dict[str, Any]] = []
    for region, revenue in ranked:
        share = revenue / total
        importance = "Core" if share >= 0.35 else "Major" if share >= 0.20 else "Material" if share >= 0.10 else "Limited"
        analyzed_rows.append({
            "region": region,
            "revenue": revenue,
            "share": share,
            "strategic_importance": importance,
            "hhi_contribution": share ** 2,
        })

    hhi = float(sum(row["hhi_contribution"] for row in analyzed_rows))
    effective_regions = float(1.0 / hhi) if hhi > 0 else 0.0
    top_share = float(analyzed_rows[0]["share"])
    top_two_share = float(sum(row["share"] for row in analyzed_rows[:2]))
    aggregate_tokens = ("international", "non-us", "rest of world", "other countries", "foreign")
    limited_granularity = any(
        any(token in row["region"].lower() for token in aggregate_tokens)
        for row in analyzed_rows
    )
    if limited_granularity and len(analyzed_rows) == 2 and top_share <= 0.55:
        score, label, concentration = 3, "Balanced split, limited detail", "Indeterminate"
    elif top_share <= 0.40 and effective_regions >= 3.0:
        score, label, concentration = 5, "Highly diversified", "Low"
    elif top_share <= 0.50 and effective_regions >= 2.5:
        score, label, concentration = 4, "Diversified", "Moderate-low"
    elif top_share <= 0.65 and effective_regions >= 2.0:
        score, label, concentration = 3, "Balanced but concentrated", "Moderate"
    elif top_share <= 0.80:
        score, label, concentration = 2, "Concentrated", "High"
    else:
        score, label, concentration = 1, "Highly concentrated", "Very high"

    largest = analyzed_rows[0]
    strengths: list[str] = []
    risks: list[str] = []
    if score >= 4:
        strengths.append("Revenue is spread across several economically meaningful regions, reducing dependence on one market.")
    elif score == 3:
        strengths.append("The company has more than one meaningful revenue engine, although concentration remains material.")
    if top_share > 0.50:
        risks.append(f"{largest['region']} generates {top_share:.1%} of disclosed revenue; local demand, regulation and currency moves can dominate results.")
    if top_two_share > 0.80 and len(analyzed_rows) > 2:
        risks.append(f"The two largest regions account for {top_two_share:.1%} of revenue.")
    if limited_granularity:
        risks.append("At least one disclosure bucket aggregates multiple countries, so true country-level concentration cannot be measured.")
    undisclosed = next((row for row in analyzed_rows if row["region"].lower() == "not separately disclosed"), None)
    if undisclosed and undisclosed["share"] >= 0.10:
        risks.append(f"{undisclosed['share']:.1%} of revenue is not allocated to a named region, limiting transparency.")
    if not risks:
        risks.append("Regional shares alone do not measure margins, growth, political risk or currency hedging; review those separately.")

    return {
        "available": True,
        "rows": analyzed_rows,
        "total_revenue": total,
        "score": score,
        "max_score": 5,
        "label": label,
        "concentration": concentration,
        "hhi": hhi,
        "effective_regions": effective_regions,
        "top_region": largest["region"],
        "top_region_share": top_share,
        "top_two_share": top_two_share,
        "limited_granularity": limited_granularity,
        "strengths": strengths,
        "risks": risks,
        "interpretation": (
            f"{largest['region']} is the largest disclosed market at {top_share:.1%}. "
            f"The distribution is equivalent to {effective_regions:.1f} equally sized regions and is rated {label.lower()}."
        ),
        "warning": "This rating measures revenue diversification, not regional profitability, growth quality or investment attractiveness.",
    }


def fetch_geographic_revenue(
    ticker: str,
    sec_filings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fetch the latest reliable geographic revenue disclosure from SEC filings."""
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return {"available": False, "error": "Ticker is required."}
    mirror_result: dict[str, Any] | None = None
    try:
        # Yahoo exposes direct mirrored copies of SEC exhibits. Prefer that
        # already-resolved annual filing URL because SEC can reject generic
        # cloud IPs even when the request is otherwise compliant.
        for filing in sec_filings or []:
            form = str(filing.get("type") or "").upper()
            if form not in SEC_ANNUAL_FORMS:
                continue
            exhibits = filing.get("exhibits") if isinstance(filing.get("exhibits"), Mapping) else {}
            filing_url = str(exhibits.get(form) or "")
            if not filing_url:
                filing_url = next(
                    (str(url) for exhibit_type, url in exhibits.items() if str(exhibit_type).upper().startswith(form)),
                    "",
                )
            if not filing_url:
                continue
            filing_date = str(filing.get("date") or "")
            extracted = _extract_geographic_revenue_from_ixbrl(_sec_request(filing_url))
            if extracted.get("available"):
                mirror_result = {
                    **extracted,
                    "analysis": analyze_geographic_revenue(extracted["rows"]),
                    "source_name": f"SEC EDGAR {form} via Yahoo Finance filing mirror",
                    "source_url": filing_url,
                    "filing_date": filing_date,
                    "report_date": extracted.get("fiscal_end", ""),
                }
                break

        # The SEC asks automated clients to identify a real organization and
        # contact. Without that deployment setting, use the filing mirror or
        # the manual UI fallback instead of sending an undeclared request.
        if not os.environ.get("SEC_USER_AGENT"):
            if mirror_result is not None:
                return mirror_result
            return {
                "available": False,
                "error": "No parseable annual filing mirror was found. Configure SEC_USER_AGENT for official EDGAR access or enter annual-report values manually.",
                "source_name": "SEC EDGAR annual filing",
            }

        mapping = _sec_ticker_map().get(symbol)
        if not mapping:
            return mirror_result or {"available": False, "error": "Ticker is not mapped to an SEC registrant."}
        cik = str(int(mapping["cik_str"])).zfill(10)
        submissions = _sec_request_json(SEC_SUBMISSIONS_URL.format(cik=cik))
        recent = submissions.get("filings", {}).get("recent", {})
        forms = list(recent.get("form", []))
        selected_index = next((index for index, form in enumerate(forms) if form in SEC_ANNUAL_FORMS), None)
        if selected_index is None:
            return {"available": False, "error": "No recent 10-K, 20-F or 40-F filing was found."}
        accession_number = str(recent.get("accessionNumber", [])[selected_index])
        document_name = str(recent.get("primaryDocument", [])[selected_index])
        filing_date = str(recent.get("filingDate", [])[selected_index])
        report_date = str(recent.get("reportDate", [])[selected_index])
        accession_path = accession_number.replace("-", "")
        filing_url = SEC_ARCHIVES_URL.format(
            cik=str(int(cik)),
            accession=accession_path,
            document=quote(document_name),
        )
        extracted = _extract_geographic_revenue_from_ixbrl(_sec_request(filing_url), report_date=report_date)
        if not extracted.get("available"):
            if mirror_result is not None:
                return mirror_result
            return {
                **extracted,
                "source_name": "SEC EDGAR annual filing",
                "source_url": filing_url,
                "filing_date": filing_date,
                "report_date": report_date,
            }
        analysis = analyze_geographic_revenue(extracted["rows"])
        return {
            **extracted,
            "analysis": analysis,
            "source_name": f"SEC EDGAR {forms[selected_index]}",
            "source_url": filing_url,
            "filing_date": filing_date,
            "report_date": report_date,
            "cik": cik,
        }
    except Exception as exc:
        if mirror_result is not None:
            return mirror_result
        return {
            "available": False,
            "error": f"SEC geographic revenue extraction failed: {exc}",
            "source_name": "SEC EDGAR annual filing",
        }


def fetch_company_data(ticker: str) -> dict[str, Any]:
    """Fetch a broad company snapshot from Yahoo Finance with graceful gaps."""
    import yfinance as yf

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker is required.")
    company = yf.Ticker(symbol)
    info = company.info or {}
    history = company.history(period="5y", interval="1d", auto_adjust=False)
    try:
        news = company.news or []
    except Exception:
        news = []
    try:
        sec_filings = company.sec_filings or []
    except Exception:
        sec_filings = []
    officers = info.get("companyOfficers") if isinstance(info.get("companyOfficers"), list) else []
    scalar_metrics = {
        key: value
        for key, value in info.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    geographic_revenue = fetch_geographic_revenue(symbol, sec_filings=sec_filings)
    return {
        "ticker": symbol,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "info": info,
        "metrics": scalar_metrics,
        "officers": officers,
        "history": history,
        "news": news[:20],
        "sec_filings": sec_filings,
        "income_statement": company.income_stmt,
        "balance_sheet": company.balance_sheet,
        "cash_flow": company.cashflow,
        "quarterly_income_statement": company.quarterly_income_stmt,
        "quarterly_balance_sheet": company.quarterly_balance_sheet,
        "quarterly_cash_flow": company.quarterly_cashflow,
        "geographic_revenue": geographic_revenue,
    }
