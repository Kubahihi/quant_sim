"""Evidence-oriented company deep dive and discounted cash-flow helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
import json
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


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
            plain = _html_to_text(html_value)
            if plain:
                chunks.append(plain)
        return "\n\n".join(chunks)

    intro_data = _wikipedia_request(
        {"action": "query", "prop": "extracts", "titles": title, "exintro": 1, "explaintext": 1}
    )
    pages = intro_data.get("query", {}).get("pages", [])
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
    officers = info.get("companyOfficers") if isinstance(info.get("companyOfficers"), list) else []
    scalar_metrics = {
        key: value
        for key, value in info.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    return {
        "ticker": symbol,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "info": info,
        "metrics": scalar_metrics,
        "officers": officers,
        "history": history,
        "news": news[:20],
        "income_statement": company.income_stmt,
        "balance_sheet": company.balance_sheet,
        "cash_flow": company.cashflow,
        "quarterly_income_statement": company.quarterly_income_stmt,
        "quarterly_balance_sheet": company.quarterly_balance_sheet,
        "quarterly_cash_flow": company.quarterly_cashflow,
    }
