from __future__ import annotations

from typing import Any, Mapping
import json
import re

import pandas as pd
from openai import OpenAI
import streamlit as st

from src.ai.ai_review import DEFAULT_GROQ_MODEL, resolve_groq_api_key
from src.stock_picker.screener import (
    apply_classic_filters,
    apply_technical_indicators,
    calculate_quant_score,
    rank_stocks,
)


def _default_parse_payload(query: str, explanation: str) -> dict[str, Any]:
    return {
        "query": query,
        "filters": {},
        "weights": {},
        "sort": {"by": "QuantScore", "ascending": False},
        "explanation": explanation,
        "parse_source": "fallback",
    }


def _extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("AI parser returned empty text.")

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No valid JSON object in AI parser response.")

    payload = json.loads(cleaned[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("AI parser JSON payload must be an object.")
    return payload


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _heuristic_parse_query(query: str) -> dict[str, Any]:
    text = f" {query.lower()} "
    filters: dict[str, Any] = {}
    explanation: list[str] = []

    if "large cap" in text or "mega cap" in text:
        filters["market_cap_min"] = 10_000_000_000
        explanation.append("Applied large-cap minimum market cap.")
    if "small cap" in text:
        filters["market_cap_max"] = 2_000_000_000
        explanation.append("Applied small-cap maximum market cap.")
    if "low pe" in text or "undervalued" in text or "value" in text:
        filters.setdefault("valuation", {})["pe_max"] = 20.0
        explanation.append("Applied conservative P/E ceiling.")
    if "growth" in text:
        filters.setdefault("growth", {})["revenue_growth_min"] = 0.08
        explanation.append("Applied revenue growth floor.")
    if "high dividend" in text or "dividend" in text:
        filters.setdefault("dividend", {})["dividend_yield_min"] = 0.02
        explanation.append("Applied dividend yield minimum.")
    if "momentum" in text:
        filters.setdefault("momentum", {})["return_52w_min"] = 0.10
        explanation.append("Applied 52-week momentum filter.")
    if "low beta" in text:
        filters["beta_max"] = 1.0
        explanation.append("Applied beta ceiling.")

    # Naive sector extraction: words after "sector".
    sector_matches = re.findall(r"sector(?:s)?\s+([a-z\s,&-]+)", query, flags=re.IGNORECASE)
    if sector_matches:
        raw_sector = sector_matches[0]
        sectors = [item.strip() for item in re.split(r",|and|&", raw_sector) if item.strip()]
        if sectors:
            filters["sectors"] = sectors
            explanation.append("Applied sector hints from query text.")

    parsed = _default_parse_payload(
        query=query,
        explanation=" ".join(explanation) if explanation else "Used keyword fallback parsing.",
    )
    parsed["filters"] = filters
    return parsed


def _normalize_parsed_payload(query: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    filters = dict(payload.get("filters") or {})
    weights = dict(payload.get("weights") or payload.get("weight_preferences") or {})
    sort = dict(payload.get("sort") or payload.get("sort_preference") or {})

    normalized_filters: dict[str, Any] = {
        "market_cap_min": _to_float(filters.get("market_cap_min")),
        "market_cap_max": _to_float(filters.get("market_cap_max")),
        "price_min": _to_float(filters.get("price_min")),
        "price_max": _to_float(filters.get("price_max")),
        "beta_min": _to_float(filters.get("beta_min")),
        "beta_max": _to_float(filters.get("beta_max")),
        "avg_volume_min": _to_float(filters.get("avg_volume_min")),
        "sectors": _to_str_list(filters.get("sectors")),
        "industries": _to_str_list(filters.get("industries")),
        "exchanges": _to_str_list(filters.get("exchanges")),
        "liquidity_prefilter": bool(filters.get("liquidity_prefilter", False)),
        "valuation": {
            "pe_min": _to_float((filters.get("valuation") or {}).get("pe_min")),
            "pe_max": _to_float((filters.get("valuation") or {}).get("pe_max")),
            "forward_pe_min": _to_float((filters.get("valuation") or {}).get("forward_pe_min")),
            "forward_pe_max": _to_float((filters.get("valuation") or {}).get("forward_pe_max")),
            "peg_min": _to_float((filters.get("valuation") or {}).get("peg_min")),
            "peg_max": _to_float((filters.get("valuation") or {}).get("peg_max")),
        },
        "growth": {
            "revenue_growth_min": _to_float((filters.get("growth") or {}).get("revenue_growth_min")),
            "revenue_growth_max": _to_float((filters.get("growth") or {}).get("revenue_growth_max")),
            "earnings_growth_min": _to_float((filters.get("growth") or {}).get("earnings_growth_min")),
            "earnings_growth_max": _to_float((filters.get("growth") or {}).get("earnings_growth_max")),
        },
        "quality": {
            "roe_min": _to_float((filters.get("quality") or {}).get("roe_min")),
            "roe_max": _to_float((filters.get("quality") or {}).get("roe_max")),
            "roa_min": _to_float((filters.get("quality") or {}).get("roa_min")),
            "roa_max": _to_float((filters.get("quality") or {}).get("roa_max")),
        },
        "momentum": {
            "return_52w_min": _to_float((filters.get("momentum") or {}).get("return_52w_min")),
            "return_52w_max": _to_float((filters.get("momentum") or {}).get("return_52w_max")),
        },
        "dividend": {
            "dividend_yield_min": _to_float((filters.get("dividend") or {}).get("dividend_yield_min")),
            "dividend_yield_max": _to_float((filters.get("dividend") or {}).get("dividend_yield_max")),
        },
    }

    normalized_weights = {
        "value": _to_float(weights.get("value")),
        "growth": _to_float(weights.get("growth")),
        "quality": _to_float(weights.get("quality")),
        "momentum": _to_float(weights.get("momentum")),
        "stability": _to_float(weights.get("stability")),
        "dividend": _to_float(weights.get("dividend")),
    }
    normalized_weights = {key: value for key, value in normalized_weights.items() if value is not None}

    normalized_sort = {
        "by": str(sort.get("by", "QuantScore")),
        "ascending": bool(sort.get("ascending", False)),
    }

    return {
        "query": query,
        "filters": normalized_filters,
        "weights": normalized_weights,
        "sort": normalized_sort,
        "explanation": str(payload.get("explanation", "")).strip() or "AI parser translated the request into structured filters.",
        "parse_source": str(payload.get("parse_source", "groq")),
    }


def parse_ai_query(query: str) -> dict[str, Any]:
    """
    Parse natural language screener intent into structured filter JSON.

    Falls back to deterministic keyword parser if Groq is unavailable or fails.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return _default_parse_payload(query="", explanation="No query provided.")

    try:
        streamlit_secrets = st.secrets
    except Exception:
        streamlit_secrets = None

    api_key = resolve_groq_api_key(streamlit_secrets)
    if not api_key:
        return _heuristic_parse_query(cleaned_query)

    system_prompt = (
        "You are an equity screener query parser. "
        "Return strict JSON only. "
        "Schema: {"
        "\"filters\": {"
        "\"market_cap_min\": number|null, \"market_cap_max\": number|null, "
        "\"price_min\": number|null, \"price_max\": number|null, "
        "\"beta_min\": number|null, \"beta_max\": number|null, "
        "\"avg_volume_min\": number|null, "
        "\"sectors\": [string], \"industries\": [string], \"exchanges\": [string], "
        "\"liquidity_prefilter\": boolean, "
        "\"valuation\": {\"pe_min\": number|null, \"pe_max\": number|null, \"forward_pe_min\": number|null, \"forward_pe_max\": number|null, \"peg_min\": number|null, \"peg_max\": number|null}, "
        "\"growth\": {\"revenue_growth_min\": number|null, \"revenue_growth_max\": number|null, \"earnings_growth_min\": number|null, \"earnings_growth_max\": number|null}, "
        "\"quality\": {\"roe_min\": number|null, \"roe_max\": number|null, \"roa_min\": number|null, \"roa_max\": number|null}, "
        "\"momentum\": {\"return_52w_min\": number|null, \"return_52w_max\": number|null}, "
        "\"dividend\": {\"dividend_yield_min\": number|null, \"dividend_yield_max\": number|null}"
        "}, "
        "\"weights\": {\"value\": number|null, \"growth\": number|null, \"quality\": number|null, \"momentum\": number|null, \"stability\": number|null, \"dividend\": number|null}, "
        "\"sort\": {\"by\": string, \"ascending\": boolean}, "
        "\"explanation\": string"
        "}"
    )

    user_prompt = (
        "Parse this stock screener request into the schema exactly. "
        "Do not include comments.\n\n"
        f"REQUEST: {cleaned_query}"
    )

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        completion = client.chat.completions.create(
            model=DEFAULT_GROQ_MODEL,
            temperature=0.0,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content if completion.choices else ""
        payload = _extract_json_payload(str(content))
        normalized = _normalize_parsed_payload(cleaned_query, payload)
        normalized["parse_source"] = "groq"
        return normalized
    except Exception:
        return _heuristic_parse_query(cleaned_query)


def _range_tuple(minimum: float | None, maximum: float | None) -> tuple[float, float] | None:
    if minimum is None and maximum is None:
        return None
    if minimum is None:
        minimum = float("-inf")
    if maximum is None:
        maximum = float("inf")
    return (minimum, maximum)


def _build_filter_maps(parsed_filters: Mapping[str, Any]) -> dict[str, Any]:
    valuation = parsed_filters.get("valuation") or {}
    growth = parsed_filters.get("growth") or {}
    quality = parsed_filters.get("quality") or {}
    momentum = parsed_filters.get("momentum") or {}
    dividend = parsed_filters.get("dividend") or {}

    valuation_filters = {
        "PE": (valuation.get("pe_min"), valuation.get("pe_max")),
        "ForwardPE": (valuation.get("forward_pe_min"), valuation.get("forward_pe_max")),
        "PEG": (valuation.get("peg_min"), valuation.get("peg_max")),
    }
    valuation_filters = {
        key: value for key, value in valuation_filters.items() if not (value[0] is None and value[1] is None)
    }

    growth_filters = {
        "RevenueGrowth": (growth.get("revenue_growth_min"), growth.get("revenue_growth_max")),
        "EarningsGrowth": (growth.get("earnings_growth_min"), growth.get("earnings_growth_max")),
    }
    growth_filters = {key: value for key, value in growth_filters.items() if not (value[0] is None and value[1] is None)}

    quality_filters = {
        "ROE": (quality.get("roe_min"), quality.get("roe_max")),
        "ROA": (quality.get("roa_min"), quality.get("roa_max")),
    }
    quality_filters = {
        key: value for key, value in quality_filters.items() if not (value[0] is None and value[1] is None)
    }

    momentum_filters = {
        "Return52W": (momentum.get("return_52w_min"), momentum.get("return_52w_max")),
    }
    momentum_filters = {
        key: value for key, value in momentum_filters.items() if not (value[0] is None and value[1] is None)
    }

    dividend_filters = {
        "DividendYield": (dividend.get("dividend_yield_min"), dividend.get("dividend_yield_max")),
    }
    dividend_filters = {
        key: value for key, value in dividend_filters.items() if not (value[0] is None and value[1] is None)
    }

    return {
        "valuation_filters": valuation_filters,
        "growth_filters": growth_filters,
        "quality_filters": quality_filters,
        "momentum_filters": momentum_filters,
        "dividend_filters": dividend_filters,
    }


def apply_ai_query(
    query: str,
    df: pd.DataFrame,
    parsed_query: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Apply an AI-parsed natural-language query to a cached universe dataframe.

    Returns:
    - Filtered/ranked dataframe
    - Human-readable explanation
    """
    if df.empty:
        return df.copy(), "Universe snapshot is empty."

    parsed = dict(parsed_query or parse_ai_query(query))
    filters = dict(parsed.get("filters") or {})
    filter_maps = _build_filter_maps(filters)

    filtered = apply_classic_filters(
        df=df,
        market_cap_range=_range_tuple(filters.get("market_cap_min"), filters.get("market_cap_max")),
        sectors=filters.get("sectors"),
        industries=filters.get("industries"),
        exchanges=filters.get("exchanges"),
        beta_range=_range_tuple(filters.get("beta_min"), filters.get("beta_max")),
        price_range=_range_tuple(filters.get("price_min"), filters.get("price_max")),
        min_avg_volume=filters.get("avg_volume_min"),
        valuation_filters=filter_maps["valuation_filters"],
        growth_filters=filter_maps["growth_filters"],
        quality_filters=filter_maps["quality_filters"],
        momentum_filters=filter_maps["momentum_filters"],
        dividend_filters=filter_maps["dividend_filters"],
        liquidity_prefilter=bool(filters.get("liquidity_prefilter", False)),
    )

    if filtered.empty:
        explanation = parsed.get("explanation", "No matches found for the interpreted query.")
        return filtered, explanation

    scored = calculate_quant_score(filtered, weight_preferences=parsed.get("weights"))

    technical_limit = min(250, len(scored))
    pre_ranked = rank_stocks(scored, sort_by="QuantScore", ascending=False, top_n=technical_limit)
    technical = apply_technical_indicators(pre_ranked)
    technical_columns = [column for column in ["Ticker", "RSI", "MACD", "Volatility", "Drawdown"] if column in technical.columns]
    merged = scored.merge(technical[technical_columns], on="Ticker", how="left")
    merged = calculate_quant_score(merged, weight_preferences=parsed.get("weights"))

    sort = parsed.get("sort") or {}
    sort_alias = {
        "quantscore": "QuantScore",
        "marketcap": "MarketCap",
        "price": "Price",
        "beta": "Beta",
        "pe": "PE",
        "return52w": "Return52W",
        "rsi": "RSI",
        "macd": "MACD",
        "volatility": "Volatility",
        "drawdown": "Drawdown",
    }
    sort_by_raw = str(sort.get("by", "QuantScore"))
    sort_by = sort_alias.get(sort_by_raw.replace("_", "").replace(" ", "").lower(), sort_by_raw)
    ascending = bool(sort.get("ascending", False))

    ranked = rank_stocks(merged, sort_by=sort_by, ascending=ascending)
    explanation = str(parsed.get("explanation", "")).strip() or "Applied AI-translated filters."
    if len(scored) > technical_limit:
        explanation = (
            f"{explanation} Technical indicators were computed on top {technical_limit} "
            "ranked symbols after first-stage filtering."
        )
    return ranked, explanation

