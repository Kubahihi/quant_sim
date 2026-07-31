"""Grounded Groq synthesis for company and management analysis."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Optional

import openai
import numpy as np
from openai import OpenAI

from .ai_review import DEFAULT_GROQ_MODEL, _extract_json_payload, _extract_message_text


def _normalize_dcf_rate(value: Any, fallback: float, lower: float, upper: float) -> float:
    """Normalize an AI-proposed decimal rate and keep it inside audit-safe bounds."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(fallback)
    if not np.isfinite(result):
        result = float(fallback)
    # Accept a model returning 10 for 10%, while storing all rates as decimals.
    if 1.0 < abs(result) <= 100.0:
        result /= 100.0
    return min(max(result, lower), upper)


def generate_dcf_assumptions(
    evidence: Mapping[str, Any],
    api_key: Optional[str],
    model: str = DEFAULT_GROQ_MODEL,
) -> dict[str, Any]:
    """Generate bounded lifecycle judgments for the multi-stage DCF.

    Only forecast judgments are delegated to the model. Reported FCFF, cash,
    debt, shares, and price always continue to come from deterministic inputs.
    Legacy aliases remain in the result for older callers.
    """
    if not api_key:
        return {"available": False, "source": "ai_unavailable", "error": "GROQ_API_KEY was not provided."}

    from src.analytics.dcf import default_multistage_dcf_assumptions, prepare_dcf_inputs

    fallback = default_multistage_dcf_assumptions(prepare_dcf_inputs(evidence))
    system_prompt = (
        "You are an equity valuation analyst selecting company-specific multi-stage FCFF DCF judgments. "
        "Use only the supplied metrics and never reverse-engineer assumptions merely to match the market price. "
        "Distinguish a short near-term growth stage from a competitive fade; do not return generic 10% WACC, "
        "2.5% terminal growth and five years unless the evidence specifically supports all three. Return strict JSON "
        "with exactly: lifecycle (high_growth, transition, mature, contracting), initial_growth_rate (decimal -0.20 "
        "to 0.60), near_term_years (integer 2 to 5), fade_years (integer 2 to 10), discount_rate (decimal WACC "
        "0.05 to 0.25), terminal_growth_rate (decimal -0.02 to 0.04), confidence (0 to 1), rationale (one concise "
        "sentence), and warnings (array of strings). WACC must exceed terminal growth by at least 0.02. "
        "When evidence is sparse, stay close to the supplied deterministic anchors."
    )
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        completion = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=350,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"COMPANY_METRICS_JSON:\n{json.dumps(dict(evidence), ensure_ascii=False, default=str)}"},
            ],
        )
        content = _extract_message_text(completion.choices[0].message) if completion.choices else ""
        parsed = _extract_json_payload(content)
        growth_rate = _normalize_dcf_rate(
            parsed.get("initial_growth_rate", parsed.get("growth_rate")),
            float(fallback["initial_growth_rate"]),
            -0.20,
            0.60,
        )
        discount_rate = _normalize_dcf_rate(parsed.get("discount_rate"), float(fallback["discount_rate"]), 0.05, 0.25)
        terminal_growth_rate = _normalize_dcf_rate(
            parsed.get("terminal_growth_rate"),
            float(fallback["terminal_growth_rate"]),
            -0.02,
            0.04,
        )
        if discount_rate - terminal_growth_rate < 0.025:
            terminal_growth_rate = max(-0.02, discount_rate - 0.025)
        def bounded_years(key: str, fallback_value: int, lower: int, upper: int) -> int:
            try:
                value = int(round(float(parsed.get(key, fallback_value))))
            except (TypeError, ValueError):
                value = int(fallback_value)
            return min(max(value, lower), upper)

        near_term_years = bounded_years("near_term_years", int(fallback["near_term_years"]), 2, 5)
        fade_years = bounded_years("fade_years", int(fallback["fade_years"]), 2, 10)
        legacy_years = bounded_years("years", near_term_years + fade_years, 3, 10)
        lifecycle = str(parsed.get("lifecycle") or fallback["lifecycle"]).strip().lower()
        if lifecycle not in {"high_growth", "transition", "mature", "contracting"}:
            lifecycle = str(fallback["lifecycle"])
        assumptions = {
            "schema_version": 2,
            "lifecycle": lifecycle,
            "initial_growth_rate": growth_rate,
            "growth_rate": growth_rate,
            "near_term_years": near_term_years,
            "fade_years": fade_years,
            "discount_rate": discount_rate,
            "terminal_growth_rate": terminal_growth_rate,
            "years": legacy_years,
        }
        return {
            "available": True,
            "source": "groq",
            "assumptions": assumptions,
            "rationale": str(parsed.get("rationale") or "AI proposal based on the supplied company metrics.").strip(),
            "confidence": _normalize_dcf_rate(parsed.get("confidence"), 0.5, 0.0, 1.0),
            "warnings": [str(item) for item in parsed.get("warnings", []) if str(item).strip()],
        }
    except Exception as exc:
        return {"available": False, "source": "ai_error", "error": str(exc)}


def generate_company_deep_dive(
    evidence: Mapping[str, Any],
    api_key: Optional[str],
    model: str = DEFAULT_GROQ_MODEL,
) -> dict[str, Any]:
    """Synthesize supplied evidence without inventing management biography."""
    if not api_key:
        return {"available": False, "source": "ai_unavailable", "error": "GROQ_API_KEY was not provided."}

    system_prompt = (
        "You are a forensic equity-research analyst. Use only the supplied company data, officer records, "
        "financial metrics, price history summary, and numbered news headlines. Never invent a manager's "
        "former employer, tenure, achievement, failure, or causal attribution. If evidence is missing, say "
        "'Not established by available evidence'. Separate company outcomes from individual management claims. "
        "Cite news evidence inline as [N1], [N2], etc. Return strict JSON with exactly these keys: "
        "company_summary (string), management_history (string), successes (array of strings), failures "
        "(array of strings), moat_analysis (string), risks (array of strings), investment_view (string), "
        "evidence_limitations (string)."
    )
    
    max_retries = 3
    base_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            payload = json.dumps(dict(evidence), ensure_ascii=False, default=str)
            completion = client.chat.completions.create(
                model=model,
                temperature=0.1,
                max_tokens=1400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"EVIDENCE_JSON:\n{payload}"},
                ],
            )
            content = _extract_message_text(completion.choices[0].message) if completion.choices else ""
            parsed = _extract_json_payload(content)
            return {
                "available": True,
                "source": "groq",
                "company_summary": str(parsed.get("company_summary") or ""),
                "management_history": str(parsed.get("management_history") or ""),
                "successes": [str(item) for item in parsed.get("successes", [])],
                "failures": [str(item) for item in parsed.get("failures", [])],
                "moat_analysis": str(parsed.get("moat_analysis") or ""),
                "risks": [str(item) for item in parsed.get("risks", [])],
                "investment_view": str(parsed.get("investment_view") or ""),
                "evidence_limitations": str(parsed.get("evidence_limitations") or ""),
            }
        except openai.RateLimitError as exc:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            return {"available": False, "source": "ai_error", "error": f"Rate limited: {exc}"}
        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "too many requests" in err_str or "rate limit" in err_str:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
            return {"available": False, "source": "ai_error", "error": str(exc)}
    
    return {"available": False, "source": "ai_error", "error": "Max retries exceeded"}
