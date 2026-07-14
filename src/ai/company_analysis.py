"""Grounded Groq synthesis for company and management analysis."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from openai import OpenAI

from .ai_review import DEFAULT_GROQ_MODEL, _extract_json_payload, _extract_message_text


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
    except Exception as exc:
        return {"available": False, "source": "ai_error", "error": str(exc)}
