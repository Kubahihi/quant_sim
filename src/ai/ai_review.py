from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Mapping, Optional

from openai import OpenAI


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def resolve_groq_api_key(
    streamlit_secrets: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Resolve Groq key from Streamlit secrets first, then environment."""
    if streamlit_secrets is not None:
        try:
            secret_value = streamlit_secrets.get("GROQ_API_KEY")
            if isinstance(secret_value, str) and secret_value.strip():
                return secret_value.strip()
        except Exception:
            pass

    env_value = os.getenv("GROQ_API_KEY", "").strip()
    return env_value or None


def _extract_json_payload(text: str) -> Dict[str, Any]:
    """Parse JSON object from model output safely."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty AI response.")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError("No JSON object in AI response.")

    parsed = json.loads(text[start_idx : end_idx + 1])
    if not isinstance(parsed, dict):
        raise ValueError("AI response JSON is not an object.")
    return parsed


def _extract_message_text(message: Any) -> str:
    """Normalize OpenAI-compatible message content into plain text."""
    if message is None:
        return ""

    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue

            if isinstance(part, dict):
                text_value = part.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                    continue
                if isinstance(text_value, dict) and isinstance(text_value.get("value"), str):
                    parts.append(text_value["value"])
                    continue

            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                parts.append(part_text)
                continue

            part_value = getattr(part_text, "value", None)
            if isinstance(part_value, str):
                parts.append(part_value)

        return "\n".join(item.strip() for item in parts if item and item.strip()).strip()

    return str(content).strip()


def _extract_section(raw_text: str, label: str) -> str:
    pattern = rf"(?is)(?:^|\n)\s*(?:{label})\s*[:\-]\s*(.*?)(?=\n\s*[A-Za-z_ ]+\s*[:\-]|\Z)"
    match = re.search(pattern, raw_text)
    return match.group(1).strip() if match else ""


def _fallback_text_to_review(raw_text: str) -> Dict[str, str]:
    """Map structured prose into review fields when JSON is unavailable."""
    cleaned = (raw_text or "").strip().strip("`")

    summary = _extract_section(cleaned, "summary")
    risks = _extract_section(cleaned, "risks")
    improvements = _extract_section(cleaned, "improvements")
    verdict = _extract_section(cleaned, "verdict")

    if not any([summary, risks, improvements, verdict]):
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
        if paragraphs:
            summary = paragraphs[0]
        if len(paragraphs) > 1:
            risks = paragraphs[1]
        if len(paragraphs) > 2:
            improvements = paragraphs[2]
        if len(paragraphs) > 3:
            verdict = paragraphs[3]

    return {
        "summary": summary,
        "risks": risks,
        "improvements": improvements,
        "verdict": verdict,
    }


def _normalize_review_payload(payload: Dict[str, Any], raw_text: str) -> Dict[str, str]:
    fallback = _fallback_text_to_review(raw_text)
    normalized = {
        "summary": str(payload.get("summary") or fallback.get("summary") or "").strip(),
        "risks": str(payload.get("risks") or fallback.get("risks") or "").strip(),
        "improvements": str(
            payload.get("improvements") or fallback.get("improvements") or ""
        ).strip(),
        "verdict": str(payload.get("verdict") or fallback.get("verdict") or "").strip(),
    }

    if not any(normalized.values()) and raw_text.strip():
        normalized["summary"] = raw_text.strip()

    return normalized


def generate_ai_review(
    summary_payload: Dict[str, Any],
    api_key: Optional[str],
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 340,
) -> Dict[str, Any]:
    """Generate concise portfolio review with Groq OpenAI-compatible API."""
    if not api_key:
        return {
            "source": "ai_unavailable",
            "available": False,
            "error": "GROQ_API_KEY was not provided.",
        }

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        prompt_json = json.dumps(summary_payload, ensure_ascii=False, separators=(",", ":"))
        messages = [
    {
        "role": "system",
        "content": (
            "You are a senior portfolio quant analyst. "
            "Analyze only the metric summary provided to you; do not assume access to raw historical data. "
            "Respond in English. Be concise, factual, and practical. "
            "Prefer a strict JSON object with exactly these keys: "
            "\"summary\", \"risks\", \"improvements\", \"verdict\". "
            "If you cannot produce valid JSON, return the same four sections as short plain text. "
            "Do not add any extra commentary, introductions, or formatting."
        ),
    },
    {
        "role": "user",
        "content": (
            "Evaluate the portfolio using only the metric summary below. "
            "Base your judgment on what the metrics imply, without requesting or relying on raw historical data. "
            "Return a concise assessment with practical insights.\n\n"
            f"INPUT_JSON: {prompt_json}"
        ),
    },
]

        json_mode_error: Optional[str] = None
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception as exc:
            json_mode_error = str(exc)
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )

        content = _extract_message_text(completion.choices[0].message) if completion.choices else ""
        parsed: Dict[str, Any] = {}

        if content:
            try:
                parsed = _extract_json_payload(content)
            except Exception:
                parsed = {}

        normalized = _normalize_review_payload(parsed, content)

        if not any(normalized.values()):
            raise ValueError("AI response did not contain usable review fields.")

        return {
            "source": "groq",
            "available": True,
            "summary": normalized["summary"],
            "risks": normalized["risks"],
            "improvements": normalized["improvements"],
            "verdict": normalized["verdict"],
            "raw_response": content,
            "json_mode_error": json_mode_error,
        }
    except Exception as exc:
        return {
            "source": "ai_error",
            "available": False,
            "error": str(exc),
        }
