from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional

from openai import OpenAI


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


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
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Jsi senior portfolio analytik. Odpovidej cesky, vecne, kratce a prakticky. "
                        "Vrat POUZE validni JSON s poli: summary, risks, improvements, verdict."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Vyhodnot portfolio na zaklade shrnuti metrik (bez raw dat). "
                        "Bud konkretni. Rizika a doporuceni max 4 body dohromady.\n"
                        f"INPUT_JSON: {prompt_json}"
                    ),
                },
            ],
        )

        content = completion.choices[0].message.content if completion.choices else ""
        parsed = _extract_json_payload(content or "")

        return {
            "source": "groq",
            "available": True,
            "summary": str(parsed.get("summary", "")).strip(),
            "risks": str(parsed.get("risks", "")).strip(),
            "improvements": str(parsed.get("improvements", "")).strip(),
            "verdict": str(parsed.get("verdict", "")).strip(),
        }
    except Exception as exc:
        return {
            "source": "ai_error",
            "available": False,
            "error": str(exc),
        }
