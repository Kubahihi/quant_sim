from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.economics import normalize_question

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def _extract_message_text(message: Any) -> str:
    if message is None:
        return ""

    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
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


def _extract_json_payload(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Empty AI response.")

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError("No JSON object found in AI response.")

    parsed = json.loads(cleaned[start_idx : end_idx + 1])
    if not isinstance(parsed, dict):
        raise ValueError("AI response JSON is not an object.")
    return parsed


def generate_economics_questions(
    analyzed_context: Dict[str, Any],
    api_key: Optional[str],
    n_questions: int = 6,
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.25,
    max_tokens: int = 1800,
) -> Dict[str, Any]:
    if not api_key:
        return {
            "source": "ai_unavailable",
            "available": False,
            "error": "GROQ_API_KEY was not provided.",
            "questions": [],
        }

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        context_json = json.dumps(analyzed_context, ensure_ascii=False, separators=(",", ":"))
        question_count = max(1, min(int(n_questions), 20))

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an economics learning designer. Generate high-quality multiple-choice "
                    "economics questions from analyzed portfolio and macro context. "
                    "Return only a strict JSON object with one key: questions. "
                    "questions must be a list of objects using this schema: "
                    "id, language, question_text, options, correct_answer, explanation, topic, "
                    "difficulty, source, option_explanations, translations. "
                    "Rules: options length must be exactly 4; correct_answer must be integer index 0-3; "
                    "difficulty must be easy/medium/hard; source must be ai. "
                    "translations must include both en and cs variants when possible. "
                    "option_explanations should explain why each option is correct/incorrect. "
                    "Do not wrap JSON in markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate {question_count} economics questions using this analyzed context only: "
                    f"{context_json}"
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
        payload = _extract_json_payload(content)
        raw_questions = payload.get("questions", [])
        if not isinstance(raw_questions, list):
            raise ValueError("AI response is missing a valid 'questions' list.")

        normalized_questions: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_questions, start=1):
            if not isinstance(item, dict):
                continue
            enriched = {
                **item,
                "id": item.get("id") or f"eco_ai_{idx}_{question_count}",
                "source": "ai",
            }
            normalized = normalize_question(enriched, default_source="ai")
            if normalized is not None:
                normalized["source"] = "ai"
                normalized_questions.append(normalized)

        if not normalized_questions:
            raise ValueError("AI response did not contain valid questions after normalization.")

        return {
            "source": "groq",
            "available": True,
            "questions": normalized_questions,
            "raw_response": content,
            "json_mode_error": json_mode_error,
        }
    except Exception as exc:
        return {
            "source": "ai_error",
            "available": False,
            "error": str(exc),
            "questions": [],
        }
