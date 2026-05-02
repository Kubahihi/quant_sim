from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from openai import OpenAI

from src.ai.ai_review import DEFAULT_GROQ_MODEL, resolve_groq_api_key


def resolve_swing_tracker_api_key(
    streamlit_secrets: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    return resolve_groq_api_key(streamlit_secrets=streamlit_secrets)


def _extract_message_text(message: Any) -> str:
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


def _extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("AI response is empty.")
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError("AI response does not contain a valid JSON object.")
    payload = json.loads(cleaned[start_idx : end_idx + 1])
    if not isinstance(payload, dict):
        raise ValueError("AI response JSON must be an object.")
    return payload


def _validate_string(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Field '{key}' must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"Field '{key}' cannot be empty.")
    return cleaned


def _validate_optional_string(value: Any, key: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Field '{key}' must be a string.")
    return value.strip()


def _validate_string_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Field '{key}' must be a list.")
    items: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text)
    if not items:
        raise ValueError(f"Field '{key}' cannot be an empty list.")
    return items


def _validate_float(value: Any, key: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"Field '{key}' must be numeric.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"Field '{key}' must be between {minimum} and {maximum}.")
    return float(parsed)


def _call_json_task(
    *,
    task_name: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    api_key: Optional[str],
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 450,
) -> dict[str, Any]:
    if not api_key:
        return {"available": False, "error": "GROQ_API_KEY was not provided."}

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Task: {task_name}. Respond as strict JSON object only.\n"
                    f"INPUT_JSON: {json.dumps(user_payload, ensure_ascii=False, separators=(',', ':'))}"
                ),
            },
        ]

        json_mode_error: str | None = None
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
        return {
            "available": True,
            "payload": payload,
            "raw_response": content,
            "json_mode_error": json_mode_error,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _heuristic_setup_type(thesis: str) -> tuple[str, list[str]]:
    text = str(thesis or "").lower()
    tags: list[str] = []
    if "breakout" in text:
        tags.append("breakout")
    if "pullback" in text or "retest" in text:
        tags.append("pullback")
    if "mean reversion" in text or "oversold" in text:
        tags.append("mean_reversion")
    if "earnings" in text or "event" in text:
        tags.append("event_driven")
    if not tags:
        tags.append("trend_continuation")
    return tags[0], tags


def summarize_trade_thesis(
    *,
    ticker: str,
    direction: str,
    thesis: str,
    api_key: Optional[str],
) -> dict[str, Any]:
    fallback = {
        "available": False,
        "thesis_summary": str(thesis or "").strip()[:320] or f"{ticker} {direction} trade thesis.",
        "risk_highlights": ["Respect predefined stop-loss and invalidation criteria."],
        "execution_focus": ["Keep position size aligned with the planned risk budget."],
    }
    result = _call_json_task(
        task_name="summarize_trade_thesis",
        system_prompt=(
            "You summarize swing trade thesis in a machine-readable structure. "
            "Output schema must be exactly: "
            "{\"thesis_summary\": string, \"risk_highlights\": [string], \"execution_focus\": [string]}. "
            "Do not add extra keys."
        ),
        user_payload={
            "ticker": ticker,
            "direction": direction,
            "thesis": thesis,
        },
        api_key=api_key,
    )
    if not result.get("available"):
        fallback["error"] = result.get("error", "AI unavailable.")
        return fallback

    try:
        payload = result["payload"]
        return {
            "available": True,
            "thesis_summary": _validate_string(payload.get("thesis_summary"), "thesis_summary"),
            "risk_highlights": _validate_string_list(payload.get("risk_highlights"), "risk_highlights"),
            "execution_focus": _validate_string_list(payload.get("execution_focus"), "execution_focus"),
            "raw_response": result.get("raw_response", ""),
            "json_mode_error": result.get("json_mode_error"),
        }
    except Exception as exc:
        fallback["error"] = f"Invalid AI output: {exc}"
        return fallback


def classify_setup_type(
    *,
    ticker: str,
    direction: str,
    thesis: str,
    api_key: Optional[str],
) -> dict[str, Any]:
    heuristic_type, tags = _heuristic_setup_type(thesis)
    fallback = {
        "available": False,
        "setup_type": heuristic_type,
        "confidence": 0.35,
        "reasoning_tags": tags,
    }
    result = _call_json_task(
        task_name="classify_setup_type",
        system_prompt=(
            "Classify discretionary swing trade setup. "
            "Return strict JSON with exactly: "
            "{\"setup_type\": string, \"confidence\": number, \"reasoning_tags\": [string]}. "
            "confidence must be between 0 and 1."
        ),
        user_payload={
            "ticker": ticker,
            "direction": direction,
            "thesis": thesis,
        },
        api_key=api_key,
    )
    if not result.get("available"):
        fallback["error"] = result.get("error", "AI unavailable.")
        return fallback

    try:
        payload = result["payload"]
        return {
            "available": True,
            "setup_type": _validate_string(payload.get("setup_type"), "setup_type"),
            "confidence": _validate_float(payload.get("confidence"), "confidence", 0.0, 1.0),
            "reasoning_tags": _validate_string_list(payload.get("reasoning_tags"), "reasoning_tags"),
            "raw_response": result.get("raw_response", ""),
            "json_mode_error": result.get("json_mode_error"),
        }
    except Exception as exc:
        fallback["error"] = f"Invalid AI output: {exc}"
        return fallback


def generate_stop_rationale(
    *,
    trade_payload: Mapping[str, Any],
    api_key: Optional[str],
) -> dict[str, Any]:
    stop_type = str(trade_payload.get("stop_type", "structural"))
    fallback = {
        "available": False,
        "stop_rationale_summary": (
            f"{stop_type} stop configured at {trade_payload.get('stop_loss')} with clear invalidation level."
        ),
        "invalidators": ["Price closes through the hard stop.", "Thesis conditions are no longer valid."],
        "checklist": ["Confirm stop placement before entry.", "Respect time-stop without exceptions."],
        "time_stop_rule": f"Exit when holding exceeds {trade_payload.get('time_stop_days', 'planned')} day(s).",
    }

    result = _call_json_task(
        task_name="generate_stop_rationale",
        system_prompt=(
            "Generate structured stop-loss rationale for a swing trade. "
            "Stop itself is already fixed and must not be changed. "
            "Return strict JSON with exactly: "
            "{\"stop_rationale_summary\": string, \"invalidators\": [string], "
            "\"checklist\": [string], \"time_stop_rule\": string}. "
            "Do not propose a different stop level."
        ),
        user_payload=dict(trade_payload),
        api_key=api_key,
    )
    if not result.get("available"):
        fallback["error"] = result.get("error", "AI unavailable.")
        return fallback

    try:
        payload = result["payload"]
        return {
            "available": True,
            "stop_rationale_summary": _validate_string(
                payload.get("stop_rationale_summary"),
                "stop_rationale_summary",
            ),
            "invalidators": _validate_string_list(payload.get("invalidators"), "invalidators"),
            "checklist": _validate_string_list(payload.get("checklist"), "checklist"),
            "time_stop_rule": _validate_optional_string(payload.get("time_stop_rule"), "time_stop_rule"),
            "raw_response": result.get("raw_response", ""),
            "json_mode_error": result.get("json_mode_error"),
        }
    except Exception as exc:
        fallback["error"] = f"Invalid AI output: {exc}"
        return fallback


def summarize_post_trade_review(
    *,
    trade_payload: Mapping[str, Any],
    review_notes: str,
    api_key: Optional[str],
) -> dict[str, Any]:
    fallback = {
        "available": False,
        "review_summary": str(review_notes or "").strip()[:320] or "Post-trade review captured.",
        "discipline_observations": ["Evaluate whether the plan was followed from entry to exit."],
        "process_improvements": ["Document one concrete rule refinement for the next trade."],
    }
    result = _call_json_task(
        task_name="summarize_post_trade_review",
        system_prompt=(
            "Summarize post-trade review in strict JSON. "
            "Output schema: {\"review_summary\": string, "
            "\"discipline_observations\": [string], \"process_improvements\": [string]}. "
            "No extra keys."
        ),
        user_payload={
            "trade": dict(trade_payload),
            "review_notes": review_notes,
        },
        api_key=api_key,
    )
    if not result.get("available"):
        fallback["error"] = result.get("error", "AI unavailable.")
        return fallback

    try:
        payload = result["payload"]
        return {
            "available": True,
            "review_summary": _validate_string(payload.get("review_summary"), "review_summary"),
            "discipline_observations": _validate_string_list(
                payload.get("discipline_observations"),
                "discipline_observations",
            ),
            "process_improvements": _validate_string_list(
                payload.get("process_improvements"),
                "process_improvements",
            ),
            "raw_response": result.get("raw_response", ""),
            "json_mode_error": result.get("json_mode_error"),
        }
    except Exception as exc:
        fallback["error"] = f"Invalid AI output: {exc}"
        return fallback
