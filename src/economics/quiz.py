from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

SUPPORTED_LANGUAGES = {"en", "cs"}
SUPPORTED_DIFFICULTIES = {"easy", "medium", "hard"}
SUPPORTED_SOURCES = {"ai", "custom"}


DEFAULT_ECONOMICS_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "eco_cpi_real_rate",
        "language": "en",
        "question_text": "If inflation rises and the nominal policy rate stays unchanged, what happens to the real policy rate?",
        "options": [
            "It increases",
            "It decreases",
            "It stays exactly the same",
            "It becomes equal to GDP growth",
        ],
        "correct_answer": 1,
        "explanation": "Real policy rate is approximately nominal rate minus inflation. If inflation rises while nominal rate is flat, the real rate declines.",
        "topic": "Inflation",
        "difficulty": "easy",
        "source": "custom",
        "option_explanations": {
            "0": "Incorrect. A higher inflation rate pushes the real rate lower, not higher.",
            "1": "Correct. Real rate is nominal minus inflation.",
            "2": "Incorrect. It would stay the same only if inflation stayed unchanged too.",
            "3": "Incorrect. Real policy rate and GDP growth are different concepts.",
        },
        "translations": {
            "cs": {
                "question_text": "Pokud inflace roste a nominální sazba centrální banky zůstane stejná, co se stane s reálnou sazbou?",
                "options": [
                    "Zvýší se",
                    "Sníží se",
                    "Zůstane přesně stejná",
                    "Bude rovna růstu HDP",
                ],
                "explanation": "Reálná sazba je přibližně nominální sazba minus inflace. Když inflace roste a nominální sazba je beze změny, reálná sazba klesá.",
                "option_explanations": {
                    "0": "Nesprávně. Vyšší inflace snižuje reálnou sazbu.",
                    "1": "Správně. Reálná sazba = nominální sazba minus inflace.",
                    "2": "Nesprávně. Stejná by zůstala jen při stejné inflaci.",
                    "3": "Nesprávně. Reálná sazba a růst HDP nejsou stejné veličiny.",
                },
            }
        },
    },
    {
        "id": "eco_yield_curve_recession",
        "language": "en",
        "question_text": "What does an inverted yield curve most often signal in macro analysis?",
        "options": [
            "Stronger inflation expectations in the very short run",
            "Potential economic slowdown or recession risk",
            "Immediate guarantee of equity market gains",
            "Lower long-term bond duration risk",
        ],
        "correct_answer": 1,
        "explanation": "An inverted curve (short rates above long rates) has historically been associated with tighter conditions and elevated recession probability.",
        "topic": "Rates",
        "difficulty": "medium",
        "source": "custom",
        "option_explanations": {
            "0": "Misleading. Short-run inflation can matter, but inversion is more strongly tied to growth expectations.",
            "1": "Correct. It often reflects tighter policy and weaker expected growth.",
            "2": "Incorrect. It is not a guarantee of equity gains.",
            "3": "Incorrect. Duration risk is a separate concept from curve inversion.",
        },
        "translations": {
            "cs": {
                "question_text": "Co inverzní výnosová křivka v makro analýze nejčastěji signalizuje?",
                "options": [
                    "Silnější očekávání inflace ve velmi krátkém období",
                    "Možné ekonomické zpomalení nebo riziko recese",
                    "Okamžitou jistotu růstu akciového trhu",
                    "Nižší dlouhodobé duracní riziko dluhopisů",
                ],
                "explanation": "Inverzní křivka (krátké sazby nad dlouhými) historicky souvisí s utaženými podmínkami a vyšší pravděpodobností recese.",
                "option_explanations": {
                    "0": "Zavádějící. Krátkodobá inflace může hrát roli, ale inverze více souvisí s očekávaným růstem.",
                    "1": "Správně. Často odráží restriktivní politiku a slabší očekávaný růst.",
                    "2": "Nesprávně. Není to garance růstu akcií.",
                    "3": "Nesprávně. Durace je jiný koncept než inverze křivky.",
                },
            }
        },
    },
    {
        "id": "eco_fiscal_multiplier",
        "language": "en",
        "question_text": "In a demand-constrained economy, what is the most likely short-term effect of targeted fiscal expansion?",
        "options": [
            "Lower aggregate demand and lower output",
            "Higher aggregate demand and potentially higher output",
            "No effect unless interest rates are zero",
            "Automatic decline in unemployment to zero",
        ],
        "correct_answer": 1,
        "explanation": "Fiscal expansion can lift aggregate demand and output in the short run, especially when private demand is weak.",
        "topic": "Fiscal policy",
        "difficulty": "medium",
        "source": "custom",
        "option_explanations": {
            "0": "Incorrect. This is opposite to the expected demand effect.",
            "1": "Correct. Additional public demand can increase near-term economic activity.",
            "2": "Incorrect. Effects can exist even when rates are above zero.",
            "3": "Incorrect. Policy can help labor markets but not instantly eliminate unemployment.",
        },
        "translations": {
            "cs": {
                "question_text": "Jaký je v ekonomice s omezenou poptávkou nejpravděpodobnější krátkodobý efekt cílené fiskální expanze?",
                "options": [
                    "Nižší agregátní poptávka a nižší výstup",
                    "Vyšší agregátní poptávka a potenciálně vyšší výstup",
                    "Žádný efekt, pokud sazby nejsou na nule",
                    "Automatický pokles nezaměstnanosti na nulu",
                ],
                "explanation": "Fiskální expanze může v krátkém období zvýšit agregátní poptávku i výstup, zejména když je soukromá poptávka slabá.",
                "option_explanations": {
                    "0": "Nesprávně. Je to opačný směr očekávaného efektu.",
                    "1": "Správně. Vyšší veřejná poptávka může zvýšit ekonomickou aktivitu.",
                    "2": "Nesprávně. Efekt může existovat i při sazbách nad nulou.",
                    "3": "Nesprávně. Politika může pomoci, ale neodstraní nezaměstnanost okamžitě.",
                },
            }
        },
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_data_dir(base_dir: str | Path = "data/economics_questions") -> Path:
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_id(raw_id: Any) -> str:
    text = str(raw_id or "").strip().lower()
    if not text:
        return f"eco_{uuid4().hex[:10]}"
    slug = re.sub(r"[^a-z0-9_-]+", "_", text)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or f"eco_{uuid4().hex[:10]}"


def _normalize_language(value: Any) -> str:
    lang = str(value or "").strip().lower()
    return lang if lang in SUPPORTED_LANGUAGES else "en"


def _normalize_difficulty(value: Any) -> str:
    difficulty = str(value or "").strip().lower()
    return difficulty if difficulty in SUPPORTED_DIFFICULTIES else "medium"


def _normalize_source(value: Any, default_source: str = "custom") -> str:
    source = str(value or "").strip().lower() or default_source
    return source if source in SUPPORTED_SOURCES else default_source


def _normalize_options(raw: Any) -> List[str]:
    if isinstance(raw, list):
        options = [str(item).strip() for item in raw if str(item).strip()]
        return options
    if isinstance(raw, str):
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return lines
    return []


def _normalize_correct_answer(raw: Any, option_count: int) -> int:
    if option_count < 2:
        return -1

    if isinstance(raw, int):
        return raw if 0 <= raw < option_count else -1

    text = str(raw or "").strip().upper()
    if not text:
        return -1

    if text.isdigit():
        parsed = int(text)
        if 0 <= parsed < option_count:
            return parsed
        if 1 <= parsed <= option_count:
            return parsed - 1
        return -1

    letter = text[0]
    if "A" <= letter <= "Z":
        index = ord(letter) - ord("A")
        if 0 <= index < option_count:
            return index
    return -1


def _normalize_option_explanations(raw: Any, option_count: int) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            index = _normalize_correct_answer(key, option_count)
            text = str(value or "").strip()
            if index >= 0 and text:
                normalized[str(index)] = text
    return normalized


def normalize_question(payload: Dict[str, Any], default_source: str = "custom") -> Dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    base_language = _normalize_language(payload.get("language", "en"))
    base_options = _normalize_options(payload.get("options", payload.get("answers", [])))
    base_question_text = str(payload.get("question_text") or payload.get("question") or "").strip()
    base_explanation = str(payload.get("explanation") or "").strip()
    base_correct_answer = _normalize_correct_answer(payload.get("correct_answer"), len(base_options))

    if len(base_options) < 2 or not base_question_text or base_correct_answer < 0:
        return None

    base_option_explanations = _normalize_option_explanations(
        payload.get("option_explanations", {}),
        len(base_options),
    )

    translations: Dict[str, Dict[str, Any]] = {}
    translations_raw = payload.get("translations", {})
    if not isinstance(translations_raw, dict):
        translations_raw = {}

    for language in ("en", "cs"):
        raw_lang = translations_raw.get(language, {})
        if not isinstance(raw_lang, dict):
            raw_lang = {}

        lang_question_text = str(
            raw_lang.get("question_text")
            or payload.get(f"question_text_{language}")
            or (base_question_text if language == base_language else "")
        ).strip()

        lang_options = _normalize_options(
            raw_lang.get("options")
            or payload.get(f"options_{language}")
            or (base_options if language == base_language else [])
        )

        lang_explanation = str(
            raw_lang.get("explanation")
            or payload.get(f"explanation_{language}")
            or (base_explanation if language == base_language else "")
        ).strip()

        lang_option_explanations = _normalize_option_explanations(
            raw_lang.get("option_explanations")
            or payload.get(f"option_explanations_{language}")
            or (base_option_explanations if language == base_language else {}),
            len(lang_options),
        )

        if lang_question_text and len(lang_options) >= 2:
            translations[language] = {
                "question_text": lang_question_text,
                "options": lang_options,
                "explanation": lang_explanation or base_explanation,
                "option_explanations": lang_option_explanations,
            }

    if base_language not in translations:
        translations[base_language] = {
            "question_text": base_question_text,
            "options": base_options,
            "explanation": base_explanation,
            "option_explanations": base_option_explanations,
        }

    primary_translation = translations.get(base_language, {})
    primary_options = _normalize_options(primary_translation.get("options", base_options))
    primary_correct_answer = _normalize_correct_answer(base_correct_answer, len(primary_options))
    if primary_correct_answer < 0:
        return None

    normalized = {
        "id": _sanitize_id(payload.get("id")),
        "language": base_language,
        "question_text": str(primary_translation.get("question_text") or base_question_text).strip(),
        "options": primary_options,
        "correct_answer": primary_correct_answer,
        "explanation": str(primary_translation.get("explanation") or base_explanation).strip(),
        "topic": str(payload.get("topic") or "General economics").strip() or "General economics",
        "difficulty": _normalize_difficulty(payload.get("difficulty")),
        "source": _normalize_source(payload.get("source"), default_source=default_source),
        "option_explanations": _normalize_option_explanations(
            primary_translation.get("option_explanations", {}),
            len(primary_options),
        ),
        "translations": translations,
        "created_at": str(payload.get("created_at") or _now_iso()),
    }
    return normalized


def _dedupe_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    output: List[Dict[str, Any]] = []
    for question in questions:
        question_id = str(question.get("id") or "").strip().lower()
        if not question_id or question_id in seen:
            continue
        seen.add(question_id)
        output.append(question)
    return output


def load_question_bank(base_dir: str | Path = "data/economics_questions") -> List[Dict[str, Any]]:
    data_dir = _ensure_data_dir(base_dir)
    path = data_dir / "questions.json"

    if not path.exists():
        default_rows = [normalize_question(item, default_source="custom") for item in DEFAULT_ECONOMICS_QUESTIONS]
        default_rows = [row for row in default_rows if row is not None]
        path.write_text(json.dumps(default_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return default_rows

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raw = []

    rows: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            normalized = normalize_question(
                item if isinstance(item, dict) else {},
                default_source=str(item.get("source", "custom") if isinstance(item, dict) else "custom"),
            )
            if normalized is not None:
                rows.append(normalized)

    if not rows:
        rows = [normalize_question(item, default_source="custom") for item in DEFAULT_ECONOMICS_QUESTIONS]
        rows = [row for row in rows if row is not None]

    deduped = _dedupe_questions(rows)
    return sorted(deduped, key=lambda q: str(q.get("created_at", "")))


def save_question_bank(
    questions: List[Dict[str, Any]],
    base_dir: str | Path = "data/economics_questions",
) -> None:
    data_dir = _ensure_data_dir(base_dir)
    path = data_dir / "questions.json"

    normalized_rows: List[Dict[str, Any]] = []
    for item in questions:
        source_hint = str(item.get("source", "custom")) if isinstance(item, dict) else "custom"
        normalized = normalize_question(item if isinstance(item, dict) else {}, default_source=source_hint)
        if normalized is not None:
            normalized_rows.append(normalized)

    deduped = _dedupe_questions(normalized_rows)
    path.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")


def load_attempt_log(base_dir: str | Path = "data/economics_questions") -> List[Dict[str, Any]]:
    data_dir = _ensure_data_dir(base_dir)
    path = data_dir / "attempts.json"
    if not path.exists():
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    attempts: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("question_id") or "").strip()
            if not qid:
                continue
            attempts.append({
                "attempt_id": str(item.get("attempt_id") or f"attempt_{uuid4().hex[:10]}"),
                "question_id": qid,
                "selected_answer": int(item.get("selected_answer", -1)),
                "correct_answer": int(item.get("correct_answer", -1)),
                "is_correct": bool(item.get("is_correct", False)),
                "topic": str(item.get("topic") or "General economics"),
                "difficulty": _normalize_difficulty(item.get("difficulty")),
                "source": _normalize_source(item.get("source"), default_source="custom"),
                "language": _normalize_language(item.get("language")),
                "timestamp": str(item.get("timestamp") or _now_iso()),
            })

    return sorted(attempts, key=lambda row: str(row.get("timestamp", "")))


def save_attempt_log(
    attempts: List[Dict[str, Any]],
    base_dir: str | Path = "data/economics_questions",
) -> None:
    data_dir = _ensure_data_dir(base_dir)
    path = data_dir / "attempts.json"
    path.write_text(json.dumps(attempts, ensure_ascii=False, indent=2), encoding="utf-8")


def get_localized_question(question: Dict[str, Any], language: str) -> Dict[str, Any]:
    target_language = _normalize_language(language)
    translations = question.get("translations", {})
    if not isinstance(translations, dict):
        translations = {}

    localized_payload = translations.get(target_language)
    if not isinstance(localized_payload, dict):
        localized_payload = translations.get(question.get("language", "en"), {})
    if not isinstance(localized_payload, dict):
        localized_payload = {}

    options = _normalize_options(localized_payload.get("options", question.get("options", [])))
    if len(options) < 2:
        options = _normalize_options(question.get("options", []))

    option_explanations = _normalize_option_explanations(
        localized_payload.get("option_explanations", question.get("option_explanations", {})),
        len(options),
    )

    return {
        "id": question.get("id"),
        "language": target_language,
        "question_text": str(
            localized_payload.get("question_text") or question.get("question_text") or ""
        ).strip(),
        "options": options,
        "correct_answer": int(question.get("correct_answer", -1)),
        "explanation": str(
            localized_payload.get("explanation") or question.get("explanation") or ""
        ).strip(),
        "topic": str(question.get("topic") or "General economics"),
        "difficulty": _normalize_difficulty(question.get("difficulty")),
        "source": _normalize_source(question.get("source"), default_source="custom"),
        "option_explanations": option_explanations,
    }


def get_unresolved_mistake_ids(attempts: List[Dict[str, Any]]) -> List[str]:
    latest_by_question: Dict[str, Dict[str, Any]] = {}
    for attempt in sorted(attempts, key=lambda row: str(row.get("timestamp", ""))):
        question_id = str(attempt.get("question_id") or "").strip()
        if question_id:
            latest_by_question[question_id] = attempt

    unresolved = [
        question_id
        for question_id, attempt in latest_by_question.items()
        if not bool(attempt.get("is_correct", False))
    ]
    return sorted(unresolved)


def _compute_streaks(attempts: List[Dict[str, Any]]) -> Dict[str, int]:
    sorted_attempts = sorted(attempts, key=lambda row: str(row.get("timestamp", "")))
    best = 0
    running = 0
    for attempt in sorted_attempts:
        if bool(attempt.get("is_correct", False)):
            running += 1
            best = max(best, running)
        else:
            running = 0

    current = 0
    for attempt in reversed(sorted_attempts):
        if bool(attempt.get("is_correct", False)):
            current += 1
        else:
            break

    return {"current_streak": current, "best_streak": best}


def build_economics_stats(
    questions: List[Dict[str, Any]],
    attempts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    question_map = {
        str(item.get("id")): item
        for item in questions
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }

    total_answered = len(attempts)
    correct = sum(1 for attempt in attempts if bool(attempt.get("is_correct", False)))
    incorrect = total_answered - correct
    accuracy = (correct / total_answered) if total_answered else 0.0

    unique_answered = len({str(attempt.get("question_id") or "") for attempt in attempts if attempt.get("question_id")})
    total_questions = len(question_map)
    completion_rate = (unique_answered / total_questions) if total_questions else 0.0

    streaks = _compute_streaks(attempts)

    topic_counter: Dict[str, Dict[str, int]] = defaultdict(lambda: {"attempts": 0, "correct": 0})
    for attempt in attempts:
        question_id = str(attempt.get("question_id") or "")
        fallback_topic = str(question_map.get(question_id, {}).get("topic") or "General economics")
        topic = str(attempt.get("topic") or fallback_topic)
        topic_counter[topic]["attempts"] += 1
        if bool(attempt.get("is_correct", False)):
            topic_counter[topic]["correct"] += 1

    topic_performance = []
    for topic_name, values in sorted(topic_counter.items(), key=lambda item: item[0]):
        attempts_count = int(values["attempts"])
        correct_count = int(values["correct"])
        topic_performance.append({
            "Topic": topic_name,
            "Attempts": attempts_count,
            "Correct": correct_count,
            "Incorrect": attempts_count - correct_count,
            "Accuracy": (correct_count / attempts_count) if attempts_count else 0.0,
        })

    unresolved_mistakes = get_unresolved_mistake_ids(attempts)

    return {
        "total_questions": total_questions,
        "total_answered": total_answered,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "completion_rate": completion_rate,
        "current_streak": streaks["current_streak"],
        "best_streak": streaks["best_streak"],
        "topic_performance": topic_performance,
        "unresolved_mistakes": unresolved_mistakes,
    }

