from __future__ import annotations

from pathlib import Path

from src.economics import (
    build_economics_stats,
    get_localized_question,
    get_unresolved_mistake_ids,
    load_question_bank,
    normalize_question,
    save_attempt_log,
)


def test_normalize_question_supports_bilingual_payload():
    payload = {
        "id": "custom_macro_1",
        "language": "en",
        "question_text": "What happens to real rates when inflation rises and policy rates stay unchanged?",
        "options": ["Rise", "Fall", "Stay flat", "Equal GDP growth"],
        "correct_answer": 1,
        "explanation": "Real rate is nominal minus inflation.",
        "topic": "Inflation",
        "difficulty": "easy",
        "source": "custom",
        "translations": {
            "cs": {
                "question_text": "Co se stane s realnymi sazbami, kdyz inflace roste a nominalni sazby zustanou stejne?",
                "options": ["Vzrostou", "Klesnou", "Zustanou stejne", "Budou rovny rustu HDP"],
                "explanation": "Realna sazba je nominalni sazba minus inflace.",
            }
        },
    }

    normalized = normalize_question(payload, default_source="custom")
    assert normalized is not None
    assert normalized["id"] == "custom_macro_1"
    assert normalized["correct_answer"] == 1
    assert "cs" in normalized["translations"]

    localized_cs = get_localized_question(normalized, "cs")
    assert localized_cs["question_text"].startswith("Co se stane")
    assert len(localized_cs["options"]) == 4


def test_build_stats_tracks_accuracy_streak_and_completion():
    questions = [
        normalize_question(
            {
                "id": "q1",
                "language": "en",
                "question_text": "Q1?",
                "options": ["A", "B", "C", "D"],
                "correct_answer": 0,
                "explanation": "E1",
                "topic": "Rates",
                "difficulty": "easy",
                "source": "custom",
            },
            default_source="custom",
        ),
        normalize_question(
            {
                "id": "q2",
                "language": "en",
                "question_text": "Q2?",
                "options": ["A", "B", "C", "D"],
                "correct_answer": 1,
                "explanation": "E2",
                "topic": "Inflation",
                "difficulty": "medium",
                "source": "ai",
            },
            default_source="custom",
        ),
    ]
    questions = [item for item in questions if item is not None]

    attempts = [
        {
            "attempt_id": "a1",
            "question_id": "q1",
            "selected_answer": 0,
            "correct_answer": 0,
            "is_correct": True,
            "topic": "Rates",
            "difficulty": "easy",
            "source": "custom",
            "language": "en",
            "timestamp": "2026-04-19T10:00:00+00:00",
        },
        {
            "attempt_id": "a2",
            "question_id": "q2",
            "selected_answer": 2,
            "correct_answer": 1,
            "is_correct": False,
            "topic": "Inflation",
            "difficulty": "medium",
            "source": "ai",
            "language": "en",
            "timestamp": "2026-04-19T10:01:00+00:00",
        },
        {
            "attempt_id": "a3",
            "question_id": "q2",
            "selected_answer": 1,
            "correct_answer": 1,
            "is_correct": True,
            "topic": "Inflation",
            "difficulty": "medium",
            "source": "ai",
            "language": "en",
            "timestamp": "2026-04-19T10:02:00+00:00",
        },
    ]

    stats = build_economics_stats(questions, attempts)
    assert stats["total_questions"] == 2
    assert stats["total_answered"] == 3
    assert stats["correct"] == 2
    assert stats["incorrect"] == 1
    assert stats["accuracy"] == 2 / 3
    assert stats["completion_rate"] == 1.0
    assert stats["current_streak"] == 1
    assert stats["best_streak"] == 1


def test_unresolved_mistakes_keep_latest_incorrect_only():
    attempts = [
        {"question_id": "q1", "is_correct": False, "timestamp": "2026-04-19T10:00:00+00:00"},
        {"question_id": "q1", "is_correct": True, "timestamp": "2026-04-19T10:01:00+00:00"},
        {"question_id": "q2", "is_correct": False, "timestamp": "2026-04-19T10:02:00+00:00"},
    ]
    unresolved = get_unresolved_mistake_ids(attempts)
    assert unresolved == ["q2"]


def test_question_bank_and_attempt_log_persist(tmp_path: Path):
    bank = load_question_bank(base_dir=tmp_path / "economics_questions")
    assert len(bank) >= 1

    attempts = [
        {
            "attempt_id": "a1",
            "question_id": bank[0]["id"],
            "selected_answer": 0,
            "correct_answer": 1,
            "is_correct": False,
            "topic": bank[0]["topic"],
            "difficulty": bank[0]["difficulty"],
            "source": bank[0]["source"],
            "language": "en",
            "timestamp": "2026-04-19T10:00:00+00:00",
        }
    ]
    save_attempt_log(attempts, base_dir=tmp_path / "economics_questions")

    attempts_file = tmp_path / "economics_questions" / "attempts.json"
    assert attempts_file.exists()
