from __future__ import annotations

from datetime import datetime, timezone
import random
from typing import Any, Dict, List
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.ai import generate_economics_questions, resolve_groq_api_key
from src.economics import (
    build_economics_stats,
    get_localized_question,
    get_unresolved_mistake_ids,
    load_attempt_log,
    load_question_bank,
    normalize_question,
    save_attempt_log,
    save_question_bank,
)


TEXT: Dict[str, Dict[str, str]] = {
    "en": {
        "section_title": "Economics Questions",
        "section_caption": "Learning-oriented economics quiz with bilingual questions, explanations, review mode, and Groq-assisted generation.",
        "tab_quiz": "Quiz",
        "tab_stats": "Stats",
        "tab_review": "Review mistakes",
        "tab_create": "Create question",
        "tab_ai": "Generate with Groq",
        "language_label": "Display language",
        "quiz_settings": "Quiz settings",
        "source_filter": "Question source",
        "difficulty_filter": "Difficulty",
        "topic_filter": "Topic",
        "mode_label": "Mode",
        "mode_standard": "Standard quiz",
        "mode_review": "Review unresolved mistakes",
        "question_count": "Questions in this quiz",
        "shuffle": "Shuffle question order",
        "start_quiz": "Start quiz",
        "no_questions": "No questions found for the selected filters.",
        "no_mistakes": "No unresolved mistakes right now. Great progress.",
        "progress": "Progress",
        "submit": "Submit answer",
        "next": "Next question",
        "finish": "Finish quiz",
        "correct": "Correct",
        "incorrect": "Incorrect",
        "explanation": "Explanation",
        "correct_answer": "Correct answer",
        "option_reasons": "Why each option is right/wrong",
        "quiz_complete": "Quiz completed",
        "retry_wrong": "Retry incorrect questions from this quiz",
        "clear_quiz": "Clear active quiz",
        "stats_caption": "Performance dashboard",
        "total_answered": "Total answered",
        "accuracy": "Accuracy",
        "correct_incorrect": "Correct / Incorrect",
        "completion_rate": "Completion rate",
        "current_streak": "Current streak",
        "best_streak": "Best streak",
        "topic_breakdown": "Topic performance",
        "review_list": "Questions to revisit",
        "start_review_quiz": "Start retry quiz",
        "review_history": "Recent incorrect attempts",
        "create_caption": "Add manual custom questions (same format as AI questions).",
        "topic": "Topic",
        "difficulty": "Difficulty",
        "correct_option": "Correct option",
        "question_en": "Question (English)",
        "question_cs": "Question (Czech, optional)",
        "option": "Option",
        "explanation_en": "Explanation (English)",
        "explanation_cs": "Explanation (Czech, optional)",
        "option_exp_en": "Option explanations EN (optional, one line per option: A: reason)",
        "option_exp_cs": "Option explanations CS (optional, one line per option: A: reason)",
        "save_question": "Save custom question",
        "create_success": "Custom question saved.",
        "create_error": "Question could not be saved. Check required fields.",
        "ai_caption": "Generate economics quiz questions from already analyzed quant output.",
        "ai_count": "Number of generated questions",
        "ai_focus": "Topic focus hint (optional)",
        "ai_generate": "Generate questions with Groq",
        "ai_success": "New AI questions were generated and stored.",
        "ai_error": "Groq generation failed",
        "ai_preview": "Generated questions preview",
        "ai_json_fallback": "JSON mode fallback",
        "ai_raw": "Raw AI response",
        "source_all": "all",
    },
    "cs": {
        "section_title": "Ekonomicke Otazky",
        "section_caption": "Vzdelavaci modul s kvizem, vysvetlenim odpovedi, revizi chyb a generovanim pres Groq.",
        "tab_quiz": "Kviz",
        "tab_stats": "Statistiky",
        "tab_review": "Revize chyb",
        "tab_create": "Vytvorit otazku",
        "tab_ai": "Generovat pres Groq",
        "language_label": "Jazyk zobrazeni",
        "quiz_settings": "Nastaveni kvizu",
        "source_filter": "Zdroj otazek",
        "difficulty_filter": "Obtiznost",
        "topic_filter": "Tema",
        "mode_label": "Rezim",
        "mode_standard": "Standardni kviz",
        "mode_review": "Revize nevyresenych chyb",
        "question_count": "Pocet otazek v kvizu",
        "shuffle": "Nahodne poradi",
        "start_quiz": "Spustit kviz",
        "no_questions": "Pro zvolene filtry nebyly nalezeny zadne otazky.",
        "no_mistakes": "Zadne nevyresene chyby. Skvela prace.",
        "progress": "Postup",
        "submit": "Potvrdit odpoved",
        "next": "Dalsi otazka",
        "finish": "Dokoncit kviz",
        "correct": "Spravne",
        "incorrect": "Nespravne",
        "explanation": "Vysvetleni",
        "correct_answer": "Spravna odpoved",
        "option_reasons": "Proc jsou moznosti spravne nebo chybne",
        "quiz_complete": "Kviz je dokoncen",
        "retry_wrong": "Zkusit znovu chybne otazky z tohoto kvizu",
        "clear_quiz": "Vymazat aktivni kviz",
        "stats_caption": "Prehled vykonnosti",
        "total_answered": "Celkem zodpovezeno",
        "accuracy": "Uspesnost",
        "correct_incorrect": "Spravne / Nespravne",
        "completion_rate": "Mira pokryti",
        "current_streak": "Aktualni serie",
        "best_streak": "Nejlepsi serie",
        "topic_breakdown": "Vykon podle temat",
        "review_list": "Otazky k opakovani",
        "start_review_quiz": "Spustit opakovaci kviz",
        "review_history": "Nedavne chybne pokusy",
        "create_caption": "Pridani vlastnich otazek se stejnym modelem jako AI.",
        "topic": "Tema",
        "difficulty": "Obtiznost",
        "correct_option": "Spravna moznost",
        "question_en": "Otazka (anglicky)",
        "question_cs": "Otazka (cesky, volitelne)",
        "option": "Moznost",
        "explanation_en": "Vysvetleni (anglicky)",
        "explanation_cs": "Vysvetleni (cesky, volitelne)",
        "option_exp_en": "Vysvetleni moznosti EN (volitelne, jeden radek na moznost: A: duvod)",
        "option_exp_cs": "Vysvetleni moznosti CS (volitelne, jeden radek na moznost: A: duvod)",
        "save_question": "Ulozit vlastni otazku",
        "create_success": "Vlastni otazka byla ulozena.",
        "create_error": "Otazku se nepodarilo ulozit. Zkontrolujte povinna pole.",
        "ai_caption": "Generovani ekonomickych otazek z jiz analyzovaneho quant vystupu.",
        "ai_count": "Pocet generovanych otazek",
        "ai_focus": "Hint na tematicky fokus (volitelne)",
        "ai_generate": "Generovat otazky pres Groq",
        "ai_success": "Nove AI otazky byly vygenerovany a ulozeny.",
        "ai_error": "Generovani pres Groq selhalo",
        "ai_preview": "Nahled vygenerovanych otazek",
        "ai_json_fallback": "Fallback bez JSON modu",
        "ai_raw": "Raw AI odpoved",
        "source_all": "vse",
    },
}


def _t(language: str, key: str) -> str:
    return TEXT.get(language, TEXT["en"]).get(key, key)


def _parse_option_explanations(raw_text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in (raw_text or "").splitlines():
        cleaned = line.strip()
        if not cleaned or ":" not in cleaned:
            continue
        left, right = cleaned.split(":", 1)
        token = left.strip().upper()
        value = right.strip()
        if not value:
            continue

        index = -1
        if token.isdigit():
            parsed = int(token)
            if 0 <= parsed <= 3:
                index = parsed
            elif 1 <= parsed <= 4:
                index = parsed - 1
        elif token and "A" <= token[0] <= "D":
            index = ord(token[0]) - ord("A")

        if 0 <= index <= 3:
            result[str(index)] = value
    return result


def _build_analysis_context(analysis_result: Dict[str, Any], topic_focus: str) -> Dict[str, Any]:
    score_result = analysis_result.get("score_result", {})
    summary_result = analysis_result.get("summary_result")
    news_result = analysis_result.get("news_result")
    metrics = analysis_result.get("metrics", {})

    context = {
        "run_id": getattr(analysis_result.get("run_record"), "run_id", ""),
        "tickers": analysis_result.get("tickers", []),
        "risk_profile": analysis_result.get("risk_profile", "balanced"),
        "horizon_days": int(analysis_result.get("horizon_days", 252) or 252),
        "score": {
            "value": int(score_result.get("score", 0) or 0),
            "rating": str(score_result.get("rating", "")),
            "flags": list(score_result.get("flags", [])),
        },
        "metrics": {
            "annualized_return": float(metrics.get("annualized_return", 0.0) or 0.0),
            "volatility": float(metrics.get("volatility", 0.0) or 0.0),
            "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0) or 0.0),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0) or 0.0),
            "avg_correlation": float(metrics.get("avg_correlation", 0.0) or 0.0),
        },
        "summary": {
            "regime": str(getattr(summary_result, "regime_label", "")),
            "highlights": list(getattr(summary_result, "highlights", [])),
            "risk_flags": list(getattr(summary_result, "risk_flags", [])),
            "news_implication": str(getattr(summary_result, "news_implication", "")),
        },
        "news": {
            "sentiment_score": float(getattr(news_result, "sentiment_score", 0.0) if news_result else 0.0),
            "sentiment_dispersion": float(getattr(news_result, "sentiment_dispersion", 0.0) if news_result else 0.0),
        },
        "topic_focus": str(topic_focus or "").strip(),
    }
    return context


def _init_state() -> None:
    if "eco_question_bank" not in st.session_state:
        st.session_state["eco_question_bank"] = load_question_bank()

    if "eco_attempt_log" not in st.session_state:
        st.session_state["eco_attempt_log"] = load_attempt_log()

    defaults = {
        "eco_ui_language": "en",
        "eco_quiz_queue": [],
        "eco_quiz_position": 0,
        "eco_quiz_submitted": False,
        "eco_last_result": None,
        "eco_wrong_ids_current_quiz": [],
        "eco_last_generated_questions": [],
        "eco_last_ai_raw_response": "",
        "eco_last_ai_json_mode_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_question_map(question_bank: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(question.get("id")): question
        for question in question_bank
        if isinstance(question, dict) and question.get("id")
    }


def _start_quiz(question_ids: List[str]) -> None:
    st.session_state["eco_quiz_queue"] = list(dict.fromkeys(question_ids))
    st.session_state["eco_quiz_position"] = 0
    st.session_state["eco_quiz_submitted"] = False
    st.session_state["eco_last_result"] = None
    st.session_state["eco_wrong_ids_current_quiz"] = []


def _append_attempt(
    attempts: List[Dict[str, Any]],
    question: Dict[str, Any],
    selected_index: int,
    is_correct: bool,
    language: str,
) -> List[Dict[str, Any]]:
    next_attempts = list(attempts)
    next_attempts.append({
        "attempt_id": f"attempt_{uuid4().hex[:10]}",
        "question_id": str(question.get("id")),
        "selected_answer": int(selected_index),
        "correct_answer": int(question.get("correct_answer", -1)),
        "is_correct": bool(is_correct),
        "topic": str(question.get("topic") or "General economics"),
        "difficulty": str(question.get("difficulty") or "medium"),
        "source": str(question.get("source") or "custom"),
        "language": language,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    save_attempt_log(next_attempts)
    return next_attempts


def _difficulty_labels(language: str) -> Dict[str, str]:
    if language == "cs":
        return {
            "all": "vse",
            "easy": "lehka",
            "medium": "stredni",
            "hard": "tezka",
        }
    return {
        "all": "all",
        "easy": "easy",
        "medium": "medium",
        "hard": "hard",
    }


def _source_labels(language: str) -> Dict[str, str]:
    if language == "cs":
        return {"all": "vse", "ai": "AI", "custom": "vlastni"}
    return {"all": "all", "ai": "AI", "custom": "custom"}


def _format_option(index: int, option: str) -> str:
    return f"{chr(ord('A') + index)}. {option}"


def _render_quiz_tab(language: str) -> None:
    question_bank = st.session_state["eco_question_bank"]
    attempts = st.session_state["eco_attempt_log"]
    question_map = _get_question_map(question_bank)
    unresolved_mistakes = set(get_unresolved_mistake_ids(attempts))

    source_labels = _source_labels(language)
    difficulty_labels = _difficulty_labels(language)

    st.subheader(_t(language, "quiz_settings"))
    settings_col1, settings_col2, settings_col3 = st.columns(3)

    source_value = settings_col1.selectbox(
        _t(language, "source_filter"),
        options=list(source_labels.keys()),
        format_func=lambda key: source_labels.get(key, key),
        key="eco_source_filter",
    )

    difficulty_value = settings_col2.selectbox(
        _t(language, "difficulty_filter"),
        options=list(difficulty_labels.keys()),
        format_func=lambda key: difficulty_labels.get(key, key),
        key="eco_difficulty_filter",
    )

    topics = sorted({str(item.get("topic") or "General economics") for item in question_bank})
    topic_options = [_t(language, "source_all"), *topics]
    topic_label = settings_col3.selectbox(
        _t(language, "topic_filter"),
        options=topic_options,
        key="eco_topic_filter",
    )

    mode_col1, mode_col2, mode_col3 = st.columns(3)
    mode_value = mode_col1.selectbox(
        _t(language, "mode_label"),
        options=["standard", "review"],
        format_func=lambda key: _t(language, "mode_standard") if key == "standard" else _t(language, "mode_review"),
        key="eco_mode_filter",
    )

    filtered_questions: List[Dict[str, Any]] = []
    for question in question_bank:
        if source_value != "all" and str(question.get("source")) != source_value:
            continue
        if difficulty_value != "all" and str(question.get("difficulty")) != difficulty_value:
            continue
        if topic_label != _t(language, "source_all") and str(question.get("topic")) != topic_label:
            continue
        if mode_value == "review" and str(question.get("id")) not in unresolved_mistakes:
            continue
        filtered_questions.append(question)

    max_count = max(1, len(filtered_questions))
    question_count = mode_col2.slider(
        _t(language, "question_count"),
        min_value=1,
        max_value=max_count,
        value=min(8, max_count),
        step=1,
        disabled=len(filtered_questions) == 0,
        key="eco_quiz_count",
    )

    shuffle_enabled = mode_col3.checkbox(_t(language, "shuffle"), value=True, key="eco_shuffle")

    start_clicked = st.button(
        _t(language, "start_quiz"),
        use_container_width=True,
        type="primary",
    )

    if start_clicked:
        if not filtered_questions:
            st.warning(_t(language, "no_questions"))
        else:
            selected = list(filtered_questions)
            if shuffle_enabled:
                random.shuffle(selected)
            selected_ids = [str(item.get("id")) for item in selected[:question_count]]
            _start_quiz(selected_ids)
            st.rerun()

    queue = st.session_state.get("eco_quiz_queue", [])
    if not queue:
        if mode_value == "review" and not unresolved_mistakes:
            st.info(_t(language, "no_mistakes"))
        else:
            st.info(_t(language, "no_questions"))
        return

    position = int(st.session_state.get("eco_quiz_position", 0) or 0)
    submitted = bool(st.session_state.get("eco_quiz_submitted", False))

    if position >= len(queue):
        st.success(_t(language, "quiz_complete"))
        wrong_ids = list(st.session_state.get("eco_wrong_ids_current_quiz", []))
        finished = len(queue)
        wrong_count = len(wrong_ids)
        accuracy = ((finished - wrong_count) / finished) if finished else 0.0

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric(_t(language, "total_answered"), finished)
        summary_col2.metric(_t(language, "accuracy"), f"{accuracy:.1%}")
        summary_col3.metric(_t(language, "correct_incorrect"), f"{finished - wrong_count} / {wrong_count}")

        action_col1, action_col2 = st.columns(2)
        if action_col1.button(_t(language, "retry_wrong"), use_container_width=True, disabled=wrong_count == 0):
            _start_quiz(wrong_ids)
            st.rerun()

        if action_col2.button(_t(language, "clear_quiz"), use_container_width=True):
            _start_quiz([])
            st.rerun()
        return

    current_question_id = str(queue[position])
    current_question = question_map.get(current_question_id)
    if current_question is None:
        st.warning(f"Question not found: {current_question_id}")
        if st.button(_t(language, "next"), use_container_width=True):
            st.session_state["eco_quiz_position"] = position + 1
            st.rerun()
        return

    localized = get_localized_question(current_question, language)
    options = localized.get("options", [])
    if len(options) < 2:
        st.warning(f"Invalid options for question: {current_question_id}")
        if st.button(_t(language, "next"), use_container_width=True):
            st.session_state["eco_quiz_position"] = position + 1
            st.rerun()
        return

    progress = (position + (1 if submitted else 0)) / len(queue)
    st.caption(f"{_t(language, 'progress')}: {position + 1}/{len(queue)}")
    st.progress(progress)

    st.markdown(f"**{localized.get('topic', '-') }** | {localized.get('difficulty', '-') } | {localized.get('source', '-')}")
    st.markdown(f"### {localized.get('question_text', '')}")

    option_indices = list(range(len(options)))
    radio_key = f"eco_choice_{current_question_id}_{position}"
    selected_index = st.radio(
        "",
        options=option_indices,
        format_func=lambda idx: _format_option(idx, options[idx]),
        key=radio_key,
        disabled=submitted,
        label_visibility="collapsed",
    )

    if not submitted:
        if st.button(_t(language, "submit"), type="primary", use_container_width=True):
            correct_index = int(localized.get("correct_answer", -1))
            is_correct = selected_index == correct_index

            if not is_correct:
                wrong_ids = list(st.session_state.get("eco_wrong_ids_current_quiz", []))
                if current_question_id not in wrong_ids:
                    wrong_ids.append(current_question_id)
                st.session_state["eco_wrong_ids_current_quiz"] = wrong_ids

            updated_attempts = _append_attempt(
                attempts=attempts,
                question=current_question,
                selected_index=selected_index,
                is_correct=is_correct,
                language=language,
            )
            st.session_state["eco_attempt_log"] = updated_attempts
            st.session_state["eco_quiz_submitted"] = True
            st.session_state["eco_last_result"] = {
                "question_id": current_question_id,
                "selected": selected_index,
                "correct": correct_index,
                "is_correct": is_correct,
            }
            st.rerun()
        return

    result = st.session_state.get("eco_last_result") or {}
    is_correct = bool(result.get("is_correct", False))
    correct_index = int(localized.get("correct_answer", -1))

    if is_correct:
        st.success(_t(language, "correct"))
    else:
        st.error(_t(language, "incorrect"))

    if 0 <= correct_index < len(options):
        st.markdown(f"**{_t(language, 'correct_answer')}:** {_format_option(correct_index, options[correct_index])}")

    st.markdown(f"**{_t(language, 'explanation')}:** {localized.get('explanation', '-')}")

    option_explanations = localized.get("option_explanations", {})
    if isinstance(option_explanations, dict) and option_explanations:
        st.caption(_t(language, "option_reasons"))
        for idx, option in enumerate(options):
            marker = "" if idx != correct_index else " (correct)"
            reason = str(option_explanations.get(str(idx)) or "").strip()
            if reason:
                st.write(f"{_format_option(idx, option)}{marker}: {reason}")

    next_label = _t(language, "next") if position < len(queue) - 1 else _t(language, "finish")
    if st.button(next_label, use_container_width=True):
        st.session_state["eco_quiz_position"] = position + 1
        st.session_state["eco_quiz_submitted"] = False
        st.session_state["eco_last_result"] = None
        st.rerun()


def _render_stats_tab(language: str) -> None:
    question_bank = st.session_state["eco_question_bank"]
    attempts = st.session_state["eco_attempt_log"]
    stats = build_economics_stats(question_bank, attempts)

    st.caption(_t(language, "stats_caption"))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(_t(language, "total_answered"), stats["total_answered"])
    col2.metric(_t(language, "accuracy"), f"{stats['accuracy']:.1%}")
    col3.metric(_t(language, "correct_incorrect"), f"{stats['correct']} / {stats['incorrect']}")
    col4.metric(_t(language, "completion_rate"), f"{stats['completion_rate']:.1%}")

    col5, col6 = st.columns(2)
    col5.metric(_t(language, "current_streak"), stats["current_streak"])
    col6.metric(_t(language, "best_streak"), stats["best_streak"])

    topic_rows = stats.get("topic_performance", [])
    if topic_rows:
        topic_df = pd.DataFrame(topic_rows)
        topic_df["Accuracy"] = topic_df["Accuracy"].map(lambda value: f"{float(value):.1%}")
        st.caption(_t(language, "topic_breakdown"))
        st.dataframe(topic_df, use_container_width=True, hide_index=True)
    else:
        st.info(_t(language, "no_questions"))


def _render_review_tab(language: str) -> None:
    question_bank = st.session_state["eco_question_bank"]
    attempts = st.session_state["eco_attempt_log"]
    question_map = _get_question_map(question_bank)

    unresolved_ids = get_unresolved_mistake_ids(attempts)
    st.caption(_t(language, "review_list"))
    if not unresolved_ids:
        st.info(_t(language, "no_mistakes"))
    else:
        review_rows: List[Dict[str, Any]] = []
        attempts_by_question: Dict[str, List[Dict[str, Any]]] = {}
        for attempt in attempts:
            qid = str(attempt.get("question_id") or "")
            if qid:
                attempts_by_question.setdefault(qid, []).append(attempt)

        for qid in unresolved_ids:
            question = question_map.get(qid)
            if not question:
                continue
            localized = get_localized_question(question, language)
            q_attempts = attempts_by_question.get(qid, [])
            wrong_count = sum(1 for item in q_attempts if not bool(item.get("is_correct", False)))
            last_seen = q_attempts[-1].get("timestamp", "") if q_attempts else ""
            review_rows.append({
                "ID": qid,
                "Topic": localized.get("topic", ""),
                "Difficulty": localized.get("difficulty", ""),
                "Source": localized.get("source", ""),
                "Question": localized.get("question_text", ""),
                "Wrong Attempts": wrong_count,
                "Last Attempt": last_seen,
            })

        if review_rows:
            st.dataframe(pd.DataFrame(review_rows), use_container_width=True, hide_index=True)
        if st.button(_t(language, "start_review_quiz"), use_container_width=True, type="primary"):
            _start_quiz(unresolved_ids)
            st.rerun()

    wrong_attempt_rows = [
        {
            "Timestamp": item.get("timestamp", ""),
            "Question ID": item.get("question_id", ""),
            "Topic": item.get("topic", ""),
            "Selected": item.get("selected_answer", ""),
            "Correct": item.get("correct_answer", ""),
            "Language": item.get("language", ""),
        }
        for item in attempts
        if not bool(item.get("is_correct", False))
    ]

    if wrong_attempt_rows:
        st.caption(_t(language, "review_history"))
        st.dataframe(pd.DataFrame(wrong_attempt_rows).tail(40), use_container_width=True, hide_index=True)


def _render_create_tab(language: str) -> None:
    st.caption(_t(language, "create_caption"))

    with st.form("eco_create_form", clear_on_submit=False):
        topic = st.text_input(_t(language, "topic"), value="Macroeconomics")
        difficulty = st.selectbox(_t(language, "difficulty"), options=["easy", "medium", "hard"], index=1)
        correct_option = st.selectbox(
            _t(language, "correct_option"),
            options=[0, 1, 2, 3],
            format_func=lambda idx: chr(ord("A") + idx),
            index=0,
        )

        question_en = st.text_area(_t(language, "question_en"), height=80)
        option_en_a = st.text_input(f"{_t(language, 'option')} A (EN)")
        option_en_b = st.text_input(f"{_t(language, 'option')} B (EN)")
        option_en_c = st.text_input(f"{_t(language, 'option')} C (EN)")
        option_en_d = st.text_input(f"{_t(language, 'option')} D (EN)")
        explanation_en = st.text_area(_t(language, "explanation_en"), height=100)
        option_exp_en = st.text_area(_t(language, "option_exp_en"), height=80)

        question_cs = st.text_area(_t(language, "question_cs"), height=80)
        option_cs_a = st.text_input(f"{_t(language, 'option')} A (CS)")
        option_cs_b = st.text_input(f"{_t(language, 'option')} B (CS)")
        option_cs_c = st.text_input(f"{_t(language, 'option')} C (CS)")
        option_cs_d = st.text_input(f"{_t(language, 'option')} D (CS)")
        explanation_cs = st.text_area(_t(language, "explanation_cs"), height=100)
        option_exp_cs = st.text_area(_t(language, "option_exp_cs"), height=80)

        save_clicked = st.form_submit_button(_t(language, "save_question"), type="primary", use_container_width=True)

    if not save_clicked:
        return

    options_en = [option_en_a.strip(), option_en_b.strip(), option_en_c.strip(), option_en_d.strip()]
    options_cs = [option_cs_a.strip(), option_cs_b.strip(), option_cs_c.strip(), option_cs_d.strip()]

    payload = {
        "id": f"custom_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
        "language": "en",
        "question_text": question_en.strip(),
        "options": options_en,
        "correct_answer": int(correct_option),
        "explanation": explanation_en.strip(),
        "topic": topic.strip() or "Macroeconomics",
        "difficulty": difficulty,
        "source": "custom",
        "option_explanations": _parse_option_explanations(option_exp_en),
        "translations": {
            "en": {
                "question_text": question_en.strip(),
                "options": options_en,
                "explanation": explanation_en.strip(),
                "option_explanations": _parse_option_explanations(option_exp_en),
            },
            "cs": {
                "question_text": question_cs.strip(),
                "options": options_cs,
                "explanation": explanation_cs.strip(),
                "option_explanations": _parse_option_explanations(option_exp_cs),
            },
        },
    }

    normalized = normalize_question(payload, default_source="custom")
    if normalized is None:
        st.error(_t(language, "create_error"))
        return

    question_bank = list(st.session_state["eco_question_bank"])
    question_bank.append(normalized)
    save_question_bank(question_bank)
    st.session_state["eco_question_bank"] = load_question_bank()
    st.success(_t(language, "create_success"))


def _render_ai_tab(language: str, analysis_result: Dict[str, Any]) -> None:
    st.caption(_t(language, "ai_caption"))

    n_questions = st.slider(
        _t(language, "ai_count"),
        min_value=1,
        max_value=15,
        value=6,
        step=1,
        key="eco_ai_count",
    )
    topic_focus = st.text_input(_t(language, "ai_focus"), key="eco_ai_focus")

    if st.button(_t(language, "ai_generate"), type="primary", use_container_width=True):
        try:
            streamlit_secrets = st.secrets
        except Exception:
            streamlit_secrets = None

        api_key = resolve_groq_api_key(streamlit_secrets)
        context_payload = _build_analysis_context(analysis_result, topic_focus)

        with st.spinner("Generating economics questions..."):
            ai_result = generate_economics_questions(
                analyzed_context=context_payload,
                api_key=api_key,
                n_questions=n_questions,
            )

        if not ai_result.get("available", False):
            st.error(f"{_t(language, 'ai_error')}: {ai_result.get('error', 'unknown error')}")
            return

        question_bank = list(st.session_state["eco_question_bank"])
        existing_ids = {str(item.get("id")): item for item in question_bank}

        for question in ai_result.get("questions", []):
            question_id = str(question.get("id") or "").strip()
            candidate = dict(question)
            if not question_id or question_id in existing_ids:
                candidate["id"] = f"eco_ai_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
            existing_ids[str(candidate["id"])] = candidate

        merged_questions = list(existing_ids.values())
        save_question_bank(merged_questions)
        reloaded = load_question_bank()
        st.session_state["eco_question_bank"] = reloaded
        st.session_state["eco_last_generated_questions"] = ai_result.get("questions", [])
        st.session_state["eco_last_ai_raw_response"] = str(ai_result.get("raw_response", "") or "")
        st.session_state["eco_last_ai_json_mode_error"] = str(ai_result.get("json_mode_error", "") or "")

        st.success(_t(language, "ai_success"))
        st.caption(f"Generated: {len(ai_result.get('questions', []))}, stored total: {len(reloaded)}")
        if ai_result.get("json_mode_error"):
            st.caption(f"{_t(language, 'ai_json_fallback')}: {ai_result['json_mode_error']}")

    generated_rows = st.session_state.get("eco_last_generated_questions", [])
    if generated_rows:
        preview_data = [
            {
                "ID": row.get("id", ""),
                "Topic": row.get("topic", ""),
                "Difficulty": row.get("difficulty", ""),
                "Source": row.get("source", ""),
                "Question": get_localized_question(row, language).get("question_text", ""),
            }
            for row in generated_rows
        ]
        st.caption(_t(language, "ai_preview"))
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)

    last_json_mode_error = str(st.session_state.get("eco_last_ai_json_mode_error", "") or "")
    if last_json_mode_error:
        st.caption(f"{_t(language, 'ai_json_fallback')}: {last_json_mode_error}")

    last_raw_response = str(st.session_state.get("eco_last_ai_raw_response", "") or "").strip()
    if last_raw_response:
        with st.expander(_t(language, "ai_raw"), expanded=False):
            st.code(last_raw_response, language="json")


def render_economics_questions_section(analysis_result: Dict[str, Any]) -> None:
    _init_state()

    selected_language = st.selectbox(
        "Language / Jazyk",
        options=["English", "Čeština"],
        index=0 if st.session_state.get("eco_ui_language", "en") == "en" else 1,
        key="eco_lang_selector",
    )
    language = "cs" if selected_language == "Čeština" else "en"
    st.session_state["eco_ui_language"] = language

    st.header(_t(language, "section_title"))
    st.caption(_t(language, "section_caption"))

    tabs = st.tabs([
        _t(language, "tab_quiz"),
        _t(language, "tab_stats"),
        _t(language, "tab_review"),
        _t(language, "tab_create"),
        _t(language, "tab_ai"),
    ])

    with tabs[0]:
        _render_quiz_tab(language)

    with tabs[1]:
        _render_stats_tab(language)

    with tabs[2]:
        _render_review_tab(language)

    with tabs[3]:
        _render_create_tab(language)

    with tabs[4]:
        _render_ai_tab(language, analysis_result)
