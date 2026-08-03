"""Transparent client-behaviour questionnaire and decision guardrails.

This module is an investment-governance aid, not a clinical or psychometric
assessment. Every score is a direct transformation of visible questionnaire
answers so the team can explain and challenge the result.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd


BEHAVIORAL_QUESTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "loss_1",
        "category": "loss_aversion",
        "category_label": "Loss aversion",
        "statement": "A portfolio loss feels materially more important than an equally large gain.",
    },
    {
        "id": "loss_2",
        "category": "loss_aversion",
        "category_label": "Loss aversion",
        "statement": "After a sharp decline, I would prefer to reduce risk before reviewing the original plan.",
    },
    {
        "id": "confidence_1",
        "category": "overconfidence",
        "category_label": "Overconfidence",
        "statement": "I am usually more confident in my investment judgement than in diversified market evidence.",
    },
    {
        "id": "confidence_2",
        "category": "overconfidence",
        "category_label": "Overconfidence",
        "statement": "When an investment performs well, I tend to attribute the result mainly to decision skill.",
    },
    {
        "id": "recency_1",
        "category": "recency_bias",
        "category_label": "Recency bias",
        "statement": "Recent market performance strongly changes what I expect over the long term.",
    },
    {
        "id": "recency_2",
        "category": "recency_bias",
        "category_label": "Recency bias",
        "statement": "I am tempted to increase exposure after a strong run and reduce it after a weak run.",
    },
    {
        "id": "herding_1",
        "category": "herding",
        "category_label": "Herding / social proof",
        "statement": "I become uncomfortable holding a different view from peers, advisers, or market consensus.",
    },
    {
        "id": "herding_2",
        "category": "herding",
        "category_label": "Herding / social proof",
        "statement": "A popular investment feels safer to me even when its valuation or evidence has weakened.",
    },
    {
        "id": "anchor_1",
        "category": "anchoring",
        "category_label": "Anchoring",
        "statement": "My view of fair value remains influenced by the original purchase price.",
    },
    {
        "id": "disposition_1",
        "category": "disposition_effect",
        "category_label": "Disposition effect",
        "statement": "I prefer to hold a losing investment until it returns to my purchase price.",
    },
    {
        "id": "action_1",
        "category": "action_bias",
        "category_label": "Action bias",
        "statement": "When markets are volatile, taking immediate action feels better than waiting for a scheduled review.",
    },
    {
        "id": "action_2",
        "category": "action_bias",
        "category_label": "Action bias",
        "statement": "I would change the portfolio in response to a compelling headline before the evidence is fully checked.",
    },
)

LIKERT_OPTIONS: tuple[str, ...] = (
    "1 - Strongly disagree",
    "2 - Disagree",
    "3 - Neither agree nor disagree",
    "4 - Agree",
    "5 - Strongly agree",
)

DRAWDOWN_ACTION_SCORES: dict[str, float] = {
    "Sell all risk assets": 0.0,
    "Sell part of the portfolio": 25.0,
    "Seek advice before acting": 70.0,
    "Hold the strategic allocation": 80.0,
    "Rebalance back to policy targets": 100.0,
    "Add risk beyond policy targets": 35.0,
}

DEFAULT_DRAWDOWN_ACTIONS: dict[str, str] = {
    "-10%": "Hold the strategic allocation",
    "-20%": "Seek advice before acting",
    "-30%": "Seek advice before acting",
}

_CATEGORY_GUARDRAILS: dict[str, tuple[str, str, str]] = {
    "loss_aversion": (
        "Pre-commit the drawdown response",
        "Write the permitted action at each loss threshold before the threshold is reached.",
        "High",
    ),
    "overconfidence": (
        "Independent challenge before concentration",
        "Require a disconfirming-evidence review and a second reviewer before exceeding the standard position limit.",
        "High",
    ),
    "recency_bias": (
        "Use base rates before changing long-term assumptions",
        "Show full-cycle evidence and the policy horizon before extrapolating the latest market move.",
        "Medium",
    ),
    "herding": (
        "Separate popularity from evidence",
        "Document why the thesis remains valid without using peer ownership or consensus as the primary reason.",
        "Medium",
    ),
    "anchoring": (
        "Refresh fair value independently of purchase price",
        "Re-underwrite from current cash flows, risks, and alternatives; hide the entry price during the first review pass.",
        "Medium",
    ),
    "disposition_effect": (
        "Apply symmetric sell discipline",
        "Use the same forward-looking thesis and opportunity-cost test for winners and losers.",
        "High",
    ),
    "action_bias": (
        "Cooling-off period for unscheduled trades",
        "Apply a 48-hour pause and written evidence checklist unless a pre-agreed risk limit is breached.",
        "High",
    ),
}


def parse_likert_answer(value: Any) -> int:
    """Convert a numeric or display-label Likert response into an integer 1-5."""
    if isinstance(value, str):
        value = value.strip().split("-", 1)[0].strip()
    try:
        answer = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("Behavioural answers must use the 1-5 scale.") from exc
    if answer < 1 or answer > 5:
        raise ValueError("Behavioural answers must be between 1 and 5.")
    return answer


def _vulnerability_band(score: float) -> str:
    if score < 25.0:
        return "Low"
    if score < 50.0:
        return "Moderate"
    if score < 70.0:
        return "Elevated"
    return "High"


def _risk_tolerance_consistency(risk_tolerance: str, resilience: float | None) -> tuple[str, str]:
    if resilience is None:
        return "Not assessed", "Add intended actions for the drawdown scenarios."
    tolerance = str(risk_tolerance or "").strip().casefold()
    expected = {
        "conservative": 40.0,
        "moderate": 55.0,
        "growth": 70.0,
        "aggressive": 80.0,
    }.get(tolerance)
    if expected is None:
        return "Not assessed", "Specify the client's declared risk tolerance in the mandate."
    gap = resilience - expected
    if gap < -15.0:
        return (
            "Behaviour below declared tolerance",
            "Planned drawdown actions are less resilient than the declared risk tolerance; use the lower level until reconciled.",
        )
    if gap > 20.0:
        return (
            "Behaviour above declared tolerance",
            "Planned actions appear more resilient than the declared tolerance; confirm that this is informed willingness, not pressure to take risk.",
        )
    return "Broadly consistent", "Declared tolerance and planned drawdown behaviour are broadly aligned."


def _decision_style(category_scores: Mapping[str, float], overall: float) -> str:
    if overall < 25.0:
        return "Process-oriented"
    if not category_scores:
        return "Not assessed"
    leading = max(category_scores, key=category_scores.get)
    if category_scores[leading] < 50.0:
        return "Mixed / context-dependent"
    return {
        "loss_aversion": "Loss-sensitive",
        "overconfidence": "High-conviction / self-directed",
        "recency_bias": "Recent-performance sensitive",
        "herding": "Consensus-sensitive",
        "anchoring": "Reference-point anchored",
        "disposition_effect": "Entry-price sensitive",
        "action_bias": "Action-oriented / reactive",
    }.get(leading, "Mixed / context-dependent")


def _build_guardrails(category_scores: Mapping[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "Priority": "High",
            "Trigger": "Any material allocation change",
            "Guardrail": "Policy and goal check",
            "Implementation": "Record which client goal, risk limit, or new fact justifies the change before execution.",
        }
    ]
    labels = {question["category"]: question["category_label"] for question in BEHAVIORAL_QUESTIONS}
    for category, score in sorted(category_scores.items(), key=lambda item: item[1], reverse=True):
        if score < 50.0:
            continue
        title, implementation, priority = _CATEGORY_GUARDRAILS[category]
        rows.append(
            {
                "Priority": priority,
                "Trigger": f"{labels[category]} score {score:.0f}/100",
                "Guardrail": title,
                "Implementation": implementation,
            }
        )
    rows.append(
        {
            "Priority": "Medium",
            "Trigger": "Market stress or emotionally salient news",
            "Guardrail": "Scheduled communication before trading",
            "Implementation": "Restate goals, remaining liquidity, policy ranges, and the pre-agreed drawdown plan before proposing a trade.",
        }
    )
    return pd.DataFrame(rows)


def _communication_plan(category_scores: Mapping[str, float]) -> list[str]:
    recommendations = [
        "Lead with progress toward client goals and policy ranges, not short-term benchmark noise.",
    ]
    if category_scores.get("loss_aversion", 0.0) >= 50.0:
        recommendations.append("Show downside ranges in currency amounts and repeat the pre-agreed response before discussing returns.")
    if category_scores.get("overconfidence", 0.0) >= 50.0:
        recommendations.append("Present the strongest counter-case, forecast error range, and position-limit impact.")
    if max(category_scores.get("recency_bias", 0.0), category_scores.get("herding", 0.0)) >= 50.0:
        recommendations.append("Pair current news with long-horizon base rates and evidence from more than one market regime.")
    if category_scores.get("action_bias", 0.0) >= 50.0:
        recommendations.append("Use scheduled decision windows and make 'no trade' an explicit available outcome.")
    return recommendations


def assess_behavioral_profile(
    answers: Mapping[str, Any],
    *,
    drawdown_actions: Mapping[str, str] | None = None,
    risk_tolerance: str = "",
) -> dict[str, Any]:
    """Score visible answers and produce explainable governance recommendations."""
    if not isinstance(answers, Mapping):
        raise TypeError("answers must be a mapping keyed by question id.")
    copied_answers = deepcopy(dict(answers))
    question_by_id = {question["id"]: question for question in BEHAVIORAL_QUESTIONS}
    unknown = sorted(set(copied_answers).difference(question_by_id))
    if unknown:
        raise ValueError(f"Unknown behavioural question ids: {', '.join(unknown)}.")

    parsed: dict[str, int] = {
        question_id: parse_likert_answer(value)
        for question_id, value in copied_answers.items()
        if value is not None and str(value).strip()
    }
    grouped: dict[str, list[int]] = {}
    category_labels: dict[str, str] = {}
    for question in BEHAVIORAL_QUESTIONS:
        category_labels[question["category"]] = question["category_label"]
        if question["id"] in parsed:
            grouped.setdefault(question["category"], []).append(parsed[question["id"]])

    category_scores = {
        category: float((np.mean(values) - 1.0) / 4.0 * 100.0)
        for category, values in grouped.items()
    }
    overall = float(np.mean([(value - 1.0) / 4.0 * 100.0 for value in parsed.values()])) if parsed else 0.0
    coverage = len(parsed) / len(BEHAVIORAL_QUESTIONS)
    bias_rows = [
        {
            "Bias": category_labels[category],
            "Category": category,
            "Score": score,
            "Band": _vulnerability_band(score),
            "AnsweredItems": len(grouped[category]),
        }
        for category, score in sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
    ]

    actions = dict(drawdown_actions or {})
    action_scores: list[tuple[float, float]] = []
    severity_weights = {"-10%": 1.0, "-20%": 2.0, "-30%": 3.0}
    for threshold, action in actions.items():
        if action not in DRAWDOWN_ACTION_SCORES:
            raise ValueError(f"Unknown drawdown action for {threshold}: {action}.")
        action_scores.append((DRAWDOWN_ACTION_SCORES[action], severity_weights.get(str(threshold), 1.0)))
    resilience = (
        float(sum(score * weight for score, weight in action_scores) / sum(weight for _, weight in action_scores))
        if action_scores
        else None
    )
    consistency_status, consistency_note = _risk_tolerance_consistency(risk_tolerance, resilience)
    top_biases = [row["Bias"] for row in bias_rows if row["Score"] >= 50.0][:3]

    return {
        "OverallVulnerabilityScore": overall,
        "VulnerabilityBand": _vulnerability_band(overall),
        "DecisionStyle": _decision_style(category_scores, overall),
        "CoveragePct": coverage,
        "AnsweredQuestions": len(parsed),
        "TotalQuestions": len(BEHAVIORAL_QUESTIONS),
        "TopBiases": top_biases,
        "BiasScores": pd.DataFrame(bias_rows),
        "DrawdownResilienceScore": resilience,
        "RiskToleranceConsistency": consistency_status,
        "RiskToleranceConsistencyNote": consistency_note,
        "Guardrails": _build_guardrails(category_scores),
        "CommunicationPlan": _communication_plan(category_scores),
        "ParsedAnswers": parsed,
    }


__all__ = [
    "BEHAVIORAL_QUESTIONS",
    "LIKERT_OPTIONS",
    "DRAWDOWN_ACTION_SCORES",
    "DEFAULT_DRAWDOWN_ACTIONS",
    "parse_likert_answer",
    "assess_behavioral_profile",
]
