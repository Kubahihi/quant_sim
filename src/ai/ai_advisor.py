from __future__ import annotations

import json
from typing import Any, Dict, Optional

from openai import OpenAI
from src.ai.ai_review import DEFAULT_GROQ_MODEL, _extract_message_text


SYSTEM_PROMPTS = {
    "risk_cockpit": (
        "You are a senior quantitative risk manager at a top hedge fund. "
        "Review the provided portfolio risk metrics (VaR, CVaR, Drawdown, Volatility). "
        "Provide a concise, 2-3 sentence objective assessment of the tail risk. "
        "State whether the risk is excessive and provide a specific, actionable recommendation "
        "(e.g. 'Add 10% exposure to short-term Treasuries like SHY to reduce tail risk')."
    ),
    "factor_exposure": (
        "You are an expert factor investing strategist. "
        "Review the provided Fama-French and Smart Beta factor exposures of the portfolio. "
        "Provide a concise, 2-3 sentence assessment of the factor tilts. "
        "Highlight any dangerous concentration (e.g. massive overweight in Momentum/Growth) "
        "and suggest 1-2 specific actions or asset classes to balance the exposure."
    ),
    "regime_detection": (
        "You are a macro-quant strategist. "
        "Review the provided current market regime state detected by a Markov Switching Model. "
        "Provide a concise, 2-3 sentence tactical playbook. "
        "If it's a high-volatility/stress regime, recommend defensive positioning. "
        "If it's a low-volatility/calm regime, recommend maximizing equity risk premia."
    ),
    "news_synthesis": (
        "You are a top-tier financial news analyst. "
        "Review the provided list of recent news headlines related to the portfolio's assets. "
        "Synthesize the underlying fundamental or macroeconomic narrative in exactly 2-3 sentences. "
        "Focus on systemic risks or major tailwinds, ignoring noise."
    ),
    "optimizer_explain": (
        "You are a portfolio manager explaining an algorithmic trade execution. "
        "Review the provided outputs from a Cost-Aware Rebalancing Optimizer (current vs optimal weights, turnover, transaction costs). "
        "Explain in 2-3 sentences why the optimizer chose its path. Focus on the trade-off "
        "between expected Sharpe ratio improvement and the drag of transaction costs/turnover."
    )
}


def generate_advisor_insight(
    context_data: Dict[str, Any],
    prompt_type: str,
    api_key: Optional[str],
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 250,
) -> Dict[str, Any]:
    """Generate a specific advisory insight using Groq API."""
    if not api_key:
        return {
            "source": "ai_unavailable",
            "available": False,
            "error": "GROQ_API_KEY was not provided.",
        }

    if prompt_type not in SYSTEM_PROMPTS:
        return {
            "source": "ai_error",
            "available": False,
            "error": f"Unknown prompt_type: {prompt_type}",
        }

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        prompt_json = json.dumps(context_data, ensure_ascii=False, indent=2)
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPTS[prompt_type] + " Respond purely in English, with no pleasantries or filler text. Be direct and objective."
            },
            {
                "role": "user",
                "content": f"Context Data:\n{prompt_json}"
            },
        ]

        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )

        content = _extract_message_text(completion.choices[0].message) if completion.choices else ""
        
        if not content:
            raise ValueError("Empty response from AI.")

        return {
            "source": "groq",
            "available": True,
            "insight": content.strip(),
        }

    except Exception as exc:
        return {
            "source": "ai_error",
            "available": False,
            "error": str(exc),
        }
