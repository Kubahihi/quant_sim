from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ai import company_analysis


def test_ai_dcf_assumptions_are_normalized_bounded_and_reproducible(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='''{
            "growth_rate": 12,
            "discount_rate": 6,
            "terminal_growth_rate": 4,
            "years": 14,
            "rationale": "Growth and risk reflect the supplied metrics."
        }'''))]
    )

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["temperature"] == 0.0
            return response

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(company_analysis, "OpenAI", lambda **kwargs: fake_client)

    evidence = {
        "freeCashflow": 1_000_000_000,
        "revenueGrowth": 0.12,
        "earningsGrowth": 0.10,
        "marketCap": 20_000_000_000,
        "totalDebt": 2_000_000_000,
        "sharesOutstanding": 100_000_000,
        "currentPrice": 50.0,
    }
    result = company_analysis.generate_dcf_assumptions(evidence, "test-key")

    assert result["available"] is True
    assert result["assumptions"]["growth_rate"] == pytest.approx(0.12)
    assert result["assumptions"]["discount_rate"] == pytest.approx(0.06)
    assert result["assumptions"]["terminal_growth_rate"] == pytest.approx(0.035)
    assert result["assumptions"]["years"] == 10


def test_ai_dcf_assumptions_report_missing_key_without_calling_provider():
    result = company_analysis.generate_dcf_assumptions({}, None)

    assert result["available"] is False
    assert result["source"] == "ai_unavailable"
