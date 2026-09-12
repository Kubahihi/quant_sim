from collections import Counter
from threading import Barrier, Lock, get_ident

import pandas as pd
import pytest
import yfinance as yf

from src.analytics import company_analysis


@pytest.mark.parametrize("failed_field", [None, "quarterly_cashflow"])
def test_company_fetch_is_bounded_complete_and_keeps_tickers_on_one_worker(
    monkeypatch, failed_field,
):
    first_requests = Barrier(4, timeout=5)
    lock = Lock()
    calls = Counter()
    active = 0
    peak = 0
    ownership_errors = []
    filings = [{"type": "10-K", "date": "2025-12-31"}]

    class Ticker:
        def __init__(self, symbol):
            assert symbol == "TEST"
            self.owner = get_ident()

        def _load(self, field):
            nonlocal active, peak
            with lock:
                if get_ident() != self.owner:
                    ownership_errors.append(field)
                calls[field] += 1
                active += 1
                peak = max(peak, active)
            try:
                if field in {"info", "history", "news", "sec_filings"}:
                    first_requests.wait()
                if field == failed_field:
                    raise RuntimeError("test endpoint unavailable")
                if field == "info":
                    return {"shortName": "Test", "companyOfficers": [{"name": "Example"}]}
                if field == "news":
                    return [{"title": str(index)} for index in range(25)]
                if field == "sec_filings":
                    return filings
                return pd.DataFrame({field: [1.0, 2.0]})
            finally:
                with lock:
                    active -= 1

        def history(self, **kwargs):
            assert kwargs == {"period": "5y", "interval": "1d", "auto_adjust": False}
            return self._load("history")

        def __getattr__(self, field):
            return self._load(field)

    def geographic_revenue(symbol, *, sec_filings):
        assert symbol == "TEST"
        assert sec_filings is filings
        return {"available": True, "source_name": "Test filing"}

    monkeypatch.setattr(yf, "Ticker", Ticker)
    monkeypatch.setattr(company_analysis, "fetch_geographic_revenue", geographic_revenue)
    result = company_analysis.fetch_company_data(" test ")

    assert peak == 4
    assert not ownership_errors
    expected_fields = {
        "info", "history", "news", "sec_filings", "income_stmt", "balance_sheet",
        "cashflow", "quarterly_income_stmt", "quarterly_balance_sheet", "quarterly_cashflow",
    }
    assert calls == Counter(dict.fromkeys(expected_fields, 1))
    assert result["ticker"] == "TEST"
    assert result["info"]["shortName"] == "Test"
    assert result["metrics"] == {"shortName": "Test"}
    assert result["officers"] == [{"name": "Example"}]
    assert result["sec_filings"] is filings
    assert len(result["news"]) == 20
    assert result["geographic_revenue"]["available"] is True
    for key, attribute in (
        ("history", "history"),
        ("income_statement", "income_stmt"),
        ("quarterly_income_statement", "quarterly_income_stmt"),
        ("balance_sheet", "balance_sheet"),
        ("quarterly_balance_sheet", "quarterly_balance_sheet"),
        ("cash_flow", "cashflow"),
        ("quarterly_cash_flow", "quarterly_cashflow"),
    ):
        if attribute == failed_field:
            assert result[key].empty
        else:
            pd.testing.assert_frame_equal(result[key], pd.DataFrame({attribute: [1.0, 2.0]}))
    if failed_field:
        assert result["provider_status"] == "degraded"
        assert result["provider_errors"] == [
            "quarterly cash flow: RuntimeError: test endpoint unavailable"
        ]
    else:
        assert result["provider_status"] == "live"
        assert result["provider_errors"] == []
