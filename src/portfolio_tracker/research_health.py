"""Factual research-governance and review-queue diagnostics.

This module deliberately does not produce an investment score.  It reports
coverage, dates, and overdue work so the UI can distinguish missing evidence
from an unfavorable investment view.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any


FRESHNESS_THRESHOLDS_DAYS = {
    "price": 1,
    "news": 7,
    "quarterly_filing": 150,
    "annual_filing": 450,
    "manual_evidence": 180,
}


def _payload(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    nested = record.get("payload")
    return dict(nested) if isinstance(nested, Mapping) else dict(record)


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [dict(item) for item in value.values() if isinstance(item, Mapping)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _as_of_day(value: date | datetime | None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    return value.date() if isinstance(value, datetime) else value


def _ticker(record: Mapping[str, Any]) -> str:
    payload = _payload(record)
    return str(record.get("ticker") or payload.get("ticker") or "").strip().upper()


def _freshness_status(observed: date | None, *, as_of: date, threshold_days: int) -> str:
    if observed is None:
        return "missing_date"
    age = (as_of - observed).days
    if age < 0:
        return "future_date"
    return "fresh" if age <= threshold_days else "review_due"


def assess_research_health(
    tickers: Sequence[str],
    *,
    theses: Any = (),
    sources: Any = (),
    catalysts: Any = (),
    price_observations: Mapping[str, Any] | None = None,
    as_of: date | datetime | None = None,
) -> dict[str, Any]:
    """Return transparent per-ticker evidence and review diagnostics.

    ``price_observations`` values may be dates, timestamps, or mappings with
    ``observed_at``/``fetched_at`` and ``source``.  Macro data is intentionally
    outside freshness scoring because the app locks it to a reference year.
    """

    today = _as_of_day(as_of)
    codes = list(dict.fromkeys(str(item or "").strip().upper() for item in tickers if str(item or "").strip()))
    thesis_by_ticker = {_ticker(item): item for item in _records(theses) if _ticker(item)}
    source_rows = _records(sources)
    catalyst_rows = _records(catalysts)
    observations = {str(key).upper(): value for key, value in (price_observations or {}).items()}

    rows: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    for code in codes:
        thesis_record = thesis_by_ticker.get(code, {})
        thesis = _payload(thesis_record)
        review_date = _day(
            thesis_record.get("next_review_at")
            if isinstance(thesis_record, Mapping)
            else None
        ) or _day(thesis.get("review_date"))
        thesis_present = bool(thesis_record)
        thesis_review_status = (
            "missing" if not thesis_present
            else "overdue" if review_date is not None and review_date < today
            else "scheduled" if review_date is not None
            else "unscheduled"
        )

        ticker_sources = [item for item in source_rows if _ticker(item) in {"", code}]
        primary_sources = sum(
            bool(item.get("primary_source", _payload(item).get("primary_source")))
            for item in ticker_sources
        )
        verified_sources = sum(
            bool(item.get("verified_at") or item.get("verified_by") or _payload(item).get("verified"))
            for item in ticker_sources
        )
        source_dates = [
            parsed
            for item in ticker_sources
            if (parsed := _day(
                item.get("accessed_at")
                or item.get("published_at")
                or item.get("period_end")
                or item.get("updated_at")
            )) is not None
        ]
        latest_source_date = max(source_dates) if source_dates else None
        evidence_status = _freshness_status(
            latest_source_date,
            as_of=today,
            threshold_days=FRESHNESS_THRESHOLDS_DAYS["manual_evidence"],
        ) if ticker_sources else "missing"

        ticker_catalysts = [item for item in catalyst_rows if _ticker(item) == code]
        upcoming = 0
        overdue = 0
        for item in ticker_catalysts:
            payload = _payload(item)
            status = str(item.get("status") or payload.get("status") or "expected").strip().lower()
            window_start = _day(item.get("window_start") or payload.get("window_start"))
            window_end = _day(item.get("window_end") or payload.get("window_end")) or window_start
            if status == "expected" and window_end is not None and window_end < today:
                overdue += 1
            elif status == "expected" and window_start is not None:
                upcoming += 1

        raw_observation = observations.get(code)
        if isinstance(raw_observation, Mapping):
            observed_at = _day(raw_observation.get("observed_at") or raw_observation.get("fetched_at"))
            price_source = str(raw_observation.get("source") or "Observed")
        else:
            observed_at = _day(raw_observation)
            price_source = "Observed" if raw_observation is not None else "Missing"
        price_status = _freshness_status(
            observed_at,
            as_of=today,
            threshold_days=FRESHNESS_THRESHOLDS_DAYS["price"],
        )

        row = {
            "ticker": code,
            "price_status": price_status,
            "price_observed_at": observed_at.isoformat() if observed_at else None,
            "price_source": price_source,
            "thesis_status": thesis_review_status,
            "thesis_review_date": review_date.isoformat() if review_date else None,
            "source_count": len(ticker_sources),
            "primary_source_count": primary_sources,
            "verified_source_count": verified_sources,
            "latest_source_date": latest_source_date.isoformat() if latest_source_date else None,
            "evidence_status": evidence_status,
            "catalyst_count": len(ticker_catalysts),
            "upcoming_catalysts": upcoming,
            "overdue_catalysts": overdue,
        }
        rows.append(row)
        if price_status != "fresh":
            review_queue.append({"ticker": code, "area": "price", "status": price_status})
        if thesis_review_status in {"missing", "overdue", "unscheduled"}:
            review_queue.append({"ticker": code, "area": "thesis", "status": thesis_review_status})
        if evidence_status in {"missing", "missing_date", "review_due", "future_date"}:
            review_queue.append({"ticker": code, "area": "evidence", "status": evidence_status})
        if overdue:
            review_queue.append({"ticker": code, "area": "catalyst", "status": "outcome_overdue", "count": overdue})

    count = len(rows)
    summary = {
        "ticker_count": count,
        "fresh_price_coverage_pct": 100.0 * sum(row["price_status"] == "fresh" for row in rows) / count if count else 0.0,
        "thesis_coverage_pct": 100.0 * sum(row["thesis_status"] != "missing" for row in rows) / count if count else 0.0,
        "scheduled_thesis_review_pct": 100.0 * sum(row["thesis_status"] == "scheduled" for row in rows) / count if count else 0.0,
        "evidence_coverage_pct": 100.0 * sum(row["source_count"] > 0 for row in rows) / count if count else 0.0,
        "primary_source_coverage_pct": 100.0 * sum(row["primary_source_count"] > 0 for row in rows) / count if count else 0.0,
        "overdue_thesis_count": sum(row["thesis_status"] == "overdue" for row in rows),
        "overdue_catalyst_count": sum(int(row["overdue_catalysts"]) for row in rows),
        "review_queue_count": len(review_queue),
    }
    return {
        "as_of": today.isoformat(),
        "summary": summary,
        "tickers": rows,
        "review_queue": review_queue,
        "freshness_thresholds_days": dict(FRESHNESS_THRESHOLDS_DAYS),
        "macro_policy": "Reference-year locked; macro observations are labelled by year rather than scored as stale.",
        "methodology": "Factual coverage and review status only; no investment or return score is produced.",
    }


__all__ = ["FRESHNESS_THRESHOLDS_DAYS", "assess_research_health"]
