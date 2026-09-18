import sqlite3

from src.portfolio_tracker.laura_case import (
    cashflow_calendar, check_ips_text, check_trading_notes, deliverables_calendar,
    save_case_section, word_count,
)
from src.portfolio_tracker.strategy_store import load_client_mandate, save_client_mandate


def test_calendar_keeps_deposits_and_beginning_of_year_operating_flows_separate():
    rows = cashflow_calendar()
    assert sum(r["deposit_usd"] for r in rows) == 450_000
    assert sum(r["operating_payment_usd"] for r in rows) == 500_000
    assert [r["calendar_year"] for r in rows if r["operating_payment_usd"]] == list(range(2033, 2043))
    assert all(r["model_year"] == r["calendar_year"] - 2026 for r in rows)


def test_deadline_conversion_respects_dst_in_both_timezones():
    rows = deliverables_calendar()
    assert len(rows) == 4
    assert rows[0]["deadline_et"].endswith("17:00:00-04:00")
    assert rows[2]["deadline_et"].endswith("17:00:00-05:00")
    assert all("T23:00:00" in row["deadline_prague"] for row in rows)


def test_trading_notes_enforce_three_distinct_executed_references_and_individual_limits():
    notes = [{"execution_reference": f"WInS-{i}", "note": "Original exact note.",
              "reflection": "word " * 100, "verbatim_confirmed": True} for i in range(3)]
    assert all(c["passed"] for c in check_trading_notes(notes))
    notes[1]["reflection"] += "extra"
    assert sum(not c["passed"] for c in check_trading_notes(notes)) == 1
    notes[1]["execution_reference"] = notes[0]["execution_reference"]
    assert sum(not c["passed"] for c in check_trading_notes(notes)) == 2
    notes[0]["verbatim_confirmed"] = "yes"
    assert sum(not c["passed"] for c in check_trading_notes(notes)) == 3


def test_pitch_and_ips_have_independent_word_limits_and_link_check():
    assert all(c["passed"] for c in check_ips_text("word " * 50, "word " * 500))
    assert not check_ips_text("word " * 51, "short IPS")[0]["passed"]
    assert not check_ips_text("pitch", "word " * 501)[1]["passed"]
    assert not check_ips_text("pitch", "See https://example.com")[2]["passed"]
    assert word_count("  Mixed\n whitespace\tworks  ") == 3


def test_case_sections_preserve_other_mandate_fields_and_each_other():
    with sqlite3.connect(":memory:") as conn:
        save_client_mandate(conn, {"client_name": "Existing", "behavioral_profile": {"score": 7},
                                  "policy_benchmark": "existing"}, updated_by="team")
        save_case_section(conn, "planning", {"policy": {"confidence_target": .95}}, updated_by="team")
        save_case_section(conn, "deliverables", {"pitch": "Our pitch"}, updated_by="team")
        payload = load_client_mandate(conn)["payload"]
    assert payload["client_name"] == "Existing"
    assert payload["behavioral_profile"] == {"score": 7}
    assert payload["policy_benchmark"] == "existing"
    assert payload["laura_case"]["planning"]["policy"]["confidence_target"] == .95
    assert payload["laura_case"]["deliverables"]["pitch"] == "Our pitch"
