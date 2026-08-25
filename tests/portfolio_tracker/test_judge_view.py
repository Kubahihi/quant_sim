from __future__ import annotations

import json

import pytest

from src.portfolio_tracker.judge_view import build_judge_view_model


def test_report_bound_reconciled_snapshot_has_highest_portfolio_authority():
    model = build_judge_view_model(
        {
            "report_record": {
                "payload": {
                    "portfolio_snapshot": {
                        "snapshot_id": "report-7",
                        "as_of": "2026-08-20T18:00:00+00:00",
                        "source": "Final report",
                        "reconciled": True,
                        "positions": [
                            {
                                "security_id": "AAA",
                                "ticker": "AAA",
                                "weight": 0.7,
                                "market_value": 350_000,
                                "currency": "USD",
                            }
                        ],
                    },
                    "performance_attribution": {"portfolio_return": 0.08},
                }
            },
            "pipeline": {
                "authority": "wins_reconciled",
                "canonical_snapshot": {
                    "snapshot_id": "wins-8",
                    "observed_at": "2026-08-21T18:00:00+00:00",
                    "payload": {
                        "positions": [{"ticker": "BBB", "market_value": 100_000}],
                    },
                },
            },
            "tracker_performance": {
                "positions": [{"ticker": "CCC", "current_value": 50_000}],
                "equity": 500_000,
            },
        }
    )

    portfolio = model["portfolio"]
    assert portfolio["source_tier"] == "report_bound"
    assert portfolio["source_label"] == "Report-bound reconciled snapshot"
    assert portfolio["snapshot_id"] == "report-7"
    assert portfolio["reconciled"] is True
    assert portfolio["positions"][0]["ticker"] == "AAA"
    assert portfolio["positions"][0]["weight_pct"] == pytest.approx(70.0)
    assert portfolio["total_return_pct"] == pytest.approx(8.0)
    assert portfolio["equity"] is None


def test_wins_canonical_snapshot_beats_tracker_when_report_snapshot_is_not_reconciled():
    model = build_judge_view_model(
        {
            "report_record": {
                "payload": {
                    "portfolio_snapshot": {
                        "snapshot_id": "draft-1",
                        "reconciled": False,
                        "positions": [{"ticker": "DRAFT", "weight": 1}],
                    }
                }
            },
            "pipeline": {
                "authority": "wins_reconciled",
                "consumer_bindings": {"reporting": {"allowed": True}},
                "canonical_snapshot": {
                    "snapshot_id": "wins-9",
                    "observed_at": "2026-08-22T12:30:00Z",
                    "payload": {
                        "positions": [
                            {
                                "ticker": "wins",
                                "asset_type": "Equity",
                                "weight_pct": 60,
                                "market_value": 300_000,
                            }
                        ],
                        "cash_value": 200_000,
                        "total_value": 500_000,
                    },
                },
            },
            "tracker_performance": {
                "positions": [{"ticker": "TRACK", "current_value": 1}],
            },
        }
    )

    portfolio = model["portfolio"]
    assert portfolio["source_tier"] == "wins_reconciled"
    assert portfolio["snapshot_id"] == "wins-9"
    assert portfolio["as_of"] == "2026-08-22T12:30:00Z"
    assert portfolio["positions"][0]["ticker"] == "WINS"
    assert portfolio["positions"][0]["weight"] == pytest.approx(0.6)
    assert portfolio["cash"] == pytest.approx(200_000)
    assert portfolio["equity"] == pytest.approx(500_000)
    assert model["integrity"]["reporting_binding_allowed"] is True


def test_invalid_report_snapshot_hash_cannot_override_reconciled_wins_snapshot():
    model = build_judge_view_model(
        {
            "report_record": {
                "payload": {
                    "portfolio_snapshot": {
                        "snapshot_id": "tampered-report",
                        "reconciled": True,
                        "positions": [{"ticker": "BAD", "market_value": 499_999}],
                    }
                }
            },
            "report_validation": {
                "is_ready": False,
                "issues": [
                    {
                        "code": "portfolio_snapshot_hash",
                        "message": "Portfolio snapshot failed its integrity check.",
                    }
                ],
            },
            "pipeline": {
                "authority": "wins_reconciled",
                "canonical_snapshot": {
                    "snapshot_id": "wins-clean",
                    "observed_at": "2026-08-22T12:30:00Z",
                    "payload": {
                        "positions": [{"ticker": "GOOD", "market_value": 500_000}],
                        "total_value": 500_000,
                    },
                },
            },
        }
    )

    assert model["portfolio"]["source_tier"] == "wins_reconciled"
    assert model["portfolio"]["snapshot_id"] == "wins-clean"
    assert model["portfolio"]["positions"][0]["ticker"] == "GOOD"


def test_tracker_is_explicitly_provisional_and_normalises_current_holdings():
    model = build_judge_view_model(
        {
            "pipeline": {
                "authority": "provisional",
                "canonical_snapshot": {
                    "snapshot_id": "manual-2",
                    "payload": {"positions": [{"ticker": "MANUAL"}]},
                },
            },
            "tracker_performance": {
                "equity": 505_000,
                "cash_before_pnl": 400_000,
                "realized_pnl": 5_000,
                "total_pnl": 5_000,
                "total_return_pct": 1.0,
                "positions": [
                    {
                        "ticker": "aaa",
                        "status": "open",
                        "quantity": 100,
                        "current_price": 1_000,
                        "return_pct": 4.5,
                    },
                    {
                        "ticker": "OLD",
                        "status": "closed",
                        "current_value": 20_000,
                    },
                ],
            },
        }
    )

    portfolio = model["portfolio"]
    assert portfolio["source_tier"] == "tracker_provisional"
    assert portfolio["source_label"] == "Tracker fallback — provisional"
    assert portfolio["reconciled"] is False
    assert portfolio["snapshot_id"] is None
    assert [item["ticker"] for item in portfolio["positions"]] == ["AAA"]
    assert portfolio["positions"][0]["market_value"] == pytest.approx(100_000)
    assert portfolio["positions"][0]["weight"] == pytest.approx(100_000 / 505_000)
    assert portfolio["positions"][0]["weight_pct"] == pytest.approx(
        100_000 / 505_000 * 100
    )
    assert portfolio["cash"] == pytest.approx(405_000)


def test_tracker_fallback_discloses_entry_price_valuation_and_withholds_return():
    model = build_judge_view_model(
        {
            "tracker_performance": {
                "equity": 507_000,
                "cash_before_pnl": 400_000,
                "realized_pnl": 5_000,
                "open_cash_income": 2_000,
                "total_return_pct": 1.4,
                "positions": [
                    {
                        "ticker": "AAA",
                        "quantity": 100,
                        "current_price": 1_000,
                        "price_source": "entry fallback",
                        "valuation_source": "WInS stored mark",
                        "price_observed_at": "2026-08-23T15:30:00Z",
                    }
                ],
            }
        }
    )

    portfolio = model["portfolio"]
    assert portfolio["cash"] == pytest.approx(407_000)
    assert portfolio["as_of"] == "2026-08-23T15:30:00Z"
    assert portfolio["valuation_sources"] == ["WInS stored mark (entry fallback)"]
    assert portfolio["return_suppressed"] is True
    assert portfolio["total_return_pct"] is None
    assert "entry price" in portfolio["valuation_note"]


def test_last_known_good_wins_snapshot_exposes_reporting_blockers():
    model = build_judge_view_model(
        {
            "pipeline": {
                "authority": "wins_reconciled",
                "last_known_good": True,
                "selection": {"is_fresh": False},
                "canonical_snapshot": {
                    "snapshot_id": "wins-old",
                    "observed_at": "2026-08-20T12:00:00Z",
                    "payload": {
                        "positions": [{"ticker": "AAA", "market_value": 500_000}],
                        "total_value": 500_000,
                    },
                },
                "consumer_bindings": {
                    "reporting": {
                        "allowed": False,
                        "blockers": [
                            "snapshot_stale",
                            "latest_wins_reconciliation_not_clean",
                        ],
                    }
                },
            }
        }
    )

    portfolio = model["portfolio"]
    assert portfolio["source_tier"] == "wins_reconciled"
    assert portfolio["last_known_good"] is True
    assert portfolio["snapshot_fresh"] is False
    assert portfolio["reporting_ready"] is False
    assert portfolio["reporting_blockers"] == [
        "snapshot_stale",
        "latest_wins_reconciliation_not_clean",
    ]


@pytest.mark.parametrize("proposed_weight_pct", [0.5, 1.0])
def test_proposed_weight_pct_remains_in_percentage_points(proposed_weight_pct):
    model = build_judge_view_model(
        {
            "investment_cases": [
                {
                    "ticker": "AAA",
                    "proposal": {"proposed_weight_pct": proposed_weight_pct},
                }
            ]
        }
    )

    assert model["decisions"]["items"][0]["proposed_weight_pct"] == pytest.approx(
        proposed_weight_pct
    )


def test_fully_voted_count_requires_two_submissions_in_both_rounds():
    model = build_judge_view_model(
        {
            "investment_cases": [
                {
                    "ticker": "AAA",
                    "pre_vote": {"revealed": True, "submitted_count": 1},
                    "post_vote": {"revealed": True, "submitted_count": 2},
                    "final_approval_complete": True,
                }
            ]
        }
    )

    assert model["decisions"]["fully_voted_count"] == 0


def test_holding_coverage_and_dossier_governance_are_explicit():
    model = build_judge_view_model(
        {
            "pipeline": {
                "authority": "wins_reconciled",
                "canonical_snapshot": {
                    "snapshot_id": "wins-coverage",
                    "payload": {
                        "positions": [
                            {"ticker": "AAA", "market_value": 200_000},
                            {"ticker": "BBB", "market_value": 150_000},
                            {"ticker": "CCC", "market_value": 150_000},
                        ],
                        "total_value": 500_000,
                    },
                },
            },
            "theses": [
                {
                    "ticker": "AAA",
                    "status": "active",
                    "canonical_dossier_status": "frozen",
                    "canonical_dossier_version": 3,
                    "source_system": "canonical_security_dossier",
                    "content_hash": "abc123",
                    "payload": {"investment_thesis": "Governed case."},
                }
            ],
            "approved_securities": [
                {
                    "ticker": "AAA",
                    "approved": True,
                    "eligibility": "eligible",
                    "provenance_status": "verified",
                    "payload": {"source_name": "Wharton universe"},
                },
                {
                    "ticker": "BBB",
                    "approved": False,
                    "eligibility": "ineligible",
                    "provenance_status": "verified",
                },
            ],
        }
    )

    dossier = model["dossiers"][0]
    assert dossier["dossier_status"] == "frozen"
    assert dossier["dossier_version"] == 3
    assert dossier["content_hash_present"] is True
    assert dossier["eligibility"] == "eligible"
    assert dossier["universe_provenance"] == "verified"

    coverage = model["integrity"]["holding_coverage"]
    assert coverage["dossier_coverage_pct"] == pytest.approx(100 / 3)
    assert coverage["eligible_universe_coverage_pct"] == pytest.approx(100 / 3)
    assert coverage["holdings_without_dossier"] == ["BBB", "CCC"]
    assert coverage["holdings_without_universe_record"] == ["CCC"]
    assert coverage["holdings_not_eligible"] == ["BBB"]
    assert coverage["all_holdings_covered"] is False


def test_missing_report_validation_and_reconciliation_remain_unknown():
    model = build_judge_view_model(
        {
            "sources": [
                {
                    "title": "Issuer filing",
                    "primary_source": True,
                    "as_of": "2026-08-22T10:00:00Z",
                }
            ],
            "thesis_reviews": [{"created_at": "2026-08-23T10:00:00Z"}],
        }
    )

    assert model["report"]["validation_ready"] is None
    assert model["integrity"]["reconciliation"]["available"] is False
    assert model["integrity"]["reconciliation"]["open_exception_count"] is None
    assert model["evidence"]["as_of"] == "2026-08-23T10:00:00Z"


def test_explicit_pipeline_reconciliation_exception_count_is_preserved():
    model = build_judge_view_model(
        {
            "reconciliation_record": {
                "payload": {
                    "status": "blocked",
                    "reconciliation_id": "recon-4",
                    "open_exception_count": 2,
                }
            }
        }
    )

    reconciliation = model["integrity"]["reconciliation"]
    assert reconciliation["available"] is True
    assert reconciliation["open_exception_count"] == 2


def test_empty_input_returns_stable_empty_sections_and_default_questions():
    model = build_judge_view_model(None)

    assert set(model) == {
        "mandate",
        "strategy",
        "readiness",
        "portfolio",
        "dossiers",
        "decisions",
        "evidence",
        "report",
        "compliance",
        "integrity",
        "app_record",
        "questions",
    }
    assert model["portfolio"]["available"] is False
    assert model["portfolio"]["source_tier"] == "none"
    assert model["portfolio"]["positions"] == []
    assert model["dossiers"] == []
    assert model["decisions"]["items"] == []
    assert model["report"]["available"] is False
    assert model["compliance"]["check_count"] == 0
    assert model["integrity"]["reporting_binding_allowed"] is None
    assert model["integrity"]["report_snapshot_matches_pipeline"] is None
    assert model["app_record"]["module_count"] == 11
    assert len(model["app_record"]["modules"]) == 11
    assert len(model["questions"]) == 3


def test_model_aggregates_judge_evidence_without_exposing_ballot_details():
    model = build_judge_view_model(
        {
            "mandate_record": {
                "payload": {
                    "client_name": "Case Family",
                    "goals": [{"name": "Education", "target_weight": 0.4}],
                    "risk_tolerance": "Moderate",
                    "policy_benchmark": "60/40",
                }
            },
            "strategy_record": {
                "payload": {
                    "name": "Goals first",
                    "thesis": "Match durable cash flows to client goals.",
                    "max_position_weight": 0.12,
                }
            },
            "theses": [
                {
                    "ticker": "AAA",
                    "payload": {
                        "investment_thesis": "Recurring revenue compounds.",
                        "primary_goal": "Education",
                        "invalidation_condition": "Retention below 85%.",
                        "fair_value_bear": 80,
                        "fair_value_base": 110,
                        "fair_value_bull": 140,
                    },
                }
            ],
            "readiness": {
                "overall_score": 78,
                "status": "Evidence build",
                "constitution": {"score": 88},
                "dossier_score": 75,
                "governance": {"score": 65},
                "dossiers": [
                    {
                        "ticker": "AAA",
                        "score": 75,
                        "status": "Developing",
                        "missing": [{"key": "source", "label": "Primary source"}],
                        "source_count": 2,
                        "primary_source_count": 1,
                        "catalyst_count": 1,
                    }
                ],
                "operating_gates": {"investment_committee": True},
            },
            "investment_cases": [
                {
                    "id": 41,
                    "ticker": "AAA",
                    "state": "sizing",
                    "pre_vote": {
                        "status": "closed",
                        "revealed": True,
                        "eligible_count": 4,
                        "submitted_count": 4,
                        "outcome": "buy",
                        "ballots": [
                            {"member_id": "secret-member", "rationale": "secret-rationale"}
                        ],
                    },
                    "post_vote": {
                        "status": "closed",
                        "revealed": True,
                        "eligible_count": 4,
                        "submitted_count": 4,
                        "outcome": "buy",
                        "dissent": [{"strongest_objection": "secret-objection"}],
                    },
                    "final_approval_complete": True,
                    "final_approvals": [
                        {"member_id": "one", "decision": "approve"},
                        {"member_id": "two", "decision": "approve"},
                    ],
                    "audit": {"valid": True},
                    "audit_events": [
                        {"event_type": "vote_closed", "payload": {"secret": "secret-event"}}
                    ],
                }
            ],
            "sources": [
                {
                    "ticker": "AAA",
                    "title": "Annual report",
                    "primary_source": True,
                    "citation": "Issuer, 2026",
                }
            ],
            "catalysts": [{"ticker": "AAA"}],
            "red_team_reviews": [{"ticker": "AAA"}],
            "ai_usage": [{"tool": "assistant"}],
            "qa_rounds": [{"score": 80}],
            "approved_securities": [{"ticker": "AAA"}],
            "report_record": {
                "payload": {
                    "report_id": "final-1",
                    "title": "Final report",
                    "status": "frozen",
                    "sections": {"strategy": {"status": "ready"}},
                    "freeze": {"content_hash": "abc"},
                }
            },
            "report_validation": {
                "is_ready": False,
                "issue_count": 1,
                "issues": [{"code": "snapshot", "message": "Snapshot missing."}],
            },
            "compliance_checks": [
                {"status": "pass", "rule": "Team size", "detail": "Confirmed."},
                {"status": "pending", "rule": "New limits", "detail": "Awaiting rules."},
            ],
            "questions": ["How is the downside case funded?"],
        }
    )

    assert model["mandate"]["client_name"] == "Case Family"
    assert model["strategy"]["max_position_weight"] == pytest.approx(0.12)
    assert model["dossiers"][0]["thesis"] == "Recurring revenue compounds."
    assert model["dossiers"][0]["fair_values"] == {
        "bear": 80.0,
        "base": 110.0,
        "bull": 140.0,
    }
    decision = model["decisions"]["items"][0]
    assert decision["pre_vote"]["submitted_count"] == 4
    assert decision["pre_vote"]["result"] == "buy"
    assert model["decisions"]["fully_voted_count"] == 1
    assert model["evidence"]["primary_source_count"] == 1
    assert model["report"]["frozen"] is True
    assert model["compliance"]["pass_count"] == 1
    assert model["compliance"]["pending_count"] == 1
    assert model["compliance"]["all_clear"] is False
    assert "How is the downside case funded?" in model["questions"]

    serialised = json.dumps(model)
    assert "ballot" not in serialised.casefold()
    assert "secret-member" not in serialised
    assert "secret-rationale" not in serialised
    assert "secret-objection" not in serialised
    assert "secret-event" not in serialised


def test_complete_app_record_mirrors_every_workspace_and_removes_private_fields():
    model = build_judge_view_model(
        {
            "app_record": {
                "collaboration": {
                    "tasks": [{"id": 1, "task_text": "Review case", "password": "nope"}],
                    "files": [
                        {
                            "filename": "case.pdf",
                            "file_path": "/private/server/path/case.pdf",
                        }
                    ],
                },
                "strategy": {
                    "versions": [
                        {
                            "version": 1,
                            "is_active": True,
                            "payload": {"name": "Client-first strategy"},
                        }
                    ]
                },
                "research": {
                    "macro_snapshots": [
                        {
                            "economy_code": "USA",
                            "payload": {"reference_year": 2024},
                        }
                    ]
                },
                "governance": {
                    "decision_log": [
                        {
                            "ticker": "AAA",
                            "action": "Buy",
                            "quant_snapshot": {"score": 82},
                            "ballots": [{"member_id": "private-voter"}],
                        }
                    ]
                },
                "operations": {
                    "qa_workspace": {
                        "record_type": "qa_workspace",
                        "payload": {
                            "questions": [{"prompt": "Defend the downside."}],
                            "secret_note": "private-note",
                        },
                    }
                },
            }
        }
    )

    app_record = model["app_record"]
    assert [item["label"] for item in app_record["modules"]] == [
        "Overview & Tasks",
        "Competition Readiness",
        "Mandate & Strategy",
        "Research Workspace",
        "Security Dossiers",
        "Investment Committee",
        "Portfolio Overview",
        "WInS & Reconciliation",
        "Risk & Scenarios",
        "Report & Pitch",
        "Rules & Compliance",
    ]
    assert app_record["available_module_count"] >= 5
    serialised = json.dumps(app_record)
    assert "Client-first strategy" in serialised
    assert "Defend the downside" in serialised
    assert "private-voter" not in serialised
    assert "private-note" not in serialised
    assert "/private/server/path" not in serialised
    assert "nope" not in serialised


def test_empty_app_record_does_not_count_derived_shells_as_saved_data():
    model = build_judge_view_model(None)

    assert model["app_record"]["available_module_count"] == 0
    assert all(
        module["status"] == "Not used yet"
        for module in model["app_record"]["modules"]
    )


@pytest.mark.parametrize("weight_pct", [0.5, 1.0])
def test_small_explicit_percentage_point_weights_stay_percentage_points(weight_pct):
    model = build_judge_view_model(
        {
            "tracker_performance": {
                "equity": 100_000,
                "positions": [
                    {
                        "ticker": "AAA",
                        "weight_pct": weight_pct,
                        "current_value": 1_000,
                    }
                ],
            }
        }
    )

    position = model["portfolio"]["positions"][0]
    assert position["weight"] == pytest.approx(weight_pct / 100.0)
    assert position["weight_pct"] == pytest.approx(weight_pct)


def test_all_cash_tracker_is_available_as_a_provisional_portfolio():
    model = build_judge_view_model(
        {
            "tracker_performance": {
                "initial_capital": 100_000,
                "equity": 100_000,
                "cash_before_pnl": 100_000,
                "positions": [],
            }
        }
    )

    assert model["portfolio"]["available"] is True
    assert model["portfolio"]["source_tier"] == "tracker_provisional"
    assert model["portfolio"]["cash"] == pytest.approx(100_000)
    assert model["portfolio"]["equity"] == pytest.approx(100_000)
    assert model["portfolio"]["positions"] == []


def test_report_projection_keeps_judge_safe_performance_attribution():
    model = build_judge_view_model(
        {
            "report_record": {
                "payload": {
                    "status": "frozen",
                    "performance_attribution": {
                        "as_of": "2026-08-24",
                        "benchmark": "ACWI",
                        "portfolio_return": 0.08,
                        "benchmark_return": 0.05,
                        "active_return": 0.03,
                        "attributed_return": 0.075,
                        "residual": 0.005,
                        "methodology": "Ledger-linked contribution analysis.",
                        "contributions": [
                            {"id": "selection", "label": "Selection", "contribution": 0.04}
                        ],
                    },
                }
            }
        }
    )

    attribution = model["report"]["performance_attribution"]
    assert attribution["available"] is True
    assert attribution["portfolio_return_pct"] == pytest.approx(8.0)
    assert attribution["benchmark_return_pct"] == pytest.approx(5.0)
    assert attribution["active_return_pct"] == pytest.approx(3.0)
    assert attribution["contributions"] == [
        {"label": "Selection", "contribution_pct": pytest.approx(4.0)}
    ]
