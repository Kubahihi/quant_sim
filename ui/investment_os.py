"""Focused Streamlit surfaces for the competition investment operating system.

The large analytical dashboard stays responsible for analytics.  This module
owns the smaller, stateful operating workflows whose value comes from explicit
transitions, immutable snapshots, and independent sign-off.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from io import BytesIO
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import streamlit as st


ConnectionFactory = Callable[[], Any]


def _actor(profile: Mapping[str, Any]) -> str:
    return str(profile.get("username") or "").strip()


def _team_names(team_members: Sequence[Any]) -> list[str]:
    names = [
        str(item.get("username") if isinstance(item, Mapping) else item).strip()
        for item in team_members
    ]
    return [name for name in names if name]


def _lines(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n;]+", value or "") if item.strip()]


def _current_document(
    get_connection: ConnectionFactory,
    record_type: str,
    record_id: str,
    default: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    from src.portfolio_tracker.operating_system_store import get_current_record

    with get_connection() as connection:
        record = get_current_record(connection, record_type, record_id)
    if not record:
        return dict(default or {}), 0
    return dict(record["payload"]), int(record["version"])


def _save_document(
    get_connection: ConnectionFactory,
    record_type: str,
    record_id: str,
    payload: Mapping[str, Any],
    *,
    actor: str,
    status: str = "",
    expected_version: int | None = None,
) -> dict[str, Any]:
    from src.portfolio_tracker.operating_system_store import save_record

    with get_connection() as connection:
        return save_record(
            connection,
            record_type,
            record_id,
            payload,
            actor=actor,
            status=status,
            expected_version=expected_version,
        )


def _error_or_rerun(action: Callable[[], Any], success: str) -> None:
    try:
        action()
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        st.error(str(exc))
    else:
        st.success(success)
        st.rerun()


def _call_db(get_connection: ConnectionFactory, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with get_connection() as connection:
        return function(connection, *args, **kwargs)


def render_security_dossiers(
    profile: Mapping[str, Any],
    get_connection: ConnectionFactory,
    team_members: Sequence[Any],
) -> None:
    """Render the canonical thesis, KPI monitor, and universe authority."""
    from src.portfolio_tracker.authoritative_universe_store import (
        get_active_authoritative_universe,
        list_authoritative_universe_snapshots,
        publish_authoritative_universe,
    )
    from src.portfolio_tracker.security_dossier_store import (
        append_dossier_version,
        append_kpi_observation,
        create_security_dossier,
        freeze_dossier,
        get_kpi_monitor,
        list_kpi_definitions,
        list_security_dossiers,
        upsert_kpi_definition,
    )

    actor = _actor(profile)
    members = _team_names(team_members)
    st.markdown("### Security Dossiers")
    st.caption(
        "One canonical record now owns the thesis, catalysts, invalidation, portfolio role, "
        "sell discipline, evidence links, and monitored KPIs. Committee proposals freeze a version of it."
    )
    dossier_tab, kpi_tab, universe_tab = st.tabs(
        ["Canonical dossier", "KPI thesis monitor", "Authoritative universe"]
    )

    with get_connection() as connection:
        dossiers = list_security_dossiers(connection)

    with dossier_tab:
        with st.expander("Create security dossier", expanded=not dossiers):
            with st.form("ios_create_dossier", clear_on_submit=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    ticker = st.text_input("Ticker / security ID").strip().upper()
                    client_goal = st.text_input("Client goal")
                with c2:
                    portfolio_role = st.text_input("Portfolio role")
                    owner = st.selectbox("Dossier owner", members, index=members.index(actor) if actor in members else 0)
                with c3:
                    candidate_source = st.text_input("Candidate source", placeholder="Screener, analyst, task")
                    asset_type = st.selectbox("Instrument type", ["Stock", "ETF", "Bond", "Commodity", "Other"])
                thesis = st.text_area("Investment thesis", height=110)
                catalysts = st.text_area("Catalysts / dated thesis tests", height=80)
                invalidation = st.text_area("Observable invalidation condition", height=75)
                sell_discipline = st.text_area("Sell / review discipline", height=75)
                d1, d2 = st.columns(2)
                with d1:
                    bull_case = st.text_area("Bull case", height=65)
                    risks = st.text_area("Key risks", height=65)
                with d2:
                    bear_case = st.text_area("Bear case", height=65)
                    evidence_refs = st.text_area("Evidence IDs / citations", height=65)
                create_clicked = st.form_submit_button("Create canonical dossier", type="primary", use_container_width=True)
            if create_clicked:
                payload = {
                    "thesis": thesis.strip(),
                    "catalysts": _lines(catalysts),
                    "invalidation_condition": invalidation.strip(),
                    "portfolio_role": portfolio_role.strip(),
                    "sell_discipline": sell_discipline.strip(),
                    "client_goal": client_goal.strip(),
                    "asset_type": asset_type,
                    "owner": owner,
                    "bull_case": bull_case.strip(),
                    "bear_case": bear_case.strip(),
                    "risks": _lines(risks),
                    "evidence_refs": _lines(evidence_refs),
                }
                _error_or_rerun(
                    lambda: (
                        (_ for _ in ()).throw(ValueError("Ticker and all five core thesis fields are required."))
                        if not ticker or any(
                            not payload[key]
                            for key in ("thesis", "catalysts", "invalidation_condition", "portfolio_role", "sell_discipline")
                        )
                        else _call_db(
                            get_connection,
                            create_security_dossier,
                            ticker,
                            payload,
                            candidate={"source": candidate_source.strip(), "asset_type": asset_type},
                            created_by=actor,
                        )
                    ),
                    f"Canonical dossier {ticker} created.",
                )

        if not dossiers:
            st.info("Create the first dossier; thesis data should no longer be copied into position and bond forms.")
        else:
            selected_id = st.selectbox(
                "Dossier",
                [int(item["id"]) for item in dossiers],
                format_func=lambda value: next(
                    f"{item['ticker']} · v{item['current_version']['version']} · {item['current_version']['status']}"
                    for item in dossiers if int(item["id"]) == value
                ),
            )
            selected = next(item for item in dossiers if int(item["id"]) == selected_id)
            version = selected["current_version"]
            payload = version["payload"]
            metrics = st.columns(4)
            metrics[0].metric("Ticker", selected["ticker"])
            metrics[1].metric("Version", version["version"])
            metrics[2].metric("State", version["status"].title())
            metrics[3].metric("KPI definitions", len(version.get("kpi_snapshot") or []))
            st.dataframe(
                pd.DataFrame([
                    {"Field": key.replace("_", " ").title(), "Canonical value": value}
                    for key, value in payload.items()
                ]),
                hide_index=True,
                use_container_width=True,
            )
            with st.expander("Append a revised dossier version"):
                with st.form(f"ios_revise_dossier_{selected_id}"):
                    revised_thesis = st.text_area("Thesis", value=str(payload.get("thesis") or ""))
                    revised_catalysts = st.text_area("Catalysts", value="\n".join(payload.get("catalysts") or []))
                    revised_invalidation = st.text_area("Invalidation", value=str(payload.get("invalidation_condition") or ""))
                    revised_role = st.text_input("Portfolio role", value=str(payload.get("portfolio_role") or ""))
                    revised_sell = st.text_area("Sell discipline", value=str(payload.get("sell_discipline") or ""))
                    revision_reason = st.text_input("Revision reason")
                    revise_clicked = st.form_submit_button("Append revision", use_container_width=True)
                if revise_clicked:
                    revised = {
                        **payload,
                        "thesis": revised_thesis.strip(),
                        "catalysts": _lines(revised_catalysts),
                        "invalidation_condition": revised_invalidation.strip(),
                        "portfolio_role": revised_role.strip(),
                        "sell_discipline": revised_sell.strip(),
                        "revision_reason": revision_reason.strip(),
                    }
                    _error_or_rerun(
                        lambda: _call_db(
                            get_connection,
                            append_dossier_version,
                            selected_id,
                            revised,
                            created_by=actor,
                            expected_current_version=int(version["version"]),
                        ),
                        "A new dossier version was appended.",
                    )
            if version["status"] == "draft":
                if st.button("Freeze dossier for committee", type="primary", use_container_width=True):
                    _error_or_rerun(
                        lambda: _call_db(get_connection, freeze_dossier, selected_id, frozen_by=actor),
                        "Dossier frozen. Committee proposals can now reference its immutable hash.",
                    )

    with kpi_tab:
        if not dossiers:
            st.info("Create a security dossier first.")
        else:
            kpi_dossier_id = st.selectbox(
                "Dossier for KPI monitor",
                [int(item["id"]) for item in dossiers],
                key="ios_kpi_dossier",
                format_func=lambda value: next(item["ticker"] for item in dossiers if int(item["id"]) == value),
            )
            with get_connection() as connection:
                definitions = list_kpi_definitions(connection, kpi_dossier_id)
                monitor = get_kpi_monitor(connection, kpi_dossier_id)
            summary = st.columns(5)
            for column, key in zip(summary, ["on_track", "watch", "breach", "missing", "stale"]):
                column.metric(key.replace("_", " ").title(), monitor["counts"].get(key, 0))
            if monitor["items"]:
                st.dataframe(pd.DataFrame([
                    {
                        "KPI": item["definition"]["name"],
                        "Owner": item["definition"]["owner"],
                        "Frequency": item["definition"]["frequency"],
                        "Latest": (item.get("latest_observation") or {}).get("observed_value"),
                        "Health": item["health_status"],
                        "Stale": item["is_stale"],
                        "Source": item["definition"]["source"],
                    }
                    for item in monitor["items"]
                ]), hide_index=True, use_container_width=True)
            with st.expander("Define or revise KPI", expanded=not definitions):
                with st.form("ios_kpi_definition", clear_on_submit=True):
                    k1, k2, k3 = st.columns(3)
                    with k1:
                        kpi_key = st.text_input("KPI key", placeholder="retention_rate")
                        kpi_name = st.text_input("Display name")
                        unit = st.text_input("Unit", value="%")
                    with k2:
                        baseline = st.number_input("Baseline", value=0.0)
                        expected_min = st.number_input("Expected minimum", value=0.0)
                        expected_max = st.number_input("Expected maximum", value=100.0)
                    with k3:
                        breach_below = st.number_input("Breach at or below", value=0.0)
                        breach_above = st.number_input("Breach at or above", value=100.0)
                        frequency = st.selectbox("Frequency", ["daily", "weekly", "monthly", "quarterly", "annual", "event_driven"])
                    source = st.text_input("Primary source / evidence reference")
                    owner = st.selectbox("KPI owner", members, index=members.index(actor) if actor in members else 0)
                    define_clicked = st.form_submit_button("Save KPI definition", use_container_width=True)
                if define_clicked:
                    _error_or_rerun(
                        lambda: _call_db(
                            get_connection, upsert_kpi_definition, kpi_dossier_id, kpi_key,
                            name=kpi_name, baseline=baseline,
                            expected_min=expected_min, expected_max=expected_max,
                            breach_below=breach_below, breach_above=breach_above,
                            unit=unit, source=source, frequency=frequency, owner=owner,
                            updated_by=actor,
                        ),
                        "KPI definition saved as an immutable revision.",
                    )
            if definitions:
                with st.form("ios_kpi_observation", clear_on_submit=True):
                    selected_kpi = st.selectbox("KPI", [item["kpi_key"] for item in definitions])
                    observed_value = st.number_input("Observed value", value=0.0)
                    observed_at = st.date_input("Observed on", value=date.today())
                    source_ref = st.text_input("Observation source")
                    observe_clicked = st.form_submit_button("Append KPI observation", type="primary", use_container_width=True)
                if observe_clicked:
                    _error_or_rerun(
                        lambda: _call_db(
                            get_connection, append_kpi_observation, kpi_dossier_id, selected_kpi, observed_value,
                            observed_at=observed_at, source_ref=source_ref or None, recorded_by=actor,
                        ),
                        "KPI observation appended.",
                    )

    with universe_tab:
        with get_connection() as connection:
            active_universe = get_active_authoritative_universe(connection)
            universe_history = list_authoritative_universe_snapshots(connection)
        if active_universe:
            badges = st.columns(4)
            badges[0].metric("Version", active_universe["version"])
            badges[1].metric("Entries", active_universe["entry_count"])
            badges[2].metric("Authority", active_universe["provenance_status"].replace("_", " ").title())
            badges[3].metric("As of", active_universe["as_of_date"])
            st.dataframe(pd.DataFrame(active_universe["entries"]), hide_index=True, use_container_width=True)
        else:
            st.warning("No authoritative universe snapshot exists. Trading must fail closed.")
        with st.expander("Publish complete universe snapshot", expanded=active_universe is None):
            with st.form("ios_publish_universe"):
                raw_entries = st.text_area(
                    "One security per line: TICKER, eligibility, type",
                    placeholder="MSFT, eligible, Stock\nXYZ, ineligible, Stock",
                    height=180,
                )
                u1, u2, u3 = st.columns(3)
                with u1:
                    source_name = st.text_input("Source name")
                    source_url = st.text_input("Direct source URL")
                with u2:
                    provenance = st.selectbox("Snapshot authority", ["official", "analyst_assumption", "outdated", "not_checked"])
                    as_of = st.date_input("As-of date", value=date.today())
                with u3:
                    confirmation = st.checkbox("Replace the active snapshot in full")
                    st.caption("Every publication is immutable and retains the prior active version.")
                publish_clicked = st.form_submit_button("Publish and activate snapshot", type="primary", use_container_width=True)
            if publish_clicked:
                entries = []
                for line in _lines(raw_entries):
                    parts = [part.strip() for part in line.split(",")]
                    if parts and parts[0]:
                        entries.append({
                            "ticker": parts[0],
                            "eligibility": parts[1].lower() if len(parts) > 1 else "unknown",
                            "security_type": parts[2] if len(parts) > 2 else "",
                            "provenance_status": provenance,
                        })
                def publish() -> Any:
                    if not confirmation:
                        raise ValueError("Confirm that this is a complete replacement snapshot.")
                    if provenance == "official" and not source_url.strip():
                        raise ValueError("Official status requires a direct official source URL.")
                    return _call_db(
                        get_connection, publish_authoritative_universe, entries,
                        source_name=source_name, source_url=source_url,
                        provenance_status=provenance, as_of_date=as_of, published_by=actor,
                        expected_active_snapshot_id=(int(active_universe["id"]) if active_universe else None),
                    )
                _error_or_rerun(publish, "A new authoritative universe snapshot is active.")
        if universe_history:
            st.markdown("#### Immutable publication history")
            st.dataframe(pd.DataFrame(universe_history), hide_index=True, use_container_width=True)


def _committee_roster(team_members: Sequence[Any]) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    """Build a stable four-person roster with two investment and two advisory votes."""
    raw_members = [dict(item) if isinstance(item, Mapping) else {"username": str(item)} for item in team_members]
    if len(raw_members) < 2:
        raise ValueError("Investment Committee requires at least two team members.")
    identities = {
        str(item.get("username") or "").strip(): (
            "member-" + hashlib.sha256(
                str(item.get("username") or "").strip().casefold().encode("utf-8")
            ).hexdigest()[:12]
        )
        for item in raw_members
        if str(item.get("username") or "").strip()
    }
    captains = [
        identities[str(item.get("username") or "").strip()]
        for item in raw_members
        if "captain" in str(item.get("role") or "").lower()
        and str(item.get("username") or "").strip() in identities
    ]
    if len(captains) < 2:
        captains = list(identities.values())[:2]
    captains = captains[:2]
    non_captains = [
        item for item in raw_members
        if identities.get(str(item.get("username") or "").strip()) not in captains
    ]

    def advisory_priority(item: Mapping[str, Any]) -> tuple[int, str]:
        role = str(item.get("role") or "").casefold()
        if "geo" in role or "communication" in role:
            priority = 0
        elif "risk" in role or "logistic" in role or "client" in role:
            priority = 1
        else:
            priority = 2
        return priority, str(item.get("username") or "").casefold()

    sorted_advisers = sorted(non_captains, key=advisory_priority)
    advisory_role_by_id = {
        identities[str(item.get("username") or "").strip()]: role
        for item, role in zip(
            sorted_advisers,
            ("clarity_reviewer", "client_fit_reviewer"),
        )
    }
    roster: list[dict[str, Any]] = []
    for item in raw_members:
        name = str(item.get("username") or "").strip()
        if not name:
            continue
        member_id = identities[name]
        if member_id in captains:
            roster.append({
                "member_id": member_id,
                "name": name,
                "role": "member",
                "vote_scope": "investment",
            })
        else:
            role = advisory_role_by_id.get(member_id, "observer")
            roster.append({
                "member_id": member_id,
                "name": name,
                "role": role,
                "vote_scope": "advisory" if role != "observer" else "observer",
            })
    return roster, identities, captains


def _lifecycle_execution(lifecycle: Mapping[str, Any]) -> dict[str, Any]:
    execution_record = lifecycle.get("wins_execution") or {}
    execution = dict(execution_record.get("execution") or {})
    if not execution:
        raise ValueError("A WInS execution must exist before reconciliation.")
    currency = str(execution.get("currency") or "").strip().upper()
    if not currency:
        raise ValueError("Execution currency is required; the tracker will not assume USD.")
    execution["currency"] = currency
    return execution


def _tracker_projection_rows(connection: Any, lifecycle_id: int) -> list[Any]:
    return connection.execute(
        """
        SELECT id, ticker, quantity, entry_price, currency, status
        FROM competition_positions
        WHERE lifecycle_id = ?
        ORDER BY id
        """,
        (int(lifecycle_id),),
    ).fetchall()


def _validate_tracker_projection(
    row: Any,
    lifecycle: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
    ticker = str(lifecycle.get("ticker") or "").strip().upper()
    currency = str(execution.get("currency") or "").strip().upper()
    if str(row[1] or "").strip().upper() != ticker:
        raise ValueError("The staged tracker ticker does not match the investment lifecycle.")
    if abs(float(row[2]) - float(execution["quantity"])) > 1e-8:
        raise ValueError("The staged tracker quantity does not match the WInS execution.")
    if abs(float(row[3]) - float(execution["average_price"])) > 0.01:
        raise ValueError("The staged tracker entry price does not match the WInS execution.")
    if str(row[4] or "").strip().upper() != currency:
        raise ValueError("The staged tracker currency does not match the WInS execution.")


def _stage_pending_tracker_position(
    connection: Any,
    lifecycle: Mapping[str, Any],
    *,
    actor: str,
) -> dict[str, Any]:
    """Idempotently stage an executed trade for full-portfolio reconciliation."""
    from src.portfolio_tracker.security_dossier_store import get_dossier_version

    lifecycle_id = int(lifecycle["id"])
    execution = _lifecycle_execution(lifecycle)
    if str(execution.get("side") or "buy").strip().lower() != "buy":
        raise ValueError("Pending tracker staging supports buy executions; exits use the lifecycle exit flow.")
    existing = _tracker_projection_rows(connection, lifecycle_id)
    if len(existing) > 1:
        raise ValueError("More than one tracker projection exists for this lifecycle.")
    if existing:
        row = existing[0]
        _validate_tracker_projection(row, lifecycle, execution)
        status = str(row[5] or "").strip().lower()
        if status not in {"pending_reconciliation", "open"}:
            raise ValueError(f"The existing tracker projection has unexpected status {status!r}.")
        return {"id": int(row[0]), "status": status, "created": False}

    dossier = get_dossier_version(
        connection, int(lifecycle["dossier_id"]), int(lifecycle["dossier_version"])
    )
    dossier_payload = dict((dossier or {}).get("payload") or {})
    security_type = str(
        dossier_payload.get("asset_type")
        or dossier_payload.get("security_type")
        or lifecycle.get("proposal", {}).get("security_type")
        or "Other"
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    executed_at = str(execution["executed_at"])
    cursor = connection.execute(
        """
        INSERT INTO competition_positions (
            ticker, security_type, quantity, entry_price, entry_date,
            opened_by, opened_at, last_price, notes, status, currency,
            lifecycle_id, competition_eligibility_status,
            eligibility_source, eligibility_checked_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending_reconciliation', ?, ?,
                  'Verified eligible', ?, ?)
        """,
        (
            str(lifecycle["ticker"]), security_type, float(execution["quantity"]),
            float(execution["average_price"]), executed_at[:10], actor, timestamp,
            float(execution["average_price"]),
            (
                f"Pending canonical reconciliation; lifecycle #{lifecycle_id}; "
                f"WInS {execution['wins_transaction_id']}"
            ),
            str(execution["currency"]), lifecycle_id,
            f"Authoritative universe snapshot {lifecycle['universe_snapshot_id']}",
            timestamp,
        ),
    )
    inserted = getattr(cursor, "lastrowid", None)
    if inserted is None:
        inserted_row = connection.execute(
            "SELECT id FROM competition_positions WHERE lifecycle_id = ? ORDER BY id DESC LIMIT 1",
            (lifecycle_id,),
        ).fetchone()
        if inserted_row is None:
            raise RuntimeError("The pending tracker projection could not be read after insertion.")
        inserted = inserted_row[0]
    connection.commit()
    sync = getattr(connection, "sync", None)
    if callable(sync):
        sync()
    return {"id": int(inserted), "status": "pending_reconciliation", "created": True}


def _record_wins_execution_and_stage_tracker_position(
    connection: Any,
    lifecycle_id: int,
    execution: Mapping[str, Any],
    *,
    actor: str,
) -> dict[str, Any]:
    """Record an execution, then create its idempotent pending tracker projection."""
    from src.portfolio_tracker.investment_lifecycle_store import record_wins_execution

    if str(execution.get("side") or "").strip().lower() != "buy":
        raise ValueError(
            "The staged Investment Committee execution path accepts only buy executions."
        )
    try:
        lifecycle = record_wins_execution(
            connection,
            lifecycle_id,
            execution,
            recorded_by=actor,
            commit=False,
        )
        _stage_pending_tracker_position(connection, lifecycle, actor=actor)
        return lifecycle
    except Exception:
        connection.rollback()
        raise


def _activate_reconciled_tracker_position(
    connection: Any,
    lifecycle: Mapping[str, Any],
    *,
    actor: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Delegate atomic activation to the canonical lifecycle domain guard."""
    from src.portfolio_tracker.investment_lifecycle_store import record_wins_reconciliation

    return record_wins_reconciliation(
        connection,
        int(lifecycle["id"]),
        recorded_by=actor,
        now=now,
    )


def render_investment_committee(
    profile: Mapping[str, Any],
    get_connection: ConnectionFactory,
    strategy_data: Mapping[str, Any],
    team_members: Sequence[Any],
) -> None:
    """Render the locked two-round Investment Committee operating workflow."""
    from src.portfolio_tracker.authoritative_universe_store import (
        get_active_authoritative_universe,
    )
    from src.portfolio_tracker.investment_lifecycle_store import (
        append_position_review,
        close_vote_round,
        create_investment_proposal,
        get_investment_lifecycle,
        has_member_submitted_vote,
        list_lifecycle_audit_events,
        list_investment_lifecycles,
        lock_proposal_dossier,
        open_pre_vote,
        record_committee_discussion,
        record_position_exit,
        record_position_sizing,
        record_rule_check,
        submit_committee_vote,
        submit_final_approval,
        update_committee_member_status,
    )
    from src.portfolio_tracker.security_dossier_store import list_security_dossiers

    actor = _actor(profile)
    roster, identities, captain_ids = _committee_roster(team_members)
    actor_id = identities.get(actor, "")
    name_by_id = {item["member_id"]: item["name"] for item in roster}
    mandate_record = dict(strategy_data.get("mandate_record") or {})
    rulebook_record = dict(strategy_data.get("strategy_record") or {})

    with get_connection() as connection:
        dossiers = list_security_dossiers(connection)
        universe = get_active_authoritative_universe(connection)
        lifecycles = list_investment_lifecycles(connection)

    st.markdown("### Investment Committee")
    st.caption(
        "Proposal → frozen dossier → blind pre-vote → bull/bear discussion → blind post-vote → "
        "rule check → two-person approval → sizing → WInS execution → reconciliation. "
        "No ballot is revealed until every eligible member has submitted."
    )
    create_tab, workflow_tab, audit_tab = st.tabs(["New proposal", "Committee workflow", "Audit trail"])

    with create_tab:
        frozen = [item for item in dossiers if item.get("latest_frozen_version")]
        if universe is None:
            st.warning("Publish an official authoritative universe before creating a proposal.")
        if not frozen:
            st.warning("Freeze a complete Security Dossier with at least one KPI before creating a proposal.")
        if universe and frozen:
            with st.form("ios_create_investment_proposal", clear_on_submit=True):
                p1, p2, p3 = st.columns(3)
                with p1:
                    dossier_id = st.selectbox(
                        "Frozen dossier",
                        [int(item["id"]) for item in frozen],
                        format_func=lambda value: next(
                            f"{item['ticker']} · v{item['latest_frozen_version']['version']}"
                            for item in frozen if int(item["id"]) == value
                        ),
                    )
                    action = st.selectbox(
                        "Executable proposal action", ["buy"],
                        help="WATCH and REJECT are committee ballot outcomes, not executable proposals.",
                    )
                with p2:
                    owner_id = st.selectbox(
                        "Proposal owner", captain_ids,
                        format_func=lambda value: name_by_id[value],
                    )
                    challenger_id = st.selectbox(
                        "Independent challenger", captain_ids,
                        index=1 if len(captain_ids) > 1 else 0,
                        format_func=lambda value: name_by_id[value],
                    )
                with p3:
                    proposed_weight = st.number_input(
                        "Initial requested weight (%)", min_value=0.0, max_value=100.0, value=5.0
                    )
                    client_goal = st.text_input("Linked client goal")
                rationale = st.text_area("Proposal rationale and client fit", height=100)
                starter_conditions = st.text_area(
                    "Starter-position conditions (if applicable)", height=70
                )
                create_clicked = st.form_submit_button(
                    "Create locked committee workflow", type="primary", use_container_width=True
                )
            if create_clicked:
                selected_dossier = next(item for item in frozen if int(item["id"]) == dossier_id)
                frozen_version = selected_dossier["latest_frozen_version"]

                def create() -> Any:
                    if owner_id == challenger_id:
                        raise ValueError("Owner and challenger must be different people.")
                    return _call_db(
                        get_connection,
                        create_investment_proposal,
                        security_ticker=selected_dossier["ticker"],
                        dossier_id=dossier_id,
                        dossier_version=int(frozen_version["version"]),
                        universe_snapshot_id=int(universe["id"]),
                        proposal={
                            "action": action,
                            "rationale": rationale.strip(),
                            "proposed_weight_pct": proposed_weight,
                            "client_goal": client_goal.strip(),
                            "starter_position_conditions": starter_conditions.strip(),
                            "security_type": frozen_version.get("payload", {}).get("asset_type", ""),
                        },
                        committee_members=roster,
                        owner_id=owner_id,
                        challenger_id=challenger_id,
                        required_approvers=captain_ids,
                        quorum=2,
                        created_by=actor,
                    )

                _error_or_rerun(create, "Proposal created with its dossier and universe versions locked.")

    with workflow_tab:
        if not lifecycles:
            st.info("Create the first committee proposal.")
        else:
            lifecycle_id = st.selectbox(
                "Investment lifecycle",
                [int(item["id"]) for item in lifecycles],
                format_func=lambda value: next(
                    f"#{item['id']} · {item['ticker']} · {item['state'].replace('_', ' ').title()}"
                    for item in lifecycles if int(item["id"]) == value
                ),
            )
            with get_connection() as connection:
                lifecycle = get_investment_lifecycle(connection, lifecycle_id)
            if lifecycle is None:
                st.error("The selected lifecycle cannot be read.")
                return
            state = str(lifecycle["state"])
            state_order = [
                "proposal", "dossier_frozen", "pre_vote", "discussion", "post_vote",
                "rule_check", "final_approval", "sizing", "wins_execution",
                "reconciliation", "active", "exited",
            ]
            headline = st.columns(5)
            headline[0].metric("Ticker", lifecycle["ticker"])
            headline[1].metric("State", state.replace("_", " ").title())
            headline[2].metric("Dossier", f"v{lifecycle['dossier_version']}")
            headline[3].metric("Universe", f"#{lifecycle['universe_snapshot_id']}")
            headline[4].metric("Audit chain", "Valid" if lifecycle["audit"]["valid"] else "Broken")
            if state in state_order:
                st.progress((state_order.index(state) + 1) / len(state_order))
            st.dataframe(pd.DataFrame(lifecycle["committee"]), hide_index=True, use_container_width=True)

            if state == "proposal":
                with st.expander("Attendance and conflict declaration", expanded=True):
                    status_member = st.selectbox(
                        "Member",
                        [item["member_id"] for item in lifecycle["committee"]],
                        format_func=lambda value: name_by_id.get(value, value),
                        key=f"ios_status_member_{lifecycle_id}",
                    )
                    stored_status = next(
                        item for item in lifecycle["committee"]
                        if item["member_id"] == status_member
                    )
                    with st.form(f"ios_committee_status_{lifecycle_id}"):
                        present = st.checkbox(
                            "Present",
                            value=bool(stored_status["present"]),
                        )
                        conflicted = st.checkbox(
                            "Conflicted / recused",
                            value=bool(stored_status["conflicted"]),
                        )
                        conflict_reason = st.text_input(
                            "Conflict reason",
                            value=str(stored_status.get("conflict_reason") or ""),
                        )
                        status_clicked = st.form_submit_button("Save attendance declaration")
                    if status_clicked:
                        _error_or_rerun(
                            lambda: _call_db(
                                get_connection, update_committee_member_status,
                                lifecycle_id, status_member, present=present,
                                conflicted=conflicted, conflict_reason=conflict_reason,
                                updated_by=actor,
                            ),
                            "Attendance and conflict status recorded.",
                        )
                if st.button("Lock dossier and roster", type="primary", use_container_width=True):
                    _error_or_rerun(
                        lambda: _call_db(
                            get_connection, lock_proposal_dossier, lifecycle_id, locked_by=actor
                        ),
                        "Dossier hash, roster and policy are locked.",
                    )

            if state == "dossier_frozen":
                if st.button("Open blind pre-vote", type="primary", use_container_width=True):
                    _error_or_rerun(
                        lambda: _call_db(get_connection, open_pre_vote, lifecycle_id, opened_by=actor),
                        "Pre-vote opened. Ballots remain hidden until completion.",
                    )

            if state in {"pre_vote", "post_vote"}:
                round_name = "pre" if state == "pre_vote" else "post"
                round_view = lifecycle[f"{round_name}_vote"]
                vote_metrics = st.columns(4)
                vote_metrics[0].metric("Submitted", round_view["submitted_count"])
                vote_metrics[1].metric("Eligible", round_view["eligible_count"])
                vote_metrics[2].metric("Remaining", round_view["remaining_count"])
                vote_metrics[3].metric("Reveal", "Open" if round_view["revealed"] else "Locked")
                eligible_ids = set(lifecycle["committee_status"]["eligible_member_ids"])
                with get_connection() as connection:
                    actor_has_voted = bool(
                        actor_id
                        and has_member_submitted_vote(
                            connection, lifecycle_id, round_name, actor_id
                        )
                    )
                if actor_id in eligible_ids and not actor_has_voted and not round_view["revealed"]:
                    with st.form(f"ios_vote_{round_name}_{lifecycle_id}", clear_on_submit=True):
                        decision = st.selectbox("Independent decision", ["buy", "watch", "reject"])
                        weight = st.number_input(
                            "Proposed position size (%)", min_value=0.0, max_value=100.0, value=5.0
                        )
                        confidence = st.slider("Confidence", 1, 5, 3)
                        rationale = st.text_area("Independent rationale", height=85)
                        objection = st.text_area("Strongest unresolved objection", height=70)
                        v1, v2 = st.columns(2)
                        clarity = v1.slider("Clarity", 1, 5, 3)
                        client_fit = v2.slider("Client fit", 1, 5, 3)
                        vote_clicked = st.form_submit_button(
                            "Submit immutable blind ballot", type="primary", use_container_width=True
                        )
                    if vote_clicked:
                        _error_or_rerun(
                            lambda: _call_db(
                                get_connection, submit_committee_vote,
                                lifecycle_id, round_name, actor_id,
                                decision=decision,
                                proposed_weight_pct=weight if decision == "buy" else None,
                                confidence=confidence,
                                rationale=rationale,
                                strongest_objection=objection,
                                dimensions={"clarity": clarity, "client_fit": client_fit},
                            ),
                            "Blind ballot submitted. It remains sealed until every eligible vote arrives.",
                        )
                elif actor_has_voted and not round_view["revealed"]:
                    st.success("Your sealed ballot was received. Waiting for the remaining members.")
                elif actor_id not in eligible_ids:
                    st.info("Your committee status is observer, absent, or conflicted; no ballot is available.")
                if round_view["revealed"]:
                    st.success(f"Ballots revealed · outcome: {round_view['outcome']}")
                    st.dataframe(pd.DataFrame(round_view["ballots"]), hide_index=True, use_container_width=True)
                    if round_view["dissent"]:
                        st.warning("Recorded dissent is preserved in the audit trail.")
                        st.dataframe(pd.DataFrame(round_view["dissent"]), hide_index=True, use_container_width=True)
                    if round_view["status"] == "ready_to_close" and st.button(
                        f"Close {round_name}-vote round", type="primary", use_container_width=True
                    ):
                        _error_or_rerun(
                            lambda: _call_db(
                                get_connection, close_vote_round,
                                lifecycle_id, round_name, closed_by=actor,
                            ),
                            f"{round_name.title()}-vote closed and transition recorded.",
                        )

            if state == "discussion":
                with st.form(f"ios_discussion_{lifecycle_id}"):
                    bull_case = st.text_area("Owner bull case", height=100)
                    bear_case = st.text_area("Challenger bear case", height=100)
                    question = st.text_area("Committee question", height=65)
                    answer = st.text_area("Evidence-backed answer", height=80)
                    evidence = st.text_input("Evidence / citation IDs")
                    notes = st.text_area("Discussion notes", height=60)
                    discussion_clicked = st.form_submit_button(
                        "Freeze discussion and open post-vote", type="primary", use_container_width=True
                    )
                if discussion_clicked:
                    _error_or_rerun(
                        lambda: _call_db(
                            get_connection, record_committee_discussion, lifecycle_id,
                            bull_case=bull_case, bear_case=bear_case,
                            q_and_a=[{
                                "question": question, "answer": answer,
                                "primary_responder": actor,
                                "evidence_refs": _lines(evidence),
                            }],
                            notes=notes, recorded_by=actor,
                        ),
                        "Discussion frozen; post-vote is now open.",
                    )

            if state == "rule_check":
                if not mandate_record or not rulebook_record:
                    st.error("An active mandate and active rulebook are mandatory before approval.")
                else:
                    rulebook_payload = dict(rulebook_record.get("payload") or rulebook_record)
                    max_position_limit_pct = float(
                        rulebook_payload.get("max_position_weight") or 0.0
                    ) * 100.0
                    requested_weight_pct = float(
                        lifecycle["proposal"].get("proposed_weight_pct") or 0.0
                    )
                    active_universe_matches = bool(
                        universe
                        and int(universe["id"]) == int(lifecycle["universe_snapshot_id"])
                        and str(universe.get("provenance_status") or "") == "official"
                    )
                    st.caption(
                        f"Requested {requested_weight_pct:g}% · rulebook maximum "
                        f"{max_position_limit_pct:g}% · authoritative universe current: "
                        f"{'yes' if active_universe_matches else 'no'}"
                    )
                    with st.form(f"ios_rule_check_{lifecycle_id}"):
                        r1, r2 = st.columns(2)
                        max_position_ok = r1.checkbox(
                            "Requested size obeys max-position rule",
                            value=bool(
                                max_position_limit_pct > 0
                                and requested_weight_pct <= max_position_limit_pct
                            ),
                        )
                        sector_budget_ok = r1.checkbox("Sector risk budget remains valid", value=True)
                        client_fit_ok = r2.checkbox("Client goal and constraints pass", value=True)
                        liquidity_ok = r2.checkbox("Liquidity and cash limits pass", value=True)
                        use_override = st.checkbox("Request explicit named override")
                        override_reason = st.text_area("Override reason", disabled=not use_override)
                        override_authorizer = st.selectbox(
                            "Override authorizer", captain_ids,
                            format_func=lambda value: name_by_id[value],
                            disabled=not use_override,
                        )
                        check_clicked = st.form_submit_button(
                            "Run mandatory rule check", type="primary", use_container_width=True
                        )
                    if check_clicked:
                        checks = [
                            {
                                "rule_id": "max_position",
                                "passed": max_position_ok,
                                "actual_pct": requested_weight_pct,
                                "limit_pct": max_position_limit_pct,
                            },
                            {"rule_id": "sector_risk_budget", "passed": sector_budget_ok},
                            {"rule_id": "client_goal_fit", "passed": client_fit_ok},
                            {"rule_id": "liquidity_and_cash", "passed": liquidity_ok},
                        ]
                        override = None if not use_override else {
                            "reason": override_reason,
                            "authorized_by": override_authorizer,
                            "scope": [
                                *[item["rule_id"] for item in checks if not item["passed"]],
                                "locked_dossier_integrity",
                                "authoritative_universe_current",
                            ],
                        }
                        _error_or_rerun(
                            lambda: _call_db(
                                get_connection, record_rule_check, lifecycle_id,
                                rulebook_version=int(rulebook_record["version"]),
                                mandate_version=int(mandate_record["version"]),
                                checks=checks, evaluated_by=actor, override=override,
                            ),
                            "Rule check recorded; any override is named and immutable.",
                        )

            if state == "final_approval":
                st.dataframe(pd.DataFrame(lifecycle["final_approvals"]), hide_index=True, use_container_width=True)
                required = set(lifecycle["committee_status"]["required_approver_ids"])
                submitted = {item["member_id"] for item in lifecycle["final_approvals"]}
                if actor_id in required - submitted:
                    with st.form(f"ios_final_approval_{lifecycle_id}"):
                        final_decision = st.selectbox("Final decision", ["approve", "reject"])
                        final_comment = st.text_area("Independent sign-off comment")
                        final_clicked = st.form_submit_button(
                            "Submit final sign-off", type="primary", use_container_width=True
                        )
                    if final_clicked:
                        _error_or_rerun(
                            lambda: _call_db(
                                get_connection, submit_final_approval,
                                lifecycle_id, actor_id, decision=final_decision,
                                comment=final_comment,
                            ),
                            "Final sign-off appended.",
                        )
                elif actor_id not in required:
                    st.info("Final approval is restricted to the two designated co-captains.")

            if state == "sizing":
                with st.form(f"ios_sizing_{lifecycle_id}"):
                    target_weight = st.number_input(
                        "Final target weight (%)", min_value=0.01, max_value=100.0,
                        value=float(lifecycle["proposal"].get("proposed_weight_pct") or 5.0),
                    )
                    sizing_rationale = st.text_area("Rule- and risk-linked sizing rationale")
                    starter = st.checkbox("Starter position")
                    expansion = st.text_area("Expansion conditions", disabled=not starter)
                    sizing_clicked = st.form_submit_button(
                        "Approve position size", type="primary", use_container_width=True
                    )
                if sizing_clicked:
                    _error_or_rerun(
                        lambda: _call_db(
                            get_connection, record_position_sizing, lifecycle_id,
                            {
                                "target_weight_pct": target_weight,
                                "rationale": sizing_rationale,
                                "starter_position": starter,
                                "expansion_conditions": expansion.strip() if starter else "",
                            },
                            sized_by=actor,
                        ),
                        "Position size locked to the committee case.",
                    )

            if state == "wins_execution":
                with st.form(f"ios_wins_execution_{lifecycle_id}"):
                    e1, e2, e3 = st.columns(3)
                    transaction_id = e1.text_input("WInS transaction ID")
                    quantity = e2.number_input("Executed quantity", min_value=0.00000001, value=1.0)
                    average_price = e3.number_input("Average execution price", min_value=0.00000001, value=1.0)
                    side = "buy"
                    st.caption("Execution side: BUY · locked to the approved proposal.")
                    currency = st.text_input("Execution currency (required)", placeholder="USD").upper()
                    executed_at = st.text_input(
                        "Executed at (ISO timestamp)", value=datetime.now(timezone.utc).isoformat()
                    )
                    execution_clicked = st.form_submit_button(
                        "Record actual WInS execution", type="primary", use_container_width=True
                    )
                if execution_clicked:
                    def execute_and_stage() -> Any:
                        with get_connection() as connection:
                            return _record_wins_execution_and_stage_tracker_position(
                                connection,
                                lifecycle_id,
                                {
                                    "wins_transaction_id": transaction_id,
                                    "side": side,
                                    "quantity": quantity,
                                    "average_price": average_price,
                                    "executed_at": executed_at,
                                    "currency": currency,
                                },
                                actor=actor,
                            )

                    _error_or_rerun(
                        execute_and_stage,
                        "WInS execution linked and its pending tracker projection staged.",
                    )

            if state == "reconciliation":
                st.info(
                    "Lifecycle activation is derived from the latest fresh, integrity-valid, independently "
                    "approved full-portfolio reconciliation. Status and snapshot IDs cannot be entered manually."
                )
                with get_connection() as connection:
                    staged_rows = _tracker_projection_rows(connection, lifecycle_id)
                pending_staged = (
                    len(staged_rows) == 1
                    and str(staged_rows[0][5] or "").strip().lower() == "pending_reconciliation"
                )
                if pending_staged:
                    st.success("Pending tracker projection is staged for canonical reconciliation.")
                else:
                    st.warning(
                        "This execution has no pending tracker projection. Stage it before importing the next "
                        "full WInS snapshot."
                    )
                    if st.button(
                        "Stage pending tracker projection",
                        key=f"ios_stage_pending_{lifecycle_id}",
                        use_container_width=True,
                    ):
                        _error_or_rerun(
                            lambda: _call_db(
                                get_connection,
                                _stage_pending_tracker_position,
                                lifecycle,
                                actor=actor,
                            ),
                            "Pending tracker projection staged.",
                        )

                st.caption(
                    "Import and independently approve the full account snapshot in Live Portfolio & Data "
                    "Reliability, then return here to activate this lifecycle."
                )
                if st.button(
                    "Activate from latest signed canonical reconciliation",
                    type="primary",
                    disabled=not pending_staged,
                    key=f"ios_activate_canonical_{lifecycle_id}",
                    use_container_width=True,
                ):
                    def activate_from_pipeline() -> Any:
                        with get_connection() as connection:
                            current = get_investment_lifecycle(connection, lifecycle_id)
                            if current is None:
                                raise ValueError("The investment lifecycle no longer exists.")
                            return _activate_reconciled_tracker_position(
                                connection,
                                current,
                                actor=actor,
                            )

                    _error_or_rerun(
                        activate_from_pipeline,
                        "Signed canonical reconciliation linked; the tracker projection is now active.",
                    )

            if state == "active":
                review_tab, exit_tab = st.tabs(["Append thesis review", "Exit position"])
                with review_tab:
                    with st.form(f"ios_position_review_{lifecycle_id}"):
                        review_outcome = st.selectbox("Thesis outcome", ["confirmed", "watch", "invalidated"])
                        kpi_status = st.text_input("KPI monitor status")
                        next_action = st.text_area("Next analytical / portfolio action")
                        review_clicked = st.form_submit_button("Append immutable position review")
                    if review_clicked:
                        _error_or_rerun(
                            lambda: _call_db(
                                get_connection, append_position_review, lifecycle_id,
                                {"kpi_status": kpi_status, "next_action": next_action},
                                outcome=review_outcome, reviewed_by=actor,
                            ),
                            "Position review appended to the lifecycle.",
                        )
                with exit_tab:
                    st.warning(
                        "This records the WInS exit and closes the linked tracker row. Import the next full "
                        "WInS snapshot in Live Portfolio immediately afterward; reporting remains blocked "
                        "until that snapshot is independently reconciled."
                    )
                    with st.form(f"ios_position_exit_{lifecycle_id}"):
                        exit_transaction = st.text_input("Exit WInS transaction ID")
                        exit_price = st.number_input("Exit price", min_value=0.00000001, value=1.0)
                        exit_at = st.text_input(
                            "Exit executed at", value=datetime.now(timezone.utc).isoformat()
                        )
                        exit_reason = st.text_area("Exit reason / invalidation")
                        exit_clicked = st.form_submit_button("Record WInS exit", type="primary")
                    if exit_clicked:
                        def exit_position() -> Any:
                            with get_connection() as connection:
                                try:
                                    connection.execute(
                                        """
                                        UPDATE competition_positions
                                        SET status = 'closed', exit_price = ?, exit_date = ?, closed_by = ?
                                        WHERE lifecycle_id = ? AND status = 'open'
                                        """,
                                        (exit_price, exit_at[:10], actor, lifecycle_id),
                                    )
                                    return record_position_exit(
                                        connection, lifecycle_id,
                                        {
                                            "wins_transaction_id": exit_transaction,
                                            "executed_at": exit_at,
                                            "reason": exit_reason,
                                            "exit_price": exit_price,
                                            "full_snapshot_reconciliation": "required",
                                        },
                                        recorded_by=actor,
                                    )
                                except Exception:
                                    connection.rollback()
                                    raise
                        _error_or_rerun(
                            exit_position,
                            "WInS exit linked and tracker position closed; reconcile the next full snapshot.",
                        )

            if state in {"rejected", "withdrawn", "exited"}:
                st.info(f"This lifecycle is terminal: {state.replace('_', ' ')}.")

    with audit_tab:
        if not lifecycles:
            st.info("Audit events appear after the first proposal.")
        else:
            audit_lifecycle_id = st.selectbox(
                "Lifecycle audit", [int(item["id"]) for item in lifecycles],
                key="ios_audit_lifecycle",
            )
            with get_connection() as connection:
                audited = get_investment_lifecycle(connection, audit_lifecycle_id)
                audit_events = list_lifecycle_audit_events(connection, audit_lifecycle_id)
            if audited:
                st.metric(
                    "Cryptographic event chain",
                    "Valid" if audited["audit"]["valid"] else "Broken",
                    help=f"{audited['audit']['checked_events']} append-only events checked",
                )
                st.dataframe(pd.DataFrame(audit_events), hide_index=True, use_container_width=True)


def _tracker_snapshot_rows(positions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in positions:
        if str(raw.get("status") or "open").lower() not in {
            "open",
            "pending_reconciliation",
        }:
            continue
        quantity = float(raw.get("quantity") or 0.0)
        price = float(raw.get("last_price") or raw.get("entry_price") or 0.0)
        rows.append({
            "ticker": str(raw.get("ticker") or "").upper(),
            "quantity": quantity,
            "current_price": price,
            "market_value": quantity * price,
            "total_cost": quantity * float(raw.get("entry_price") or 0.0),
            "asset_type": str(raw.get("security_type") or "Unknown"),
            "currency": str(raw.get("currency") or "").strip().upper() or None,
        })
    return rows


def _tracker_cash_value(
    positions: Sequence[Mapping[str, Any]],
    *,
    initial_capital: float = 500_000.0,
) -> float:
    """Derive competition cash after open costs, realised P/L, and cash income."""
    from src.portfolio_tracker.wharton_competition import calculate_portfolio_performance

    performance = calculate_portfolio_performance(
        positions,
        live_prices={},
        initial_capital=initial_capital,
    )
    return float(
        performance["cash_before_pnl"]
        + performance["realized_pnl"]
        + performance["open_cash_income"]
    )


def _save_pipeline_workspace(
    get_connection: ConnectionFactory,
    workspace: Mapping[str, Any],
    *,
    actor: str,
    expected_version: int,
) -> None:
    from src.portfolio_tracker.reconciliation_ledger import (
        latest_reconciliation,
        reconciliation_readiness_gate,
    )

    _save_document(
        get_connection,
        "portfolio_pipeline",
        "competition",
        workspace,
        actor=actor,
        status="active",
        expected_version=expected_version,
    )
    latest = latest_reconciliation(workspace.get("ledger") or {"reconciliations": [], "events": []})
    gate = reconciliation_readiness_gate(
        workspace.get("ledger") or {"reconciliations": [], "events": []},
        max_age_seconds=86_400,
    )
    latest_projection = {
        "status": "clean" if gate["ready"] else "blocked",
        "open_exceptions": (
            [item for item in (latest or {}).get("exceptions", []) if item.get("status") != "closed"]
        ),
        "reconciliation_id": gate.get("latest_reconciliation_id"),
        "wins_snapshot_id": gate.get("wins_snapshot_id"),
        "blockers": gate.get("blockers", []),
        "signed_off_by": gate.get("signed_off_by"),
        "as_of": (latest or {}).get("wins_observed_at"),
    }
    _, projection_version = _current_document(
        get_connection, "workspace", "latest_reconciliation"
    )
    _save_document(
        get_connection,
        "workspace",
        "latest_reconciliation",
        latest_projection,
        actor=actor,
        status=latest_projection["status"],
        expected_version=projection_version,
    )


def render_live_portfolio_pipeline(
    profile: Mapping[str, Any],
    get_connection: ConnectionFactory,
    strategy_data: Mapping[str, Any],
    tracker_positions: Sequence[Mapping[str, Any]],
    team_members: Sequence[Any],
) -> None:
    """Import, reconcile, sign, and bind one portfolio to every consumer."""
    from src.data.reliability import initial_circuit_state, record_circuit_result
    from src.portfolio_tracker.portfolio_pipeline import (
        build_live_portfolio_pipeline,
        create_portfolio_snapshot,
        snapshot_badges,
    )
    from src.portfolio_tracker.reconciliation_ledger import (
        append_reconciliation,
        assign_exception,
        latest_reconciliation,
        materialize_reconciliation,
        migrate_reconciliation_ledger,
        new_reconciliation_ledger,
        reconciliation_history,
        resolve_exception,
        sign_off_exception,
        sign_off_reconciliation,
    )

    actor = _actor(profile)
    members = _team_names(team_members)
    default = {
        "snapshots": [],
        "ledger": new_reconciliation_ledger(),
        "expected_return_assumptions": {},
        "provider_circuits": {"Yahoo": initial_circuit_state("Yahoo")},
    }
    workspace, version = _current_document(
        get_connection, "portfolio_pipeline", "competition", default
    )
    for key, value in default.items():
        workspace.setdefault(key, value)
    migrated_ledger = migrate_reconciliation_ledger(workspace.get("ledger"))
    ledger_was_migrated = migrated_ledger != workspace.get("ledger")
    if ledger_was_migrated:
        workspace = {**workspace, "ledger": migrated_ledger}
        _save_pipeline_workspace(
            get_connection,
            workspace,
            actor=f"{actor}:automatic-ledger-migration",
            expected_version=version,
        )
        version += 1

    mandate_record = strategy_data.get("mandate_record") or {}
    strategy_record = strategy_data.get("strategy_record") or {}
    mandate = dict(mandate_record.get("payload") or mandate_record) if isinstance(mandate_record, Mapping) else {}
    rulebook = dict(strategy_record.get("payload") or strategy_record) if isinstance(strategy_record, Mapping) else {}
    pipeline = build_live_portfolio_pipeline(
        workspace["snapshots"],
        workspace["ledger"],
        mandate=mandate,
        rulebook=rulebook,
        expected_return_assumptions=workspace["expected_return_assumptions"],
        max_age_seconds=86_400,
    )

    st.markdown("### Live Portfolio & Data Reliability")
    st.caption(
        "A single reconciled WInS snapshot feeds Tracker, Quant, Risk, Factors, Scenarios, FX, and Reporting. "
        "Every consumer displays the same snapshot ID, timestamp, source, completeness, and freshness."
    )
    if ledger_was_migrated:
        st.success(
            "Legacy reconciliation history was migrated losslessly and saved as schema v1."
        )
    top = st.columns(4)
    top[0].metric("Pipeline", pipeline["status"].title())
    top[1].metric("Authority", pipeline["authority"].replace("_", " ").title())
    top[2].metric("Canonical snapshot", pipeline.get("canonical_snapshot", {}).get("snapshot_id", "None") if pipeline.get("canonical_snapshot") else "None")
    top[3].metric("Last-known-good", "In use" if pipeline["last_known_good"] else "No")

    import_tab, ledger_tab, bindings_tab, reliability_tab = st.tabs(
        ["Import & reconcile", "Exception ledger", "Consumer bindings", "Provider reliability"]
    )
    with import_tab:
        st.markdown("#### WInS snapshot import")
        uploaded = st.file_uploader(
            "WInS positions CSV or Excel",
            type=["csv", "xlsx", "xls"],
            key="ios_wins_snapshot",
        )
        i1, i2, i3 = st.columns(3)
        with i1:
            observed_on = st.date_input("WInS as-of date", value=date.today())
        with i2:
            source_reference = st.text_input("Statement / export reference")
        with i3:
            cash_value = st.number_input("WInS cash (USD)", value=0.0, min_value=0.0)
        import_clicked = st.button(
            "Create snapshots and append reconciliation",
            type="primary",
            disabled=uploaded is None,
            use_container_width=True,
        )
        if import_clicked and uploaded is not None:
            def do_import() -> None:
                if str(uploaded.name).lower().endswith((".xlsx", ".xls")):
                    rows = pd.read_excel(BytesIO(uploaded.getvalue())).to_dict("records")
                else:
                    rows = pd.read_csv(BytesIO(uploaded.getvalue()), sep=None, engine="python").to_dict("records")
                if not rows:
                    raise ValueError("The WInS file contains no positions.")
                timestamp = datetime.combine(observed_on, datetime.min.time(), tzinfo=timezone.utc)
                wins_snapshot = create_portfolio_snapshot(
                    rows, provider="WInS", observed_at=timestamp,
                    method="manual_import", source_reference=source_reference,
                    imported_by=actor, cash_value=cash_value,
                )
                tracked_rows = _tracker_snapshot_rows(tracker_positions)
                tracker_snapshot = create_portfolio_snapshot(
                    tracked_rows, provider="Portfolio Tracker", observed_at=datetime.now(timezone.utc),
                    method="cache", source_reference="competition_positions",
                    imported_by=actor, cash_value=_tracker_cash_value(tracker_positions),
                )
                updated = dict(workspace)
                updated["snapshots"] = [*workspace["snapshots"], tracker_snapshot, wins_snapshot]
                updated["ledger"] = append_reconciliation(
                    workspace["ledger"], wins_snapshot, tracker_snapshot, owner=actor
                )
                _save_pipeline_workspace(
                    get_connection, updated, actor=actor, expected_version=version
                )
            _error_or_rerun(do_import, "WInS and tracker snapshots were frozen and reconciled.")

        if workspace["snapshots"]:
            st.markdown("#### Snapshot history")
            badge_rows = []
            for snapshot in reversed(workspace["snapshots"]):
                badge = snapshot_badges(snapshot, max_age_seconds=86_400)
                badge_rows.append({
                    "Snapshot": badge["snapshot_id"], "As of": badge["as_of"],
                    "Source": badge["source"], "Freshness": badge["freshness"],
                    "Completeness %": badge["completeness_pct"],
                    "Integrity": badge["integrity_valid"], "Status": badge["status"],
                })
            st.dataframe(pd.DataFrame(badge_rows), hide_index=True, use_container_width=True)

        st.markdown("#### Expected-return assumption set")
        current_assumptions = workspace.get("expected_return_assumptions") or {}
        with st.form("ios_expected_returns"):
            assumption_name = st.text_input("Assumption-set name", value=str(current_assumptions.get("name") or "IC approved expected returns"))
            assumption_source = st.text_input("Evidence / approval reference", value=str(current_assumptions.get("source") or ""))
            assumption_values = st.text_area(
                "Annual return assumptions (TICKER=percent)",
                value="\n".join(
                    f"{key}={float(value) * 100:g}"
                    for key, value in (current_assumptions.get("values") or {}).items()
                ),
            )
            assumptions_clicked = st.form_submit_button("Save active assumption set", use_container_width=True)
        if assumptions_clicked:
            def save_assumptions() -> None:
                values: dict[str, float] = {}
                for line in _lines(assumption_values):
                    if "=" not in line:
                        raise ValueError("Each return assumption must use TICKER=percent.")
                    ticker, raw = line.split("=", 1)
                    values[ticker.strip().upper()] = float(raw.strip().rstrip("%")) / 100.0
                updated = {
                    **workspace,
                    "expected_return_assumptions": {
                        "name": assumption_name.strip(), "source": assumption_source.strip(),
                        "values": values, "active": True, "status": "active",
                        "approved_by": actor, "approved_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
                _save_pipeline_workspace(get_connection, updated, actor=actor, expected_version=version)
            _error_or_rerun(save_assumptions, "Expected-return assumptions saved and bound to Quant.")

    with ledger_tab:
        history = reconciliation_history(workspace["ledger"])
        if history:
            st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
            latest = latest_reconciliation(workspace["ledger"])
            assert latest is not None
            gate = pipeline["reconciliation_gate"]
            st.info(
                "Report gate: " + ("READY" if gate["ready"] else "BLOCKED · " + ", ".join(gate["blockers"]))
            )
            if latest["exceptions"]:
                st.dataframe(pd.DataFrame(latest["exceptions"]), hide_index=True, use_container_width=True)
                open_exceptions = [item for item in latest["exceptions"] if item["status"] != "closed"]
                if open_exceptions:
                    selected_exception = st.selectbox(
                        "Exception",
                        [item["exception_id"] for item in open_exceptions],
                        format_func=lambda value: next(
                            f"{item.get('ticker') or 'portfolio'} · {item['category']} · {item['status']}"
                            for item in open_exceptions if item["exception_id"] == value
                        ),
                    )
                    action = st.radio("Workflow action", ["Assign", "Resolve", "Sign off"], horizontal=True)
                    if action == "Assign":
                        owner = st.selectbox("Owner", members, key="ios_exception_owner")
                        if st.button("Assign exception", use_container_width=True):
                            _error_or_rerun(
                                lambda: _save_pipeline_workspace(
                                    get_connection,
                                    {**workspace, "ledger": assign_exception(
                                        workspace["ledger"], latest["reconciliation_id"], selected_exception,
                                        owner=owner, assigned_by=actor,
                                    )},
                                    actor=actor, expected_version=version,
                                ),
                                "Exception assigned.",
                            )
                    elif action == "Resolve":
                        resolution_type = st.selectbox("Resolution", ["tracker corrected", "WInS corrected", "explained", "mapping fixed"])
                        resolution_summary = st.text_area("Resolution and evidence")
                        if st.button("Submit resolution", use_container_width=True):
                            _error_or_rerun(
                                lambda: _save_pipeline_workspace(
                                    get_connection,
                                    {**workspace, "ledger": resolve_exception(
                                        workspace["ledger"], latest["reconciliation_id"], selected_exception,
                                        resolution_type=resolution_type, summary=resolution_summary,
                                        resolved_by=actor, evidence_refs=_lines(resolution_summary),
                                    )},
                                    actor=actor, expected_version=version,
                                ),
                                "Resolution submitted for independent sign-off.",
                            )
                    else:
                        decision = st.selectbox("Decision", ["approved", "rejected"])
                        if st.button("Sign off exception", use_container_width=True):
                            _error_or_rerun(
                                lambda: _save_pipeline_workspace(
                                    get_connection,
                                    {**workspace, "ledger": sign_off_exception(
                                        workspace["ledger"], latest["reconciliation_id"], selected_exception,
                                        decision=decision, signed_off_by=actor,
                                    )},
                                    actor=actor, expected_version=version,
                                ),
                                "Exception sign-off recorded.",
                            )
            if latest["base_is_clean"] and latest["all_exceptions_closed"] and not latest.get("sign_off"):
                st.warning("A different team member must sign the clean reconciliation before reporting can proceed.")
                if st.button("Approve clean reconciliation", type="primary", use_container_width=True):
                    _error_or_rerun(
                        lambda: _save_pipeline_workspace(
                            get_connection,
                            {**workspace, "ledger": sign_off_reconciliation(
                                workspace["ledger"], latest["reconciliation_id"],
                                decision="approved", signed_off_by=actor,
                            )},
                            actor=actor, expected_version=version,
                        ),
                        "Clean WInS reconciliation independently approved.",
                    )
        else:
            st.info("Import a WInS snapshot to start the append-only reconciliation ledger.")

    with bindings_tab:
        rows = [
            {
                "Consumer": name.title(),
                "Snapshot ID": binding.get("snapshot_id"),
                "Allowed": binding.get("allowed"),
                "Blockers": ", ".join(binding.get("blockers") or []),
            }
            for name, binding in pipeline["consumer_bindings"].items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "The rulebook is mandatory for competition Quant runs. Reporting additionally requires fresh data, "
            "active expected-return assumptions, and the latest clean signed WInS reconciliation."
        )

    with reliability_tab:
        circuits = workspace.get("provider_circuits") or {}
        st.dataframe(pd.DataFrame(list(circuits.values())), hide_index=True, use_container_width=True)
        provider = st.text_input("Provider", value="Yahoo")
        result = st.radio("Record provider result", ["Success", "Failure"], horizontal=True)
        error_text = st.text_input("Failure summary", disabled=result == "Success")
        if st.button("Append provider health event", use_container_width=True):
            def save_circuit() -> None:
                current = circuits.get(provider) or initial_circuit_state(provider)
                next_state = record_circuit_result(
                    current, succeeded=result == "Success", error=error_text,
                    failure_threshold=3, cooldown_seconds=300,
                )
                updated = {**workspace, "provider_circuits": {**circuits, provider: next_state}}
                _save_pipeline_workspace(get_connection, updated, actor=actor, expected_version=version)
            _error_or_rerun(save_circuit, "Provider circuit state updated.")


def render_report_evidence_studio(
    profile: Mapping[str, Any],
    get_connection: ConnectionFactory,
    team_members: Sequence[Any],
    strategy_data: Mapping[str, Any] | None = None,
) -> None:
    """Render mid-project/final report workspaces and their evidence graph."""
    from src.reporting.evidence_studio import (
        add_decision_case_study,
        add_report_claim,
        assign_report_section,
        build_export_ready_report,
        create_report_workspace,
        finalise_report,
        freeze_report,
        record_report_approval,
        register_report_evidence,
        register_report_figure,
        set_performance_attribution,
        set_report_portfolio_snapshot,
        set_report_section_content,
        validate_report_workspace,
    )
    from src.portfolio_tracker.portfolio_pipeline import build_live_portfolio_pipeline

    actor = _actor(profile)
    members = _team_names(team_members)
    st.markdown("### Report Evidence Studio")
    st.caption(
        "Build the mid-project and final deliverables from linked claims, evidence, figures, decisions, "
        "performance attribution, and one reconciled as-of portfolio snapshot."
    )
    report_type = st.radio("Workspace", ["mid_project", "final"], horizontal=True)
    workspace, version = _current_document(
        get_connection, "report_workspace", report_type
    )

    if not workspace:
        with st.form(f"ios_create_report_{report_type}"):
            title = st.text_input(
                "Report title",
                value="Wharton Mid-Project Report" if report_type == "mid_project" else "Wharton Final Report",
            )
            page_budget = st.number_input(
                "Total page budget", min_value=1.0,
                value=7.0 if report_type == "mid_project" else 12.0,
            )
            approvers = st.multiselect(
                "Required final approvers", members,
                default=members[:2] if len(members) >= 2 else members,
            )
            create_clicked = st.form_submit_button("Create flexible report workspace", type="primary", use_container_width=True)
        if create_clicked:
            _error_or_rerun(
                lambda: _save_document(
                    get_connection, "report_workspace", report_type,
                    create_report_workspace(
                        f"wharton-{report_type}", report_type, title,
                        created_by=actor, page_budget=page_budget,
                        required_approvers=approvers,
                    ),
                    actor=actor, status="draft", expected_version=0,
                ),
                "Report workspace created.",
            )
        return

    def save(next_workspace: Mapping[str, Any], message: str) -> None:
        _error_or_rerun(
            lambda: _save_document(
                get_connection, "report_workspace", report_type, next_workspace,
                actor=actor, status=str(next_workspace.get("status") or "draft"),
                expected_version=version,
            ),
            message,
        )

    validation = validate_report_workspace(workspace)
    top = st.columns(5)
    top[0].metric("State", str(workspace["status"]).title())
    top[1].metric("Pages", f"{validation['estimated_pages']:g}/{validation['page_budget']:g}")
    top[2].metric("Claims", validation["claim_count"])
    top[3].metric("Evidence", validation["evidence_count"])
    top[4].metric("Freeze blockers", validation["issue_count"])

    sections_tab, evidence_tab, figures_tab, portfolio_tab, freeze_tab = st.tabs(
        ["Sections", "Claims & evidence", "Figures & case studies", "Portfolio & attribution", "Freeze & export"]
    )
    with sections_tab:
        section_id = st.selectbox(
            "Section",
            workspace["section_order"],
            format_func=lambda value: workspace["sections"][value]["title"],
        )
        section = workspace["sections"][section_id]
        with st.form("ios_report_section"):
            s1, s2, s3 = st.columns(3)
            with s1:
                owner = st.selectbox(
                    "Owner", members,
                    index=members.index(section.get("owner")) if section.get("owner") in members else 0,
                )
            with s2:
                reviewer_options = [name for name in members if name != owner]
                reviewer = st.selectbox(
                    "Independent reviewer", reviewer_options,
                    index=(reviewer_options.index(section.get("reviewer")) if section.get("reviewer") in reviewer_options else 0),
                )
            with s3:
                estimated_pages = st.number_input(
                    "Estimated pages", min_value=0.0,
                    max_value=float(section["page_budget"]),
                    value=float(section.get("estimated_pages") or 0.0), step=0.1,
                )
            content = st.text_area("Section working copy", value=str(section.get("content") or ""), height=260)
            ready = st.checkbox("Owner and reviewer mark this section ready", value=section.get("status") == "ready")
            section_clicked = st.form_submit_button("Save section", use_container_width=True)
        if section_clicked:
            revised = assign_report_section(workspace, section_id, owner=owner, reviewer=reviewer)
            revised = set_report_section_content(
                revised, section_id, content=content,
                estimated_pages=estimated_pages, ready_for_freeze=ready,
            )
            save(revised, "Section assignment and content saved.")
        st.dataframe(pd.DataFrame([
            {
                "Section": item["title"], "Owner": item.get("owner") or "—",
                "Reviewer": item.get("reviewer") or "—", "Budget": item["page_budget"],
                "Estimate": item["estimated_pages"], "State": item["status"],
                "Claims": len(item["claim_ids"]), "Figures": len(item["figure_ids"]),
            }
            for item in workspace["sections"].values()
        ]), hide_index=True, use_container_width=True)

    with evidence_tab:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Register verified evidence")
            with st.form("ios_report_evidence", clear_on_submit=True):
                evidence_id = st.text_input("Evidence ID", placeholder="src-10k-msft")
                evidence_title = st.text_input("Title")
                citation = st.text_area("Citation")
                locator = st.text_input("Source URL / file / passage")
                source_type = st.selectbox("Type", ["official", "filing", "dataset", "analysis", "interview", "other"])
                evidence_clicked = st.form_submit_button("Register evidence", use_container_width=True)
            if evidence_clicked:
                save(
                    register_report_evidence(
                        workspace, evidence_id, title=evidence_title,
                        citation=citation, source_locator=locator,
                        source_type=source_type, verified_by=actor,
                        accessed_at=date.today(),
                    ),
                    "Evidence registered with an integrity hash.",
                )
        with right:
            st.markdown("#### Add evidence-backed claim")
            with st.form("ios_report_claim", clear_on_submit=True):
                claim_id = st.text_input("Claim ID", placeholder="claim-client-fit")
                claim_section = st.selectbox("Report section", workspace["section_order"])
                statement = st.text_area("Exact report claim")
                evidence_ids = st.multiselect("Supporting evidence", list(workspace["evidence"]))
                claim_clicked = st.form_submit_button("Link claim to evidence", use_container_width=True)
            if claim_clicked:
                save(
                    add_report_claim(
                        workspace, claim_id, section_id=claim_section,
                        statement=statement, evidence_ids=evidence_ids,
                        created_by=actor,
                    ),
                    "Claim linked to evidence.",
                )
        if workspace["claims"]:
            st.dataframe(pd.DataFrame(workspace["claims"].values()), hide_index=True, use_container_width=True)

    with figures_tab:
        figure_col, case_col = st.columns(2)
        with figure_col:
            st.markdown("#### Figure register")
            with st.form("ios_report_figure", clear_on_submit=True):
                figure_id = st.text_input("Figure ID")
                figure_section = st.selectbox("Section", workspace["section_order"], key="ios_figure_section")
                figure_title = st.text_input("Figure title")
                figure_caption = st.text_area("Caption and interpretation")
                artifact = st.text_input("Artifact locator")
                figure_evidence = st.multiselect("Evidence", list(workspace["evidence"]), key="ios_figure_evidence")
                figure_clicked = st.form_submit_button("Register figure", use_container_width=True)
            if figure_clicked:
                save(
                    register_report_figure(
                        workspace, figure_id, section_id=figure_section,
                        title=figure_title, caption=figure_caption,
                        artifact_locator=artifact, evidence_ids=figure_evidence,
                        data_as_of=date.today(), owner=actor,
                    ),
                    "Figure registered.",
                )
        with case_col:
            st.markdown("#### Decision case study")
            with st.form("ios_report_case", clear_on_submit=True):
                case_id = st.text_input("Case-study ID")
                case_section = st.selectbox("Section", workspace["section_order"], key="ios_case_section")
                decision_id = st.text_input("Canonical decision ID")
                case_ticker = st.text_input("Ticker")
                case_title = st.text_input("Case title")
                process_summary = st.text_area("Process summary")
                outcome_summary = st.text_area("Outcome summary")
                lesson = st.text_area("Lesson learned")
                case_evidence = st.multiselect("Evidence", list(workspace["evidence"]), key="ios_case_evidence")
                case_clicked = st.form_submit_button("Add case study", use_container_width=True)
            if case_clicked:
                save(
                    add_decision_case_study(
                        workspace, case_id, section_id=case_section,
                        decision_id=decision_id, ticker=case_ticker,
                        title=case_title, process_summary=process_summary,
                        outcome_summary=outcome_summary, lesson=lesson,
                        evidence_ids=case_evidence,
                    ),
                    "Decision case study linked.",
                )
        if workspace["figures"]:
            st.dataframe(pd.DataFrame(workspace["figures"].values()), hide_index=True, use_container_width=True)

    with portfolio_tab:
        pipeline_workspace, _ = _current_document(
            get_connection, "portfolio_pipeline", "competition"
        )
        if pipeline_workspace:
            strategy_context = dict(strategy_data or {})
            pipeline = build_live_portfolio_pipeline(
                pipeline_workspace.get("snapshots", []),
                pipeline_workspace.get("ledger", {"reconciliations": [], "events": []}),
                mandate=strategy_context.get("mandate_record"),
                rulebook=strategy_context.get("strategy_record"),
                expected_return_assumptions=pipeline_workspace.get("expected_return_assumptions"),
            )
            canonical = pipeline.get("canonical_snapshot")
            if canonical:
                st.write(
                    f"Available: `{canonical['snapshot_id']}` · {canonical['observed_at']} · "
                    f"reporting {'allowed' if pipeline['consumer_bindings']['reporting']['allowed'] else 'blocked'}"
                )
                if st.button("Attach canonical reconciled snapshot", use_container_width=True):
                    positions = [
                        {
                            "security_id": item["ticker"], "ticker": item["ticker"],
                            "weight": item.get("weight") or 0.0,
                            "market_value": item.get("market_value") or 0.0,
                            "currency": item.get("currency") or "UNKNOWN",
                        }
                        for item in canonical["payload"].get("positions", [])
                    ]
                    reconciliation_id = pipeline["reconciliation_gate"].get("latest_reconciliation_id")
                    save(
                        set_report_portfolio_snapshot(
                            workspace, canonical["snapshot_id"],
                            as_of=canonical["observed_at"], source="WInS reconciled pipeline",
                            positions=positions,
                            reconciled=bool(pipeline["consumer_bindings"]["reporting"]["allowed"]),
                            reconciliation_id=reconciliation_id,
                        ),
                        "Canonical portfolio snapshot attached.",
                    )
            else:
                st.warning("No canonical portfolio snapshot exists.")
        else:
            st.warning("Build the live portfolio pipeline first.")

        if report_type == "final":
            with st.form("ios_report_attribution"):
                a1, a2, a3 = st.columns(3)
                with a1:
                    benchmark = st.text_input("Benchmark", value="SPY")
                with a2:
                    portfolio_return = st.number_input("Portfolio return (%)", value=0.0)
                with a3:
                    benchmark_return = st.number_input("Benchmark return (%)", value=0.0)
                contribution_text = st.text_area("Contributions: label=percent", placeholder="Security selection=2.4\nAllocation=-0.5")
                methodology = st.text_area("Attribution methodology")
                attribution_clicked = st.form_submit_button("Save performance attribution", use_container_width=True)
            if attribution_clicked:
                contributions = []
                for index, line in enumerate(_lines(contribution_text), start=1):
                    label, raw = line.split("=", 1)
                    contributions.append({"id": f"c{index}", "label": label.strip(), "contribution": float(raw.rstrip("%")) / 100})
                save(
                    set_performance_attribution(
                        workspace, as_of=date.today(), benchmark=benchmark,
                        portfolio_return=portfolio_return / 100,
                        benchmark_return=benchmark_return / 100,
                        contributions=contributions, methodology=methodology,
                    ),
                    "Performance attribution saved.",
                )

    with freeze_tab:
        if validation["issues"]:
            st.dataframe(pd.DataFrame(validation["issues"]), hide_index=True, use_container_width=True)
        if workspace["status"] == "draft":
            if st.button("Freeze report", type="primary", disabled=not validation["is_ready"], use_container_width=True):
                save(freeze_report(workspace, frozen_by=actor), "Report frozen; content is now hash-locked.")
        elif workspace["status"] == "frozen":
            st.write(f"Frozen content hash: `{workspace['freeze']['content_hash']}`")
            if actor in workspace["required_approvers"] and actor not in workspace["approvals"]:
                approval_note = st.text_input("Approval note")
                if st.button("Approve frozen report", type="primary", use_container_width=True):
                    save(
                        record_report_approval(workspace, approver=actor, notes=approval_note),
                        "Report approval recorded.",
                    )
            missing = [name for name in workspace["required_approvers"] if name not in workspace["approvals"]]
            st.caption("Missing approvals: " + (", ".join(missing) if missing else "none"))
            if not missing and st.button("Finalise report", type="primary", use_container_width=True):
                save(finalise_report(workspace, finalised_by=actor), "Report finalised.")
        else:
            from src.reporting.report_documents import (
                generate_evidence_report_documents,
            )

            export_model = build_export_ready_report(workspace)
            st.success("Final report model is locked and export-ready.")
            documents = generate_evidence_report_documents(workspace)
            export_columns = st.columns(3)
            export_columns[0].download_button(
                "Download editable DOCX",
                documents["docx"],
                file_name=f"{workspace['report_id']}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
            export_columns[1].download_button(
                "Download final PDF",
                documents["pdf"],
                file_name=f"{workspace['report_id']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            export_columns[2].download_button(
                "Download audit JSON",
                json.dumps(export_model, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{workspace['report_id']}.json",
                mime="application/json",
                use_container_width=True,
            )


def render_qa_rehearsal(
    profile: Mapping[str, Any],
    get_connection: ConnectionFactory,
    team_members: Sequence[Any],
) -> None:
    """Render evidence-linked questions, timed rounds, and member history."""
    from src.portfolio_tracker.qa_rehearsal import (
        add_qa_question,
        complete_mock_round,
        create_mock_round,
        create_qa_rehearsal_workspace,
        killer_question_status,
        member_qa_history,
        record_qa_response,
    )

    actor = _actor(profile)
    members = _team_names(team_members)
    workspace, version = _current_document(get_connection, "qa_workspace", "competition")
    st.markdown("### Q&A Rehearsal Engine")
    st.caption(
        "Every question has a model answer, evidence, primary and backup responder, time limit, "
        "follow-ups, and scored rehearsal history."
    )
    if not workspace:
        if st.button("Create team rehearsal workspace", type="primary", use_container_width=True):
            _error_or_rerun(
                lambda: _save_document(
                    get_connection, "qa_workspace", "competition",
                    create_qa_rehearsal_workspace(
                        "wharton-qa", team_members=members, created_by=actor
                    ),
                    actor=actor, status="active", expected_version=0,
                ),
                "Q&A rehearsal workspace created.",
            )
        return

    def save(next_workspace: Mapping[str, Any], message: str) -> None:
        def persist() -> None:
            _save_document(
                get_connection, "qa_workspace", "competition", next_workspace,
                actor=actor, status="active", expected_version=version,
            )
            for round_id in next_workspace.get("round_order", []):
                round_data = next_workspace["rounds"][round_id]
                if round_data.get("status") != "completed":
                    continue
                existing, round_version = _current_document(
                    get_connection, "qa_round", round_id
                )
                if not existing:
                    _save_document(
                        get_connection, "qa_round", round_id, round_data,
                        actor=actor, status="completed", expected_version=round_version,
                    )
        _error_or_rerun(persist, message)

    bank_tab, round_tab, history_tab = st.tabs(["Question bank", "Mock round", "Performance"])
    with bank_tab:
        with st.form("ios_qa_question", clear_on_submit=True):
            q1, q2 = st.columns(2)
            with q1:
                question_id = st.text_input("Question ID", placeholder="q-client-fit")
                prompt = st.text_area("Question")
                category = st.text_input("Category", value="client_fit")
                time_limit = st.number_input("Time limit (seconds)", min_value=10, max_value=600, value=90)
            with q2:
                model_answer = st.text_area("Model answer")
                evidence = st.text_area("Evidence IDs", placeholder="claim-1\nsource-2")
                primary = st.selectbox("Primary responder", members)
                backups = [name for name in members if name != primary]
                backup = st.selectbox("Backup responder", backups)
            follow_ups = st.text_area("Likely follow-up questions")
            killer = st.checkbox("Killer question / unresolved challenge")
            question_clicked = st.form_submit_button("Add rehearsal-ready question", type="primary", use_container_width=True)
        if question_clicked:
            save(
                add_qa_question(
                    workspace, question_id, prompt=prompt, model_answer=model_answer,
                    evidence_ids=_lines(evidence), primary_responder=primary,
                    backup_responder=backup, time_limit_seconds=int(time_limit),
                    category=category, follow_ups=_lines(follow_ups),
                    killer_question=killer, created_by=actor,
                ),
                "Question added to the evidence-linked bank.",
            )
        if workspace["questions"]:
            st.dataframe(pd.DataFrame(workspace["questions"].values()), hide_index=True, use_container_width=True)
        else:
            st.info("Add at least one complete question before starting a round.")

    with round_tab:
        ready_ids = [
            key for key, item in workspace["questions"].items()
            if item.get("status") == "ready"
        ]
        active_rounds = [
            key for key in workspace["round_order"]
            if workspace["rounds"][key]["status"] == "active"
        ]
        if ready_ids:
            with st.form("ios_start_qa_round"):
                round_id = st.text_input("Round ID", value=f"round-{len(workspace['round_order']) + 1}")
                participants = st.multiselect("Participants", members, default=members)
                question_count = st.number_input(
                    "Random question count", min_value=1, max_value=len(ready_ids),
                    value=min(3, len(ready_ids)),
                )
                random_seed = st.number_input("Random seed", min_value=0, value=len(workspace["round_order"]) + 1)
                start_clicked = st.form_submit_button("Start frozen random round", use_container_width=True)
            if start_clicked:
                save(
                    create_mock_round(
                        workspace, round_id, started_by=actor,
                        participant_ids=participants, question_count=int(question_count),
                        random_seed=int(random_seed),
                    ),
                    "Mock round started; question order and answer keys are frozen.",
                )
        if active_rounds:
            selected_round_id = st.selectbox("Active round", active_rounds)
            selected_round = workspace["rounds"][selected_round_id]
            unanswered = [slot for slot in selected_round["slots"] if slot.get("response") is None]
            if unanswered:
                selected_question_id = st.selectbox(
                    "Question",
                    [slot["question_id"] for slot in unanswered],
                )
                slot = next(item for item in unanswered if item["question_id"] == selected_question_id)
                snapshot = slot["question_snapshot"]
                st.warning(f"{snapshot['prompt']} · limit {snapshot['time_limit_seconds']} seconds")
                st.caption("Follow-ups: " + (" | ".join(snapshot["follow_ups"]) or "none"))
                with st.form("ios_qa_response"):
                    responder = st.selectbox("Responder", slot["eligible_responders"])
                    evaluator_options = [name for name in members if name != responder]
                    evaluator = st.selectbox("Evaluator", evaluator_options)
                    answer = st.text_area("Answer delivered")
                    duration = st.number_input("Duration (seconds)", min_value=0.0, value=90.0)
                    score_cols = st.columns(3)
                    with score_cols[0]:
                        clarity = st.slider("Clarity", 1, 5, 3)
                    with score_cols[1]:
                        evidence_score = st.slider("Evidence", 1, 5, 3)
                    with score_cols[2]:
                        client_fit = st.slider("Client fit", 1, 5, 3)
                    notes = st.text_area("Evaluator notes / follow-up gaps")
                    response_clicked = st.form_submit_button("Record scored response", type="primary", use_container_width=True)
                if response_clicked:
                    save(
                        record_qa_response(
                            workspace, selected_round_id, selected_question_id,
                            responder=responder, answer=answer,
                            duration_seconds=duration,
                            scores={"clarity": clarity, "evidence": evidence_score, "client_fit": client_fit},
                            evaluator=evaluator, notes=notes,
                        ),
                        "Scored response recorded.",
                    )
            else:
                if st.button("Complete and score round", type="primary", use_container_width=True):
                    save(
                        complete_mock_round(workspace, selected_round_id, completed_by=actor),
                        "Round completed and added to member histories.",
                    )
        elif workspace["round_order"]:
            st.info("No round is currently active.")

    with history_tab:
        completed = [
            workspace["rounds"][key]
            for key in workspace["round_order"]
            if workspace["rounds"][key]["status"] == "completed"
        ]
        if completed:
            st.dataframe(pd.DataFrame([
                {"Round": item["round_id"], **item["summary"]}
                for item in completed
            ]), hide_index=True, use_container_width=True)
        member = st.selectbox("Member history", members, key="ios_qa_member")
        history = member_qa_history(workspace, member)
        st.json(history, expanded=False)
        killer_status = killer_question_status(workspace)
        st.metric("Unresolved killer questions", killer_status["unresolved_count"])
        if killer_status["unresolved"]:
            st.dataframe(pd.DataFrame(killer_status["unresolved"]), hide_index=True, use_container_width=True)


def render_official_rules_watch(
    profile: Mapping[str, Any],
    get_connection: ConnectionFactory,
    team_members: Sequence[Any],
) -> None:
    """Render immutable rules capture, change diff, and acknowledgements."""
    from src.portfolio_tracker.official_rules import (
        acknowledge_rules_snapshot,
        capture_rules_snapshot,
        create_official_rules_watch,
        current_rules_watch_status,
        diff_rules_snapshots,
        latest_rules_snapshot,
        rules_acknowledgement_status,
        verify_rules_watch_integrity,
    )

    actor = _actor(profile)
    members = _team_names(team_members)
    watch, version = _current_document(get_connection, "rules_watch", "wharton")
    st.markdown("### Official Rules Watch")
    st.caption(
        "Capture every official publication as immutable text with a SHA-256 hash, show changes, "
        "and require each team member to acknowledge the latest version."
    )
    if not watch:
        if st.button("Create 2026–2027 rules watch", type="primary", use_container_width=True):
            _error_or_rerun(
                lambda: _save_document(
                    get_connection, "rules_watch", "wharton",
                    create_official_rules_watch(
                        "wharton-2026-27", competition="Wharton Global High School Investment Competition 2026–2027",
                        team_members=members, created_by=actor,
                    ),
                    actor=actor, status="active", expected_version=0,
                ),
                "Official rules watch created.",
            )
        return

    def save(next_watch: Mapping[str, Any], message: str) -> None:
        def persist() -> None:
            _save_document(
                get_connection, "rules_watch", "wharton", next_watch,
                actor=actor, status="active", expected_version=version,
            )
            status = current_rules_watch_status(next_watch)
            latest_records = [
                latest_rules_snapshot(next_watch, ruleset)
                for ruleset in next_watch["rulesets"]
                if latest_rules_snapshot(next_watch, ruleset) is not None
            ]
            content_hash = "|".join(item["content_hash"] for item in latest_records)
            acknowledged = sorted(set.intersection(*[
                set(rules_acknowledgement_status(next_watch, item["snapshot_id"])["acknowledged_members"])
                for item in latest_records
            ])) if latest_records else []
            projection, projection_version = _current_document(
                get_connection, "workspace", "latest_rules"
            )
            _save_document(
                get_connection, "workspace", "latest_rules",
                {
                    "content_hash": content_hash,
                    "all_acknowledged": status["is_current"],
                    "acknowledged_by": acknowledged,
                    "missing_rulesets": status["missing_rulesets"],
                    "unacknowledged_rulesets": status["unacknowledged_rulesets"],
                },
                actor=actor,
                status="current" if status["is_current"] else "attention",
                expected_version=projection_version,
            )
        _error_or_rerun(persist, message)

    status = current_rules_watch_status(watch)
    integrity = verify_rules_watch_integrity(watch)
    top = st.columns(4)
    top[0].metric("Current", "Yes" if status["is_current"] else "No")
    top[1].metric("Snapshots", integrity["snapshot_count"])
    top[2].metric("Hash chain", "Valid" if integrity["is_valid"] else "Broken")
    top[3].metric("Missing publications", len(status["missing_rulesets"]))

    capture_tab, changes_tab, ack_tab = st.tabs(["Capture official source", "Change diff", "Acknowledgements"])
    with capture_tab:
        with st.form("ios_rules_capture"):
            ruleset = st.selectbox("Ruleset", list(watch["rulesets"]))
            source_title = st.text_input("Official page title")
            source_url = st.text_input("Official source URL")
            published_at = st.date_input("Published / effective date", value=date.today())
            content = st.text_area(
                "Verified page text / structured rule snapshot",
                height=320,
                help="Paste the official content or a complete structured transcription; the exact text is hashed.",
            )
            capture_clicked = st.form_submit_button("Capture immutable snapshot", type="primary", use_container_width=True)
        if capture_clicked:
            save(
                capture_rules_snapshot(
                    watch, ruleset, source_url=source_url, content=content,
                    source_title=source_title, captured_by=actor,
                    published_at=published_at,
                ),
                "Official rules snapshot captured; all acknowledgements reset for this version.",
            )
        snapshot_rows = [
            {key: value for key, value in item.items() if key != "content"}
            for item in watch["snapshots"].values()
        ]
        if snapshot_rows:
            st.dataframe(pd.DataFrame(snapshot_rows), hide_index=True, use_container_width=True)

    with changes_tab:
        selected_ruleset = st.selectbox("Ruleset", list(watch["rulesets"]), key="ios_diff_ruleset")
        ids = watch["rulesets"][selected_ruleset]["snapshot_ids"]
        if len(ids) >= 2:
            diff = diff_rules_snapshots(watch, ids[-2], ids[-1])
            d1, d2 = st.columns(2)
            d1.metric("Added lines", diff["added_line_count"])
            d2.metric("Removed lines", diff["removed_line_count"])
            st.code("\n".join(diff["unified_diff"]), language="diff")
        else:
            st.info("A diff appears after the second changed snapshot of a ruleset.")

    with ack_tab:
        latest_choices = [
            item for ruleset in watch["rulesets"]
            if (item := latest_rules_snapshot(watch, ruleset)) is not None
        ]
        if latest_choices:
            snapshot_id = st.selectbox(
                "Latest snapshot",
                [item["snapshot_id"] for item in latest_choices],
            )
            ack_status = rules_acknowledgement_status(watch, snapshot_id)
            st.write(
                f"Acknowledged {ack_status['acknowledged_count']}/{ack_status['required_count']} · "
                f"missing: {', '.join(ack_status['missing_members']) or 'none'}"
            )
            if actor in ack_status["missing_members"]:
                note = st.text_input("Acknowledgement note")
                if st.button("I reviewed this exact hashed version", type="primary", use_container_width=True):
                    save(
                        acknowledge_rules_snapshot(
                            watch, snapshot_id, member_id=actor, note=note,
                        ),
                        "Acknowledgement recorded against the content hash.",
                    )
        else:
            st.info("Capture an official source before requesting acknowledgements.")
