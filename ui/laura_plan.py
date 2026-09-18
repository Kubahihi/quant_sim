"""Laura's client plan, kept separate from the WInS trading account."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
import streamlit as st

from src.portfolio_tracker.laura_case import (
    REQUIREMENTS, SOURCES, cashflow_calendar, check_ips_text, check_trading_notes,
    deliverables_calendar, fundraising_draft, save_case_section,
)
from src.simulation.laura_funding import (
    LauraPolicy, bootstrap_laura_plan, illustrative_cases, partner_interval, project_laura_plan,
)


def _download(label: str, value: Any, filename: str, key: str) -> None:
    st.download_button(label, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                       file_name=filename, mime="application/json", key=key)


def render_case_brief() -> None:
    st.info(
        "Laura Gao · Taiwan creative residency · nominal USD. Deposit 300,000 USD at the beginning "
        "of 2027 and 150,000 USD at the beginning of 2028. Pay 50,000 USD at the beginning of every "
        "year from 2033 through 2042. Reserve the operating obligation before deciding the facility contribution."
    )
    st.caption(
        "Source: supplied assessment dated 17 September 2026, pp. 1–9, and Laura_Gao_Cashflow.xlsx. "
        "The 500,000 USD WInS account and its P&L are separate from these 450,000 USD client deposits. "
        "No additional contributions, inflation uplift or external operating support are assumed."
    )


def _render_case() -> None:
    calendar = pd.DataFrame(cashflow_calendar()).rename(columns={
        "calendar_year": "Year", "model_year": "Model year", "deposit_usd": "Deposit USD",
        "operating_payment_usd": "Operating payment USD", "timing": "Timing", "milestone": "Milestone",
    })
    calendar = calendar.drop(columns="Timing")
    calendar["Milestone"] = calendar["Year"].map({2031: "Partner contribution range", 2033: "Reserve, facility, first payment", 2042: "Final payment"}).fillna("")
    st.caption("All deposits and operating payments occur at the beginning of the year.")
    st.dataframe(calendar.style.format({"Deposit USD": "${:,.0f}", "Operating payment USD": "${:,.0f}"}),
                 hide_index=True, use_container_width=True, column_config={
                     "Year": st.column_config.NumberColumn(width="small", format="%d"),
                     "Model year": st.column_config.NumberColumn(width="small", format="%d"),
                     "Milestone": st.column_config.TextColumn(width="large"),
                 })
    st.markdown("#### Facts and team choices")
    st.write(
        "The payment dates and amounts are fixed. Allocation, reserve size, confidence threshold, "
        "fees, flexibility and the facility rule are team choices. The brief does not assign a numeric "
        "loss tolerance, mandatory ESG exclusions, Taiwan-stock allocation or building budget. "
        "Partner communication takes place in 2031; this tool explicitly chooses 1 January 2031 for valuation."
    )
    st.dataframe(pd.DataFrame(REQUIREMENTS, columns=["ID", "Requirement", "Support / remaining team work"]),
                 hide_index=True, use_container_width=True)
    st.caption("Available software support does not certify completion of the team's submission.")
    _download("Download case inputs", {"sources": SOURCES, "cashflows": cashflow_calendar(),
                                      "requirements": REQUIREMENTS}, "laura_case.json", "laura_case_download")


def _render_illustrations() -> None:
    cases = illustrative_cases()
    st.markdown("#### Cashflow workbook cross-check")
    st.caption(
        "Four deterministic illustrations from the supplied workbook: 0%, 6%, 9% growth, "
        "and a −25% return in 2032. The 80% facility share and 3% reserve discount are illustrations. "
        "These four cases have no probability weights."
    )
    rows = []
    for name, result in cases.items():
        row = result.scenario_rows()[0]
        rows.append({"Case": name, "Portfolio 2033": row["portfolio_2033"],
                     "Required reserve": row["required_reserve"], "Facility": row["facility"],
                     "Flexibility": row["flexibility"], "Unpaid total": row["total_unpaid"],
                     "All payments": "Met" if row["all_payments_met"] else "Shortfall"})
    st.dataframe(pd.DataFrame(rows).style.format({column: "${:,.0f}" for column in
                  ["Portfolio 2033", "Required reserve", "Facility", "Flexibility", "Unpaid total"]}),
                 hide_index=True, use_container_width=True)
    selected = st.selectbox("Inspect cash flows", list(cases), key="laura_illustration")
    result = cases[selected]
    with st.expander("Annual accumulation and all ten payments"):
        st.write(f"Reserve funding gap: ${result.reserve_gap[0]:,.0f}; "
                 f"first missed payment: {int(result.first_shortfall_year[0]) or 'none'}.")
        st.dataframe(pd.DataFrame({"year": result.accumulation_years,
                                  "opening": result.accumulation_opening[0],
                                  "deposit": [300_000, 150_000, 0, 0, 0, 0],
                                  "closing": result.accumulation_closing[0]}),
                     hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame(result.payment_rows()), hide_index=True, use_container_width=True)
    st.download_button("Download illustrated payment ledger", pd.DataFrame(result.payment_rows()).to_csv(index=False),
                       "laura_payments.csv", "text/csv", key="laura_payments_download")


def build_funding_run(annual: pd.DataFrame, allocation: pd.DataFrame, policy: LauraPolicy,
                      *, opening_2031: float, state_description: str,
                      n_scenarios: int = 10_000, seed: int = 2027,
                      candidates: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Testable UI orchestration; all candidates/stresses use the same sampled rows."""
    columns = [str(c) for c in annual.columns]
    if allocation["Asset"].duplicated().any() or set(allocation["Asset"]) != set(columns):
        raise ValueError("Allocation assets must match the aligned return history exactly.")
    weights = allocation.set_index("Asset").reindex(columns)
    early = weights["2027–2030 %"].to_numpy(dtype=float) / 100
    late = weights["2031–2032 %"].to_numpy(dtype=float) / 100
    reserve_early = weights["2033–2036 %"].to_numpy(dtype=float) / 100
    reserve_late = weights["2037–2041 %"].to_numpy(dtype=float) / 100
    growth_weights = np.vstack([early] * 4 + [late] * 2)
    reserve_weights = np.vstack([reserve_early] * 4 + [reserve_late] * 5)
    history = annual.to_numpy(dtype=float)
    projection, indices = bootstrap_laura_plan(history, growth_weights, reserve_weights, policy,
                                               n_scenarios=n_scenarios, random_seed=seed)
    conditional, conditional_indices = bootstrap_laura_plan(
        history, np.vstack([late] * 2), reserve_weights, policy,
        valuation_year=2031, opening_wealth_2031=opening_2031,
        n_scenarios=n_scenarios, random_seed=seed,
    )
    interval = partner_interval(conditional)
    growth = np.einsum("sta,ta->st", history[indices[:, :6]], growth_weights)
    reserve = np.einsum("sta,ta->st", history[indices[:, 6:]], reserve_weights)
    # These stresses are sensitivity choices, not probabilistic market forecasts.
    loss = growth.copy()
    loss[:, -1] = np.minimum(loss[:, -1], -0.25)
    sequence = np.sort(reserve, axis=1)
    stressed_history = np.sort(history, axis=0)  # co-rank asset losses: adverse dependence
    correlation_growth = np.einsum("sta,ta->st", stressed_history[indices[:, :6]], growth_weights)
    correlation_reserve = np.einsum("sta,ta->st", stressed_history[indices[:, 6:]], reserve_weights)
    stresses = {
        "Baseline": projection,
        "2032 return capped at −25%": project_laura_plan(loss, reserve, policy),
        "Returns lower by 2 percentage points": project_laura_plan(np.maximum(growth - .02, -1), np.maximum(reserve - .02, -1), policy),
        "Fees higher by 1 percentage point": project_laura_plan(growth, reserve, replace(policy, accumulation_fee=policy.accumulation_fee + .01, reserve_fee=policy.reserve_fee + .01)),
        "Worst reserve years first": project_laura_plan(growth, sequence, policy),
        "Asset losses occur together": project_laura_plan(correlation_growth, correlation_reserve, policy),
        "Reserve discount lower by 2 percentage points": project_laura_plan(growth, reserve, replace(policy, reserve_discount_rate=policy.reserve_discount_rate - .02)),
    }
    comparisons = [{"Portfolio": "Staged client policy", **projection.summary()}]
    for name, weights in (candidates or {}).items():
        candidate, candidate_indices = bootstrap_laura_plan(
            history, weights, reserve_weights, policy, n_scenarios=n_scenarios, random_seed=seed,
        )
        if not np.array_equal(indices, candidate_indices):
            raise ValueError("Candidate portfolios must use identical sampled scenarios.")
        comparisons.append({"Portfolio": name, **candidate.summary()})
    metadata = {
        "sources": SOURCES, "policy": asdict(policy),
        "history_years": [int(y) for y in annual.index], "history_assets": columns,
        "annual_usd_return_history": annual.to_dict(orient="split"),
        "history_sha256": hashlib.sha256(annual.to_csv().encode()).hexdigest(),
        "annual_allocation": allocation.to_dict(orient="records"),
        "random_seed": seed, "scenario_count": n_scenarios,
        "comparison_accumulation_weights": {name: np.asarray(w, dtype=float).tolist() for name, w in (candidates or {}).items()},
        "valuation_2031": {"date": "2031-01-01", "opening_wealth_usd": opening_2031,
                           "state_description": state_description},
        "method": "IID annual-row historical bootstrap; paired asset returns; annual rebalancing at explicit stage weights",
        "limitations": [
            "Conditional on history and assumptions; not a forecast or guarantee",
            "Same-history optimized allocations are in-sample exploratory, not out-of-sample evidence",
            "No serial dependence, regime estimation or unobserved future market states",
            "Flat reserve discount is a team assumption, not a locked bond ladder or ETF guarantee",
            "Fees applied multiplicatively; no taxes (outside case scope); no implicit external cash",
            "Flexibility is retained outside the operating reserve and is not reinvested or used to rescue payments",
            "Whole annual USD return history is required; return conversion and security eligibility need independent verification",
        ],
    }
    return {"projection": projection, "conditional": conditional, "interval": interval,
            "metadata": metadata, "sampled_indices": indices, "conditional_indices": conditional_indices,
            "comparison_rows": comparisons,
            "stress_rows": [{"Stress assumption": name, **p.summary()} for name, p in stresses.items()],
            "fundraising_draft": fundraising_draft(interval, state_description=state_description)}


def _render_funding(result: Mapping[str, Any], profile: Mapping[str, Any], get_connection,
                    saved: Mapping[str, Any]) -> None:
    _render_illustrations()
    st.markdown("#### Portfolio scenarios and the 2031 partner range")
    from ui.pages.wharton_dash import _annual_goal_return_history, _goal_candidate_weights

    annual = _annual_goal_return_history(result.get("returns", pd.DataFrame()))
    if len(annual) < 2:
        st.info("Run Quant Engine with at least two near-complete calendar years of aligned USD returns to test your allocations. The workbook checks above remain available without market data.")
        return
    if len(annual) < 5:
        st.warning(f"Only {len(annual)} annual observations: the bootstrap reuses a small set of regimes. Results are exploratory.")
    st.caption(
        "Set allocations for accumulation, the approach to 2033, and reserve drawdown. Each column must total 100%. "
        "Reserve assets must exist in the Quant Engine history; include the required instruments in that run. "
        "The reserve discount rate prices the liability; it does not force the reserve portfolio to earn that rate."
    )
    columns = [str(c) for c in annual.columns]
    current = _goal_candidate_weights(result, columns).get("Current strategy", np.zeros(len(columns)))
    old_allocations = {r["Asset"]: r for r in saved.get("allocation", [])}
    frame = pd.DataFrame([
        old_allocations.get(asset, {"Asset": asset, "2027–2030 %": float(current[i] * 100),
                                    "2031–2032 %": float(current[i] * 100), "2033–2036 %": 0.0, "2037–2041 %": 0.0})
        for i, asset in enumerate(columns)
    ])
    defaults = {**asdict(LauraPolicy()), **saved.get("policy", {})}
    with st.form("laura_funding_form"):
        allocation = st.data_editor(frame, disabled=["Asset"], hide_index=True, use_container_width=True,
                                    key="laura_allocation_editor")
        a, b, c = st.columns(3)
        discount = a.number_input("Reserve discount (%) · team assumption", -50.0, 50.0, float(defaults["reserve_discount_rate"] * 100))
        buffer = b.number_input("Reserve margin (%)", 0.0, 100.0, float(defaults["reserve_buffer"] * 100))
        share = c.number_input("Facility share of surplus (%)", 0.0, 100.0, float(defaults["facility_surplus_share"] * 100))
        minimum = a.number_input("Minimum flexibility (USD)", min_value=0.0, value=float(defaults["minimum_flexibility"]), step=10_000.0)
        growth_fee = b.number_input("Annual accumulation costs (%)", 0.0, 20.0, float(defaults["accumulation_fee"] * 100))
        reserve_fee = c.number_input("Annual reserve costs (%)", 0.0, 20.0, float(defaults["reserve_fee"] * 100))
        confidence = a.number_input("Team definition of high confidence (%)", 50.0, 99.9, float(defaults["confidence_target"] * 100))
        st.caption("Costs are additional to costs already embedded in return data. Avoid charging fund fees twice.")
        rationale = st.text_area("Reserve instruments, liquidity, flexibility and allocation rationale", value=str(saved.get("rationale", "")))
        # This state is explicit and fixed before the two-year projection; no hidden future-path selection.
        opening = st.number_input("Modeled portfolio at 1 January 2031 (USD)", min_value=0.0,
                                  value=float(saved.get("opening_2031", 300_000 * 1.06**4 + 150_000 * 1.06**3)), step=10_000.0)
        state = st.text_input("2031 state and evidence", value=str(saved.get("state_description", "Illustrative future state using 6% accumulation; not an observed 2031 balance")))
        confirmed = st.checkbox("I confirm that these are aligned annual total returns in nominal USD", value=False)
        run = st.form_submit_button("Save assumptions and calculate", type="primary")
    if run:
        try:
            if not confirmed or not state.strip() or not rationale.strip():
                raise ValueError("Confirm the USD return basis and document the 2031 state and allocation rationale.")
            policy = LauraPolicy(discount / 100, buffer / 100, share / 100, minimum,
                                 growth_fee / 100, reserve_fee / 100, confidence / 100)
            computed = build_funding_run(annual, allocation, policy, opening_2031=opening, state_description=state,
                                         candidates=_goal_candidate_weights(result, columns))
            saved_values = {"policy": asdict(policy), "allocation": allocation.to_dict(orient="records"),
                            "opening_2031": opening, "state_description": state, "rationale": rationale,
                            "last_run": {"metadata": computed["metadata"], "summary": computed["projection"].summary(),
                                         "partner_interval": computed["interval"]}}
            with get_connection() as conn:
                save_case_section(conn, "planning", saved_values, updated_by=str(profile["username"]))
            st.session_state["laura_funding_run"] = computed
            st.success("Assumptions and calculation summary saved with the client mandate.")
        except (TypeError, ValueError, KeyError) as exc:
            st.error(str(exc))
            return
    computed = st.session_state.get("laura_funding_run")
    if computed:
        _render_results(computed)
    elif saved.get("last_run"):
        st.caption("A previous calculation is saved. Recalculate to see trajectories or export a new evidence package.")
        st.json(saved["last_run"])


def _render_results(run: Mapping[str, Any]) -> None:
    projection = run["projection"]
    summary = projection.summary()
    st.caption("Last calculated inputs are shown below. Save and calculate again to apply form or Quant Engine changes.")
    a, b, c = st.columns(3)
    a.metric("Reserve affordable in 2033", f"{summary['reserve_funding_probability']:.1%}")
    b.metric("All ten payments on time", f"{summary['all_payments_probability']:.1%}")
    c.metric("Reserve, flexibility and payments met", f"{summary['plan_success_probability']:.1%}")
    lo, hi = summary["mc_sampling_interval_95"]
    conditional = summary["payments_success_given_funded_reserve"]
    st.caption(f"95% Monte Carlo sampling interval for all payments: {lo:.1%}–{hi:.1%}. This measures simulation sampling error only. "
               f"Payment success conditional on funding the reserve: {conditional:.1%}." if conditional is not None else
               "No scenario fully funded the reserve; conditional payment success is unavailable.")
    if not summary["meets_team_confidence_target"]:
        st.warning("This policy misses the team's confidence target. Reducing the facility contribution alone may not resolve an underfunded or risky reserve.")
    a.metric("Mean unpaid amount when payments fail", f"${summary['conditional_mean_unpaid']:,.0f}")
    b.metric("Median facility contribution", f"${summary['median_facility']:,.0f}")
    c.metric("Median retained flexibility", f"${summary['median_flexibility']:,.0f}")
    payment_table = pd.DataFrame({"Year": range(2033, 2043), "Due USD": 50_000,
                                 "P(payment in full)": (projection.unpaid <= .01).mean(axis=0),
                                 "Mean unpaid USD": projection.unpaid.mean(axis=0),
                                 "First shortfall scenarios": [summary["first_shortfall_counts"][str(y)] for y in range(2033, 2043)]})
    st.dataframe(payment_table.style.format({"Due USD": "${:,.0f}", "P(payment in full)": "{:.1%}",
                                             "Mean unpaid USD": "${:,.0f}"}), hide_index=True, use_container_width=True)
    st.caption("Per-year payment rates above are distinct from joint success across all ten years.")
    balance_bands = pd.DataFrame(np.quantile(projection.reserve_before_payment, [.1, .5, .9], axis=0).T,
                                 index=pd.Index(range(2033, 2043), name="Payment year"),
                                 columns=["10th percentile", "Median", "90th percentile"])
    st.line_chart(balance_bands, y_label="Reserve before payment (USD)")
    st.caption("Pointwise reserve percentiles before each payment; the lines are not individual scenario paths.")
    with st.expander("Common-path capital allocation and sensitivity"):
        comparison_columns = ["Portfolio", "reserve_funding_probability", "all_payments_probability", "plan_success_probability", "median_facility"]
        st.dataframe(pd.DataFrame(run["comparison_rows"])[comparison_columns], hide_index=True, use_container_width=True)
        st.caption("Comparison portfolios keep constant accumulation weights in 2027–2032 and share the same staged reserve policy and sampled years. Optimized candidates are exploratory, fitted to the same history.")
        st.dataframe(pd.DataFrame(projection.scenario_rows()[:100]), hide_index=True, use_container_width=True)
        stress_columns = ["Stress assumption", "reserve_funding_probability", "all_payments_probability", "plan_success_probability", "conditional_mean_unpaid", "median_facility"]
        st.dataframe(pd.DataFrame(run["stress_rows"])[stress_columns], hide_index=True, use_container_width=True)
        st.caption("Sensitivity assumptions are deliberately adverse, not probability-weighted forecasts. The reserve price and realized reserve returns remain separate inputs.")
    interval = run["interval"]
    st.markdown("#### Conditional partner communication · valuation 1 January 2031")
    st.write(run["metadata"]["valuation_2031"]["state_description"])
    st.write(f"5th–95th percentile range: **{interval['lower_usd']:,.0f}–{interval['upper_usd']:,.0f} USD**. "
             f"Nominal quantile coverage: 90%; observed model coverage: {interval['empirical_interval_coverage']:.1%}; "
             f"below the lower bound: {interval['below_lower_probability']:.1%}. Ties at zero can increase coverage.")
    st.write(f"Joint chance of this range and all ten payments: {interval['interval_and_all_payments_probability']:.1%}. "
             f"Conditional-plan success: {interval['plan_success_probability']:.1%}.")
    st.text_area("Fundraising working draft", run["fundraising_draft"], height=210, disabled=True)
    if interval["plan_success_probability"] < projection.policy.confidence_target:
        st.warning("The conditional plan misses the team's threshold. This range is not ready to communicate as a supported contribution expectation.")
    with st.expander("Saved inputs, history and model limits"):
        st.json(run["metadata"])
    export = {"metadata": run["metadata"], "summary": summary, "payment_summary": payment_table.to_dict(orient="records"),
              "scenarios": projection.scenario_rows(), "partner_interval": interval,
              "partner_scenarios": run["conditional"].scenario_rows(),
              "stress_results": run["stress_rows"], "portfolio_comparison": run["comparison_rows"], "fundraising_draft": run["fundraising_draft"],
              "sampled_year_indices": run["sampled_indices"].tolist(),
              "conditional_sampled_year_indices": run["conditional_indices"].tolist()}
    _download("Download reproducible planning evidence", export, "laura_funding_evidence.json", "laura_funding_export")
    st.download_button("Download fundraising working draft", run["fundraising_draft"], "laura_fundraising_draft.txt", key="laura_draft_export")


def render_deliverables(profile: Mapping[str, Any], get_connection, saved: Mapping[str, Any]) -> None:
    deadlines = pd.DataFrame(deliverables_calendar()).rename(columns={
        "deliverable": "Deliverable", "deadline_et": "Deadline ET", "deadline_prague": "Deadline Prague",
        "requirements": "Requirements", "source": "Source",
    })
    st.dataframe(deadlines, hide_index=True, use_container_width=True)
    st.caption("Deadlines: supplied assessment pp. 10–11; all are 17:00 ET / 23:00 Prague. Submit through SurveyMonkey Apply. The IPS deadline also ends trading and locks the strategy. Final Report instructions are expected 9 November; verify the final format and school template when supplied.")
    with st.form("laura_submission_form"):
        notes = []
        existing = saved.get("notes", [])
        for i in range(3):
            row = existing[i] if len(existing) > i else {}
            st.markdown(f"#### Trading Note {i + 1}")
            notes.append({
                "execution_reference": st.text_input("Executed trade / evidence reference", value=str(row.get("execution_reference", "")), key=f"laura_trade_ref_{i}"),
                "note": st.text_area("Original WInS note (verbatim)", value=str(row.get("note", "")), key=f"laura_trade_note_{i}"),
                "reflection": st.text_area("Reflection (up to 100 words)", value=str(row.get("reflection", "")), key=f"laura_reflection_{i}"),
                "verbatim_confirmed": st.checkbox("Team verified execution and exact original wording", value=bool(row.get("verbatim_confirmed", False)), key=f"laura_trade_confirmed_{i}"),
            })
        pitch = st.text_area("Elevator pitch (up to 50 words)", value=str(saved.get("pitch", "")))
        ips = st.text_area("IPS strategy (up to 500 words)", value=str(saved.get("ips", "")), height=250)
        submitted = st.form_submit_button("Save drafts and check limits")
    if submitted:
        saved = {"notes": notes, "pitch": pitch, "ips": ips}
        with get_connection() as conn:
            save_case_section(conn, "deliverables", saved, updated_by=str(profile["username"]))
        st.success("Drafts saved. No submission has been sent.")
    if saved:
        checks = check_trading_notes(saved.get("notes", [])) + check_ips_text(str(saved.get("pitch", "")), str(saved.get("ips", "")))
        st.dataframe(pd.DataFrame(checks), hide_index=True, use_container_width=True)
        _download("Download submission working drafts", saved, "laura_submission_drafts.json", "laura_submissions_export")
    st.info("IPS file review: cover ≤1 page with official team name, first names + surname initials and WInS username; strategy ≤2 pages; Times New Roman 12, double spacing, 1-inch margins; PDF ≤5 MB. No graphics, appendices, external links, footnotes or formal citations. Text checks do not verify PDF formatting, executed trades or strategy consistency.")
    st.caption("Use the existing Strategy Rulebook, Security Dossiers and Report Evidence Studio to preserve the same strategy and research evidence across all three graded outputs. No final client numbers are required in Trading Notes or IPS.")


def render_laura_plan(profile: Mapping[str, Any], result: Mapping[str, Any], get_connection) -> None:
    from src.portfolio_tracker.strategy_store import load_client_mandate

    st.markdown("### Laura Gao · Client Plan")
    render_case_brief()
    with get_connection() as conn:
        record = load_client_mandate(conn)
    saved = ((record or {}).get("payload") or {}).get("laura_case") or {}
    view = st.radio("Client plan view", ["Case & Cash Flows", "Funding Model", "Submissions"], horizontal=True, key="laura_plan_view")
    if view == "Case & Cash Flows":
        _render_case()
    elif view == "Funding Model":
        _render_funding(result, profile, get_connection, saved.get("planning") or {})
    else:
        render_deliverables(profile, get_connection, saved.get("deliverables") or {})
