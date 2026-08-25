"""Read-only competition review and complete app record for the judge role."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

import pandas as pd
import streamlit as st


_MISSING = "\u2014"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _payload(value: Any) -> dict[str, Any]:
    record = _mapping(value)
    nested = record.get("payload")
    return dict(nested) if isinstance(nested, Mapping) else record


def _items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _first(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def _display(value: Any, fallback: str = _MISSING) -> str:
    if value is None or value == "":
        return fallback
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, Mapping):
        parts = [f"{key}: {_display(item)}" for key, item in value.items() if item not in (None, "")]
        return ", ".join(parts) or fallback
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [_display(item, "") for item in value]
        return ", ".join(part for part in parts if part) or fallback
    return str(value)


def _label(value: Any, fallback: str = _MISSING) -> str:
    text = _display(value, fallback).strip()
    return text.replace("_", " ").replace("-", " ").title() if text else fallback


def _safe_markup(template: str, **values: Any) -> str:
    """Interpolate model values into HTML only after escaping every value."""
    return template.format(**{key: escape(_display(value), quote=True) for key, value in values.items()})


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score(value: Any) -> str:
    number = _number(value)
    if number is None:
        return _display(value)
    if isinstance(value, float) and not value.is_integer() and -1.0 <= number <= 1.0:
        number *= 100.0
    return f"{number:.0f}/100"


def _percent(value: Any, *, fraction: bool = False) -> str:
    number = _number(value)
    if number is None:
        return _display(value)
    if fraction and -1.0 <= number <= 1.0:
        number *= 100.0
    return f"{number:.1f}%"


def _money(value: Any) -> str:
    number = _number(value)
    if number is None:
        return _display(value)
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.0f}"


def _count(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    number = _number(value)
    return max(0, int(number)) if number is not None else 0


def _section(kicker: str, title: str, description: str) -> None:
    st.markdown(
        _safe_markup(
            "<div class='judge-section-head'>"
            "<span>{kicker}</span><h2>{title}</h2><p>{description}</p></div>",
            kicker=kicker,
            title=title,
            description=description,
        ),
        unsafe_allow_html=True,
    )


def _empty(message: str) -> None:
    st.markdown(
        _safe_markup("<div class='judge-empty'>{message}</div>", message=message),
        unsafe_allow_html=True,
    )


def _fact(label: str, value: Any) -> None:
    st.markdown(
        _safe_markup(
            "<div class='judge-fact'><span>{label}</span><strong>{value}</strong></div>",
            label=label,
            value=value,
        ),
        unsafe_allow_html=True,
    )


def _app_module(model: Mapping[str, Any], key: str) -> dict[str, Any]:
    record = _mapping(model.get("app_record"))
    for module in _items(record.get("modules")):
        if str(module.get("key") or "") == key:
            return module
    return {}


def _journey_stage(
    index: int,
    label: str,
    description: str,
    status: str,
    evidence: str,
    tone: str,
) -> dict[str, str]:
    return {
        "index": f"{index:02d}",
        "label": label,
        "description": description,
        "status": status,
        "evidence": evidence,
        "tone": tone if tone in {"good", "watch", "open", "neutral"} else "neutral",
    }


def _journey_stages(model: Mapping[str, Any]) -> list[dict[str, str]]:
    """Describe the saved investment journey without inventing an official score."""
    mandate = _payload(model.get("mandate"))
    strategy = _payload(model.get("strategy"))
    readiness = _mapping(model.get("readiness"))
    portfolio = _mapping(model.get("portfolio"))
    dossiers = _items(model.get("dossiers") or model.get("investment_cases"))
    decisions = _mapping(model.get("decisions"))
    evidence = _mapping(model.get("evidence"))
    report = _mapping(model.get("report"))
    compliance = _mapping(model.get("compliance"))
    integrity = _mapping(model.get("integrity"))
    reconciliation = _mapping(integrity.get("reconciliation"))

    mandate_available = bool(mandate.get("available", mandate))
    strategy_available = bool(strategy.get("available", strategy))
    source_count = _count(evidence.get("source_count"))
    decision_count = _count(decisions.get("count"))
    fully_voted = _count(decisions.get("fully_voted_count"))
    risk_records = _count(_app_module(model, "risk_scenarios").get("record_count"))
    review_count = _count(evidence.get("thesis_review_count"))
    report_available = bool(report.get("available", report))
    check_count = _count(compliance.get("check_count"))
    qa_rounds = _count(evidence.get("qa_round_count"))

    if dossiers:
        research_status, research_tone = "Recorded", "good"
    elif source_count:
        research_status, research_tone = "Building", "watch"
    else:
        research_status, research_tone = "Not started", "open"

    if decision_count and fully_voted >= decision_count:
        committee_status, committee_tone = "Fully voted", "good"
    elif decision_count:
        committee_status, committee_tone = "In progress", "watch"
    else:
        committee_status, committee_tone = "Not started", "open"

    if portfolio.get("reconciled"):
        execution_status, execution_tone = "Reconciled", "good"
    elif portfolio.get("available", bool(portfolio)):
        execution_status, execution_tone = "Provisional", "watch"
    else:
        execution_status, execution_tone = "Not available", "open"

    if report.get("validation_ready") is True:
        report_status, report_tone = "Validated", "good"
    elif report_available:
        report_status, report_tone = "In review", "watch"
    else:
        report_status, report_tone = "Not started", "open"

    if compliance.get("all_clear") is True:
        control_status, control_tone = "Clear", "good"
    elif check_count:
        control_status, control_tone = "Review needed", "watch"
    else:
        control_status, control_tone = "Not assessed", "open"

    return [
        _journey_stage(
            1,
            "Client mandate",
            "Goals, horizon, liquidity and risk capacity define success.",
            "Recorded" if mandate_available else "Not started",
            (
                f"Constitution {readiness.get('constitution_score', 0)}/100"
                if readiness.get("available")
                else "No mandate evidence"
            ),
            "good" if mandate_available else "open",
        ),
        _journey_stage(
            2,
            "Strategy",
            "The rulebook turns client needs into repeatable selection and sizing rules.",
            "Recorded" if strategy_available else "Not started",
            _display(_first(strategy, "name", "strategy_name"), "No active strategy"),
            "good" if strategy_available else "open",
        ),
        _journey_stage(
            3,
            "Research",
            "Eligible securities receive an evidence-backed thesis and downside case.",
            research_status,
            f"{len(dossiers)} cases · {source_count} sources",
            research_tone,
        ),
        _journey_stage(
            4,
            "Risk tests",
            "Quant runs and decision snapshots challenge the portfolio before capital moves.",
            "Recorded" if risk_records else "No saved run",
            f"{risk_records} saved analytical record{'s' if risk_records != 1 else ''}",
            "good" if risk_records else "open",
        ),
        _journey_stage(
            5,
            "Committee",
            "Blind votes, discussion, approvals and audit history authorize the decision.",
            committee_status,
            f"{fully_voted}/{decision_count} cases fully voted" if decision_count else "No committee case",
            committee_tone,
        ),
        _journey_stage(
            6,
            "Execute & reconcile",
            "Approved sizing reaches WInS, then reconciles into the authoritative portfolio.",
            execution_status,
            (
                f"Snapshot {_display(portfolio.get('snapshot_id'))}"
                if portfolio.get("available", bool(portfolio))
                else _display(reconciliation.get("status"), "No canonical snapshot")
            ),
            execution_tone,
        ),
        _journey_stage(
            7,
            "Monitor & learn",
            "Holdings stay tied to goals, thesis checks and explicit review evidence.",
            "Active" if portfolio.get("available", bool(portfolio)) else "Not started",
            f"{_count(portfolio.get('positions'))} holdings · {review_count} thesis reviews",
            "good" if portfolio.get("available", bool(portfolio)) and review_count else "watch" if portfolio.get("available", bool(portfolio)) else "open",
        ),
        _journey_stage(
            8,
            "Report & defend",
            "The final narrative binds claims, evidence and Q&A to the same portfolio snapshot.",
            report_status,
            f"{_label(_first(report, 'status', default='Not started'))} · {qa_rounds} Q&A rounds",
            report_tone,
        ),
        _journey_stage(
            9,
            "Controls",
            "Rules, disclosures and reconciliation prove the process held from end to end.",
            control_status,
            f"{_count(compliance.get('pass_count'))}/{check_count} checks passed" if check_count else "No saved checks",
            control_tone,
        ),
    ]


def _render_journey_row(stages: Sequence[Mapping[str, Any]]) -> None:
    padded = [dict(item) for item in stages[:3]]
    while len(padded) < 3:
        padded.append(_journey_stage(0, "", "", "", "", "neutral"))
    st.markdown(
        _safe_markup(
            "<div class='judge-journey-grid'>"
            "<article class='judge-journey-card is-{tone0}'><div class='judge-journey-top'>"
            "<span class='judge-journey-index'>{index0}</span><span class='judge-status'>{status0}</span></div>"
            "<h3>{label0}</h3><p>{description0}</p><small>{evidence0}</small></article>"
            "<article class='judge-journey-card is-{tone1}'><div class='judge-journey-top'>"
            "<span class='judge-journey-index'>{index1}</span><span class='judge-status'>{status1}</span></div>"
            "<h3>{label1}</h3><p>{description1}</p><small>{evidence1}</small></article>"
            "<article class='judge-journey-card is-{tone2}'><div class='judge-journey-top'>"
            "<span class='judge-journey-index'>{index2}</span><span class='judge-status'>{status2}</span></div>"
            "<h3>{label2}</h3><p>{description2}</p><small>{evidence2}</small></article>"
            "</div>",
            **{
                f"{key}{index}": stage.get(key)
                for index, stage in enumerate(padded)
                for key in ("tone", "index", "status", "label", "description", "evidence")
            },
        ),
        unsafe_allow_html=True,
    )


def _render_portfolio_journey(model: Mapping[str, Any]) -> None:
    st.markdown(
        "<div class='judge-journey-heading'><span>End-to-end evidence trail</span>"
        "<h2>Portfolio Journey</h2><p>Follow the capital in chronological order. "
        "Every status below comes from the saved, read-only competition record.</p></div>",
        unsafe_allow_html=True,
    )
    stages = _journey_stages(model)
    for start in range(0, len(stages), 3):
        _render_journey_row(stages[start : start + 3])


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --judge-ink:#102338;
            --judge-muted:#617184;
            --judge-line:#dbe4ea;
            --judge-soft:#f5f8f9;
            --judge-teal:#0d7a72;
            --judge-teal-bright:#2dd4bf;
            --judge-gold:#d7a84d;
            --judge-navy:#11283f;
        }
        [data-testid="stMainBlockContainer"] { padding-bottom:5rem !important; }
        .judge-hero {
            position:relative;
            overflow:hidden;
            margin:.15rem 0 1.55rem;
            padding:2.2rem 2.25rem;
            border:1px solid rgba(255,255,255,.09);
            border-radius:24px;
            color:#f8fafc;
            background:
                radial-gradient(circle at 93% 8%,rgba(45,212,191,.24),transparent 22rem),
                radial-gradient(circle at 4% 112%,rgba(215,168,77,.13),transparent 19rem),
                linear-gradient(132deg,#102238 0%,#173a50 62%,#12625e 132%);
            box-shadow:0 24px 60px rgba(15,35,55,.16);
        }
        .judge-hero:after {
            content:"";
            position:absolute;
            width:22rem;
            height:22rem;
            right:-13rem;
            bottom:-15rem;
            border:1px solid rgba(255,255,255,.15);
            border-radius:50%;
            box-shadow:0 0 0 3rem rgba(255,255,255,.025),0 0 0 7rem rgba(255,255,255,.018);
        }
        .judge-hero-grid {
            position:relative;
            z-index:1;
            display:grid;
            grid-template-columns:minmax(0,1fr) 15rem;
            align-items:center;
            gap:2.25rem;
        }
        .judge-hero-label {
            display:inline-flex;
            align-items:center;
            gap:.5rem;
            margin-bottom:.82rem;
            color:#9df6e8;
            font-size:.72rem;
            font-weight:850;
            letter-spacing:.14em;
            text-transform:uppercase;
        }
        .judge-hero-label:before {
            content:"";
            width:.52rem;
            height:.52rem;
            border-radius:50%;
            background:#2dd4bf;
            box-shadow:0 0 0 .25rem rgba(45,212,191,.12);
        }
        .judge-hero h1 {
            max-width:52rem;
            margin:0;
            color:#fff !important;
            font-size:clamp(2rem,3.5vw,3.05rem);
            line-height:1.03;
            letter-spacing:-.055em;
        }
        .judge-hero h1 span,.judge-hero h1 a { color:#fff !important; }
        .judge-hero p {
            max-width:48rem;
            margin:.85rem 0 0;
            color:rgba(241,245,249,.77);
            font-size:.98rem;
            line-height:1.6;
        }
        .judge-badges { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.25rem; }
        .judge-badge {
            padding:.38rem .68rem;
            border:1px solid rgba(226,232,240,.2);
            border-radius:999px;
            color:#e6f2f3;
            background:rgba(255,255,255,.07);
            font-size:.77rem;
        }
        .judge-score-panel {
            padding:1.2rem 1.15rem;
            border:1px solid rgba(255,255,255,.16);
            border-radius:19px;
            background:rgba(6,28,43,.27);
            backdrop-filter:blur(9px);
            text-align:center;
        }
        .judge-score-ring {
            display:grid;
            place-items:center;
            width:8.2rem;
            height:8.2rem;
            margin:0 auto .9rem;
            border-radius:50%;
            background:conic-gradient(#43ddc9 var(--score),rgba(255,255,255,.11) 0);
        }
        .judge-score-ring:before {
            content:"";
            grid-area:1/1;
            width:6.65rem;
            height:6.65rem;
            border-radius:50%;
            background:#17374a;
            box-shadow:inset 0 0 0 1px rgba(255,255,255,.08);
        }
        .judge-score-copy { position:relative; grid-area:1/1; }
        .judge-score-copy strong { display:block; color:#fff; font-size:2rem; line-height:1; }
        .judge-score-copy span { color:#a7c9ce; font-size:.7rem; font-weight:700; text-transform:uppercase; }
        .judge-score-panel>strong { display:block; color:#eafaf8; font-size:.83rem; }
        .judge-score-panel>small { display:block; margin-top:.35rem; color:#9ab7bd; font-size:.69rem; line-height:1.35; }
        .judge-journey-heading { margin:2.3rem 0 .9rem; }
        .judge-journey-heading>span,
        .judge-section-head>span {
            color:var(--judge-teal);
            font-size:.69rem;
            font-weight:850;
            letter-spacing:.145em;
            text-transform:uppercase;
        }
        .judge-journey-heading h2,
        .judge-section-head h2 {
            margin:.18rem 0 .22rem;
            color:var(--judge-ink);
            font-size:1.48rem;
            line-height:1.15;
            letter-spacing:-.035em;
        }
        .judge-journey-heading p,
        .judge-section-head p { max-width:62rem; margin:0; color:var(--judge-muted); font-size:.86rem; line-height:1.5; }
        .judge-journey-grid {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:.7rem;
            margin-bottom:.7rem;
        }
        .judge-journey-card {
            position:relative;
            min-height:11.2rem;
            padding:1rem 1.05rem .92rem;
            overflow:hidden;
            border:1px solid var(--judge-line);
            border-top:3px solid #9aa9b5;
            border-radius:15px;
            background:linear-gradient(155deg,#fff 0%,#f8fafb 100%);
            box-shadow:0 7px 22px rgba(16,35,56,.045);
        }
        .judge-journey-card:after {
            content:"";
            position:absolute;
            right:-2.2rem;
            bottom:-2.5rem;
            width:6.5rem;
            height:6.5rem;
            border:1px solid rgba(15,118,110,.08);
            border-radius:50%;
        }
        .judge-journey-card.is-good { border-top-color:#138a7e; }
        .judge-journey-card.is-watch { border-top-color:#d49b35; }
        .judge-journey-card.is-open { border-top-color:#a9b5bf; }
        .judge-journey-top { display:flex; align-items:center; justify-content:space-between; gap:.7rem; }
        .judge-journey-index {
            display:grid;
            place-items:center;
            width:1.9rem;
            height:1.9rem;
            border-radius:9px;
            color:#fff;
            background:var(--judge-navy);
            font-size:.68rem;
            font-weight:850;
            letter-spacing:.04em;
        }
        .judge-status {
            padding:.25rem .5rem;
            border-radius:999px;
            color:#526170;
            background:#eef2f4;
            font-size:.66rem;
            font-weight:800;
            letter-spacing:.025em;
        }
        .is-good .judge-status { color:#0a665e; background:#def5f0; }
        .is-watch .judge-status { color:#8a5c13; background:#fff1d4; }
        .judge-journey-card h3 { margin:.78rem 0 .35rem; color:var(--judge-ink); font-size:1rem; letter-spacing:-.02em; }
        .judge-journey-card p { margin:0; color:#657486; font-size:.77rem; line-height:1.48; }
        .judge-journey-card small { display:block; margin-top:.72rem; color:#293d50; font-size:.72rem; font-weight:750; }
        .judge-brief-grid {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:.7rem;
            margin:.35rem 0 1.2rem;
        }
        .judge-brief-card {
            min-height:7.2rem;
            padding:.92rem 1rem;
            border:1px solid var(--judge-line);
            border-radius:15px;
            background:#fff;
            box-shadow:0 6px 18px rgba(16,35,56,.04);
        }
        .judge-brief-card span { display:block; color:#748292; font-size:.67rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
        .judge-brief-card strong { display:block; margin:.48rem 0 .3rem; color:var(--judge-ink); font-size:1.13rem; line-height:1.2; letter-spacing:-.025em; }
        .judge-brief-card small { color:#69798a; font-size:.73rem; line-height:1.4; }
        .judge-section-head {
            position:relative;
            margin:3rem 0 1rem;
            padding:.12rem 0 .18rem 1rem;
            border-left:3px solid var(--judge-teal);
        }
        .judge-fact {
            min-height:5.4rem;
            margin-bottom:.7rem;
            padding:.88rem .95rem;
            border:1px solid var(--judge-line);
            border-radius:13px;
            background:linear-gradient(145deg,#fff,#fbfcfc);
            box-shadow:0 4px 14px rgba(16,35,56,.025);
        }
        .judge-fact span,
        .judge-mini-label {
            display:block;
            margin-bottom:.3rem;
            color:#718092;
            font-size:.66rem;
            font-weight:800;
            letter-spacing:.075em;
            text-transform:uppercase;
        }
        .judge-fact strong { color:var(--judge-ink); font-size:.88rem; font-weight:670; line-height:1.44; }
        .judge-source {
            padding:.92rem 1rem;
            margin-bottom:.85rem;
            border:1px solid #bfe1dc;
            border-left:4px solid var(--judge-teal);
            border-radius:13px;
            background:linear-gradient(90deg,#ecf9f6,#f8fcfb);
            color:#275b57;
            font-size:.83rem;
            line-height:1.45;
        }
        .judge-allocation-card {
            min-height:6.2rem;
            margin:.1rem 0 .8rem;
            padding:.88rem .92rem;
            border:1px solid var(--judge-line);
            border-radius:13px;
            background:#fff;
        }
        .judge-allocation-card header { display:flex; align-items:baseline; justify-content:space-between; gap:.6rem; }
        .judge-allocation-card h4 { margin:0; color:var(--judge-ink); font-size:1.02rem; }
        .judge-allocation-card b { color:var(--judge-teal); font-size:.82rem; }
        .judge-allocation-card p { margin:.45rem 0 0; color:#69798a; font-size:.74rem; line-height:1.4; }
        .judge-case {
            margin-bottom:.85rem;
            padding:1.15rem 1.2rem;
            border:1px solid var(--judge-line);
            border-radius:17px;
            background:#fff;
            box-shadow:0 8px 26px rgba(16,35,56,.045);
        }
        .judge-case-head { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; }
        .judge-case-title small { display:block; margin-bottom:.2rem; color:#758495; font-size:.64rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
        .judge-case h3 { margin:0; color:var(--judge-ink); font-size:1.18rem; letter-spacing:-.025em; }
        .judge-case-score { padding:.28rem .55rem; border-radius:999px; color:#086a62; background:#e1f6f2; font-size:.72rem; font-weight:850; }
        .judge-chip-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.72rem 0; }
        .judge-chip { padding:.27rem .52rem; border:1px solid #dce5e9; border-radius:999px; color:#536576; background:#f8fafb; font-size:.68rem; }
        .judge-case-thesis { margin:.72rem 0 .85rem; color:#263b4e; font-size:.91rem; font-weight:620; line-height:1.52; }
        .judge-case-grid,.judge-decision-grid {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:.55rem;
        }
        .judge-case-cell,.judge-decision-cell {
            min-height:4.65rem;
            padding:.68rem .72rem;
            border-radius:10px;
            background:#f5f8f9;
        }
        .judge-case-cell span,.judge-decision-cell span { display:block; margin-bottom:.22rem; color:#758495; font-size:.61rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; }
        .judge-case-cell strong,.judge-decision-cell strong { color:#2b4052; font-size:.74rem; font-weight:650; line-height:1.4; }
        .judge-case-downside { margin-top:.55rem; padding:.72rem .78rem; border-left:3px solid #d69b32; border-radius:0 10px 10px 0; background:#fff8e9; }
        .judge-case-downside span { color:#91621c; font-size:.64rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; }
        .judge-case-downside strong { display:block; margin-top:.2rem; color:#5e4a2b; font-size:.77rem; line-height:1.4; }
        .judge-case-meta { display:flex; flex-wrap:wrap; justify-content:space-between; gap:.55rem; margin-top:.7rem; color:#6d7d8d; font-size:.68rem; }
        .judge-decision {
            margin-bottom:.72rem;
            padding:1rem 1.05rem;
            border:1px solid var(--judge-line);
            border-radius:15px;
            background:linear-gradient(145deg,#fff,#fafcfc);
        }
        .judge-decision-head { display:flex; align-items:center; justify-content:space-between; gap:.8rem; margin-bottom:.75rem; }
        .judge-decision-head h3 { margin:0; color:var(--judge-ink); font-size:1.02rem; }
        .judge-stage-pill { padding:.29rem .55rem; border-radius:999px; color:#fff; background:var(--judge-navy); font-size:.67rem; font-weight:800; }
        .judge-decision-proposal { margin:0 0 .75rem; color:#536576; font-size:.79rem; line-height:1.45; }
        .judge-module-card {
            min-height:6.3rem;
            margin-bottom:.7rem;
            padding:.85rem .9rem;
            border:1px solid var(--judge-line);
            border-radius:13px;
            background:#fff;
        }
        .judge-module-card.is-used { border-color:#bfe0da; background:#f4fbf9; }
        .judge-module-card span { color:#738293; font-size:.64rem; font-weight:800; letter-spacing:.075em; text-transform:uppercase; }
        .judge-module-card h4 { margin:.35rem 0 .25rem; color:var(--judge-ink); font-size:.88rem; }
        .judge-module-card p { margin:0; color:#6b7b8b; font-size:.7rem; line-height:1.4; }
        .judge-empty {
            padding:1rem 1.05rem;
            border:1px dashed #bdc9d2;
            border-radius:12px;
            color:#687889;
            background:#f7f9fa;
            font-size:.84rem;
        }
        .judge-question {
            display:grid;
            grid-template-columns:2.15rem minmax(0,1fr);
            align-items:start;
            gap:.8rem;
            margin-bottom:.62rem;
            padding:.9rem 1rem;
            border:1px solid #d8e6e4;
            border-radius:13px;
            background:linear-gradient(105deg,#f0faf8,#fff);
            color:#24394c;
            font-size:.86rem;
            line-height:1.5;
        }
        .judge-question strong { display:grid; place-items:center; width:2rem; height:2rem; border-radius:9px; color:#fff; background:var(--judge-teal); font-size:.72rem; }
        div[data-testid="stDataFrame"] { margin:.35rem 0 .75rem; border:1px solid var(--judge-line); border-radius:13px; overflow:hidden; }
        div[data-testid="stExpander"] { margin-bottom:.5rem; border-color:var(--judge-line); border-radius:12px; background:#fff; }
        div[data-testid="stMetric"] { min-height:7rem; border-radius:15px !important; box-shadow:0 6px 18px rgba(16,35,56,.035) !important; }
        div[data-testid="stProgress"] { margin:.7rem 0 .25rem; }
        @media (max-width:1000px) {
            .judge-hero-grid { grid-template-columns:1fr 12rem; gap:1.4rem; }
            .judge-brief-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .judge-case-grid,.judge-decision-grid { grid-template-columns:1fr; }
        }
        @media (max-width:760px) {
            .judge-hero { padding:1.45rem 1.25rem; border-radius:18px; }
            .judge-hero-grid { grid-template-columns:1fr; }
            .judge-score-panel { display:flex; align-items:center; gap:1rem; text-align:left; }
            .judge-score-ring { width:5.2rem; height:5.2rem; flex:0 0 5.2rem; margin:0; }
            .judge-score-ring:before { width:4.25rem; height:4.25rem; }
            .judge-score-copy strong { font-size:1.35rem; }
            .judge-journey-grid,.judge-brief-grid { grid-template-columns:1fr; }
            .judge-journey-card { min-height:auto; }
            .judge-section-head { margin-top:2.35rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(model: Mapping[str, Any]) -> None:
    readiness = _mapping(model.get("readiness"))
    portfolio = _mapping(model.get("portfolio"))
    report = _mapping(model.get("report"))
    dossiers = _items(model.get("dossiers") or model.get("investment_cases"))
    decisions = _mapping(model.get("decisions"))
    readiness_value = _first(readiness, "overall_score", "score", "readiness_score")
    report_status = _first(report, "status_label", "status", default="Not prepared")
    portfolio_available = bool(portfolio.get("available", portfolio))
    source_status = (
        "Reconciled authority"
        if portfolio.get("reconciled")
        else "Provisional tracker"
        if portfolio_available
        else "No snapshot"
    )
    readiness_label = (
        _score(readiness_value)
        if readiness.get("available", bool(readiness))
        else "Not assessed"
    )
    decision_count = _count(decisions.get("count"))
    fully_voted = _count(decisions.get("fully_voted_count"))
    report_note = (
        f"{_count(report.get('ready_section_count'))}/{_count(report.get('section_count'))} sections ready"
        if report.get("available", bool(report))
        else "No report workspace saved"
    )
    st.markdown(
        _safe_markup(
            "<div class='judge-brief-grid'>"
            "<article class='judge-brief-card'><span>Internal readiness</span>"
            "<strong>{readiness}</strong><small>{readiness_note}</small></article>"
            "<article class='judge-brief-card'><span>Portfolio truth</span>"
            "<strong>{authority}</strong><small>{authority_note}</small></article>"
            "<article class='judge-brief-card'><span>Decision evidence</span>"
            "<strong>{cases}</strong><small>{decision_note}</small></article>"
            "<article class='judge-brief-card'><span>Submission</span>"
            "<strong>{report}</strong><small>{report_note}</small></article>"
            "</div>",
            readiness=readiness_label,
            readiness_note=_first(readiness, "status", default="Returns excluded"),
            authority=source_status,
            authority_note=(
                f"Snapshot {_display(portfolio.get('snapshot_id'))}"
                if portfolio_available
                else "No authoritative capital record"
            ),
            cases=f"{len(dossiers)} dossiers",
            decision_note=(
                f"{fully_voted}/{decision_count} committee cases fully voted"
                if decision_count
                else "No committee lifecycle recorded"
            ),
            report=_label(report_status),
            report_note=report_note,
        ),
        unsafe_allow_html=True,
    )

    score_number = _number(readiness_value)
    if score_number is not None and readiness.get("available", bool(readiness)):
        normalized = score_number if score_number <= 1 else score_number / 100.0
        st.progress(max(0.0, min(1.0, normalized)), text=_display(readiness.get("status"), "Readiness"))
        st.caption(
            "Internal preparation diagnostic, not an official Wharton score. "
            "Portfolio returns are excluded."
        )


def _render_client_and_strategy(model: Mapping[str, Any]) -> None:
    mandate = _payload(model.get("mandate") or model.get("client"))
    strategy = _payload(model.get("strategy"))
    _section(
        "01 · Mandate",
        "Client & Strategy",
        "The client objective, constraints, and the investment process used to serve them.",
    )
    mandate_available = bool(mandate.get("available", mandate))
    strategy_available = bool(strategy.get("available", strategy))
    if not mandate_available and not strategy_available:
        _empty("The client mandate and strategy have not been recorded yet.")
        return

    left, right = st.columns(2)
    with left:
        _fact("Client", _first(mandate, "client_name", "name"))
        _fact("Mandate", _first(mandate, "mandate_summary", "summary"))
        _fact("Goals", _first(mandate, "goals", "client_goals", "primary_goal"))
        _fact(
            "Risk tolerance / capacity",
            " / ".join(
                (
                    _display(_first(mandate, "risk_tolerance")),
                    _display(_first(mandate, "risk_capacity")),
                )
            ),
        )
        horizon = _first(mandate, "horizon_years")
        liquidity = _first(mandate, "liquidity_need_pct")
        _fact(
            "Horizon / near-term liquidity",
            {
                "horizon": f"{_display(horizon)} years" if horizon is not None else None,
                "liquidity": _percent(liquidity) if liquidity is not None else None,
            },
        )
        drawdown = _first(mandate, "max_tolerated_drawdown", "max_drawdown_pct")
        _fact(
            "Maximum tolerated drawdown",
            _percent(drawdown) if drawdown is not None else None,
        )
        _fact("Client constraints", _first(mandate, "constraints"))
        _fact("Policy benchmark", _first(mandate, "policy_benchmark", "benchmark"))
        _fact(
            "Why this benchmark",
            _first(mandate, "policy_benchmark_rationale", "benchmark_rationale"),
        )
    with right:
        _fact("Strategy", _first(strategy, "name", "strategy_name"))
        _fact(
            "Investment thesis",
            _first(strategy, "thesis", "strategy_thesis", "one_sentence_thesis"),
        )
        _fact("Process", _first(strategy, "process", "investment_process"))
        _fact("Selection factors", _first(strategy, "selection_factors", "factors"))
        guardrails = {
            "max position": _percent(
                _first(strategy, "max_position_weight", "max_position_pct"),
                fraction=True,
            ),
            "max sector": _percent(
                _first(strategy, "max_sector_weight", "max_sector_pct"),
                fraction=True,
            ),
            "min cash": _percent(_first(strategy, "min_cash_weight"), fraction=True),
            "max cash": _percent(_first(strategy, "max_cash_weight"), fraction=True),
            "drift": _percent(
                _first(
                    strategy,
                    "drift_limit",
                    "max_drift",
                    "drift_tolerance_pct",
                ),
                fraction=True,
            ),
        }
        _fact(
            "Portfolio guardrails",
            {
                key: value
                for key, value in guardrails.items()
                if value not in (None, "", _MISSING)
            },
        )
        _fact("Sell discipline", _first(strategy, "sell_discipline", "sell_rules"))
        _fact("Rebalancing policy", _first(strategy, "rebalance_policy"))


def _position_rows(portfolio: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _items(portfolio.get("positions") or portfolio.get("holdings")):
        weight_key = "weight_pct" if item.get("weight_pct") is not None else "weight"
        valuation_source = _first(item, "valuation_source")
        price_source = _first(item, "price_source")
        pricing = valuation_source or price_source
        if (
            valuation_source not in (None, "")
            and price_source not in (None, "")
            and str(valuation_source).casefold() != str(price_source).casefold()
        ):
            pricing = f"{valuation_source} ({price_source})"
        rows.append(
            {
                "Ticker": _display(_first(item, "ticker", "symbol", "security")),
                "Type": _display(_first(item, "security_type", "asset_type", "type")),
                "Weight": _percent(item.get(weight_key), fraction=weight_key == "weight"),
                "Market value": _money(_first(item, "market_value", "current_value", "value")),
                "Return": _percent(_first(item, "return_pct", "total_return_pct")),
                "P&L": _money(_first(item, "pnl", "unrealized_pnl", "profit_loss")),
                "Pricing": _display(pricing),
                "Price as of": _display(item.get("price_observed_at")),
                "Lifecycle": _label(
                    _first(item, "lifecycle_state", "lifecycle", "stage", "status")
                ),
                "Client goal": _display(_first(item, "client_goal", "goal")),
            }
        )
    return rows


def _position_weight_pct(item: Mapping[str, Any]) -> float | None:
    explicit = _number(item.get("weight_pct"))
    if explicit is not None:
        return explicit
    fraction = _number(item.get("weight"))
    return fraction * 100.0 if fraction is not None else None


def _render_portfolio(model: Mapping[str, Any]) -> None:
    portfolio = _mapping(model.get("portfolio"))
    _section(
        "05 · Capital",
        "Portfolio",
        "What the team owns now, where the marks came from, and how each position remains tied to the client mandate.",
    )
    if not portfolio.get("available", bool(portfolio)):
        _empty("No portfolio snapshot is available yet. Strategy evidence can still be reviewed above.")
        return

    reporting_ready = portfolio.get("reporting_ready")
    if portfolio.get("reconciled"):
        if portfolio.get("last_known_good"):
            authority_state = "Reconciled last-known-good · reporting blocked"
        elif reporting_ready is True:
            authority_state = "Reconciled · report-ready"
        elif reporting_ready is False:
            authority_state = "Reconciled · reporting blocked"
        else:
            authority_state = "Reconciled snapshot"
    else:
        authority_state = "Provisional view"
    st.markdown(
        _safe_markup(
            "<div class='judge-source'><strong>{source}</strong> · Snapshot {snapshot} · "
            "As of {as_of} · {state}</div>",
            source=_first(portfolio, "source_label", "source", default="Source not identified"),
            snapshot=_first(portfolio, "snapshot_id", "id"),
            as_of=_first(portfolio, "as_of", "as_of_date", "updated_at"),
            state=authority_state,
        ),
        unsafe_allow_html=True,
    )
    blockers = portfolio.get("reporting_blockers")
    if reporting_ready is False and isinstance(blockers, Sequence) and blockers:
        blocker_labels = [str(item).replace("_", " ") for item in blockers]
        st.warning(f"Reporting is currently blocked: {_display(blocker_labels)}.")
    valuation_sources = portfolio.get("valuation_sources")
    if isinstance(valuation_sources, Sequence) and valuation_sources:
        st.caption(f"Tracker pricing sources: {_display(valuation_sources)}.")
    if portfolio.get("valuation_note"):
        st.warning(str(portfolio["valuation_note"]))
    metrics = st.columns(4)
    equity_value = _first(portfolio, "equity", "equity_value")
    value_label = "Equity value"
    if equity_value is None:
        equity_value = _first(portfolio, "invested_value")
        value_label = "Invested value"
    metrics[0].metric(value_label, _money(equity_value))
    metrics[1].metric("Cash", _money(_first(portfolio, "cash", "cash_value")))
    metrics[2].metric("Total return", _percent(_first(portfolio, "total_return_pct", "return_pct")))
    position_rows = _position_rows(portfolio)
    metrics[3].metric("Holdings", str(len(position_rows)))
    if position_rows:
        raw_positions = _items(portfolio.get("positions") or portfolio.get("holdings"))
        ranked_positions = sorted(
            raw_positions,
            key=lambda item: _position_weight_pct(item) or 0.0,
            reverse=True,
        )[:3]
        if ranked_positions:
            st.caption("Largest capital expressions")
            allocation_columns = st.columns(len(ranked_positions))
            for column, item in zip(allocation_columns, ranked_positions):
                column.markdown(
                    _safe_markup(
                        "<article class='judge-allocation-card'><header><h4>{ticker}</h4>"
                        "<b>{weight}</b></header><p>{value} · {goal}<br>{lifecycle}</p></article>",
                        ticker=_first(item, "ticker", "symbol", default="Holding"),
                        weight=_percent(_position_weight_pct(item)),
                        value=_money(_first(item, "market_value", "current_value", "value")),
                        goal=_first(item, "client_goal", "goal", default="Client goal not linked"),
                        lifecycle=_label(
                            _first(
                                item,
                                "lifecycle_state",
                                "lifecycle",
                                "stage",
                                "status",
                                default="Lifecycle not recorded",
                            )
                        ),
                    ),
                    unsafe_allow_html=True,
                )
        st.dataframe(pd.DataFrame(position_rows), hide_index=True, width="stretch")
    else:
        _empty("This snapshot does not contain any holdings.")


def _case_details(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    thesis = _first(item, "thesis", "investment_thesis", default="Thesis not documented")
    nested_valuation = _first(item, "fair_values", "fair_value")
    valuation = nested_valuation if isinstance(nested_valuation, Mapping) else {
        "bear": _first(item, "fair_value_bear", "bear_value"),
        "base": _first(item, "fair_value_base", "base_value", "fair_value"),
        "bull": _first(item, "fair_value_bull", "bull_value"),
    }
    valuation = {key: value for key, value in _mapping(valuation).items() if value not in (None, "")}
    valuation_currency = _display(item.get("fair_value_currency"), "")
    valuation_text = _display(valuation)
    if valuation_currency:
        valuation_text = f"{valuation_text} {valuation_currency}"
    score = _score(_first(item, "readiness_score", "score"))
    return (
        _display(thesis),
        valuation_text,
        score,
        _display(_first(item, "ticker", "symbol", default="Untitled case")),
    )


def _render_investment_cases(model: Mapping[str, Any]) -> None:
    dossiers = _items(model.get("dossiers") or model.get("investment_cases"))
    _section(
        "02 · Research",
        "Investment Cases",
        "Why each security belongs: universe eligibility, client purpose, valuation, evidence depth, and the condition that breaks the thesis.",
    )
    coverage = _mapping(_mapping(model.get("integrity")).get("holding_coverage"))
    holding_count = _count(coverage.get("holding_count"))
    if holding_count:
        coverage_metrics = st.columns(3)
        coverage_metrics[0].metric(
            "Holdings with dossier",
            _percent(coverage.get("dossier_coverage_pct")),
        )
        coverage_metrics[1].metric(
            "Eligible-universe coverage",
            _percent(coverage.get("eligible_universe_coverage_pct")),
        )
        gap_tickers = list(
            dict.fromkeys(
                [
                    *list(coverage.get("holdings_without_dossier") or []),
                    *list(coverage.get("holdings_without_universe_record") or []),
                    *list(coverage.get("holdings_not_eligible") or []),
                ]
            )
        )
        coverage_metrics[2].metric("Coverage gaps", str(len(gap_tickers)))
        if gap_tickers:
            st.warning(
                "Portfolio holdings needing dossier or eligibility follow-up: "
                f"{_display(gap_tickers)}."
            )
    if not dossiers:
        _empty("No documented investment cases are available for review yet.")
        return
    for item in dossiers:
        thesis, valuation, score, ticker = _case_details(item)
        catalysts = _first(item, "catalyst_count", default=0)
        missing_value = item.get("missing")
        missing = _items(missing_value)
        gap_labels = [_first(gap, "label", "name") for gap in missing]
        if (
            isinstance(missing_value, Sequence)
            and not isinstance(missing_value, (str, bytes, bytearray))
        ):
            gap_labels.extend(
                gap
                for gap in missing_value
                if isinstance(gap, str) and gap.strip()
            )
        dossier_status = _first(item, "dossier_status", "status")
        eligibility = _first(item, "eligibility", "universe_status")
        provenance = _first(item, "universe_provenance", "provenance_status")
        downside = _first(
            item,
            "invalidation_condition",
            "invalidation",
            default="Invalidation condition not documented",
        )
        counter_thesis = _first(item, "counter_thesis")
        if counter_thesis not in (None, ""):
            downside = f"{downside} · Counter-thesis: {_display(counter_thesis)}"
        evidence_summary = (
            f"{_display(item.get('primary_source_count'))} primary / "
            f"{_display(item.get('source_count'))} total sources · "
            f"{_display(catalysts)} catalysts"
        )
        gap_summary = (
            f"Open gap: {_display(gap_labels[0])}" if gap_labels else "No recorded readiness gap"
        )
        st.markdown(
            _safe_markup(
                "<article class='judge-case'><div class='judge-case-head'>"
                "<div class='judge-case-title'><small>Security case</small><h3>{ticker}</h3></div>"
                "<span class='judge-case-score'>{score}</span></div>"
                "<div class='judge-chip-row'><span class='judge-chip'>Dossier · {dossier}</span>"
                "<span class='judge-chip'>Universe · {eligibility}</span>"
                "<span class='judge-chip'>Provenance · {provenance}</span></div>"
                "<div class='judge-case-thesis'>{thesis}</div>"
                "<div class='judge-case-grid'>"
                "<div class='judge-case-cell'><span>Client purpose</span><strong>{goal}</strong></div>"
                "<div class='judge-case-cell'><span>Portfolio role</span><strong>{role}</strong></div>"
                "<div class='judge-case-cell'><span>Why now</span><strong>{why_now}</strong></div></div>"
                "<div class='judge-case-downside'><span>Downside discipline</span>"
                "<strong>{downside}</strong></div>"
                "<div class='judge-case-meta'><span>Fair value · {valuation}</span>"
                "<span>{evidence}</span><span>{gap}</span></div></article>",
                ticker=ticker,
                score=score,
                thesis=thesis,
                dossier=_label(dossier_status),
                eligibility=_label(eligibility),
                provenance=_label(provenance),
                goal=_first(
                    item,
                    "client_goal",
                    "primary_goal",
                    "goal",
                    default="Client link not documented",
                ),
                role=_first(item, "portfolio_role", default="Role not documented"),
                why_now=_first(item, "why_now", default="Timing case not documented"),
                downside=downside,
                valuation=valuation,
                evidence=evidence_summary,
                gap=gap_summary,
            ),
            unsafe_allow_html=True,
        )


def _render_risk_and_scenarios(model: Mapping[str, Any]) -> None:
    risk_module = _app_module(model, "risk_scenarios")
    groups = _items(risk_module.get("groups"))
    _section(
        "03 · Stress test",
        "Risk & Scenarios",
        "The analytical challenge between research and approval: saved Quant runs, decision-linked snapshots, and portfolio diagnostics.",
    )
    if not _count(risk_module.get("record_count")):
        _empty(
            "No saved risk or scenario record is attached to the competition journey yet."
        )
        return

    columns = st.columns(3)
    for index, label in enumerate(
        ("Saved Quant run history", "Decision-linked Quant snapshots", "Other current analytical records")
    ):
        group = next((item for item in groups if item.get("label") == label), {})
        columns[index].metric(label, str(_count(group.get("record_count"))))

    for group in groups:
        value = group.get("value")
        if not _count(group.get("record_count")) or _record_is_empty(value):
            continue
        summary_rows = _record_summary_rows(value)[:5]
        if not summary_rows:
            continue
        st.caption(_display(group.get("label"), "Saved analytical evidence"))
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")


def _decision_items(value: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _mapping(value)
    if summary:
        items = _items(summary.get("items") or summary.get("decisions") or summary.get("lifecycles"))
        return summary, items
    return {}, _items(value)


def _vote_summary(item: Mapping[str, Any], prefix: str) -> str:
    nested = _mapping(item.get(f"{prefix}_vote"))
    revealed = _first(item, f"{prefix}_vote_revealed", default=nested.get("revealed"))
    submitted = _first(
        item,
        f"{prefix}_vote_submitted_count",
        f"{prefix}_submitted_count",
        default=nested.get("submitted_count"),
    )
    result = _first(item, f"{prefix}_vote_result", default=nested.get("result"))
    if revealed is None and submitted is None and result is None:
        return _MISSING
    state = "revealed" if bool(revealed) else "not revealed"
    return f"{_display(submitted, '0')} submitted · {state} · {_label(result)}"


def _render_decisions(model: Mapping[str, Any]) -> None:
    summary, decisions = _decision_items(model.get("decisions") or model.get("decision_trail"))
    _section(
        "04 · Governance",
        "Decision Trail",
        "How a researched idea became an authorized position: proposal, two vote rounds, approval, sizing, execution, reconciliation, and audit.",
    )
    decision_count = _count(_first(summary, "count", "total", "case_count", default=decisions))
    if summary and decision_count:
        counts = st.columns(4)
        counts[0].metric(
            "Cases",
            str(decision_count),
        )
        counts[1].metric(
            "Approved or later",
            str(_count(_first(summary, "approved_or_later_count", "approved_count", "approved"))),
        )
        counts[2].metric(
            "Fully voted",
            str(_count(_first(summary, "fully_voted_count", "voted_count"))),
        )
        counts[3].metric("Lifecycle states", str(_count(summary.get("by_state"))))
    if decisions:
        for item in decisions:
            reconciliation = _mapping(item.get("reconciliation"))
            approval_value = _first(
                item,
                "final_approval_complete",
                "final_approval",
                "approval_status",
            )
            approval_label = (
                "Complete"
                if approval_value is True
                else "Incomplete"
                if approval_value is False
                else _display(approval_value, "Not recorded")
            )
            audit_label = (
                "Valid"
                if item.get("audit_valid") is True
                else "Invalid"
                if item.get("audit_valid") is False
                else "Not audited"
            )
            st.markdown(
                _safe_markup(
                    "<article class='judge-decision'><div class='judge-decision-head'>"
                    "<h3>{ticker} · Investment lifecycle</h3><span class='judge-stage-pill'>{stage}</span>"
                    "</div><p class='judge-decision-proposal'>{proposal}</p>"
                    "<div class='judge-decision-grid'>"
                    "<div class='judge-decision-cell'><span>Proposed size</span><strong>{weight}</strong></div>"
                    "<div class='judge-decision-cell'><span>Pre-committee</span><strong>{pre}</strong></div>"
                    "<div class='judge-decision-cell'><span>Post-committee</span><strong>{post}</strong></div>"
                    "<div class='judge-decision-cell'><span>Final approval</span><strong>{approval}</strong></div>"
                    "<div class='judge-decision-cell'><span>WInS reconciliation</span><strong>{reconciliation}</strong></div>"
                    "<div class='judge-decision-cell'><span>Audit chain</span><strong>{audit}</strong></div>"
                    "</div><div class='judge-case-meta'><span>Owner · {owner}</span>"
                    "<span>Challenger · {challenger}</span><span>Updated · {updated}</span></div></article>",
                    ticker=_first(item, "ticker", "symbol", default="Untitled case"),
                    stage=_label(_first(item, "stage", "state", "status", default="Unknown")),
                    proposal=_first(
                        item,
                        "proposal_summary",
                        "decision",
                        "result",
                        "final_decision",
                        default="Proposal summary not recorded.",
                    ),
                    weight=_percent(item.get("proposed_weight_pct")),
                    pre=_vote_summary(item, "pre"),
                    post=_vote_summary(item, "post"),
                    approval=approval_label,
                    reconciliation=_label(
                        _first(reconciliation, "status", default="Not recorded")
                    ),
                    audit=audit_label,
                    owner=_first(item, "owner", default="Not recorded"),
                    challenger=_first(item, "challenger", default="Not recorded"),
                    updated=_first(
                        item,
                        "updated_at",
                        "decided_at",
                        "created_at",
                        default="Not recorded",
                    ),
                ),
                unsafe_allow_html=True,
            )
    elif decision_count:
        st.caption("Aggregate lifecycle counts are available; no case-level trail is attached to this view.")
    else:
        _empty("No canonical committee decisions have been recorded yet.")


def _security_lineage_rows(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    dossiers = {
        str(_first(item, "ticker", "symbol", default="")).upper(): item
        for item in _items(model.get("dossiers") or model.get("investment_cases"))
        if _first(item, "ticker", "symbol")
    }
    _, decision_items = _decision_items(model.get("decisions") or model.get("decision_trail"))
    decisions = {
        str(_first(item, "ticker", "symbol", default="")).upper(): item
        for item in decision_items
        if _first(item, "ticker", "symbol")
    }
    portfolio = _mapping(model.get("portfolio"))
    positions = {
        str(_first(item, "ticker", "symbol", default="")).upper(): item
        for item in _items(portfolio.get("positions") or portfolio.get("holdings"))
        if _first(item, "ticker", "symbol")
    }
    rows: list[dict[str, Any]] = []
    for ticker in sorted(set(dossiers) | set(decisions) | set(positions)):
        dossier = dossiers.get(ticker, {})
        decision = decisions.get(ticker, {})
        position = positions.get(ticker, {})
        proposed_weight = _number(decision.get("proposed_weight_pct"))
        actual_weight = _position_weight_pct(position)
        rows.append(
            {
                "Security": ticker,
                "Research": (
                    f"{_label(_first(dossier, 'dossier_status', 'status', default='Not recorded'))} · "
                    f"{_label(_first(dossier, 'eligibility', 'universe_status', default='No universe record'))}"
                ),
                "Committee": (
                    f"{_label(_first(decision, 'stage', 'state', 'status', default='No case'))}"
                    + (
                        f" · {_percent(proposed_weight)} proposed"
                        if proposed_weight is not None
                        else ""
                    )
                ),
                "Capital today": (
                    f"{_percent(actual_weight)} held"
                    if position
                    else "Not in current snapshot"
                ),
                "Outcome": (
                    f"{_percent(_first(position, 'return_pct', 'total_return_pct'))} · "
                    f"{_money(_first(position, 'pnl', 'profit_loss'))}"
                    if position
                    else _MISSING
                ),
                "Monitoring link": (
                    f"{_label(_first(position, 'lifecycle_state', 'stage', 'status', default='Not recorded'))} · "
                    f"{_display(_first(position, 'client_goal', 'goal', default='Goal not linked'))}"
                    if position
                    else _display(_first(dossier, "review_date"), "No current position")
                ),
            }
        )
    return rows


def _render_security_lineage(model: Mapping[str, Any]) -> None:
    rows = _security_lineage_rows(model)
    _section(
        "04 → 05 · Proof chain",
        "Security Lineage",
        "One row per security connects the dossier, committee state, proposed size, current capital, outcome, and monitoring link.",
    )
    if not rows:
        _empty("No security-level research, decision, or holding record is available yet.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _evidence_count(evidence: Mapping[str, Any], *keys: str) -> int:
    return _count(_first(evidence, *keys, default=0))


def _render_report_and_evidence(model: Mapping[str, Any]) -> None:
    report = _mapping(model.get("report"))
    evidence = _mapping(model.get("evidence"))
    portfolio = _mapping(model.get("portfolio"))
    integrity = _mapping(model.get("integrity"))
    _section(
        "06 · Submission",
        "Report & Evidence",
        "How the portfolio becomes a defensible submission: one governed snapshot, evidence-linked claims, validation, and rehearsal.",
    )
    evidence_total = sum(
        _evidence_count(
            evidence,
            key,
        )
        for key in (
            "source_count",
            "catalyst_count",
            "thesis_review_count",
            "red_team_review_count",
            "ai_disclosure_count",
            "qa_round_count",
        )
    )
    if not report.get("available", bool(report)) and evidence_total == 0:
        _empty("No report workspace or evidence register is available yet.")
        return

    portfolio_snapshot = _first(portfolio, "snapshot_id", "id")
    report_snapshot = _first(report, "portfolio_snapshot_id", "snapshot_id")
    binding_match = integrity.get("report_snapshot_matches_pipeline")
    binding_label = (
        "Verified same snapshot"
        if binding_match is True
        else "Snapshot mismatch"
        if binding_match is False
        else "Binding not yet verifiable"
    )
    st.markdown(
        _safe_markup(
            "<div class='judge-source'><strong>Same-snapshot proof</strong> · "
            "Portfolio {portfolio_snapshot} → Report {report_snapshot} · {binding}</div>",
            portfolio_snapshot=portfolio_snapshot,
            report_snapshot=report_snapshot,
            binding=binding_label,
        ),
        unsafe_allow_html=True,
    )

    attribution = _mapping(report.get("performance_attribution"))
    if attribution.get("available", bool(attribution)):
        st.caption(
            f"Report-bound performance attribution · {_display(attribution.get('benchmark'), 'Benchmark not named')}"
        )
        outcome_metrics = st.columns(4)
        outcome_metrics[0].metric(
            "Portfolio return", _percent(attribution.get("portfolio_return_pct"))
        )
        outcome_metrics[1].metric(
            "Benchmark return", _percent(attribution.get("benchmark_return_pct"))
        )
        outcome_metrics[2].metric(
            "Active return", _percent(attribution.get("active_return_pct"))
        )
        outcome_metrics[3].metric(
            "Attribution residual", _percent(attribution.get("residual_pct"))
        )
        contributions = _items(attribution.get("contributions"))
        if contributions:
            contribution_rows = [
                {
                    "Driver": _display(_first(item, "label", "id")),
                    "Contribution": _percent(item.get("contribution_pct")),
                }
                for item in contributions
            ]
            st.dataframe(
                pd.DataFrame(contribution_rows),
                hide_index=True,
                width="stretch",
            )

    metrics = st.columns(4)
    metrics[0].metric("Report", _label(_first(report, "status_label", "status"), "Not prepared"))
    metrics[1].metric("Research sources", str(_evidence_count(evidence, "sources", "source_count")))
    metrics[2].metric(
        "Red-team reviews",
        str(_evidence_count(evidence, "red_team_reviews", "red_team_review_count", "red_team_count")),
    )
    metrics[3].metric(
        "Q&A sessions",
        str(_evidence_count(evidence, "qa_sessions", "qa_round_count", "qa_count")),
    )

    report_facts, evidence_facts = st.columns(2)
    with report_facts:
        _fact("Report title", _first(report, "title", default="Untitled report"))
        _fact("Frozen", _first(report, "frozen", "is_frozen", default=False))
        _fact("Bound portfolio snapshot", _first(report, "portfolio_snapshot_id", "snapshot_id"))
        _fact(
            "Ready sections",
            f"{_count(report.get('ready_section_count'))}/{_count(report.get('section_count'))}",
        )
        _fact(
            "Claims / figures",
            f"{_count(_first(report, 'claim_count', default=report.get('claims')))} / "
            f"{_count(_first(report, 'figure_count', default=report.get('figures')))}",
        )
        validation_ready = report.get("validation_ready")
        _fact(
            "Validation",
            "Ready" if validation_ready is True else "Needs attention" if validation_ready is False else "Not validated",
        )
        _fact(
            "Page budget / estimate",
            {
                "budget": report.get("page_budget"),
                "estimate": report.get("estimated_pages"),
            },
        )
    with evidence_facts:
        _fact("Catalysts", _evidence_count(evidence, "catalysts", "catalyst_count"))
        _fact("Thesis reviews", _evidence_count(evidence, "thesis_reviews", "thesis_review_count"))
        _fact(
            "AI disclosures",
            _evidence_count(evidence, "ai_usage", "ai_disclosure_count", "ai_usage_count"),
        )
        _fact("Latest evidence as of", _first(evidence, "as_of", "updated_at"))
        _fact("Report approvals", _count(report.get("approval_count")))

    issues = _items(report.get("issues") or report.get("validation_issues"))
    if issues:
        issue_rows = [
            {
                "Severity": _display(_first(item, "severity", "level")),
                "Area": _display(_first(item, "area", "field", "code")),
                "Finding": _display(_first(item, "message", "finding", "description")),
            }
            for item in issues
        ]
        st.dataframe(pd.DataFrame(issue_rows), hide_index=True, width="stretch")

    sources = _items(evidence.get("sources"))
    if sources:
        st.caption("Evidence register")
        source_rows = [
            {
                "Ticker": _display(item.get("ticker")),
                "Source": _display(_first(item, "title", "name")),
                "Type": _display(_first(item, "source_type", "type")),
                "Primary": "Yes" if item.get("primary_source") else "No",
                "As of": _display(_first(item, "as_of", "published_at")),
                "Verified by": _display(item.get("verified_by")),
                "Citation": _display(_first(item, "citation", "url")),
            }
            for item in sources
        ]
        st.dataframe(pd.DataFrame(source_rows), hide_index=True, width="stretch")


def _check_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                rows.append({"name": key, **dict(item)})
            elif isinstance(item, bool):
                rows.append({"name": key, "passed": item})
        return rows
    return _items(value)


def _check_passed(item: Mapping[str, Any]) -> bool:
    explicit = _first(item, "passed", "compliant")
    if explicit is not None:
        return bool(explicit)
    return str(_first(item, "status", default="")).strip().casefold() in {
        "pass",
        "passed",
        "compliant",
        "ready",
    }


def _render_rules_and_integrity(model: Mapping[str, Any]) -> None:
    compliance_value = model.get("compliance")
    compliance = _mapping(compliance_value)
    if not compliance and _items(compliance_value):
        compliance = {"checks": _items(compliance_value)}
    integrity = _mapping(model.get("integrity"))
    checks = _check_items(
        compliance.get("items") or compliance.get("checks") or compliance.get("rules")
    )
    _section(
        "07 · Controls",
        "Rules & Integrity",
        "The final trust layer: official-rule provenance, compliance, reconciliation quality, audit validity, and disclosure controls.",
    )
    if not compliance and not integrity:
        _empty("No compliance or integrity assessment is available yet.")
        return

    passed = _count(compliance.get("pass_count")) or sum(_check_passed(item) for item in checks)
    reconciliation = _mapping(integrity.get("reconciliation"))
    rules = _mapping(integrity.get("rules"))
    all_clear = compliance.get("all_clear")
    failed = _count(compliance.get("fail_count"))
    pending = _count(compliance.get("pending_count"))
    compliance_label = (
        "Needs attention"
        if failed
        else "Passes with pending items"
        if pending
        else "All passed"
        if all_clear is True
        else _display(_first(compliance, "status", "label"), "Not assessed")
    )
    metrics = st.columns(4)
    metrics[0].metric("Compliance", compliance_label)
    metrics[1].metric(
        "Checks passed",
        f"{passed}/{_count(compliance.get('check_count')) or len(checks)}" if checks else _MISSING,
    )
    metrics[2].metric(
        "Reconciliation",
        _label(
            _first(reconciliation, "status", default=_first(integrity, "reconciliation_status")),
            "Not assessed",
        ),
    )
    metrics[3].metric(
        "Open exceptions",
        (
            str(
                _count(
                    _first(
                        reconciliation,
                        "open_exception_count",
                        "open_exceptions",
                        "exceptions",
                    )
                )
            )
            if reconciliation.get("available", bool(reconciliation))
            else _MISSING
        ),
    )
    if failed:
        st.error(f"{failed} compliance check{'s' if failed != 1 else ''} need attention.")
    elif pending:
        st.warning(f"{pending} compliance check{'s are' if pending != 1 else ' is'} still pending.")

    facts_left, facts_right = st.columns(2)
    with facts_left:
        _fact(
            "Portfolio pipeline",
            {
                "status": integrity.get("pipeline_status"),
                "authority": integrity.get("pipeline_authority"),
            },
        )
        _fact(
            "Official rules snapshot",
            _first(rules, "snapshot_id", "version", default=_first(integrity, "rules_snapshot_id")),
        )
        hash_value = _first(
            rules,
            "content_hash",
            "hash",
            default=_first(integrity, "rules_hash"),
        )
        if hash_value is None and rules.get("content_hash_present") is not None:
            hash_value = "Present" if rules.get("content_hash_present") else "Missing"
        _fact("Rules hash", hash_value)
    with facts_right:
        _fact(
            "Rules acknowledged",
            _first(
                rules,
                "all_acknowledged",
                "acknowledged",
                default=_first(integrity, "rules_acknowledged"),
            ),
        )
        _fact("Report bound to pipeline", integrity.get("report_snapshot_matches_pipeline"))
        _fact("Reporting binding allowed", integrity.get("reporting_binding_allowed"))

    if checks:
        rows = [
            {
                "Check": _display(_first(item, "name", "rule", "label")),
                "Status": (
                    "Pass"
                    if _check_passed(item)
                    else "Pending"
                    if str(item.get("status")).casefold() == "pending"
                    else "Needs attention"
                ),
                "Detail": _display(_first(item, "message", "detail", "reason")),
            }
            for item in checks
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _record_is_empty(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, Mapping):
        return not value
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return not value
    return False


def _summary_cell(value: Any) -> Any:
    if isinstance(value, (Mapping, Sequence)) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        text = _display(value)
        return text if len(text) <= 180 else f"{text[:177]}..."
    return value


def _record_summary_rows(value: Any) -> list[dict[str, Any]]:
    preferred = (
        "id",
        "record_id",
        "ticker",
        "economy_code",
        "name",
        "title",
        "task_text",
        "filename",
        "action",
        "status",
        "version",
        "is_active",
        "assignee",
        "owner",
        "created_by",
        "updated_by",
        "created_at",
        "updated_at",
        "date",
        "timestamp",
    )
    if isinstance(value, Mapping):
        candidates = [dict(value)]
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        candidates = [dict(item) for item in value if isinstance(item, Mapping)]
        if not candidates and value:
            return [{"Value": _summary_cell(item)} for item in value]
    else:
        return [{"Value": _summary_cell(value)}]

    rows: list[dict[str, Any]] = []
    for item in candidates:
        keys = [key for key in preferred if key in item]
        if not keys:
            keys = list(item)[:8]
        rows.append({key: _summary_cell(item.get(key)) for key in keys})
    return rows


def _render_complete_app_record(model: Mapping[str, Any]) -> None:
    record = _mapping(model.get("app_record"))
    modules = _items(record.get("modules"))
    _section(
        "09 · Appendix",
        "Complete App Record",
        "The full audit appendix. Open a module for its readable summary and complete saved record; unused workspaces remain visible by design.",
    )
    if not modules:
        _empty("The complete application record is not available yet.")
        return

    metrics = st.columns(3)
    metrics[0].metric("App workspaces", str(_count(record.get("module_count")) or len(modules)))
    metrics[1].metric("With evidence", str(_count(record.get("available_module_count"))))
    metrics[2].metric("Access", "Read only")
    st.caption(_display(record.get("scope_note")))
    for start in range(0, len(modules), 3):
        columns = st.columns(3)
        for column, module in zip(columns, modules[start : start + 3]):
            count = _count(module.get("record_count"))
            column.markdown(
                _safe_markup(
                    "<article class='judge-module-card {used}'><span>{status}</span>"
                    "<h4>{label}</h4><p>{count} evidence item{suffix}<br>{description}</p></article>",
                    used="is-used" if count else "",
                    status=module.get("status"),
                    label=module.get("label"),
                    count=count,
                    suffix="s" if count != 1 else "",
                    description=module.get("description"),
                ),
                unsafe_allow_html=True,
            )

    for index, module in enumerate(modules, start=1):
        label = _display(module.get("label"), f"Workspace {index}")
        status = _display(module.get("status"), "Not used yet")
        count = _count(module.get("record_count"))
        with st.expander(
            f"{index:02d} · {label} · {status} · {count} evidence item{'s' if count != 1 else ''}",
            expanded=False,
        ):
            st.caption(_display(module.get("description")))
            groups = _items(module.get("groups"))
            if not count:
                _empty(
                    "This workspace has no saved competition data yet. It remains visible so the judge can verify that it was not used."
                )
                continue
            for group in groups:
                group_label = _display(group.get("label"), "Saved record")
                value = group.get("value")
                group_count = _count(group.get("record_count"))
                st.markdown(f"##### {escape(group_label)}")
                st.caption(f"{group_count} visible item{'s' if group_count != 1 else ''}")
                if _record_is_empty(value):
                    st.caption("No saved data in this part of the workspace.")
                    continue
                summary_rows = _record_summary_rows(value)
                if summary_rows:
                    st.dataframe(
                        pd.DataFrame(summary_rows),
                        hide_index=True,
                        width="stretch",
                    )
                st.caption("Complete saved record")
                st.json(value, expanded=False)


def _render_questions(model: Mapping[str, Any]) -> None:
    questions_value = model.get("questions") or model.get("judge_questions")
    if isinstance(questions_value, Mapping):
        questions_value = questions_value.get("items")
    questions = []
    if isinstance(questions_value, Sequence) and not isinstance(
        questions_value, (str, bytes, bytearray)
    ):
        questions = list(questions_value)
    _section(
        "08 · Oral defense",
        "Questions for the Team",
        "The final step in the journey: questions derived from the evidence trail, portfolio choices, and remaining gaps.",
    )
    if not questions:
        _empty("No judge question prompts have been generated yet.")
        return
    for index, question in enumerate(questions, start=1):
        if isinstance(question, Mapping):
            question = _first(question, "question", "prompt", "text")
        st.markdown(
            _safe_markup(
                "<div class='judge-question'><strong>{index}</strong><span>{question}</span></div>",
                index=index,
                question=question,
            ),
            unsafe_allow_html=True,
        )


def render_judge_view(profile: Mapping[str, Any], model: Mapping[str, Any]) -> None:
    """Render a judge-only review surface without exposing any workflow actions."""
    profile_data = _mapping(profile)
    view_model = _mapping(model)
    portfolio = _mapping(view_model.get("portfolio"))
    report = _mapping(view_model.get("report"))
    evidence = _mapping(view_model.get("evidence"))
    readiness = _mapping(view_model.get("readiness"))
    as_of = _first(
        view_model,
        "as_of",
        default=_first(
            portfolio,
            "as_of",
            "as_of_date",
            default=_first(report, "updated_at", default=evidence.get("as_of")),
        ),
    )
    reviewer = _first(profile_data, "display_name", "name", "username", default="Judge")
    score_number = _number(_first(readiness, "overall_score", "score", "readiness_score"))
    score_percent = max(0.0, min(100.0, score_number or 0.0))
    score_label = f"{score_percent:.0f}" if score_number is not None else _MISSING
    score_status = _first(readiness, "status", default="Not assessed")

    _inject_styles()
    st.markdown(
        _safe_markup(
            "<section class='judge-hero'><div class='judge-hero-grid'><div>"
            "<div class='judge-hero-label'>Judge View · Read only</div>"
            "<h1>From client mandate to defended portfolio.</h1>"
            "<p>Follow every dollar through the process: why it belongs, who challenged it, "
            "how it was authorized, what was executed, and which evidence supports the final story.</p>"
            "<div class='judge-badges'><span class='judge-badge'>Reviewer · {reviewer}</span>"
            "<span class='judge-badge'>Evidence as of · {as_of}</span>"
            "<span class='judge-badge'>Independent review surface</span></div></div>"
            "<aside class='judge-score-panel'><div class='judge-score-ring' style='--score:{score_degrees}deg'>"
            "<div class='judge-score-copy'><strong>{score}</strong><span>of 100</span></div></div>"
            "<strong>{score_status}</strong><small>Internal preparation diagnostic<br>Portfolio returns excluded</small>"
            "</aside></div></section>",
            reviewer=reviewer,
            as_of=as_of,
            score_degrees=score_percent * 3.6,
            score=score_label,
            score_status=score_status,
        ),
        unsafe_allow_html=True,
    )
    _render_overview(view_model)
    _render_portfolio_journey(view_model)
    _render_client_and_strategy(view_model)
    _render_investment_cases(view_model)
    _render_risk_and_scenarios(view_model)
    _render_decisions(view_model)
    _render_security_lineage(view_model)
    _render_portfolio(view_model)
    _render_report_and_evidence(view_model)
    _render_rules_and_integrity(view_model)
    _render_questions(view_model)
    _render_complete_app_record(view_model)


__all__ = ["render_judge_view"]
