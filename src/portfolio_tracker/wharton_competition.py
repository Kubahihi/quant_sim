"""Rules and portfolio calculations for the Wharton competition cockpit."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


INITIAL_CAPITAL_USD = 500_000.0
OFFICIAL_RULES_URL = (
    "https://globalyouth.wharton.upenn.edu/competitions/"
    "investment-competition/rules-roles/"
)
COMPETITION_URL = (
    "https://globalyouth.wharton.upenn.edu/competitions/investment-competition/"
)


def _confirmed(value: Any) -> bool:
    return bool(int(value or 0))


def evaluate_compliance(
    settings: Mapping[str, Any] | None,
    positions: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Evaluate every currently machine-checkable 2026-2027 rule."""
    values = dict(settings or {})
    team_size = int(values.get("team_size") or 0)
    leader_age = int(values.get("leader_age") or 0)
    advisor_team_count = int(values.get("advisor_team_count") or 0)

    checks: list[dict[str, str]] = []

    def add(ok: bool, rule: str, success: str, failure: str) -> None:
        checks.append(
            {
                "status": "pass" if ok else "fail",
                "rule": rule,
                "detail": success if ok else failure,
            }
        )

    add(
        4 <= team_size <= 6,
        "Team has 4-6 active students",
        f"Confirmed: {team_size} active students.",
        f"Current value is {team_size}; the official range is 4 through 6.",
    )
    add(
        _confirmed(values.get("same_school")),
        "All students attend the same school and branch",
        "Confirmed by the team.",
        "The team has not confirmed that every student attends the same school and branch.",
    )
    add(
        _confirmed(values.get("eligible_students")),
        "Students are eligible high-school students",
        "Confirmed: ages and diploma status meet the published eligibility rule.",
        "Confirm that every student is a current pre-university high-school student, age 14-18 at the start, with no diploma earned before the competition.",
    )
    add(
        _confirmed(values.get("leader_designated")) and leader_age >= 16,
        "One designated team leader is at least 16",
        f"Confirmed: designated leader age is {leader_age}.",
        (
            "A designated team leader has not been confirmed."
            if not _confirmed(values.get("leader_designated"))
            else f"Leader age is {leader_age}; the minimum is 16 at the start of the competition."
        ),
    )
    add(
        _confirmed(values.get("advisor_is_teacher")),
        "Primary advisor is a teacher at the team's school",
        "Confirmed by the team.",
        "The required primary teacher-advisor relationship has not been confirmed.",
    )
    add(
        1 <= advisor_team_count <= 5,
        "Advisor oversees no more than five teams",
        f"Confirmed: advisor oversees {advisor_team_count} team(s).",
        f"Current value is {advisor_team_count}; enter a value from 1 through 5.",
    )
    declarations = [
        ("one_wins_account", "Team shares one WInS account", "Use of exactly one shared team account has not been confirmed."),
        ("members_single_team", "No student competes on another team", "Single-team participation has not been confirmed for every student."),
        ("no_client_contact", "Team has not contacted the competition client", "The no-client-contact rule has not been confirmed; a violation causes disqualification."),
        ("no_paid_advisor", "No paid advisor or prohibited outside course", "The team has not confirmed that it uses no paid advisor, consultant, agent, or prohibited competition course."),
        ("student_owned_work", "Strategy and decisions are the students' own work", "Student ownership of the work has not been confirmed."),
        ("ai_cited", "AI-generated material is cited and not submitted as original work", "The team has not confirmed compliant AI use and citation."),
        ("sources_cited", "Sources, images, and media are properly credited", "Source and media attribution has not been confirmed."),
        ("school_permission", "Official school permission documentation is ready", "The school-letterhead permission document required with the final report is not confirmed ready."),
    ]
    for key, rule, failure in declarations:
        add(_confirmed(values.get(key)), rule, "Confirmed by the team.", failure)

    invested = sum(
        float(row.get("quantity") or 0) * float(row.get("entry_price") or 0)
        for row in positions
        if str(row.get("status") or "open") == "open"
    )
    add(
        invested <= INITIAL_CAPITAL_USD + 1e-6,
        "Open positions do not exceed $500,000 starting capital",
        f"Open-position cost is ${invested:,.2f}.",
        f"Open-position cost is ${invested:,.2f}, exceeding starting capital by ${invested - INITIAL_CAPITAL_USD:,.2f}.",
    )
    add(
        all(
            str(row.get("opened_by") or "").strip()
            and float(row.get("quantity") or 0) > 0
            and float(row.get("entry_price") or 0) > 0
            for row in positions
        ),
        "Every tracked position has valid size, entry price, and author",
        "All tracked positions contain the required audit fields.",
        "At least one position is missing its author or has a non-positive quantity/entry price.",
    )
    checks.append(
        {
            "status": "pending",
            "rule": "2026-2027 trading limits and approved-security lists",
            "detail": "Wharton currently says 'More information coming soon.' This check will remain pending instead of applying last year's limits.",
        }
    )
    checks.append(
        {
            "status": "pending",
            "rule": "2026-2027 case study and deliverable details",
            "detail": "The new case objectives, deadlines, and detailed deliverable requirements have not yet been published on the official competition pages.",
        }
    )
    return checks


def calculate_portfolio_performance(
    positions: Sequence[Mapping[str, Any]],
    live_prices: Mapping[str, float] | None = None,
    initial_capital: float = INITIAL_CAPITAL_USD,
) -> dict[str, Any]:
    """Calculate realized, unrealized, per-position, and total performance."""
    prices = {str(key).upper(): float(value) for key, value in (live_prices or {}).items()}
    rows: list[dict[str, Any]] = []
    realized_pnl = 0.0
    unrealized_pnl = 0.0
    open_cost = 0.0

    for item in positions:
        row = dict(item)
        ticker = str(row.get("ticker") or "").upper()
        quantity = float(row.get("quantity") or 0)
        entry_price = float(row.get("entry_price") or 0)
        status = str(row.get("status") or "open")
        cost = quantity * entry_price

        if status == "closed":
            current_price = float(row.get("exit_price") or entry_price)
            price_source = "exit"
            pnl = quantity * (current_price - entry_price)
            realized_pnl += pnl
        else:
            stored_price = float(row.get("last_price") or 0)
            current_price = prices.get(ticker) or stored_price or entry_price
            price_source = "live" if ticker in prices else "manual" if stored_price else "entry fallback"
            pnl = quantity * (current_price - entry_price)
            unrealized_pnl += pnl
            open_cost += cost

        rows.append(
            {
                **row,
                "ticker": ticker,
                "cost": cost,
                "current_price": current_price,
                "current_value": quantity * current_price,
                "pnl": pnl,
                "return_pct": (pnl / cost * 100.0) if cost else 0.0,
                "price_source": price_source,
            }
        )

    total_pnl = realized_pnl + unrealized_pnl
    equity = initial_capital + total_pnl
    return {
        "initial_capital": initial_capital,
        "equity": equity,
        "cash_before_pnl": initial_capital - open_cost,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "total_return_pct": (total_pnl / initial_capital * 100.0) if initial_capital else 0.0,
        "positions": rows,
    }
