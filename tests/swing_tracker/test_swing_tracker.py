from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.swing_tracker.logic import apply_trade_lifecycle, compute_capital_trapped_overdue
from src.swing_tracker.manager import (
    close_trade,
    create_trade,
    load_trade_book,
    save_trade_book,
)
from src.swing_tracker.stop_logic import calculate_stop_loss


def test_position_size_and_fixed_risk_stop_are_deterministic():
    stop_loss = calculate_stop_loss(
        direction="long",
        entry_price=50.0,
        stop_type="fixed_risk",
        fixed_risk_percent=5.0,
    )
    assert stop_loss == pytest.approx(47.5)

    trade = create_trade(
        ticker="AAPL",
        direction="long",
        setup_type="breakout",
        thesis="Daily breakout with volume expansion.",
        entry_price=50.0,
        stop_loss=stop_loss,
        stop_type="fixed_risk",
        stop_rationale="Stop under breakout base low.",
        target_price=58.0,
        targets=[58.0, 62.0],
        time_stop_days=10,
        planned_holding_days=12,
        risk_percent=1.0,
        position_size=400.0,
        status="open",
        entry_date=date.today() - timedelta(days=2),
    )
    assert trade.position_size == pytest.approx(400.0)
    assert trade.discipline_score is not None


def test_overdue_status_when_holding_exceeds_time_rules():
    trade = create_trade(
        ticker="MSFT",
        direction="long",
        setup_type="pullback",
        thesis="Trend pullback to 20 EMA.",
        entry_price=100.0,
        stop_loss=95.0,
        stop_type="structural",
        stop_rationale="Under prior swing low.",
        target_price=112.0,
        targets=[112.0],
        time_stop_days=7,
        planned_holding_days=5,
        risk_percent=1.0,
        position_size=200.0,
        status="open",
        entry_date=date.today() - timedelta(days=9),
    )
    updated = apply_trade_lifecycle(trade, as_of=date.today())
    assert updated.status == "overdue"
    assert updated.actual_holding_days == 9


def test_close_trade_computes_realized_pnl_and_r_multiple():
    base_trade = create_trade(
        ticker="NVDA",
        direction="long",
        setup_type="continuation",
        thesis="Continuation after earnings gap.",
        entry_price=100.0,
        stop_loss=95.0,
        stop_type="structural",
        stop_rationale="Below post-gap reclaim level.",
        target_price=120.0,
        targets=[120.0],
        time_stop_days=15,
        planned_holding_days=12,
        risk_percent=1.0,
        position_size=100.0,
        status="open",
        entry_date=date.today() - timedelta(days=4),
    )

    trades, closed_trade = close_trade(
        [base_trade],
        trade_id=base_trade.id,
        exit_price=112.0,
        exit_date=date.today(),
        exit_reason="target_hit",
        notes="Followed plan and scaled once.",
        final_status="closed",
    )

    assert len(trades) == 1
    assert closed_trade.status == "closed"
    assert closed_trade.realized_pnl == pytest.approx(1200.0)
    assert closed_trade.realized_r_multiple == pytest.approx(2.4)
    assert closed_trade.actual_holding_days == 4


def test_capital_trapped_in_overdue_trades():
    overdue_trade = create_trade(
        ticker="TSLA",
        direction="long",
        setup_type="event_driven",
        thesis="Post-event trend continuation.",
        entry_price=200.0,
        stop_loss=188.0,
        stop_type="fixed_risk",
        stop_rationale="Risk defined to 6% from entry.",
        target_price=230.0,
        targets=[230.0],
        time_stop_days=5,
        planned_holding_days=6,
        risk_percent=1.0,
        position_size=50.0,
        status="open",
        entry_date=date.today() - timedelta(days=8),
    )
    overdue_trade = apply_trade_lifecycle(overdue_trade, as_of=date.today())
    assert overdue_trade.status == "overdue"
    assert compute_capital_trapped_overdue([overdue_trade]) == pytest.approx(10000.0)


def test_trade_book_persistence_round_trip(tmp_path):
    trade = create_trade(
        ticker="AMD",
        direction="short",
        setup_type="breakdown",
        thesis="Breakdown under range support.",
        entry_price=120.0,
        stop_loss=126.0,
        stop_type="structural",
        stop_rationale="Invalidation above broken support.",
        target_price=108.0,
        targets=[108.0, 102.0],
        time_stop_days=8,
        planned_holding_days=10,
        risk_percent=0.8,
        position_size=150.0,
        status="open",
        entry_date=date.today() - timedelta(days=1),
    )
    storage_path = tmp_path / "swing_trades.json"
    save_trade_book([trade], storage_path=storage_path)
    loaded = load_trade_book(storage_path=storage_path)

    assert storage_path.exists()
    assert len(loaded) == 1
    assert loaded[0].ticker == "AMD"
    assert loaded[0].direction == "short"
