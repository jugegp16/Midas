"""Tests for the backtest summary terminal rendering."""

from __future__ import annotations

from datetime import date

from midas.output import console, print_backtest_summary
from midas.results import BacktestResult
from midas.tax import AnnualTaxSummary


def _result_with_tax() -> BacktestResult:
    curve = [(date(2026, 1, 2), 10_000.0), (date(2027, 12, 30), 12_000.0)]
    return BacktestResult(
        trades=[],
        final_value=12_000.0,
        starting_value=10_000.0,
        buy_and_hold_value=11_000.0,
        train_trades=[],
        test_trades=[],
        train_return=0.0,
        test_return=0.0,
        train_bh_return=0.0,
        test_bh_return=0.0,
        split_date=None,
        twr=0.2,
        equity_curve=curve,
        total_days=727,
        train_days=0,
        test_days=0,
        cagr=0.096,
        max_drawdown=0.1,
        sharpe_ratio=1.0,
        sortino_ratio=1.2,
        win_rate=0.5,
        profit_factor=1.5,
        avg_win=100.0,
        avg_loss=-50.0,
        efficiency_ratio=0.0,
        strategy_stats=[],
        unrealized_pnl=0.0,
        unrealized_pnl_by_ticker={},
        basis_per_sell=[],
        after_tax_final_value=11_500.0,
        after_tax_total_return=0.15,
        after_tax_cagr=0.073,
        tax_summary=[
            AnnualTaxSummary(
                year=2026,
                st_realized=-5_000.0,
                lt_realized=0.0,
                net_after_cross=-5_000.0,
                deductible_loss=3_000.0,
                carry_forward=2_000.0,
                tax_owed=-900.0,
                payment_date=date(2027, 4, 15),
            ),
            AnnualTaxSummary(
                year=2027,
                st_realized=1_000.0,
                lt_realized=500.0,
                net_after_cross=-500.0,  # 1,500 gain less 2,000 carry-in
                deductible_loss=500.0,
                carry_forward=0.0,
                tax_owed=-150.0,
                payment_date=date(2027, 12, 30),
            ),
        ],
    )


def _capture_summary() -> str:
    # The summary tables are BACKTEST_TABLE_WIDTH (100) wide; a captured
    # console defaults to 80 and would crop the right-justified values.
    old_width = console.width
    console.width = 160
    try:
        with console.capture() as capture:
            print_backtest_summary(_result_with_tax(), show_charts=False)
    finally:
        console.width = old_width
    return capture.get()


def test_after_tax_metrics_live_in_after_tax_block_only() -> None:
    text = _capture_summary()
    assert "After-Tax Final Value" in text
    assert "$11,500.00" in text
    assert "After-Tax Total Return" in text
    assert "15.00%" in text
    # The main Performance table stays gross-only.
    assert "Final Value (After Tax)" not in text
    assert "Total Return (After Tax)" not in text


def test_tax_table_total_row_sums_raw_buckets_not_carried_nets() -> None:
    text = _capture_summary()
    assert "Total" in text
    # ST total -4,000; LT total +500; Net = raw ST+LT = -3,500 (NOT the
    # column sum -5,500, which would double-count the carried loss).
    assert "$-4,000.00" in text
    assert "$-3,500.00" in text
    assert "5,500.00" not in text
    # Tax Owed total = -1,050; ending Carry Forward = final year's 0.
    assert "$-1,050.00" in text
