"""Tests for the `midas tax-report` subcommand."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from click.testing import CliRunner

from midas.cli import cli
from midas.models import Direction, HoldingPeriod, TradeRecord
from midas.trade_log import append_trade

PORTFOLIO_YAML = (
    "portfolio:\n  - ticker: AAPL\n    shares: 10\navailable_cash: 1000.0\n"
    "tax:\n  short_term_rate: 0.30\n  long_term_rate: 0.15\n"
)


def _sell(
    log: Path,
    *,
    day: date,
    shares: float,
    price: float,
    cost_basis: float | None,
    holding_period: HoldingPeriod | None = HoldingPeriod.SHORT_TERM,
    purchase_date: date | None = None,
) -> None:
    append_trade(
        log,
        TradeRecord(
            date=day,
            ticker="AAPL",
            direction=Direction.SELL,
            shares=shares,
            price=price,
            strategy_name="StopLoss",
            holding_period=holding_period,
            purchase_date=purchase_date,
        ),
        cost_basis=cost_basis,
        purchase_date=purchase_date,
    )


def _seed_log(path: Path) -> None:
    # 2026 ST: 10sh, basis $20/sh, sold $30 -> proceeds 300, basis 200, pnl +100
    _sell(
        path,
        day=date(2026, 6, 1),
        shares=10.0,
        price=30.0,
        cost_basis=20.0,
        holding_period=HoldingPeriod.SHORT_TERM,
        purchase_date=date(2026, 1, 1),
    )
    # 2026 LT: 5sh, basis $15/sh, sold $40 -> proceeds 200, basis 75, pnl +125
    _sell(
        path,
        day=date(2026, 7, 1),
        shares=5.0,
        price=40.0,
        cost_basis=15.0,
        holding_period=HoldingPeriod.LONG_TERM,
        purchase_date=date(2024, 1, 1),
    )


def _run_tax_report(log: Path, tmp_path: Path, *extra_args: str, portfolio_text: str = PORTFOLIO_YAML):
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(portfolio_text)
    runner = CliRunner()
    return runner.invoke(
        cli,
        ["tax-report", "--portfolio", str(portfolio), "--from-trades", str(log), *extra_args],
    )


def test_tax_report_emits_csv_and_prints_totals(tmp_path: Path) -> None:
    log = tmp_path / "trades.csv"
    output = tmp_path / "schedule_d_2026.csv"
    _seed_log(log)

    result = _run_tax_report(log, tmp_path, "--year", "2026", "--output", str(output))
    assert result.exit_code == 0, result.output
    assert "AAPL" in result.output
    # Per-year footer: ST +100, LT +125, tax = 100*0.30 + 125*0.15 = 48.75.
    assert "Net +225.00" in result.output
    assert "Tax +48.75" in result.output

    with open(output, newline="") as handle:
        rows = list(csv.reader(handle))
    # Data rows only — per-year aggregates live in the companion summary CSV.
    assert len(rows) == 3
    header, first, second = rows
    assert header[4:7] == ["cost_basis", "proceeds", "realized_pnl"]
    # cost_basis, proceeds, realized_pnl are all totals: proceeds - basis = pnl.
    assert [float(cell) for cell in first[4:7]] == [200.0, 300.0, 100.0]
    assert [float(cell) for cell in second[4:7]] == [75.0, 200.0, 125.0]

    with open(output.with_suffix(".summary.csv"), newline="") as handle:
        summary_rows = list(csv.reader(handle))
    assert summary_rows[0] == [
        "year",
        "st_realized",
        "lt_realized",
        "net_after_cross",
        "deductible_loss",
        "carry_forward",
        "tax_owed",
        "payment_date",
    ]
    (year_row,) = summary_rows[1:]
    assert year_row[0] == "2026"
    assert float(year_row[3]) == 225.0
    assert float(year_row[6]) == 48.75
    # Natural payment date (Dec 31 + 105-day lag), not clamped to the window.
    assert year_row[7] == "2027-04-15"


def test_tax_report_applies_carryforward_from_before_report_window(tmp_path: Path) -> None:
    log = tmp_path / "trades.csv"
    # 2025: -$5,000 ST loss -> $3K deducted in 2025, $2,000 carries into 2026.
    _sell(
        log,
        day=date(2025, 6, 1),
        shares=100.0,
        price=30.0,
        cost_basis=80.0,
        purchase_date=date(2025, 1, 2),
    )
    # 2026: +$100 ST gain; after the $2,000 carry-in the net is -$1,900.
    _sell(
        log,
        day=date(2026, 6, 1),
        shares=10.0,
        price=30.0,
        cost_basis=20.0,
        purchase_date=date(2026, 1, 2),
    )

    result = _run_tax_report(log, tmp_path, "--year", "2026", "--output", str(tmp_path / "out.csv"))
    assert result.exit_code == 0, result.output
    assert "Year 2026" in result.output
    assert "Net -1,900.00" in result.output
    assert "Deductible 1,900.00" in result.output
    assert "Tax -570.00" in result.output
    assert "Carry-Forward 0.00" in result.output
    # Only the requested year's summary (and rows) are shown.
    assert "Year 2025" not in result.output
    assert "2025-06-01" not in result.output


def test_tax_report_warns_on_missing_cost_basis(tmp_path: Path) -> None:
    log = tmp_path / "trades.csv"
    _sell(
        log,
        day=date(2026, 6, 1),
        shares=10.0,
        price=30.0,
        cost_basis=None,
        purchase_date=date(2026, 1, 2),
    )

    result = _run_tax_report(log, tmp_path, "--year", "2026", "--output", str(tmp_path / "out.csv"))
    assert result.exit_code == 0, result.output
    assert "no cost_basis" in result.stderr
    assert "Net +0.00" in result.output


def test_tax_report_excludes_rows_without_holding_period(tmp_path: Path) -> None:
    log = tmp_path / "trades.csv"
    _seed_log(log)
    _sell(
        log,
        day=date(2026, 8, 1),
        shares=3.0,
        price=50.0,
        cost_basis=10.0,
        holding_period=None,
        purchase_date=date(2026, 2, 1),
    )

    output = tmp_path / "out.csv"
    result = _run_tax_report(log, tmp_path, "--year", "2026", "--output", str(output))
    assert result.exit_code == 0, result.output
    assert "no holding_period" in result.stderr
    # The unclassifiable row is excluded from both the listing and the totals,
    # so the footer matches the seeded rows alone.
    assert "Net +225.00" in result.output
    assert "2026-08-01" not in output.read_text()


def test_tax_report_no_sells_in_year_prints_message(tmp_path: Path) -> None:
    log = tmp_path / "trades.csv"
    _seed_log(log)
    result = _run_tax_report(log, tmp_path, "--year", "2099")
    assert result.exit_code == 0
    assert "No realized sales" in result.output


def test_tax_report_no_sells_with_output_writes_header_only_csvs(tmp_path: Path) -> None:
    log = tmp_path / "trades.csv"
    _seed_log(log)
    output = tmp_path / "empty.csv"
    result = _run_tax_report(log, tmp_path, "--year", "2099", "--output", str(output))
    assert result.exit_code == 0
    assert "No realized sales" in result.output
    assert "Wrote" in result.output
    assert output.read_text().startswith("ticker,")
    assert len(output.read_text().splitlines()) == 1
    summary_text = output.with_suffix(".summary.csv").read_text()
    assert summary_text.startswith("year,")
    assert len(summary_text.splitlines()) == 1


def test_tax_report_corrupt_log_is_clean_cli_error(tmp_path: Path) -> None:
    """A corrupt trade log surfaces as a CLI error with a line number, not a traceback."""
    log = tmp_path / "trades.csv"
    _seed_log(log)
    with open(log, "a") as handle:
        handle.write("2026-06-02,AAPL,SELL,abc,30.0,StopLoss,short-term,2026-01-01,20.0,100.0,0.5\n")

    result = _run_tax_report(log, tmp_path, "--year", "2026")
    assert result.exit_code == 1
    assert result.exception is None or not isinstance(result.exception, ValueError)
    assert "line 4" in result.stderr  # header + two seeded rows precede it


def test_tax_report_rejects_out_of_range_year(tmp_path: Path) -> None:
    log = tmp_path / "trades.csv"
    _seed_log(log)
    result = _run_tax_report(log, tmp_path, "--year", "12026")
    assert result.exit_code != 0
    assert "calendar year" in result.stderr


def test_tax_report_rejects_start_after_end(tmp_path: Path) -> None:
    log = tmp_path / "trades.csv"
    _seed_log(log)
    result = _run_tax_report(log, tmp_path, "--start", "2026-12-31", "--end", "2026-01-01")
    assert result.exit_code != 0
    assert "on or before" in result.stderr


def test_tax_report_rejects_year_combined_with_start_end(tmp_path: Path) -> None:
    log = tmp_path / "trades.csv"
    _seed_log(log)
    result = _run_tax_report(log, tmp_path, "--year", "2026", "--start", "2026-01-01", "--end", "2026-06-30")
    assert result.exit_code != 0
    assert "cannot be combined" in result.stderr


def test_tax_report_requires_tax_block(tmp_path: Path) -> None:
    log = tmp_path / "trades.csv"
    _seed_log(log)
    no_tax = "portfolio:\n  - ticker: AAPL\n    shares: 10\navailable_cash: 1000.0\n"
    result = _run_tax_report(log, tmp_path, "--year", "2026", portfolio_text=no_tax)
    assert result.exit_code != 0
    assert "no `tax:` block" in result.stderr


def test_tax_report_resolves_log_from_portfolio_state_file(tmp_path: Path) -> None:
    state = tmp_path / "live.state.yaml"
    log = tmp_path / "live.state.yaml.trades.csv"
    _seed_log(log)
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(f"{PORTFOLIO_YAML}state_file: {state}\n")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tax-report",
            "--portfolio",
            str(portfolio),
            "--year",
            "2026",
            "--output",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Net +225.00" in result.output
