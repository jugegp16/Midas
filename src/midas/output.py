"""Rich terminal output for alerts and status messages."""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from midas.models import Direction, Order
from midas.strategies.base import Strategy

if TYPE_CHECKING:
    from midas.backtest import BacktestResult

console = Console()


def print_alert(
    order: Order,
    remaining_cash: float,
    timestamp: datetime,
    *,
    dry_run: bool = False,
) -> None:
    color = "green" if order.direction == Direction.BUY else "red"
    prefix = "[DRY RUN] " if dry_run else ""

    ctx = order.context
    dominant = ctx.source

    lines = [
        f"[bold]{order.ticker}[/bold] — ${order.price:,.2f}",
        ctx.reason,
        f"Target weight: {ctx.target_weight:.1%} | Current: {ctx.current_weight:.1%}",
        f"Blended score: {ctx.blended_score:+.3f}",
        f"Primary strategy: {dominant}",
        f"Suggested order: {order.shares} share{'s' if order.shares != 1 else ''} "
        f"@ ${order.price:,.2f} = ${order.estimated_value:,.2f}",
    ]

    lines.append(f"Available cash after: ${remaining_cash:,.2f}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"{prefix}[{color}][{order.direction.value}][/{color}]",
            border_style=color,
            subtitle=timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
    )


def print_status(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")


def print_strategy_table(strategies: list[Strategy]) -> None:
    table = Table(title="Available Strategies")
    table.add_column("Name", style="bold")
    table.add_column("Tier", style="magenta")
    table.add_column("Description")
    table.add_column("Suitability", style="cyan")

    for s in strategies:
        tags = ", ".join(t.value for t in s.suitability)
        table.add_row(s.name, s.tier.value, s.description, tags)

    console.print(table, justify="center")


BACKTEST_TABLE_WIDTH = 100
# Split table in half so the column divider is centered.
# Account for 4 chars of box borders/separator (outer borders + center separator).
METRIC_COL_WIDTH = (BACKTEST_TABLE_WIDTH - 4) // 2
VALUE_COL_WIDTH = BACKTEST_TABLE_WIDTH - 4 - METRIC_COL_WIDTH


def color_signed(value: float, fmt: str = ".2%") -> str:
    """Color-code a numeric value green/red based on sign."""
    style = "green" if value >= 0 else "red"
    return f"[{style}]{value:{fmt}}[/{style}]"


def make_metric_table(title: str) -> Table:
    """2-column metric/value table — the canonical layout for summary outputs."""
    table = Table(title=title, show_lines=True, width=BACKTEST_TABLE_WIDTH)
    table.add_column("Metric", style="bold", width=METRIC_COL_WIDTH)
    table.add_column("Value", justify="right", width=VALUE_COL_WIDTH)
    return table


def make_wide_table(title: str, width: int = BACKTEST_TABLE_WIDTH) -> Table:
    """Multi-column table at the standard width (caller adds columns)."""
    return Table(title=title, title_style="bold", show_lines=True, width=width)


def print_centered(table: Table) -> None:
    """Centered render — used by all summary tables for visual consistency."""
    console.print(table, justify="center")


def print_run_info(rows: list[tuple[str, str]], title: str = "Run Info") -> None:
    """Render a small key/value table for run metadata (trials, output path, etc)."""
    table = make_metric_table(title)
    for k, v in rows:
        table.add_row(k, v)
    print_centered(table)


def print_params_table(
    title: str,
    params: dict[str, dict[str, float]],
    global_key: str | None = None,
) -> None:
    """Render an optimizer's per-strategy parameter table.

    `global_key`, if supplied, is the synthetic strategy name used to hold
    portfolio-wide allocation knobs; it's relabeled as "Global" for display.
    """
    table = make_wide_table(title)
    table.add_column("Strategy", style="bold")
    table.add_column("Parameters")
    for name, p in params.items():
        display = "Global" if global_key is not None and name == global_key else name
        table.add_row(display, ", ".join(f"{k}={v}" for k, v in p.items()))
    print_centered(table)


def print_backtest_summary(result: BacktestResult) -> None:
    sv, fv, bhv = result.starting_value, result.final_value, result.buy_and_hold_value
    total_return = (fv - sv) / sv if sv > 0 else 0
    bh_return = (bhv - sv) / sv if sv > 0 else 0

    # --- Performance ---
    perf = make_metric_table("Performance")
    perf.add_row("Starting Value", f"${sv:,.2f}")
    perf.add_row("Final Value", f"${fv:,.2f}")
    perf.add_row("Total Return", color_signed(total_return))
    perf.add_row("CAGR", color_signed(result.cagr))
    perf.add_row("Time-Weighted Return", color_signed(result.twr))
    perf.add_row("Buy & Hold Value", f"${bhv:,.2f}")
    perf.add_row("Buy & Hold Return", color_signed(bh_return))
    perf.add_row("Total Trades", str(len(result.trades)))
    print_centered(perf)

    # --- Train / Test Split ---
    if result.split_date:
        split = make_metric_table("Train / Test Split")
        split.add_row("Split Date", result.split_date.isoformat())
        split.add_row("Train Return", color_signed(result.train_return))
        split.add_row("Train B&H Return", color_signed(result.train_bh_return))
        split.add_row("Train Trades", str(len(result.train_trades)))
        split.add_row("Test Return", color_signed(result.test_return))
        split.add_row("Test B&H Return", color_signed(result.test_bh_return))
        split.add_row("Test Trades", str(len(result.test_trades)))
        split.add_row("Efficiency Ratio", f"{result.efficiency_ratio:.0%}")
        print_centered(split)

    # --- Risk Metrics ---
    risk_table = make_metric_table("Risk Metrics")
    risk_table.add_row("Max Drawdown", f"[red]{result.max_drawdown:.2%}[/red]")
    risk_table.add_row("Sharpe Ratio", color_signed(result.sharpe_ratio, fmt=".2f"))
    risk_table.add_row("Sortino Ratio", color_signed(result.sortino_ratio, fmt=".2f"))
    print_centered(risk_table)

    # --- Trade Quality ---
    if any(t.direction == Direction.SELL for t in result.trades):
        trade_table = make_metric_table("Trade Quality")
        trade_table.add_row("Win Rate", color_signed(result.win_rate))
        pf_str = f"{result.profit_factor:.2f}" if not math.isinf(result.profit_factor) else "∞"
        trade_table.add_row("Profit Factor", pf_str)
        trade_table.add_row("Avg Win", f"[green]${result.avg_win:,.2f}[/green]")
        trade_table.add_row("Avg Loss", f"[red]${result.avg_loss:,.2f}[/red]")
        print_centered(trade_table)

    # --- Per-Strategy Breakdown ---
    if result.strategy_stats:
        strat_table = make_wide_table("Strategy Breakdown")
        strat_table.add_column("Strategy", style="bold")
        strat_table.add_column("Trades", justify="right")
        strat_table.add_column("Buys", justify="right")
        strat_table.add_column("Sells", justify="right")
        strat_table.add_column("Win Rate", justify="right")
        strat_table.add_column("P&L", justify="right")

        for s in result.strategy_stats:
            pnl_style = "green" if s.pnl >= 0 else "red"
            strat_table.add_row(
                s.name,
                str(s.trades),
                str(s.buys),
                str(s.sells),
                f"{s.win_rate:.0%}" if s.sells > 0 else "—",
                f"[{pnl_style}]${s.pnl:,.2f}[/{pnl_style}]" if s.sells > 0 else "—",
            )

        print_centered(strat_table)
        # Per-strategy P&L is currently credited to whichever strategy
        # triggered the *exit*, not the strategy that found the entry.
        # Exit strategies (e.g. ProfitTaking, TrailingStop) therefore
        # capture gains that entry strategies (e.g. Momentum) actually
        # sourced. Trust the activity counts; take the P&L column with
        # a grain of salt until the tier redesign lands. See #26.
        console.print(
            "[dim italic]Note: P&L is credited to the exit strategy, not the "
            "entry strategy. Activity counts are accurate; P&L attribution is "
            "known to be misleading. See issue #26.[/dim italic]",
            justify="center",
        )
