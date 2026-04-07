"""Rich terminal output for alerts and status messages."""

from __future__ import annotations

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


def _color(value: float, fmt: str = ".2%") -> str:
    """Color-code a numeric value green/red based on sign."""
    style = "green" if value >= 0 else "red"
    return f"[{style}]{value:{fmt}}[/{style}]"


def print_backtest_summary(result: BacktestResult) -> None:
    import math

    sv, fv, bhv = result.starting_value, result.final_value, result.buy_and_hold_value
    total_return = (fv - sv) / sv if sv > 0 else 0
    bh_return = (bhv - sv) / sv if sv > 0 else 0

    # --- Returns ---
    table = Table(title="Backtest Summary", show_lines=True, width=BACKTEST_TABLE_WIDTH)
    table.add_column("Metric", style="bold", width=METRIC_COL_WIDTH)
    table.add_column("Value", justify="right", width=VALUE_COL_WIDTH)

    table.add_row("Starting Value", f"${sv:,.2f}")
    table.add_row("Final Value", f"${fv:,.2f}")
    table.add_row("Total Return", _color(total_return))
    table.add_row("CAGR", _color(result.cagr))
    table.add_row("Time-Weighted Return", _color(result.twr))
    table.add_row("Buy & Hold Value", f"${bhv:,.2f}")
    table.add_row("Buy & Hold Return", _color(bh_return))
    table.add_row("Total Trades", str(len(result.trades)))

    if result.split_date:
        table.add_section()
        table.add_row("Split Date", result.split_date.isoformat())
        table.add_row("Train Return", _color(result.train_return))
        table.add_row("Train B&H Return", _color(result.train_bh_return))
        table.add_row("Train Trades", str(len(result.train_trades)))
        table.add_row("Test Return", _color(result.test_return))
        table.add_row("Test B&H Return", _color(result.test_bh_return))
        table.add_row("Test Trades", str(len(result.test_trades)))
        table.add_row("Efficiency Ratio", f"{result.efficiency_ratio:.0%}")

    console.print(table, justify="center")

    # --- Risk Metrics ---
    risk_table = Table(title="Risk Metrics", show_lines=True, width=BACKTEST_TABLE_WIDTH)
    risk_table.add_column("Metric", style="bold", width=METRIC_COL_WIDTH)
    risk_table.add_column("Value", justify="right", width=VALUE_COL_WIDTH)

    risk_table.add_row("Max Drawdown", f"[red]{result.max_drawdown:.2%}[/red]")
    risk_table.add_row("Sharpe Ratio", _color(result.sharpe_ratio, fmt=".2f"))
    risk_table.add_row("Sortino Ratio", _color(result.sortino_ratio, fmt=".2f"))

    console.print(risk_table, justify="center")

    # --- Trade Quality ---
    sells = [t for t in result.trades if t.direction == Direction.SELL]
    if sells:
        trade_table = Table(title="Trade Quality", show_lines=True, width=BACKTEST_TABLE_WIDTH)
        trade_table.add_column("Metric", style="bold", width=METRIC_COL_WIDTH)
        trade_table.add_column("Value", justify="right", width=VALUE_COL_WIDTH)

        trade_table.add_row("Win Rate", _color(result.win_rate))
        pf_str = f"{result.profit_factor:.2f}" if not math.isinf(result.profit_factor) else "∞"
        trade_table.add_row("Profit Factor", pf_str)
        trade_table.add_row("Avg Win", f"[green]${result.avg_win:,.2f}[/green]")
        trade_table.add_row("Avg Loss", f"[red]${result.avg_loss:,.2f}[/red]")

        console.print(trade_table, justify="center")

    # --- Per-Strategy Breakdown ---
    if result.strategy_stats:
        strat_table = Table(title="Strategy Breakdown", show_lines=True, width=BACKTEST_TABLE_WIDTH)
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

        console.print(strat_table, justify="center")
