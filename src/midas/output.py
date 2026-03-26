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
    sig = order.signal
    color = "green" if sig.direction == Direction.BUY else "red"
    prefix = "[DRY RUN] " if dry_run else ""

    lines = [
        f"[bold]{sig.ticker}[/bold] — ${sig.price:,.2f}",
        sig.reasoning,
        f"Strength: {sig.strength}",
        f"Strategy: {sig.strategy_name}",
        f"Suggested order: {order.shares} share{'s' if order.shares != 1 else ''} "
        f"@ ${sig.price:,.2f} = ${order.estimated_value:,.2f}",
    ]

    if sig.direction == Direction.BUY:
        lines.append(f"Available cash after: ${remaining_cash:,.2f}")

    if order.relies_on_pending_cash:
        lines.append("[yellow]⚠ Includes pending cash infusion[/yellow]")

    console.print(Panel(
        "\n".join(lines),
        title=f"{prefix}[{color}][{sig.direction.value}][/{color}]",
        border_style=color,
        subtitle=timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
    ))


def print_circuit_breaker_alert(order: Order, *, dry_run: bool = False) -> None:
    sig = order.signal
    prefix = "[DRY RUN] " if dry_run else ""
    console.print(Panel(
        f"[bold]{sig.ticker}[/bold] — ${sig.price:,.2f}\n"
        f"{sig.reasoning}\n"
        f"Suggested order: 0 shares — [yellow]circuit breaker active[/yellow] "
        f"(daily deployment limit reached)",
        title=f"{prefix}[yellow][{sig.direction.value}][/yellow]",
        border_style="yellow",
    ))


def print_status(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")


def print_strategy_table(strategies: list[Strategy]) -> None:
    table = Table(title="Available Strategies")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Suitability", style="cyan")

    for s in strategies:
        tags = ", ".join(t.value for t in s.suitability)
        table.add_row(s.name, s.description, tags)

    console.print(table)


def print_backtest_summary(result: BacktestResult) -> None:
    sv, fv, bhv = result.starting_value, result.final_value, result.buy_and_hold_value
    total_return = (fv - sv) / sv if sv > 0 else 0
    bh_return = (bhv - sv) / sv if sv > 0 else 0

    table = Table(title="Backtest Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Starting Value", f"${sv:,.2f}")
    table.add_row("Final Value", f"${fv:,.2f}")
    table.add_row("Total Return", f"{total_return:.2%}")
    table.add_row("Buy & Hold Value", f"${bhv:,.2f}")
    table.add_row("Buy & Hold Return", f"{bh_return:.2%}")
    table.add_row("Total Trades", str(len(result.trades)))

    if result.split_date:
        table.add_section()
        table.add_row("Split Date", result.split_date.isoformat())
        table.add_row("Train Return", f"{result.train_return:.2%}")
        table.add_row("Train B&H Return", f"{result.train_bh_return:.2%}")
        table.add_row("Train Trades", str(len(result.train_trades)))
        table.add_row("Test Return", f"{result.test_return:.2%}")
        table.add_row("Test B&H Return", f"{result.test_bh_return:.2%}")
        table.add_row("Test Trades", str(len(result.test_trades)))

    console.print(table)
