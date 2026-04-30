"""Terminal ASCII charts for backtest summaries.

Built via ``plotext`` and emitted through Rich's ``Console`` so the chart
block aligns with the centered summary tables. Each function is a no-op
when its inputs are empty, so callers can render unconditionally without
guarding on ``risk_history`` shape.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import plotext as plt  # type: ignore[import-untyped]
from rich.console import Console
from rich.text import Text

if TYPE_CHECKING:
    from midas.results import BacktestResult


CHART_HEIGHT = 18
CHART_WIDTH = 100

console = Console()

# Strips CSI escape sequences (colors, styles) so visible width can be measured
# independently of the ANSI codes plotext emits.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def render_charts(result: BacktestResult) -> None:
    """Render the full chart panel — equity, drawdown, exposure, and vol target.

    Each chart is its own single-figure plot at the same ``CHART_WIDTH`` so all
    panels line up vertically. The drawdown chart always renders alongside the
    equity curve. The gross-exposure chart is emitted whenever ``risk_history``
    is populated; the predicted-vs-target vol chart only when the run was
    configured with a vol target and the history contains at least one
    non-zero predicted-vol sample.
    """
    if not result.equity_curve:
        return
    _render_equity(result)
    _render_drawdown(result)
    if result.risk_history is None or not result.risk_history.dates:
        return
    _render_gross_exposure(result)
    if (result.risk_metrics is not None and result.risk_metrics.vol_target is not None) and any(
        value > 0 for value in result.risk_history.predicted_vol
    ):
        _render_predicted_vs_target_vol(result)


def _flush_centered(title: str) -> None:
    """Print *title* on its own centered line, then the centered chart block.

    plotext anchors its own title flush-left within the chart frame, so we skip
    ``plt.title()`` and emit the heading separately through Rich — that way the
    title is centered relative to the terminal, not stuck to the left edge of
    the plot area. Chart lines vary in visible width (axis labels, legend,
    etc.); padding to a uniform width first keeps the figure as a single
    rectangular block when Rich centers it.
    """
    chart = plt.build().rstrip("\n")
    if not chart:
        return
    console.print()
    console.print(f"[bold]{title}[/bold]", justify="center")
    lines = chart.split("\n")
    visible_widths = [len(_ANSI_RE.sub("", line)) for line in lines]
    max_width = max(visible_widths) if visible_widths else 0
    padded = [line + " " * (max_width - width) for line, width in zip(lines, visible_widths, strict=True)]
    console.print(Text.from_ansi("\n".join(padded)), justify="center")


def _drawdown_pct_series(result: BacktestResult, dates: list[str]) -> list[float]:
    """Drawdown as a negative percentage so the chart reads as a downward dip."""
    if result.risk_history is not None and len(result.risk_history.drawdown) == len(dates):
        return [-value * 100.0 for value in result.risk_history.drawdown]
    equity = [value for _, value in result.equity_curve]
    peak = equity[0] if equity else 0.0
    out: list[float] = []
    for value in equity:
        peak = max(peak, value)
        out.append(-((peak - value) / peak) * 100.0 if peak > 0 else 0.0)
    return out


def _setup_single_figure() -> None:
    plt.clear_figure()
    plt.plot_size(CHART_WIDTH, CHART_HEIGHT)
    plt.theme("clear")
    plt.date_form("Y-m-d")


def _render_equity(result: BacktestResult) -> None:
    dates = [dt.isoformat() for dt, _ in result.equity_curve]
    equity = [value for _, value in result.equity_curve]
    _setup_single_figure()
    plt.plot(dates, equity, color="cyan", label="Strategy", marker="braille")
    if result.bh_equity_curve and len(result.bh_equity_curve) == len(dates):
        bh = [value for _, value in result.bh_equity_curve]
        plt.plot(dates, bh, color="yellow", label="Buy & Hold", marker="braille")
    plt.ylabel("Portfolio $")
    _flush_centered("Equity Curve")


def _render_drawdown(result: BacktestResult) -> None:
    dates = [dt.isoformat() for dt, _ in result.equity_curve]
    drawdown_pct = _drawdown_pct_series(result, dates)
    _setup_single_figure()
    plt.plot(dates, drawdown_pct, color="red", marker="braille")
    plt.ylabel("DD %")
    _flush_centered("Drawdown (%)")


def _render_gross_exposure(result: BacktestResult) -> None:
    history = result.risk_history
    assert history is not None  # checked by caller
    dates = [dt.isoformat() for dt in history.dates]
    gross_pct = [value * 100.0 for value in history.gross_exposure]

    _setup_single_figure()
    plt.plot(dates, gross_pct, color="green", label="Gross Exposure", marker="braille")

    if any(scale < 1.0 for scale in history.cppi_scale):
        cppi_pct = [scale * 100.0 for scale in history.cppi_scale]
        plt.plot(dates, cppi_pct, color="orange", label="CPPI Scale", marker="braille")

    plt.ylabel("%")
    _flush_centered("Gross Exposure / CPPI Scaling (%)")


def _render_predicted_vs_target_vol(result: BacktestResult) -> None:
    history = result.risk_history
    assert history is not None  # checked by caller
    metrics = result.risk_metrics
    assert metrics is not None and metrics.vol_target is not None

    dates = [dt.isoformat() for dt in history.dates]
    predicted_pct = [value * 100.0 for value in history.predicted_vol]
    target_pct = [metrics.vol_target * 100.0] * len(dates)

    _setup_single_figure()
    plt.plot(dates, predicted_pct, color="cyan", label="Predicted Vol", marker="braille")
    plt.plot(dates, target_pct, color="red", label="Target Vol", marker="braille")
    plt.ylabel("Vol %")
    _flush_centered("Predicted vs Target Annualized Vol (%)")
