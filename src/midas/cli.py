"""CLI interface — the sole user-facing interface for MVP."""

from __future__ import annotations

import csv
import sys
from collections.abc import Sequence
from datetime import MAXYEAR, MINYEAR, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import click
import pandas as pd
from rich.console import Console

from midas.allocator import Allocator
from midas.backtest import DEFAULT_TRAIN_PCT, BacktestEngine, ExecutionMode
from midas.config import load_portfolio, load_strategies
from midas.data import CachedYFinanceProvider
from midas.metrics import SHORT_WINDOW_THRESHOLD_DAYS
from midas.models import (
    DEFAULT_OBJECTIVE,
    AlertsConfig,
    AllocationConstraints,
    Direction,
    ForecastScaling,
    Objective,
    PortfolioConfig,
    RiskConfig,
    StrategyConfig,
    TaxConfig,
    TradeRecord,
)
from midas.order_sizer import OrderSizer
from midas.output import (
    color_signed,
    console,
    make_metric_table,
    make_wide_table,
    print_backtest_summary,
    print_centered,
    print_params_table,
    print_run_info,
    print_status,
    print_strategy_table,
)
from midas.results import write_backtest_results
from midas.strategies import STRATEGY_REGISTRY, EntrySignal, ExitRule, Strategy
from midas.strategies.base import max_warmup, warmup_bars_to_calendar_days
from midas.sweep import run_sweep
from midas.tax import TAX_BRACKET_YEAR, AnnualTaxSummary, compute_tax_summary, derive_tax_rates
from midas.trade_log import LoggedTrade, TradeLogError, read_trades

if TYPE_CHECKING:
    from midas.optimizer import FoldResult, OptimizeResult, WalkForwardResult

TAX_REPORT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "shares",
    "purchase_date",
    "sale_date",
    "cost_basis",
    "proceeds",
    "realized_pnl",
    "holding_period_days",
    "classification",
)

TAX_SUMMARY_COLUMNS: tuple[str, ...] = (
    "year",
    "st_realized",
    "lt_realized",
    "net_after_cross",
    "deductible_loss",
    "carry_forward",
    "tax_owed",
    "payment_date",
)


def _build_strategy(cfg: StrategyConfig) -> Strategy:
    """Instantiate a registered strategy from its config."""
    cls = STRATEGY_REGISTRY.get(cfg.name)
    if cls is None:
        msg = f"Unknown strategy '{cfg.name}'. Available: {', '.join(STRATEGY_REGISTRY)}"
        raise click.ClickException(msg)
    return cls(**cfg.params)


def _build_components(
    strategy_configs: list[StrategyConfig] | None,
    constraints: AllocationConstraints,
    n_tickers: int,
    risk_config: RiskConfig | None = None,
) -> tuple[Allocator, OrderSizer, list[ExitRule]]:
    """Build allocator, order sizer, and exit rules from config."""
    configs = strategy_configs or [StrategyConfig(name=name) for name in STRATEGY_REGISTRY]

    entries: list[tuple[EntrySignal, float]] = []
    exits: list[ExitRule] = []

    for cfg in configs:
        strategy = _build_strategy(cfg)

        if isinstance(strategy, ExitRule):
            exits.append(strategy)
        elif isinstance(strategy, EntrySignal):
            entries.append((strategy, cfg.weight))
        else:
            msg = f"Strategy {cfg.name!r} is neither EntrySignal nor ExitRule"
            raise click.ClickException(msg)

    allocator = Allocator(entries, constraints, n_tickers, risk_config=risk_config)
    order_sizer = OrderSizer()

    return allocator, order_sizer, exits


def _to_date(dt: date | datetime) -> date:
    """Coerce click.DateTime output to a plain date."""
    return dt.date() if isinstance(dt, datetime) else dt


def _load_strategy_bundle(
    strategies: str | None,
) -> tuple[list[StrategyConfig] | None, AllocationConstraints, RiskConfig]:
    """Load strategy configs from YAML, or defaults when no path is given."""
    if strategies:
        return load_strategies(Path(strategies))
    return None, AllocationConstraints(), RiskConfig()


def _resolve_state_path(port: PortfolioConfig, portfolio_path: Path) -> Path:
    """State file from the portfolio config, defaulting to `<portfolio>.state.yaml` beside it."""
    return port.state_file if port.state_file is not None else portfolio_path.with_suffix(".state.yaml")


def _stdin_is_interactive() -> bool:
    """Whether both stdin and stdout are TTYs, i.e. it is safe to prompt."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_tax_rates() -> tuple[float, float, str]:
    """Ask for capital-gains rates, directly or derived from income.

    Returns:
        ``(short_term_rate, long_term_rate, provenance)`` where provenance
        is the YAML comment describing where the numbers came from.
    """
    if click.confirm("Do you know your short/long-term capital-gains rates?", default=False):
        short_term = click.prompt("Short-term capital-gains rate (decimal fraction)", type=float, default=0.37)
        long_term = click.prompt("Long-term capital-gains rate (decimal fraction)", type=float, default=0.20)
        return short_term, long_term, "configured interactively"
    click.echo("No problem — they can be derived from your federal bracket.")
    status = click.prompt(
        "Filing status (married = filing jointly)",
        type=click.Choice(["single", "married"]),
        default="single",
    )
    income = click.prompt("Approximate annual taxable income (USD)", type=float)
    short_term, long_term = derive_tax_rates(status, income)
    click.echo(
        f"Derived rates: short-term {short_term:.1%}, long-term {long_term:.1%} "
        f"({TAX_BRACKET_YEAR} federal brackets; add state taxes to both if applicable)"
    )
    return short_term, long_term, f"derived from ~${income:,.0f} {status}, {TAX_BRACKET_YEAR} federal brackets"


def _ensure_tax_config(port: PortfolioConfig, portfolio_path: Path) -> TaxConfig | None:
    """Return the portfolio's tax config, offering one-time interactive setup.

    When the portfolio file has no ``tax:`` key at all and the session is
    interactive, ask whether to set up after-tax accounting and append the
    outcome to the file: a ``tax:`` block on yes, ``tax: off`` on no. A
    file that declares ``tax:`` either way is never prompted about. The
    income figure used for derivation is never written to the file.
    """
    if port.tax_declared:
        return port.tax_config
    if not _stdin_is_interactive():
        return None
    if not click.confirm(
        f"No tax config in {portfolio_path} — set up after-tax accounting now?",
        default=False,
    ):
        with open(portfolio_path, "a", encoding="utf-8") as handle:
            handle.write("\ntax: off  # after-tax accounting disabled; delete this line to be asked again\n")
        click.echo(f"Wrote `tax: off` to {portfolio_path}.")
        return None
    while True:
        try:
            short_term, long_term, provenance = _prompt_tax_rates()
            cap = click.prompt("Deductible loss cap (USD/year)", type=float, default=3000.0)
            lag = click.prompt("Payment lag (days from year-end to payment)", type=int, default=105)
            tax_config = TaxConfig(
                short_term_rate=short_term,
                long_term_rate=long_term,
                deductible_loss_cap=cap,
                payment_lag_days=lag,
            )
            break
        except ValueError as exc:
            click.echo(f"Invalid tax config: {exc}. Let's try again.", err=True)
    block = (
        f"\ntax:  # {provenance}\n"
        f"  short_term_rate: {tax_config.short_term_rate}\n"
        f"  long_term_rate: {tax_config.long_term_rate}\n"
        f"  deductible_loss_cap: {tax_config.deductible_loss_cap}\n"
        f"  payment_lag_days: {tax_config.payment_lag_days}\n"
    )
    with open(portfolio_path, "a", encoding="utf-8") as handle:
        handle.write(block)
    click.echo(f"Wrote tax config to {portfolio_path}.")
    return tax_config


def _fetch_prices(
    portfolio: PortfolioConfig,
    start: date,
    end: date,
    warmup_bars: int = 0,
) -> dict[str, pd.DataFrame]:
    """Fetch price history from ``start - warmup_buffer`` through ``end``.

    ``warmup_bars`` is the maximum number of trading days any configured
    strategy needs before it can emit valid scores. The fetch start is
    shifted backward by an equivalent calendar-day buffer so strategies
    have history available on day one of the simulation.
    """
    provider = CachedYFinanceProvider()
    price_data: dict[str, pd.DataFrame] = {}
    tickers = [holding.ticker for holding in portfolio.holdings]
    buffer_days = warmup_bars_to_calendar_days(warmup_bars)
    fetch_start = start - timedelta(days=buffer_days)
    print_status(f"Fetching data for {', '.join(tickers)} (with {buffer_days}-day warmup buffer from {fetch_start})...")
    for ticker in tickers:
        try:
            price_data[ticker] = provider.get_history(ticker, fetch_start, end)
        except Exception as exc:
            print_status(f"Skipping {ticker}: {exc}")
    return price_data


@click.group()
def cli() -> None:
    """Midas — Portfolio Signal Engine."""


@cli.command()
@click.option(
    "--portfolio",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to portfolio YAML config.",
)
@click.option(
    "--strategies",
    "-s",
    default=None,
    type=click.Path(exists=True),
    help="Path to strategies YAML config. Defaults to all strategies.",
)
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--output",
    "-o",
    default="backtest_results",
    help="Output directory path.",
)
@click.option(
    "--train-pct",
    default=DEFAULT_TRAIN_PCT,
    help="Train/test split ratio (0-1).",
)
@click.option("--no-split", is_flag=True, help="Disable train/test split.")
@click.option(
    "--execution-mode",
    type=click.Choice(["close", "next_open", "next_close"]),
    default="next_open",
    show_default=True,
    help=(
        "When orders computed on day T actually fill. "
        "'close' = same day (legacy, optimistic); "
        "'next_open' = next session's open (honest default); "
        "'next_close' = next session's close."
    ),
)
@click.option(
    "--charts/--no-charts",
    default=True,
    show_default=True,
    help="Render terminal ASCII charts (equity, drawdown, exposure) after the summary.",
)
def backtest(
    portfolio: str,
    strategies: str | None,
    start: date,
    end: date,
    output: str,
    train_pct: float,
    no_split: bool,
    execution_mode: ExecutionMode,
    charts: bool,
) -> None:
    """Run a backtest over historical data."""
    port = load_portfolio(Path(portfolio))
    # Ask before the (slow) price download so the prompt isn't buried
    # behind fetch output or left waiting for a user who walked away.
    tax_config = _ensure_tax_config(port, Path(portfolio))
    strat_configs, constraints, risk_config = _load_strategy_bundle(strategies)

    start_d, end_d = _to_date(start), _to_date(end)

    allocator, order_sizer, exit_rules = _build_components(
        strat_configs,
        constraints,
        port.active_ticker_count(),
        risk_config=risk_config,
    )

    warmup_bars = max_warmup([*allocator.strategies, *exit_rules])
    price_data = _fetch_prices(port, start_d, end_d, warmup_bars=warmup_bars)

    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=order_sizer,
        exit_rules=exit_rules,
        constraints=constraints,
        train_pct=train_pct,
        enable_split=not no_split,
        log_fn=print_status,
        execution_mode=execution_mode,
        tax_config=tax_config,
    )

    print_status("Running backtest...")
    result = engine.run(port, price_data, start_d, end_d)

    out_path = Path(output)
    write_backtest_results(result, out_path)
    print_status(f"Results written to {out_path}/")
    print_backtest_summary(result, show_charts=charts)


@cli.command()
@click.option(
    "--portfolio",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to portfolio YAML config.",
)
@click.option(
    "--strategies",
    "-s",
    default=None,
    type=click.Path(exists=True),
    help="Path to strategies YAML config. Defaults to all strategies.",
)
@click.option("--interval", default=60, help="Poll interval in seconds.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Observe signals with in-memory fills: no state file, trade log, or Discord writes.",
)
@click.option(
    "--ignore-market-hours",
    is_flag=True,
    help="Poll 24/7 instead of only during US equity sessions (debugging).",
)
def live(
    portfolio: str,
    strategies: str | None,
    interval: int,
    dry_run: bool,
    ignore_market_hours: bool,
) -> None:
    """Run live analysis with real-time price polling."""
    from midas.alerts import build_discord
    from midas.live import LiveEngine

    portfolio_path = Path(portfolio)
    port = load_portfolio(portfolio_path)
    state_path = _resolve_state_path(port, portfolio_path)
    alerts_cfg = port.alerts_config or AlertsConfig()
    try:
        confirmer, reporter = build_discord(port.alerts_config)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if confirmer is not None and not dry_run:
        click.echo("Discord confirm mode: orders fill only on \u2705 reaction.")
    else:
        click.echo("Terminal-only mode: alerts assume immediate execution.")
    if reporter is not None and not ignore_market_hours and not dry_run:
        click.echo("End-of-day report: Discord.")
    # Setup opportunity: live's trade log is what tax-report consumes later.
    _ensure_tax_config(port, portfolio_path)
    strat_configs, constraints, risk_config = _load_strategy_bundle(strategies)
    provider = CachedYFinanceProvider()

    allocator, order_sizer, exit_rules = _build_components(
        strat_configs,
        constraints,
        port.active_ticker_count(),
        risk_config=risk_config,
    )

    with LiveEngine(
        portfolio=port,
        allocator=allocator,
        order_sizer=order_sizer,
        provider=provider,
        state_path=state_path,
        exit_rules=exit_rules,
        constraints=constraints,
        poll_interval=interval,
        dry_run=dry_run,
        confirmer=confirmer,
        pending_ttl_hours=alerts_cfg.pending_ttl_hours,
        realert_hours=alerts_cfg.realert_hours,
        reporter=reporter,
        market_hours_only=not ignore_market_hours,
    ) as engine:
        engine.run()


@cli.command()
@click.option(
    "-p",
    "--portfolio",
    required=True,
    type=click.Path(exists=True),
    help="Path to portfolio YAML config.",
)
@click.option(
    "--wait-reaction",
    is_flag=True,
    help="After posting the intra-day test message, wait for you to react to verify the confirm round trip.",
)
def doctor(portfolio: str, wait_reaction: bool) -> None:
    """Check that configured integrations actually work, with fix hints.

    Posts clearly-labeled test messages to the configured Discord
    channels and verifies every capability live mode depends on: token,
    channel visibility, posting, reading back (confirm-mode polling),
    and editing. Never touches live state.
    """
    import os

    from midas.alerts import DISCORD_BOT_TOKEN_ENV_VAR
    from midas.doctor import run_discord_checks

    portfolio_path = Path(portfolio)
    port = load_portfolio(portfolio_path)
    result = run_discord_checks(
        port.alerts_config,
        os.environ.get(DISCORD_BOT_TOKEN_ENV_VAR, ""),
        _resolve_state_path(port, portfolio_path),
        CachedYFinanceProvider(),
        echo=click.echo,
        wait_reaction=wait_reaction,
    )
    click.echo(
        ("healthy" if result.healthy else "NOT healthy")
        + f" — {result.checks} checks, {result.failures} failed, {result.warnings} warning(s)"
    )
    if not result.healthy:
        raise SystemExit(1)


def _resolve_report_period(
    year: int | None,
    start: datetime | None,
    end: datetime | None,
) -> tuple[date, date, str]:
    """Resolve --year or --start/--end into a report window.

    Returns:
        ``(start, end, period_label)`` for the report window.

    Raises:
        click.UsageError: If the options are missing, conflicting, or invalid.
    """
    if year is not None:
        if start is not None or end is not None:
            raise click.UsageError("--year cannot be combined with --start/--end")
        if not MINYEAR <= year <= MAXYEAR:
            raise click.UsageError(f"--year must be a calendar year ({MINYEAR}-{MAXYEAR}), got {year}")
        return date(year, 1, 1), date(year, 12, 31), str(year)
    if start is not None and end is not None:
        start_d = _to_date(start)
        end_d = _to_date(end)
        if start_d > end_d:
            raise click.UsageError(f"--start ({start_d}) must be on or before --end ({end_d})")
        return start_d, end_d, f"{start_d.isoformat()}_{end_d.isoformat()}"
    raise click.UsageError("either --year or both --start and --end must be provided")


def _resolve_trades_path(port: PortfolioConfig, portfolio_path: Path, from_trades: str | None) -> Path:
    """Locate the trade log, preferring an explicit --from-trades path.

    Raises:
        click.UsageError: If the state-file-derived trade log does not exist.
    """
    if from_trades is not None:
        return Path(from_trades)
    state_path = _resolve_state_path(port, portfolio_path)
    trades_path = state_path.with_suffix(state_path.suffix + ".trades.csv")
    if not trades_path.exists():
        msg = f"trade log not found at {trades_path}"
        raise click.UsageError(msg)
    return trades_path


def _collect_usable_sells(logged_rows: Sequence[LoggedTrade]) -> list[LoggedTrade]:
    """Filter the log to SELL rows usable for reporting, warning about gaps."""
    usable_sells: list[LoggedTrade] = []
    for row in logged_rows:
        if row.direction != Direction.SELL:
            continue
        if row.holding_period is None:
            click.echo(
                f"warning: {row.date.isoformat()} {row.ticker} SELL has no holding_period; "
                "excluded from report rows and totals",
                err=True,
            )
            continue
        if row.cost_basis is None:
            click.echo(
                f"warning: {row.date.isoformat()} {row.ticker} SELL has no cost_basis; "
                "assuming basis equals sale price (realized P&L $0)",
                err=True,
            )
        usable_sells.append(row)
    return usable_sells


def _basis_or_price(row: LoggedTrade) -> float:
    """Per-share cost basis, falling back to sale price when unrecorded."""
    return row.cost_basis if row.cost_basis is not None else row.price


def _write_empty_report(out_path: Path) -> None:
    """Write header-only report and summary CSVs for a window with no sales."""
    out_path.write_text(",".join(TAX_REPORT_COLUMNS) + "\n")
    summary_path = out_path.with_suffix(".summary.csv")
    summary_path.write_text(",".join(TAX_SUMMARY_COLUMNS) + "\n")
    click.echo(f"Wrote {out_path}")
    click.echo(f"Wrote {summary_path}")


@cli.command(name="tax-report")
@click.option(
    "--portfolio",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help=(
        "Portfolio YAML containing the tax: block; also resolves the trade log "
        "next to its state file unless --from-trades is given."
    ),
)
@click.option(
    "--from-trades",
    "from_trades",
    default=None,
    type=click.Path(exists=True),
    help="Explicit path to a trades.csv. Overrides --portfolio resolution.",
)
@click.option("--year", type=int, default=None, help="Calendar year to report (e.g. 2026).")
@click.option("--start", type=click.DateTime(formats=["%Y-%m-%d"]), default=None)
@click.option("--end", type=click.DateTime(formats=["%Y-%m-%d"]), default=None)
@click.option(
    "--output",
    "-o",
    default=None,
    help=(
        "Output CSV path. Defaults to schedule_d_<year>.csv (or schedule_d_<start>_<end>.csv). "
        "Per-year aggregates land in a companion summary CSV named by swapping the extension "
        "(schedule_d_2026.csv -> schedule_d_2026.summary.csv)."
    ),
)
def tax_report(
    portfolio: str,
    from_trades: str | None,
    year: int | None,
    start: datetime | None,
    end: datetime | None,
    output: str | None,
) -> None:
    """Year-end realized-P&L report (Schedule D-shaped) from a trade log."""
    start_d, end_d, period_label = _resolve_report_period(year, start, end)

    portfolio_path = Path(portfolio)
    port = load_portfolio(portfolio_path)
    tax_config = _ensure_tax_config(port, portfolio_path)
    if tax_config is None:
        msg = (
            "portfolio file has no `tax:` block; tax-report requires configured rates "
            "(short_term_rate, long_term_rate). Run interactively to be walked through "
            "setup (delete any `tax: off` line first), or see docs/tax-reporting.md."
        )
        raise click.UsageError(msg)

    trades_path = _resolve_trades_path(port, portfolio_path, from_trades)

    # Netting and carryforward are annual and cumulative, so the tax engine
    # must see every SELL in the log — a loss realized before the report
    # window carries into it. The window only filters what gets displayed.
    try:
        logged_rows = read_trades(trades_path)
    except TradeLogError as exc:
        raise click.ClickException(str(exc)) from exc

    usable_sells = _collect_usable_sells(logged_rows)
    rows = [row for row in usable_sells if start_d <= row.date <= end_d]

    if not rows:
        click.echo(f"No realized sales in {period_label}.")
        if output is not None:
            _write_empty_report(Path(output))
        return

    out_path = Path(output) if output is not None else Path(f"schedule_d_{period_label}.csv")

    trade_records = [
        TradeRecord(
            date=row.date,
            ticker=row.ticker,
            direction=row.direction,
            shares=row.shares,
            price=row.price,
            strategy_name=row.strategy_name,
            holding_period=row.holding_period,
            purchase_date=row.purchase_date,
        )
        for row in usable_sells
    ]
    basis_per_sell = [_basis_or_price(row) for row in usable_sells]

    # end_date only clamps payment dates to a backtest's final bar; a report
    # has no curve to clamp against, so surface the natural payment dates.
    full_summary = compute_tax_summary(trade_records, basis_per_sell, tax_config, end_date=date.max)
    summary = [entry for entry in full_summary if start_d.year <= entry.year <= end_d.year]
    period_basis = [_basis_or_price(row) for row in rows]
    _print_tax_report(rows, period_basis, summary, period_label)
    _write_tax_report_csv(rows, period_basis, out_path)
    summary_path = out_path.with_suffix(".summary.csv")
    _write_tax_summary_csv(summary, summary_path)
    click.echo(f"\nWrote {out_path}")
    click.echo(f"Wrote {summary_path}")


def _validate_optimize_options(
    walk_forward: bool,
    train_pct: float,
    wf_min_train_pct: float | None,
    wf_min_test_days: int | None,
) -> None:
    """Validate optimize-command option combinations.

    Raises:
        click.UsageError: On conflicting or out-of-range option values.
    """
    if walk_forward and train_pct != DEFAULT_TRAIN_PCT:
        raise click.UsageError("--train-pct cannot be used with --walk-forward (walk-forward has its own split logic).")
    if not walk_forward and (wf_min_train_pct is not None or wf_min_test_days is not None):
        raise click.UsageError("--wf-min-train-pct and --wf-min-test-days require --walk-forward.")
    if train_pct <= 0 or train_pct > 1:
        raise click.UsageError("--train-pct must be in (0, 1].")
    if wf_min_train_pct is not None and (wf_min_train_pct <= 0 or wf_min_train_pct >= 1):
        raise click.UsageError("--wf-min-train-pct must be in (0, 1).")
    if wf_min_test_days is not None and wf_min_test_days < 1:
        raise click.UsageError("--wf-min-test-days must be >= 1.")


@cli.command()
@click.option(
    "--portfolio",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to portfolio YAML config.",
)
@click.option(
    "--strategies",
    "-s",
    default=None,
    type=click.Path(exists=True),
    help="Strategies to optimize. Defaults to all.",
)
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--output",
    "-o",
    default="optimized_strategies.yaml",
    help="Output YAML path.",
)
@click.option(
    "--n-trials",
    "-n",
    default=200,
    show_default=True,
    help="Number of Optuna optimisation trials.",
)
@click.option(
    "--train-pct",
    default=DEFAULT_TRAIN_PCT,
    show_default=True,
    help="Train/test split ratio (0-1).",
)
@click.option(
    "--walk-forward",
    is_flag=True,
    default=False,
    help="Use walk-forward optimization (auto-determines folds from date range).",
)
@click.option(
    "--wf-min-train-pct",
    default=None,
    type=float,
    help="Walk-forward: minimum initial training window as fraction of data (0-1). Default 0.60.",
)
@click.option(
    "--wf-min-test-days",
    default=None,
    type=int,
    help="Walk-forward: minimum trading days per test fold. Default 63 (~3 months).",
)
@click.option(
    "--search-globals",
    is_flag=True,
    default=False,
    help="Legacy full search space: exit-rule parameters and allocator globals become searchable. "
    "Incompatible with a strategies file that pins exit rules.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Sampler seed. Same seed + same inputs reproduces the search exactly; vary it to measure search noise.",
)
@click.option(
    "--objective",
    type=click.Choice(get_args(Objective.__value__)),
    default=DEFAULT_OBJECTIVE,
    show_default=True,
    help="What trials maximize on the training window: calmar (return over max drawdown), "
    "sharpe (annualized Sharpe), gross (raw return), net (return after tax; needs a tax: block), "
    "ulcer (return over Ulcer Index), robust (lower-quartile block return).",
)
@click.pass_context
def optimize(
    ctx: click.Context,
    portfolio: str,
    strategies: str | None,
    start: date,
    end: date,
    output: str,
    n_trials: int,
    train_pct: float,
    walk_forward: bool,
    wf_min_train_pct: float | None,
    wf_min_test_days: int | None,
    search_globals: bool,
    seed: int,
    objective: Objective,
) -> None:
    """Find optimal strategy parameters via Bayesian optimisation (Optuna TPE).

    By default only entry-signal parameters and weights are searched; exit
    rules and allocator globals are policy-owned and pass through the search
    untouched. Without -s, all entry signals are searched and no exit rules
    run — pin exits in a strategies file for any run you intend to act on.
    """
    from midas.optimizer import max_warmup_for_search
    from midas.strategies import STRATEGY_REGISTRY
    from midas.strategies.base import EntrySignal

    _validate_optimize_options(walk_forward, train_pct, wf_min_train_pct, wf_min_test_days)

    if strategies and ctx.get_parameter_source("objective") == click.core.ParameterSource.COMMANDLINE:
        from midas.config import load_policy

        if load_policy(Path(strategies)) is not None:
            raise click.UsageError(
                "the strategies file has a policy: block whose objective governs; "
                "--objective would silently disagree with it — edit the policy instead"
            )

    port = load_portfolio(Path(portfolio))
    tax_config = _tax_config_for_objective(port, Path(portfolio), objective)

    strategy_names: list[str] | None = None
    strat_constraints = AllocationConstraints()
    risk_config: RiskConfig = RiskConfig()
    exit_configs: list[StrategyConfig] = []
    if strategies:
        strat_configs, strat_constraints, risk_config = load_strategies(Path(strategies))
        entry_configs = [c for c in strat_configs if issubclass(STRATEGY_REGISTRY[c.name], EntrySignal)]
        exit_configs = [c for c in strat_configs if c not in entry_configs]
        if search_globals:
            if exit_configs:
                raise click.UsageError(
                    "--search-globals searches exit parameters; remove the pinned exit rules "
                    f"({', '.join(c.name for c in exit_configs)}) from the strategies file"
                )
            strategy_names = [cfg.name for cfg in strat_configs]
        else:
            strategy_names = [cfg.name for cfg in entry_configs]
    min_cash_pct = strat_constraints.min_cash_pct
    forecast_scaling = strat_constraints.forecast_scaling
    exit_params = None if search_globals else {c.name: dict(c.params) for c in exit_configs}
    constraints = None if search_globals else strat_constraints

    start_d, end_d = _to_date(start), _to_date(end)
    warmup_bars = max_warmup_for_search(strategy_names, min_cash_pct, port.active_ticker_count())
    price_data = _fetch_prices(port, start_d, end_d, warmup_bars=warmup_bars)

    try:
        _run_optimizer_and_write(
            port,
            price_data,
            start_d,
            end_d,
            strategy_names=strategy_names,
            n_trials=n_trials,
            min_cash_pct=min_cash_pct,
            risk_config=risk_config,
            forecast_scaling=forecast_scaling,
            objective=objective,
            tax_config=tax_config,
            seed=seed,
            search_globals=search_globals,
            exit_params=exit_params,
            constraints=constraints,
            exit_configs=exit_configs,
            output=output,
            walk_forward=walk_forward,
            train_pct=train_pct,
            wf_min_train_pct=wf_min_train_pct,
            wf_min_test_days=wf_min_test_days,
        )
    except ValueError as exc:
        # The optimizer rejects windows it cannot split (e.g. a train_pct that
        # leaves no training days once the trading calendar is known).
        raise click.ClickException(str(exc)) from exc


def _run_optimizer_and_write(
    port: PortfolioConfig,
    price_data: dict[str, pd.DataFrame],
    start_d: date,
    end_d: date,
    *,
    strategy_names: list[str] | None,
    n_trials: int,
    min_cash_pct: float,
    risk_config: RiskConfig | None,
    forecast_scaling: ForecastScaling,
    objective: Objective,
    tax_config: TaxConfig | None,
    seed: int,
    search_globals: bool,
    exit_params: dict[str, dict[str, float | int | str]] | None,
    constraints: AllocationConstraints | None,
    exit_configs: list[StrategyConfig],
    output: str,
    walk_forward: bool,
    train_pct: float,
    wf_min_train_pct: float | None,
    wf_min_test_days: int | None,
) -> None:
    """Run the chosen optimizer mode, write the strategies YAML, and print its report."""
    from midas.optimizer import (
        WF_MIN_TEST_DAYS,
        WF_MIN_TRAIN_PCT,
        walk_forward_optimize,
        write_strategies_yaml,
    )
    from midas.optimizer import (
        optimize as run_optimize,
    )

    if walk_forward:
        wf_result = walk_forward_optimize(
            portfolio=port,
            price_data=price_data,
            start=start_d,
            end=end_d,
            strategy_names=strategy_names,
            n_trials=n_trials,
            min_cash_pct=min_cash_pct,
            min_train_pct=wf_min_train_pct or WF_MIN_TRAIN_PCT,
            min_test_days=wf_min_test_days or WF_MIN_TEST_DAYS,
            log_fn=print_status,
            risk_config=risk_config,
            forecast_scaling=forecast_scaling,
            objective=objective,
            tax_config=tax_config,
            seed=seed,
            search_globals=search_globals,
            exit_params=exit_params,
            constraints=constraints,
        )
        write_strategies_yaml(
            wf_result.best_params,
            output,
            min_cash_pct=min_cash_pct,
            risk_config=risk_config,
            forecast_scaling=forecast_scaling,
            objective=objective,
            exit_configs=exit_configs,
            constraints=constraints,
        )
        _print_walk_forward_report(wf_result, output)
    else:
        result = run_optimize(
            portfolio=port,
            price_data=price_data,
            start=start_d,
            end=end_d,
            strategy_names=strategy_names,
            n_trials=n_trials,
            min_cash_pct=min_cash_pct,
            train_pct=train_pct,
            log_fn=print_status,
            risk_config=risk_config,
            forecast_scaling=forecast_scaling,
            objective=objective,
            tax_config=tax_config,
            seed=seed,
            search_globals=search_globals,
            exit_params=exit_params,
            constraints=constraints,
        )
        write_strategies_yaml(
            result.best_params,
            output,
            min_cash_pct=min_cash_pct,
            risk_config=risk_config,
            forecast_scaling=forecast_scaling,
            objective=objective,
            exit_configs=exit_configs,
            constraints=constraints,
        )
        _print_optimize_report(result, train_pct, output)


def _tax_config_for_objective(port: PortfolioConfig, portfolio_path: Path, objective: Objective) -> TaxConfig | None:
    """Resolve the tax config the optimizer runs with.

    Any declared tax config is passed through so the final report carries
    after-tax figures under every objective. ``net`` additionally needs one:
    offer the one-time interactive setup when the file is silent on tax, and
    fail before any prices are fetched when it still isn't configured.

    Raises:
        click.ClickException: ``--objective net`` with after-tax accounting off.
    """
    if objective != "net":
        return port.tax_config
    tax_config = _ensure_tax_config(port, portfolio_path)
    if tax_config is None:
        raise click.ClickException(
            f"--objective net requires a tax: block in {portfolio_path} (after-tax accounting is off). "
            "Add one, or pick another --objective."
        )
    return tax_config


def _print_fold_table(folds: Sequence[FoldResult]) -> None:
    """Render the per-fold walk-forward results table."""
    # Per-fold results — wider since this table has 9 columns (10 with tax).
    fold_table = make_wide_table("Walk-Forward Analysis", width=140)
    fold_table.add_column("Fold", justify="center", style="bold")
    fold_table.add_column("IS Period")
    fold_table.add_column("OOS Period")
    fold_table.add_column("IS Return (Annualized)", justify="right")
    fold_table.add_column("OOS Return (Annualized)", justify="right")
    taxed = any(fold.after_tax_test_return is not None for fold in folds)
    if taxed:
        fold_table.add_column("OOS After-Tax (Annualized)", justify="right")
    fold_table.add_column("Max DD", justify="right")
    fold_table.add_column("Sharpe", justify="right")
    fold_table.add_column("Sortino", justify="right")
    fold_table.add_column("Win Rate", justify="right")
    for fold in folds:
        after_tax = (
            [color_signed(fold.after_tax_test_return) if fold.after_tax_test_return is not None else "—"]
            if taxed
            else []
        )
        fold_table.add_row(
            str(fold.fold),
            f"{fold.train_start} → {fold.train_end}",
            f"{fold.test_start} → {fold.test_end}",
            color_signed(fold.train_return),
            color_signed(fold.test_return),
            *after_tax,
            f"[red]{fold.max_drawdown:.2%}[/red]",
            color_signed(fold.sharpe_ratio, fmt=".2f"),
            color_signed(fold.sortino_ratio, fmt=".2f"),
            f"{fold.win_rate:.0%}" if fold.win_rate > 0 else "—",
        )
    print_centered(fold_table)


def _print_short_fold_note(folds: Sequence[FoldResult]) -> None:
    """Warn when some OOS windows are too short for stable annualization."""
    short_folds = [fold for fold in folds if (fold.test_end - fold.test_start).days < SHORT_WINDOW_THRESHOLD_DAYS]
    if not short_folds:
        return
    shortest = min((fold.test_end - fold.test_start).days for fold in short_folds)
    console.print(
        f"[yellow]Note: {len(short_folds)}/{len(folds)} OOS windows are "
        f"under one year (shortest: {shortest} days). Per-fold annualized returns "
        f"extrapolate from short samples and can be noisy.[/yellow]",
        justify="center",
    )


def _print_low_trials_note(wf_result: WalkForwardResult) -> None:
    """Warn when the per-fold trial budget leaves results inside sampler noise."""
    from midas.optimizer import MIN_TRIALS_PER_FOLD

    n_folds = len(wf_result.folds)
    if not n_folds:
        return
    per_fold = wf_result.total_trials // n_folds
    if per_fold >= MIN_TRIALS_PER_FOLD:
        return
    suggested = MIN_TRIALS_PER_FOLD * n_folds
    console.print(
        f"[yellow]Note: {per_fold} trials per fold is below the {MIN_TRIALS_PER_FOLD} "
        f"needed for the search to settle — results at this budget are dominated by "
        f"sampler noise, and re-running with a different --seed can move OOS return "
        f"by several points. Use -n {suggested} or more for a decision you intend to "
        f"act on.[/yellow]",
        justify="center",
    )


def _print_wf_aggregate(wf_result: WalkForwardResult) -> None:
    """Render the aggregate walk-forward metrics table."""
    # Aggregate metrics — same layout as the backtest summary tables.
    n_folds = len(wf_result.folds)
    agg = make_metric_table("Walk-Forward Aggregate")
    agg.add_row("Annualized OOS Return (CAGR)", color_signed(wf_result.annualized_return))
    if wf_result.after_tax_annualized_return is not None:
        agg.add_row("Annualized OOS Return After-Tax", color_signed(wf_result.after_tax_annualized_return))
    agg.add_row(
        "Per-Fold OOS Mean ± Std (Annualized)",
        f"{wf_result.mean_test_return:.2%} ± {wf_result.std_test_return:.2%}",
    )
    agg.add_row("Winning Folds", f"{wf_result.winning_folds}/{n_folds}")
    agg.add_row(
        "Best / Worst Fold",
        f"{color_signed(wf_result.best_fold_return)} / {color_signed(wf_result.worst_fold_return)}",
    )
    agg.add_row("Efficiency Ratio", f"{wf_result.efficiency_ratio:.0%}")
    agg.add_row("Mean Max Drawdown", f"[red]{wf_result.mean_max_drawdown:.2%}[/red]")
    agg.add_row("Mean Sharpe Ratio", color_signed(wf_result.mean_sharpe, fmt=".2f"))
    agg.add_row("Mean Sortino Ratio", color_signed(wf_result.mean_sortino, fmt=".2f"))
    agg.add_row(
        "Mean Win Rate",
        f"{wf_result.mean_win_rate:.0%}" if wf_result.mean_win_rate > 0 else "—",
    )
    print_centered(agg)


def _print_walk_forward_report(wf_result: WalkForwardResult, output: str) -> None:
    """Render the full walk-forward console report."""
    from midas.optimizer import ALLOCATION_KEY

    console.print()
    _print_fold_table(wf_result.folds)
    _print_short_fold_note(wf_result.folds)
    _print_low_trials_note(wf_result)
    _print_wf_aggregate(wf_result)
    print_run_info(
        [
            ("Objective", wf_result.objective),
            ("Total Trials", str(wf_result.total_trials)),
            ("Output", output),
        ]
    )
    print_params_table(
        "Deployed Parameters (from latest fold)",
        wf_result.best_params,
        global_key=ALLOCATION_KEY,
    )


def _print_optimize_report(result: OptimizeResult, train_pct: float, output: str) -> None:
    """Render the single-run optimization console report."""
    from midas.optimizer import ALLOCATION_KEY, format_objective

    console.print()

    # Reuse the backtest summary tables for the optimized strategy.
    assert result.best_result is not None
    print_backtest_summary(result.best_result)

    best = format_objective(result.best_objective_value, result.objective)
    print_run_info(
        [
            ("Objective", f"{result.objective} (best in-sample: {best})"),
            ("Trials", str(result.trials_run)),
            ("Train/Test Split", f"{train_pct:.0%} / {1 - train_pct:.0%}" if train_pct < 1.0 else "100% / 0%"),
            ("Output", output),
        ]
    )
    print_params_table("Optimized Parameters", result.best_params, global_key=ALLOCATION_KEY)


@cli.command()
@click.option("--portfolio", "-p", required=True, type=click.Path(exists=True), help="Path to portfolio YAML.")
@click.option(
    "--strategies", "-s", required=True, type=click.Path(exists=True), help="Strategies YAML with a policy: block."
)
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]), help="Earliest data to fit on.")
@click.option(
    "--as-of",
    default=None,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Fit on data strictly before this date (default: today).",
)
@click.option("--rollback", default=0, type=int, help="Restore the members deployed N fits ago instead of fitting.")
def fit(portfolio: str, strategies: str, start: date, as_of: date | None, rollback: int) -> None:
    """Fit deployable members per the strategies file's policy block.

    Writes the machine-owned members sidecar and a fits/ history entry;
    the strategies YAML itself is never modified. The current sidecar (if
    any) warm-starts restart 0 of the new fit.
    """
    from midas.artifacts import list_fits, read_members, write_fit
    from midas.artifacts import rollback as rollback_fit
    from midas.config import load_policy
    from midas.fitter import fit_as_of
    from midas.optimizer import max_warmup_for_search
    from midas.strategies import STRATEGY_REGISTRY
    from midas.strategies.base import EntrySignal

    strategies_path = Path(strategies)
    if rollback > 0:
        try:
            restored = rollback_fit(strategies_path, steps=rollback)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Rolled back {rollback} fit(s): deployed members from {restored.as_of.isoformat()}.")
        return

    policy = load_policy(strategies_path)
    if policy is None:
        raise click.ClickException(
            f"{strategies_path} has no policy: block — fit is policy-driven; add one (see docs/cli.md)."
        )

    port = load_portfolio(Path(portfolio))
    strat_configs, strat_constraints, risk_config = load_strategies(strategies_path)
    entry_configs = [c for c in strat_configs if issubclass(STRATEGY_REGISTRY[c.name], EntrySignal)]
    exit_configs = [c for c in strat_configs if c not in entry_configs]
    exit_params = {c.name: dict(c.params) for c in exit_configs}
    strategy_names = [c.name for c in entry_configs]

    start_d = _to_date(start)
    as_of_d = _to_date(as_of) if as_of is not None else date.today()
    warmup_bars = max_warmup_for_search(strategy_names, strat_constraints.min_cash_pct, port.active_ticker_count())
    price_data = _fetch_prices(port, start_d, as_of_d, warmup_bars=warmup_bars)

    incumbent = read_members(strategies_path)
    print_status(
        f"Fitting as of {as_of_d} — objective {policy.objective}, {policy.restarts} restarts x "
        f"{policy.budget} trials, deployment {policy.deployment}"
        + (" (warm-starting from deployed members)" if incumbent else "")
    )
    result = fit_as_of(
        port,
        price_data,
        as_of_d,
        policy,
        constraints=strat_constraints,
        exit_params=exit_params,
        risk_config=risk_config,
        min_cash_pct=strat_constraints.min_cash_pct,
        forecast_scaling=strat_constraints.forecast_scaling,
        tax_config=port.tax_config,
        incumbent=incumbent,
        strategy_names=strategy_names,
        log_fn=print_status,
    )
    entry = write_fit(strategies_path, result)
    bests = ", ".join(f"{b:.4g}" for b in result.restart_bests)
    click.echo(f"Fitted {len(result.members)} member(s) as of {result.as_of.isoformat()}.")
    click.echo(f"  restart bests ({policy.objective}): {bests}")
    click.echo(f"  members sidecar: {strategies_path.stem}.members.yaml")
    click.echo(f"  history: {entry.name} ({len(list_fits(strategies_path))} fits recorded)")


@cli.command()
@click.option("--portfolio", "-p", required=True, type=click.Path(exists=True), help="Path to portfolio YAML.")
@click.option(
    "--strategies", "-s", required=True, type=click.Path(exists=True), help="Strategies YAML with a policy: block."
)
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--min-train-pct", default=0.60, show_default=True, help="Initial training reserve before the first fold."
)
def validate(portfolio: str, strategies: str, start: date, end: date, min_train_pct: float) -> None:
    """Run the policy over history exactly as live would, and write baselines.

    Executes cadence-driven folds with fits, degradation-gated adoption,
    and a carried portfolio; the resulting per-fold distribution is what
    the live monitor compares against.
    """
    from midas.baselines import baselines_path, write_baselines
    from midas.config import load_policy
    from midas.optimizer import max_warmup_for_search
    from midas.strategies import STRATEGY_REGISTRY
    from midas.strategies.base import EntrySignal
    from midas.validator import validate_policy

    strategies_path = Path(strategies)
    policy = load_policy(strategies_path)
    if policy is None:
        raise click.ClickException(f"{strategies_path} has no policy: block — validate is policy-driven.")

    port = load_portfolio(Path(portfolio))
    strat_configs, strat_constraints, risk_config = load_strategies(strategies_path)
    entry_configs = [c for c in strat_configs if issubclass(STRATEGY_REGISTRY[c.name], EntrySignal)]
    exit_configs = [c for c in strat_configs if c not in entry_configs]
    exit_params = {c.name: dict(c.params) for c in exit_configs}
    strategy_names = [c.name for c in entry_configs]

    start_d, end_d = _to_date(start), _to_date(end)
    warmup_bars = max_warmup_for_search(strategy_names, strat_constraints.min_cash_pct, port.active_ticker_count())
    price_data = _fetch_prices(port, start_d, end_d, warmup_bars=warmup_bars)

    try:
        baselines = validate_policy(
            port,
            price_data,
            start_d,
            end_d,
            policy,
            constraints=strat_constraints,
            exit_params=exit_params,
            risk_config=risk_config,
            min_cash_pct=strat_constraints.min_cash_pct,
            forecast_scaling=strat_constraints.forecast_scaling,
            tax_config=port.tax_config,
            strategy_names=strategy_names,
            min_train_pct=min_train_pct,
            log_fn=print_status,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    write_baselines(strategies_path, baselines)
    for fold in baselines.folds:
        marker = "adopted" if fold.adopted else "held"
        click.echo(
            f"  fold {fold.fold}: {fold.test_start} → {fold.test_end}  "
            f"{fold.return_annualized:+.2%} ann.  dd {fold.drawdown:.2%}  [{marker}]"
        )
    click.echo(f"Aggregate OOS CAGR: {baselines.aggregate_oos_cagr:+.2%} over {len(baselines.folds)} folds.")
    click.echo(f"Baselines written: {baselines_path(strategies_path).name}")


@cli.command()
@click.option(
    "--portfolio",
    "-p",
    "portfolios",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="Portfolio YAML; repeatable — each is validated side by side.",
)
@click.option(
    "--strategies", "-s", required=True, type=click.Path(exists=True), help="Strategies YAML with a policy: block."
)
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--vary", default="objective", show_default=True, help="Policy field the variants differ along.")
@click.option("--value", "values", multiple=True, help="One variant per value; omit to sweep seeds only.")
@click.option("--seeds", default=3, show_default=True, help="Seed replications per cell (base_seed + i).")
@click.option(
    "--holdout-days", default=730, show_default=True, help="Calendar days withheld from the end of the range."
)
@click.option("--use-holdout", is_flag=True, default=False, help="One-shot holdout report; single variant only.")
@click.option("--budget", default=None, type=int, help="Override the policy budget (requires --force).")
@click.option("--force", is_flag=True, default=False, help="Allow a budget that differs from the policy's.")
@click.option("--min-train-pct", default=0.60, show_default=True)
def sweep(
    portfolios: tuple[str, ...],
    strategies: str,
    start: date,
    end: date,
    vary: str,
    values: tuple[str, ...],
    seeds: int,
    holdout_days: int,
    use_holdout: bool,
    budget: int | None,
    force: bool,
    min_train_pct: float,
) -> None:
    """Compare policy variants with seed replication and honest statistics.

    Differences inside the seed noise floor print as not resolvable; a
    single-portfolio verdict is qualified with "(on these windows)". The
    final --holdout-days are withheld from every sweep so policy selection
    cannot overfit the same folds it reports.
    """
    import dataclasses as _dc

    from midas.config import load_policy
    from midas.optimizer import max_warmup_for_search
    from midas.strategies import STRATEGY_REGISTRY
    from midas.strategies.base import EntrySignal

    strategies_path = Path(strategies)
    policy = load_policy(strategies_path)
    if policy is None:
        raise click.ClickException(f"{strategies_path} has no policy: block — sweep is policy-driven.")
    if budget is not None and budget != policy.budget:
        if not force:
            raise click.UsageError(
                "validating at a different budget than deployment breaks coherence; pass --force to accept"
            )
        policy = _dc.replace(policy, budget=budget)
    if use_holdout and len(values) > 1:
        raise click.UsageError("--use-holdout is a one-shot report: exactly one variant allowed")

    start_d, end_d = _to_date(start), _to_date(end)
    holdout_boundary: date | None = None
    if use_holdout:
        click.echo("HOLDOUT REPORT — one shot; do not iterate on this.")
    else:
        holdout_boundary = end_d - timedelta(days=holdout_days)
        if holdout_boundary <= start_d:
            raise click.UsageError("holdout leaves no sweep range; shrink --holdout-days")
        click.echo(f"holdout: withholding {holdout_boundary} → {end_d} from the sweep")
        end_d = holdout_boundary

    strat_configs, strat_constraints, risk_config = load_strategies(strategies_path)
    entry_configs = [c for c in strat_configs if issubclass(STRATEGY_REGISTRY[c.name], EntrySignal)]
    exit_configs = [c for c in strat_configs if c not in entry_configs]
    exit_params = {c.name: dict(c.params) for c in exit_configs}
    strategy_names = [c.name for c in entry_configs]

    ports: dict[str, PortfolioConfig] = {}
    price_by_port: dict[str, dict[str, pd.DataFrame]] = {}
    warmup_bars = max_warmup_for_search(strategy_names, strat_constraints.min_cash_pct, 1)
    for path_str in portfolios:
        path = Path(path_str)
        port = load_portfolio(path)
        ports[path.stem] = port
        price_by_port[path.stem] = _fetch_prices(port, start_d, end_d, warmup_bars=warmup_bars)

    n_variants = max(len(values), 1)
    approx_days = max((end_d - start_d).days * 5 // 7, 1)
    approx_folds = max(int(approx_days * (1 - min_train_pct)) // policy.cadence_days, 1)
    total_trials = n_variants * len(ports) * seeds * approx_folds * policy.restarts * policy.budget
    click.echo(
        f"preflight: {n_variants} variant(s) x {len(ports)} portfolio(s) x {seeds} seed(s) x "
        f"~{approx_folds} fold(s) x {policy.restarts} restart(s) x {policy.budget} trials "
        f"≈ {total_trials:,} trials"
    )

    try:
        report = run_sweep(
            ports,
            price_by_port,
            start_d,
            end_d,
            policy,
            vary,
            list(values),
            seeds,
            constraints=strat_constraints,
            exit_params=exit_params,
            risk_config=risk_config,
            min_cash_pct=strat_constraints.min_cash_pct,
            forecast_scaling=strat_constraints.forecast_scaling,
            tax_config=next(iter(ports.values())).tax_config if len(ports) == 1 else None,
            strategy_names=strategy_names,
            log_fn=print_status,
            holdout_trimmed_to=holdout_boundary,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    for line in report.summary_lines():
        click.echo(line)


@cli.command(name="strategies")
def list_strategies() -> None:
    """List all available strategies."""
    print_strategy_table([cls() for cls in STRATEGY_REGISTRY.values()])


def _sell_row_values(row: LoggedTrade, basis: float) -> tuple[float, float, float, str, int | None]:
    """Compute the derived report cells for one SELL row.

    ``basis`` is per-share; the report columns are totals (Schedule D
    column (e) is total cost basis), so proceeds - basis_total = pnl
    holds row by row.

    Returns:
        ``(basis_total, proceeds, pnl, purchase_cell, days_held)`` where
        ``days_held`` is ``None`` when the purchase date is unknown or mixed.
    """
    proceeds = row.shares * row.price
    basis_total = basis * row.shares
    pnl = proceeds - basis_total
    if isinstance(row.purchase_date, date):
        return basis_total, proceeds, pnl, row.purchase_date.isoformat(), (row.date - row.purchase_date).days
    return basis_total, proceeds, pnl, row.purchase_date or "", None


def _print_tax_report(
    rows: Sequence[LoggedTrade],
    basis_per_sell: Sequence[float],
    summary: Sequence[AnnualTaxSummary],
    period_label: str,
) -> None:
    """Render the Schedule D table and per-year summary lines to the console."""
    table_width = 140
    table = make_wide_table(f"Schedule D — {period_label}", width=table_width)
    table.add_column("Ticker", style="bold")
    table.add_column("Shares", justify="right")
    table.add_column("Purchase Date")
    table.add_column("Sale Date")
    table.add_column("Cost Basis", justify="right")
    table.add_column("Proceeds", justify="right")
    table.add_column("Realized P&L", justify="right")
    table.add_column("Days Held", justify="right")
    table.add_column("Classification")

    for row, basis in zip(rows, basis_per_sell, strict=True):
        basis_total, proceeds, pnl, purchase_disp, days_held = _sell_row_values(row, basis)
        table.add_row(
            row.ticker,
            f"{row.shares:.4f}",
            purchase_disp,
            row.date.isoformat(),
            f"${basis_total:,.2f}",
            f"${proceeds:,.2f}",
            f"${pnl:+,.2f}",
            str(days_held) if days_held is not None else "",
            row.holding_period.value if row.holding_period else "",
        )
    # Use a width-pinned console so the rendered table isn't truncated when
    # stdout isn't a TTY (e.g. piped output, CliRunner in tests).
    Console(width=table_width).print(table, justify="center")

    for entry in summary:
        click.echo(
            f"\nYear {entry.year}: ST {entry.st_realized:+,.2f}  LT {entry.lt_realized:+,.2f}  "
            f"Net {entry.net_after_cross:+,.2f}  Deductible {entry.deductible_loss:,.2f}  "
            f"Tax {entry.tax_owed:+,.2f}  Carry-Forward {entry.carry_forward:,.2f}  "
            f"Payment {entry.payment_date.isoformat()}"
        )


def _write_tax_report_csv(
    rows: Sequence[LoggedTrade],
    basis_per_sell: Sequence[float],
    path: Path,
) -> None:
    """Write the per-sale Schedule D rows to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TAX_REPORT_COLUMNS)
        for row, basis in zip(rows, basis_per_sell, strict=True):
            basis_total, proceeds, pnl, purchase_cell, days_held = _sell_row_values(row, basis)
            writer.writerow(
                [
                    row.ticker,
                    row.shares,
                    purchase_cell,
                    row.date.isoformat(),
                    round(basis_total, 4),
                    round(proceeds, 4),
                    round(pnl, 4),
                    days_held if days_held is not None else "",
                    row.holding_period.value if row.holding_period else "",
                ]
            )


def _write_tax_summary_csv(summary: Sequence[AnnualTaxSummary], path: Path) -> None:
    """Write the per-year tax aggregates to a companion CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TAX_SUMMARY_COLUMNS)
        for entry in summary:
            writer.writerow(
                [
                    entry.year,
                    round(entry.st_realized, 4),
                    round(entry.lt_realized, 4),
                    round(entry.net_after_cross, 4),
                    round(entry.deductible_loss, 4),
                    round(entry.carry_forward, 4),
                    round(entry.tax_owed, 4),
                    entry.payment_date.isoformat(),
                ]
            )
