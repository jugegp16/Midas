"""CLI interface — the sole user-facing interface for MVP."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import click
import pandas as pd

from midas.allocator import Allocator
from midas.backtest import DEFAULT_TRAIN_PCT, BacktestEngine, write_backtest_csv
from midas.config import load_portfolio, load_strategies
from midas.data import CachedYFinanceProvider
from midas.models import (
    AllocationConstraints,
    PortfolioConfig,
    StrategyConfig,
    StrategyTier,
)
from midas.output import print_backtest_summary, print_status, print_strategy_table
from midas.rebalancer import Rebalancer
from midas.strategies import STRATEGY_REGISTRY, Strategy


def _build_strategy(cfg: StrategyConfig) -> Strategy:
    cls = STRATEGY_REGISTRY.get(cfg.name)
    if cls is None:
        msg = (
            f"Unknown strategy '{cfg.name}'. "
            f"Available: {', '.join(STRATEGY_REGISTRY)}"
        )
        raise click.ClickException(msg)
    return cls(**cfg.params)


def _build_components(
    strategy_configs: list[StrategyConfig] | None,
    constraints: AllocationConstraints,
    n_tickers: int,
) -> tuple[Allocator, Rebalancer, list[Strategy]]:
    """Build allocator, rebalancer, and mechanical strategies from config."""
    configs = strategy_configs or [
        StrategyConfig(name=name) for name in STRATEGY_REGISTRY
    ]

    conviction: list[tuple[Strategy, float]] = []
    protective: list[tuple[Strategy, float]] = []
    mechanical: list[Strategy] = []

    for cfg in configs:
        strategy = _build_strategy(cfg)

        if strategy.tier == StrategyTier.PROTECTIVE:
            protective.append((strategy, cfg.veto_threshold))
        elif strategy.tier == StrategyTier.MECHANICAL:
            mechanical.append(strategy)
        else:
            conviction.append((strategy, cfg.weight))

    allocator = Allocator(conviction, protective, constraints, n_tickers)
    rebalancer = Rebalancer()

    return allocator, rebalancer, mechanical


def _to_date(dt: date | datetime) -> date:
    """Coerce click.DateTime output to a plain date."""
    return dt.date() if isinstance(dt, datetime) else dt


def _fetch_prices(
    portfolio: PortfolioConfig,
    start: date,
    end: date,
) -> dict[str, pd.Series]:
    provider = CachedYFinanceProvider()
    price_data: dict[str, pd.Series] = {}
    tickers = [h.ticker for h in portfolio.holdings]
    print_status(f"Fetching data for {', '.join(tickers)}...")
    for ticker in tickers:
        try:
            price_data[ticker] = provider.get_history(ticker, start, end)
        except Exception as e:
            print_status(f"Skipping {ticker}: {e}")
    return price_data


@click.group()
def cli() -> None:
    """Midas — Portfolio Signal Engine."""


@cli.command()
@click.option(
    "--portfolio", "-p", required=True, type=click.Path(exists=True),
    help="Path to portfolio YAML config.",
)
@click.option(
    "--strategies", "-s", default=None, type=click.Path(exists=True),
    help="Path to strategies YAML config. Defaults to all strategies.",
)
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--output", "-o", default="backtest_results.csv",
    help="Output CSV path.",
)
@click.option(
    "--train-pct", default=DEFAULT_TRAIN_PCT,
    help="Train/test split ratio (0-1).",
)
@click.option("--no-split", is_flag=True, help="Disable train/test split.")
def backtest(
    portfolio: str,
    strategies: str | None,
    start: date,
    end: date,
    output: str,
    train_pct: float,
    no_split: bool,
) -> None:
    """Run a backtest over historical data."""
    port = load_portfolio(Path(portfolio))
    strat_configs, constraints = (
        load_strategies(Path(strategies)) if strategies
        else (None, AllocationConstraints())
    )

    start_d, end_d = _to_date(start), _to_date(end)
    price_data = _fetch_prices(port, start_d, end_d)

    n_tickers = sum(1 for h in port.holdings if h.shares > 0)
    allocator, rebalancer, mechanical = _build_components(
        strat_configs, constraints, n_tickers,
    )

    engine = BacktestEngine(
        allocator=allocator,
        rebalancer=rebalancer,
        mechanical_strategies=mechanical,
        constraints=constraints,
        train_pct=train_pct,
        enable_split=not no_split,
        log_fn=print_status,
    )

    print_status("Running backtest...")
    result = engine.run(port, price_data, start_d, end_d)

    out_path = Path(output)
    write_backtest_csv(result, out_path)
    print_status(f"Results written to {out_path}")
    print_backtest_summary(result)


@cli.command()
@click.option(
    "--portfolio", "-p", required=True, type=click.Path(exists=True),
    help="Path to portfolio YAML config.",
)
@click.option(
    "--strategies", "-s", default=None, type=click.Path(exists=True),
    help="Path to strategies YAML config. Defaults to all strategies.",
)
@click.option("--interval", default=60, help="Poll interval in seconds.")
@click.option("--dry-run", is_flag=True, help="Log signals without alerts.")
def live(
    portfolio: str,
    strategies: str | None,
    interval: int,
    dry_run: bool,
) -> None:
    """Run live analysis with real-time price polling."""
    from midas.live import LiveEngine

    port = load_portfolio(Path(portfolio))
    strat_configs, constraints = (
        load_strategies(Path(strategies)) if strategies
        else (None, AllocationConstraints())
    )
    provider = CachedYFinanceProvider()

    n_tickers = sum(1 for h in port.holdings if h.shares > 0)
    allocator, rebalancer, mechanical = _build_components(
        strat_configs, constraints, n_tickers,
    )

    engine = LiveEngine(
        portfolio=port,
        allocator=allocator,
        rebalancer=rebalancer,
        provider=provider,
        mechanical_strategies=mechanical,
        constraints=constraints,
        poll_interval=interval,
        dry_run=dry_run,
    )
    engine.run()


@cli.command()
@click.option(
    "--portfolio", "-p", required=True, type=click.Path(exists=True),
    help="Path to portfolio YAML config.",
)
@click.option(
    "--strategies", "-s", default=None, type=click.Path(exists=True),
    help="Strategies to optimize. Defaults to all.",
)
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--output", "-o", default="optimized_strategies.yaml",
    help="Output YAML path.",
)
@click.option(
    "--n-trials", "-n", default=200, show_default=True,
    help="Number of Optuna optimisation trials.",
)
def optimize(
    portfolio: str,
    strategies: str | None,
    start: date,
    end: date,
    output: str,
    n_trials: int,
) -> None:
    """Find optimal strategy parameters via Bayesian optimisation (Optuna TPE)."""
    from midas.optimizer import optimize as run_optimize
    from midas.optimizer import write_strategies_yaml

    port = load_portfolio(Path(portfolio))

    strategy_names: list[str] | None = None
    min_cash_pct = AllocationConstraints().min_cash_pct
    if strategies:
        strat_configs, strat_constraints = load_strategies(Path(strategies))
        strategy_names = [c.name for c in strat_configs]
        min_cash_pct = strat_constraints.min_cash_pct

    start_d, end_d = _to_date(start), _to_date(end)
    price_data = _fetch_prices(port, start_d, end_d)

    result = run_optimize(
        portfolio=port,
        price_data=price_data,
        start=start_d,
        end=end_d,
        strategy_names=strategy_names,
        n_trials=n_trials,
        min_cash_pct=min_cash_pct,
        log_fn=print_status,
    )

    write_strategies_yaml(result.best_params, output, min_cash_pct=min_cash_pct)
    print_status(f"Optimized strategies written to {output}")

    from rich.table import Table

    table = Table(title="Optimization Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total Return", f"{result.best_return:.2%}")
    table.add_row("Buy & Hold Return", f"{result.best_bh_return:.2%}")
    table.add_row("Train Return", f"{result.best_train_return:.2%}")
    table.add_row("Test Return", f"{result.best_test_return:.2%}")
    table.add_row("Trials Run", str(result.trials_run))
    table.add_section()
    for name, params in result.best_params.items():
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        table.add_row(name, param_str)

    from midas.output import console
    console.print(table)


@cli.command(name="strategies")
def list_strategies() -> None:
    """List all available strategies."""
    print_strategy_table([cls() for cls in STRATEGY_REGISTRY.values()])
