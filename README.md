# Midas

Target-weight allocation engine for your portfolio. Strategies emit continuous conviction scores, an allocator blends them into target portfolio weights via sigmoid transformation, and a rebalancer diffs against current holdings to generate trades. Optimize strategy parameters with walk-forward optimization, backtest against years of historical data with train/test splits, and run it live with real-time polling.

## Quick Start

```bash
uv sync
uv run midas strategies
uv run midas backtest -p portfolio.yaml -s strategies.yaml --start 2023-01-01 --end 2024-12-31 -o results.csv
uv run midas live -p portfolio.yaml -s strategies.yaml
uv run midas optimize -p portfolio.yaml --start 2023-01-01 --end 2024-12-31
uv run midas optimize -p portfolio.yaml --start 2020-01-01 --end 2025-01-01 --walk-forward
```

See [Architecture](docs/architecture.md) for how the engine works and [Strategies](docs/strategies.md) for a reference of all available strategies. Pre-built strategy compositions are available in [`example-strategies/`](example-strategies/).

## Config

### Portfolio

```yaml
# portfolio.yaml
portfolio:
  - ticker: VOO
    shares: 5
    cost_basis: 420.00
  - ticker: AAPL
    shares: 10
    cost_basis: 155.00

available_cash: 2000.00

trading_restrictions:
  round_trip_days: 30  # can't buy then sell (or vice versa) the same ticker within 30 days

cash_infusion:
  amount: 1500.00
  next_date: 2026-04-03
  frequency: biweekly
```

### Strategies

```yaml
# strategies.yaml
sigmoid_steepness: 2.0        # how aggressively the allocator responds to conviction
rebalance_threshold: 0.02     # ignore weight diffs smaller than 2%
min_cash_pct: 0.05            # always keep 5% cash
# max_position_pct: 0.20      # omit to auto-compute from portfolio size

strategies:
  - name: MeanReversion
    weight: 1.5              # blending influence (conviction strategies only)
    params:
      window: 30
      threshold: 0.10
  - name: StopLoss
    veto_threshold: -0.5     # score at or below which position is force-liquidated
    params:
      loss_threshold: 0.10
  - name: DollarCostAveraging
    params:
      frequency_days: 14
      amount: 500.00        # dollar amount per DCA buy
```

See [Strategies](docs/strategies.md) for all available strategies, their parameters, and how to compose them. Pre-built configurations are in [`example-strategies/`](example-strategies/).

## Optimizer

The optimizer uses Bayesian optimization to search over all tunable parameters jointly -- strategy params, weights, veto thresholds, and global allocator settings. It outputs a strategies YAML with optimized values.

```bash
uv run midas optimize -p portfolio.yaml --start 2015-01-01 --end 2024-12-31 -o optimized.yaml
uv run midas optimize -p portfolio.yaml --start 2020-01-01 --end 2025-01-01 --walk-forward -n 200
```

See [Architecture](docs/architecture.md#optimizer) for details on standard vs. walk-forward optimization.

## Development

```bash
uv sync                     # install all deps
uv run pytest               # run tests
uv run ruff check . --fix   # lint
uv run ruff format .        # format
uv run mypy src             # type check
```

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14+.
