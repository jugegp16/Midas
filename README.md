# Midas

Target-weight allocation engine for your portfolio. Define your holdings and pick a set of strategies to fit your thesis -- Midas scores every position using technical indicators, blends those scores into target portfolio weights via a sigmoid transformation, and generates the trades needed to rebalance.

- **Optimize** strategy parameters with Bayesian search and walk-forward validation
- **Backtest** against years of historical data with train/test splits
- **Live** poll real-time prices and get trade alerts

## Quick Start

```bash
uv sync

# list available strategies
uv run midas strategies

# find optimal parameters
uv run midas optimize -p portfolio.yaml -s strategies.yaml --start 2020-01-01 --end 2025-01-01 --walk-forward -o optimized.yaml

# validate performance
uv run midas backtest -p portfolio.yaml -s strategies.yaml --start 2020-01-01 --end 2025-01-01

# real-time alerts
uv run midas live -p portfolio.yaml -s strategies.yaml
```

## Configuration

Midas takes two YAML files: a portfolio (what you own) and a strategies file (how to trade it).

### Portfolio

```yaml
portfolio:
  - ticker: VOO
    shares: 5
    cost_basis: 420.00
  - ticker: AAPL
    shares: 10
    cost_basis: 155.00

available_cash: 2000.00

trading_restrictions:
  round_trip_days: 30

cash_infusion:
  amount: 1500.00
  next_date: 2026-04-03
  frequency: biweekly
```

### Strategies

```yaml
sigmoid_steepness: 2.0
rebalance_threshold: 0.02
min_cash_pct: 0.05

strategies:
  - name: BollingerBand
    weight: 1.0
    params:
      window: 20
      num_std: 2.0
  - name: RSIOversold
    weight: 1.0
    params:
      window: 14
      oversold_threshold: 30.0
  - name: StopLoss
    veto_threshold: -0.5
    params:
      loss_threshold: 0.10
```

Don't want to build your own? Start with one of the [pre-built compositions](example-strategies/).

## Docs

- [Architecture](docs/architecture.md) -- how the engine works: strategies, allocator, rebalancer, and execution modes
- [Strategies](docs/strategies.md) -- all 13 strategies, how to compose them, and how they interact
- [CLI Reference](docs/cli.md) -- every command and option

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14+.

```bash
uv sync                     # install all deps
uv run pytest               # run tests
uv run ruff check . --fix   # lint
uv run ruff format .        # format
uv run mypy src             # type check
```
