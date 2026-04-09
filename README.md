# Midas

Target-weight allocation engine for your portfolio. Define your holdings and pick entry signals and exit rules to fit your thesis -- Midas scores buy candidates with entry signals, blends them into target portfolio weights via a softmax construct-to-budget allocator, and runs exit rules independently to trim or liquidate lots when their conditions fire.

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
softmax_temperature: 0.5
min_buy_delta: 0.02
min_cash_pct: 0.05

strategies:
  # Entry signals — score buy candidates, blended into target weights.
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
  # Exit rules — fire independently of the allocator. No weight, no score.
  - name: StopLoss
    params:
      loss_threshold: 0.10
```

Every strategy is either an **entry signal** (scores buy candidates in [0, 1]) or an **exit rule** (decides which lots to liquidate). Buys come from the allocator's softmax over entry scores. Sells come exclusively from exit rules — no entry signal can ever produce a sell. A workable strategy file pairs at least one entry signal with at least one exit rule.

Don't want to build your own? Start with one of the [pre-built compositions](example-strategies/).

## Docs

- [Architecture](docs/architecture.md) -- how the engine works: entry signals, exit rules, allocator, order sizer, and execution modes
- [Strategies](docs/strategies.md) -- all available strategies, how to compose them, and how they interact
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
