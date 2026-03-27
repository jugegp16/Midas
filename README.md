# Midas

Algorithmic signal engine for your portfolio. Point it at your holdings and it watches the market for you — mean reversion dips, momentum crossovers, profit-taking exits. When something hits, it sizes a real order (whole shares, slippage-adjusted, circuit-breaker-capped) and tells you exactly what to buy or sell. Backtest against years of historical data with train/test splits, or run it live with real-time polling. You pull the trigger.

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14+.

## Quick Start

```bash
uv sync
uv run midas strategies
uv run midas backtest -p portfolio.yaml -s strategies.yaml --start 2023-01-01 --end 2024-12-31 -o results.csv
uv run midas live -p portfolio.yaml -s strategies.yaml
uv run midas optimize -p portfolio.yaml --start 2023-01-01 --end 2024-12-31
```

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
strategies:
  - name: MeanReversion
    params:
      window: 30
      threshold: 0.10
  - name: ProfitTaking
    params:
      gain_threshold: 0.20
  - name: Momentum
    params:
      window: 20
```

To add a strategy: implement the `Strategy` base class and register it in `strategies/__init__.py`.

## Development

```bash
uv sync                     # install all deps
uv run pytest               # run tests
uv run ruff check . --fix   # lint
uv run ruff format .        # format
uv run mypy src             # type check
```
