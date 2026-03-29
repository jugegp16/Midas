# Midas

Target-weight allocation engine for your portfolio. Strategies emit continuous conviction scores, an allocator blends them into target portfolio weights via sigmoid transform, and a rebalancer diffs against current holdings to generate trades. Backtest against years of historical data with train/test splits, or run it live with real-time polling. You pull the trigger.

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
```

Each strategy has an intrinsic tier determined by its class:

| Tier | Behavior | Strategies |
|------|----------|------------|
| CONVICTION | Scores blended via weighted average into target weights | MeanReversion, Momentum, RSIOversold, RSIOverbought, BollingerBand, MACDCrossover, VWAPReversion, MovingAverageCrossover, GapDownRecovery, ProfitTaking |
| PROTECTIVE | Evaluated after blending; vetoes position (forces weight to 0) if score <= `veto_threshold` | StopLoss, TrailingStop |
| MECHANICAL | Bypasses allocator entirely; generates independent order intents | DollarCostAveraging |

**Strategy knobs:**

| Field | Scope | Default | Description |
|-------|-------|---------|-------------|
| `sigmoid_steepness` | Global | 2.0 | Controls how aggressively the allocator responds to conviction scores. Higher = more extreme position sizes |
| `rebalance_threshold` | Global | 0.02 | Minimum weight diff to trigger a rebalance trade. Higher = fewer, larger trades |
| `min_cash_pct` | Global | 0.05 | Minimum cash allocation as a fraction of portfolio value. Higher = more conservative |
| `max_position_pct` | Global | auto | Maximum weight for any single position. Omit to auto-compute from portfolio size |
| `weight` | CONVICTION only | 1.0 | How much influence this strategy has in the blended score. Higher = more influence relative to other strategies. Ignored for PROTECTIVE and MECHANICAL strategies |
| `veto_threshold` | PROTECTIVE only | -0.5 | Score at or below which the strategy forces target weight to 0.0. Lower (e.g. -0.8) = only veto on extreme conviction. Higher (e.g. -0.3) = veto more easily. Ignored for CONVICTION and MECHANICAL strategies |
| `params` | All | `{}` | Strategy-specific parameters (window sizes, thresholds, etc.) |

All knobs are tunable by the optimizer.

To add a strategy: implement the `Strategy` base class and register it in `strategies/__init__.py`.

## Optimizer

The optimizer runs a two-phase grid search (coarse then fine) over all tunable parameters. It tunes three layers jointly:

| Layer | What it controls | Examples |
|-------|-----------------|----------|
| Strategy parameters | When a strategy fires and how strong | `window`, `threshold`, `loss_threshold` |
| Strategy weights | How much influence each conviction strategy has in the blend | `_weight: 0.5` to `3.0` |
| Veto thresholds | When a protective strategy overrides the blend | `_veto_threshold: -0.8` to `-0.2` |
| Sigmoid steepness | How aggressively the allocator responds to conviction | `sigmoid_steepness: 1.0` to `5.0` |
| Rebalance threshold | Minimum weight diff to trigger a trade | `rebalance_threshold: 0.01` to `0.05` |
| Min cash % | Minimum cash reserve as fraction of portfolio | `min_cash_pct: 0.02` to `0.15` |
| Max position % | Maximum weight for any single position | `max_position_pct: 0.15` to `0.50` |

Default search ranges are defined in `PARAM_RANGES` in `optimizer.py`. The optimizer outputs a `strategies.yaml` with optimized `params`, `weight`, and `veto_threshold` per strategy.

MECHANICAL strategies (DCA) are excluded from optimization — their parameters are user preferences, not performance parameters.

```bash
uv run midas optimize -p portfolio.yaml --start 2015-01-01 --end 2024-12-31 -o optimized.yaml
```

## Development

```bash
uv sync                     # install all deps
uv run pytest               # run tests
uv run ruff check . --fix   # lint
uv run ruff format .        # format
uv run mypy src             # type check
```
