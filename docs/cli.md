# CLI Reference

Midas is invoked as `uv run midas <command>`. All commands accept `--help` for full usage.

## strategies

List all available strategies with their tier, description, and asset suitability.

```bash
uv run midas strategies
```

No options. Useful for discovering what strategies exist before building a strategy file.

## optimize

Find optimal strategy parameters via Bayesian optimization. This is typically the first step -- run the optimizer, then backtest or go live with the resulting parameters.

```bash
uv run midas optimize -p portfolio.yaml --start 2015-01-01 --end 2024-12-31 -o optimized.yaml
uv run midas optimize -p portfolio.yaml --start 2020-01-01 --end 2025-01-01 --walk-forward -n 200
```

| Option | Default | Description |
|--------|---------|-------------|
| `-p`, `--portfolio` | required | Path to portfolio YAML |
| `-s`, `--strategies` | all strategies | Path to strategies YAML. Controls which strategies are optimized and sets `min_cash_pct` |
| `--start` | required | Start date (YYYY-MM-DD) |
| `--end` | required | End date (YYYY-MM-DD) |
| `-o`, `--output` | `optimized_strategies.yaml` | Output YAML path for optimized parameters |
| `-n`, `--n-trials` | 200 | Number of Optuna optimization trials |
| `--train-pct` | 0.70 | Train/test split ratio. Cannot be used with `--walk-forward` |
| `--walk-forward` | off | Enable walk-forward optimization |
| `--wf-min-train-pct` | 0.60 | Minimum initial training window as fraction of data. Requires `--walk-forward` |
| `--wf-min-test-days` | 63 | Minimum trading days per test fold (~3 months). Requires `--walk-forward` |

## backtest

Simulate the strategy over historical data and produce a trade log, return metrics, and buy-and-hold comparison.

```bash
uv run midas backtest -p portfolio.yaml -s strategies.yaml --start 2023-01-01 --end 2024-12-31 -o results.csv
```

| Option | Default | Description |
|--------|---------|-------------|
| `-p`, `--portfolio` | required | Path to portfolio YAML |
| `-s`, `--strategies` | all strategies | Path to strategies YAML |
| `--start` | required | Start date (YYYY-MM-DD) |
| `--end` | required | End date (YYYY-MM-DD) |
| `-o`, `--output` | `backtest_results.csv` | Output CSV path for trade log and summary |
| `--train-pct` | 0.70 | Train/test split ratio |
| `--no-split` | off | Disable train/test split entirely |
| `--execution-mode` | `next_open` | When decisions fill: `close` (legacy same-day, optimistic), `next_open` (next session's open — honest default), `next_close` (next session's close) |

## live

Poll real-time prices and emit order alerts for manual execution.

```bash
uv run midas live -p portfolio.yaml -s strategies.yaml
uv run midas live -p portfolio.yaml -s strategies.yaml --interval 30 --dry-run
```

| Option | Default | Description |
|--------|---------|-------------|
| `-p`, `--portfolio` | required | Path to portfolio YAML |
| `-s`, `--strategies` | all strategies | Path to strategies YAML |
| `--interval` | 60 | Poll interval in seconds |
| `--dry-run` | off | Log signals without emitting alerts |
