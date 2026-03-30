# CLI Reference

```
midas [OPTIONS] COMMAND [ARGS]...
```

Midas — Portfolio Signal Engine.

## `backtest`

Run a backtest over historical data.

```
midas backtest [OPTIONS]
```

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `-p, --portfolio` | PATH | Yes | Path to portfolio YAML config |
| `-s, --strategies` | PATH | No | Path to strategies YAML config. Defaults to all strategies |
| `--start` | DATE | Yes | Start date (`YYYY-MM-DD`) |
| `--end` | DATE | Yes | End date (`YYYY-MM-DD`) |
| `-o, --output` | TEXT | No | Output CSV path |
| `--train-pct` | FLOAT | No | Train/test split ratio (0-1) |
| `--no-split` | FLAG | No | Disable train/test split |

## `live`

Run live analysis with real-time price polling.

```
midas live [OPTIONS]
```

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `-p, --portfolio` | PATH | Yes | Path to portfolio YAML config |
| `-s, --strategies` | PATH | No | Path to strategies YAML config. Defaults to all strategies |
| `--interval` | INTEGER | No | Poll interval in seconds |
| `--dry-run` | FLAG | No | Log signals without alerts |

## `optimize`

Find optimal strategy parameters via Bayesian optimisation (Optuna TPE).

```
midas optimize [OPTIONS]
```

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `-p, --portfolio` | PATH | Yes | Path to portfolio YAML config |
| `-s, --strategies` | PATH | No | Strategies to optimize. Defaults to all |
| `--start` | DATE | Yes | Start date (`YYYY-MM-DD`) |
| `--end` | DATE | Yes | End date (`YYYY-MM-DD`) |
| `-o, --output` | TEXT | No | Output YAML path |
| `-n, --n-trials` | INTEGER | No | Number of Optuna optimisation trials (default: 200) |

## `strategies`

List all available strategies.

```
midas strategies
```
