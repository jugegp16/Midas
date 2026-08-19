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

### Discord push notifications + fill confirmation

Live alerts can be pushed to a Discord channel, and — this is the
important part — **orders only fill when you confirm them there**.
Without Discord configured, live runs terminal-only and assumes every
alert is executed immediately (the historical behavior).

In confirm mode, each BUY/SELL alert becomes one embed (green/red) with
the strategy, reason, and estimated value, pre-seeded with ✅/❌
reactions:

- **React ✅** — you executed the trade. Midas books the fill at the
  alert price: position lots, cash, the trade log, and the round-trip
  restriction clock all update at confirmation time, not alert time.
- **React ❌** — you skipped it. Nothing is booked.
- **No reaction** — the alert expires after `pending_ttl_hours`
  (default 8) and nothing is booked.

While an order awaits confirmation, the same intent is not re-alerted;
if the engine recomputes a materially different order (>10% share
change or a direction flip), the stale alert is marked expired and
replaced. Terminal output is unchanged and remains the channel of
record; Discord/network failures never interrupt polling — a pending
order just waits for the next poll.

One-time setup (~5 minutes):

1. [discord.com/developers](https://discord.com/developers/applications)
   → New Application → **Bot** tab → Reset Token → copy the token.
2. OAuth2 → URL Generator → scope `bot` → permissions **Send Messages**,
   **Read Message History**, **Add Reactions** → open the generated URL
   to invite the bot to your server.
3. Discord settings → Advanced → Developer Mode on → right-click your
   alerts channel → Copy Channel ID.

Configure the channel in the portfolio YAML (the id is not a secret):

```yaml
alerts:
  discord_channel_id: 1400000000000000001
  timeout_seconds: 5      # optional, per-request timeout
  pending_ttl_hours: 8    # optional, confirmation window
```

The bot token IS a secret — it goes in the environment, never in YAML:

```bash
MIDAS_DISCORD_BOT_TOKEN=... uv run midas live -p portfolio.yaml
```

Both the token and the channel id are required for confirm mode;
setting only one is a startup error. `--dry-run` disables confirm mode
(terminal-only, assumed fills, alerts labeled).

> Migration note: the `discord_webhook_url` key from the earlier
> webhook-based delivery was removed and is rejected at config load —
> delete the webhook in Discord and switch to the bot setup above.
