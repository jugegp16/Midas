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
| `--objective` | `calmar` | What trials maximize: `gross` (raw return), `sharpe` (risk-adjusted), `net` (return after tax; requires a `tax:` block in the portfolio), `calmar` (return over max drawdown), `ulcer` (return over Ulcer Index), `robust` (lower-quartile block return) |

### Objectives

Every trial is scored on the **training window only** — the first
`--train-pct` of trading days in standard mode, each fold's anchored
train window in walk-forward — so no objective ever sees the test
period. The best parameters are then re-run over the full range with the
split for the printed train/test comparison, and the optimized YAML opens
with a `# optimized with --objective …` comment recording the choice.

- `calmar` (default): training return divided by max drawdown. Chosen as
  the default from a five-objective walk-forward study: it was the only
  objective that never lost badly to buy-and-hold on any basket/strategy
  combination — the property a sight-unseen default needs. The ratio is
  floored at a 1% drawdown so a drawdown-free window stays finite. Max
  drawdown is a single worst event per window, so `calmar` is noisier
  than a mean/σ ratio — in practice that noise cost it nothing.
- `sharpe`: annualized Sharpe of the training equity curve. Strong on
  symmetric (mean-reversion) strategies, but it penalizes upside
  volatility, which guts right-skewed strategies whose edge lives in a
  few big winners.
- `gross`: raw time-weighted return. The historical objective, kept for
  comparison and for portfolios where return is genuinely all that
  matters — but it overfits harder the more trials you give it.
- `net`: time-weighted return after capital-gains tax, using the
  portfolio's `tax:` block. A high-turnover strategy that wins gross can
  lose its edge after short-term gains; `net` lets the search see that.
  Refuses to start without a `tax:` block (and offers the one-time setup
  prompt when the portfolio file is silent on tax). The after-tax TWR is
  exact only without mid-period cash infusions; with them the objective
  mixes contribution timing into the tax drag it scores.
- `ulcer`: training return divided by the Ulcer Index — the RMS of
  drawdown depth across every bar (Martin ratio). Calmar's shape without
  its single-worst-event noise: long shallow bleeds count, one historical
  spike does not dominate. Floored at 0.5%.
- `robust`: the lower-quartile return across 8 contiguous blocks of the
  training window. A consistency objective rather than a risk ratio: a
  parameter set that made all its money in one lucky streak scores
  terribly even if its total return and drawdown look fine.

Every risk figure — Sharpe, Sortino, max drawdown, Ulcer Index, and the
block returns, in the optimizer and in the backtest report alike — is
computed on an inflow-adjusted return series: cash infusions and
deferred position activations are stripped out, so deposits are never
counted as performance and never mask drawdown depth.

Train/test returns and the efficiency ratio stay return-based under every
objective, so they remain comparable across runs. When the portfolio has a
`tax:` block, the walk-forward report adds an after-tax OOS column and an
after-tax OOS CAGR so `net` can be judged on its own terms — with one
caveat: each fold is a fresh backtest, so lots re-enter at the fold start
and every gain inside a short fold is short-term. Per-fold after-tax
figures overstate tax drag relative to a continuous run; compare
objectives against each other, not against the continuous backtest.

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
| `--dry-run` | off | Observe signals with in-memory fills — never writes state, trade log, or Discord |
| `--ignore-market-hours` | off | Poll 24/7 instead of only during US equity sessions (debugging) |

### Market hours

Live polls only during regular US equity sessions (9:30-16:00 ET,
weekdays, minus full-closure market holidays) — outside the session it
sleeps until the next open instead of burning API calls on prices that
cannot move. The session clock is DST-correct regardless of the host
timezone, and holidays are computed from the exchange's calendar rules
(fixed dates with weekend observation, nth-weekday floaters, Good
Friday via computus) — valid for any year, no maintenance. One-off
special closures (e.g. mourning days) are unknowable in advance; the
engine polls harmlessly through them.

### End-of-day report

Near the close of each session (~15:55 ET), live emits a one-embed
summary: equity and day-over-day change, cash, a position table
(shares, ticker, and market value at the session's last observed
price; largest portfolios capped with an explicit "+N more"), the
day's alert tally (posted / ✅ / ❌ / expired / still pending), and
risk-overlay scales when engaged. It doubles as a heartbeat — the day
ends with a push notification even when no orders fired, so silence
stops being ambiguous between "no signals" and "engine died".

The report posts to `discord_end_of_day_channel_id` when set — keeping the
alerts channel action-only — and falls back to the alerts channel
otherwise. Terminal always gets it. The day-over-day anchor persists in
the state file, so the delta survives restarts.

### Discord push notifications + fill confirmation

Live alerts can be pushed to a Discord channel, and — this is the
important part — **orders only fill when you confirm them there**.
Without Discord configured, live runs terminal-only and assumes every
alert is executed immediately (the historical behavior).

In confirm mode, each BUY/SELL alert becomes one embed (green/red) with
the strategy, reason, and estimated value. Respond by reacting to the
message:

- **React ✅** (✔️/☑️ also count) — you executed the trade. Midas books the fill at the
  alert price: position lots, cash, the trade log, and the round-trip
  restriction clock all update at confirmation time, not alert time.
- **React ❌** (❎/✖️/⛔/🚫 also count) — you skipped it. Nothing is booked, and the same trade
  (ticker + direction) is not re-alerted until the decline ages past
  `pending_ttl_hours` — a declined trade stays declined for the day.
  This applies to SELLs too: a declined exit stays silent for the full
  window even if the position keeps falling, so decline a stop-loss
  alert only if you really mean "no for today".
- **No reaction** — the alert expires after `pending_ttl_hours`
  (default 8) and nothing is booked.

While an order awaits confirmation, the same intent is not re-alerted.
Re-alerts are rate-limited by `realert_hours` (default 1): a >10% share
resize only supersedes a pending alert once it is at least that old
(prices wobble every tick — without this, alerts churned minutes
apart), and after a ✅ the same trade won't be proposed again within
the window. Direction flips supersede immediately — an exit signal
never waits. Terminal output is unchanged and remains the channel of
record; Discord/network failures never interrupt polling — a pending
order just waits for the next poll.

One-time setup (~5 minutes):

1. [discord.com/developers](https://discord.com/developers/applications)
   → New Application → **Bot** tab → Reset Token → copy the token.
2. OAuth2 → URL Generator → scope `bot` → permissions **Send Messages** and
   **Read Message History** → open the generated URL
   to invite the bot to your server.
3. Discord settings → Advanced → Developer Mode on → right-click your
   alerts channel → Copy Channel ID.

Use a **private channel** only you (and the bot) can see: Midas cannot
tell who reacted, so any channel member's ✅ books a fill against your
portfolio state.

Configure the channel in the portfolio YAML (the id is not a secret):

```yaml
alerts:
  discord_intra_day_channel_id: 1400000000000000001  # optional, order alerts + confirmation
  discord_end_of_day_channel_id: 1400000000000000002  # optional, end-of-day reports
  timeout_seconds: 5      # optional, per-request timeout
  pending_ttl_hours: 8    # optional, confirmation window
  realert_hours: 1        # optional, min gap between re-alerts of one intent
```

The two channels are independently optional: intra-day only gives
confirm-mode alerts with reports falling back to the same channel;
end-of-day only gives the daily report while alerts stay terminal-only
with assumed fills; both gives the full split.

The bot token IS a secret — it goes in the environment, never in YAML:

```bash
MIDAS_DISCORD_BOT_TOKEN=... uv run midas live -p portfolio.yaml
```

The token plus at least one channel id are required for any Discord
delivery; a token with no channel, or a channel without the token, is a
startup error. `--dry-run` is fully ephemeral: it reads existing state
for a realistic starting book but writes nothing — no state file, no
trade log rows, no Discord. Fills are simulated in memory, alerts print
labeled to the terminal, and the end-of-day report prints terminal-only.
(It still takes the state lock, so it can't run concurrently with a
real session against the same state file.)

> Migration note: the `discord_webhook_url` key from the earlier
> webhook-based delivery was removed and is rejected at config load —
> delete the webhook in Discord and switch to the bot setup above.

## doctor

Check that configured integrations actually work, with a fix hint for
every failure. Posts clearly-labeled 🧪 test messages to the configured
Discord channels and verifies each capability live mode depends on:
bot token, channel visibility, posting, reading messages back (what
confirm-mode polling uses), and editing. The end-of-day test report is
built from your real state file marked to market when possible, and
falls back to labeled sample numbers otherwise. Never touches live
state. Exits non-zero when any check fails.

```bash
uv run midas doctor -p portfolio.yaml
uv run midas doctor -p portfolio.yaml --wait-reaction
```

| Option | Default | Description |
|--------|---------|-------------|
| `-p`, `--portfolio` | required | Path to portfolio YAML |
| `--wait-reaction` | off | Wait up to 60s for you to react to the test alert, verifying the confirm round trip |
