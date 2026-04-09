# Architecture

Midas is a target-weight portfolio allocation engine. Entry signals score buy candidates, an allocator blends those scores into target weights via softmax, an order sizer turns the diff against current holdings into buy orders, and exit rules independently emit sell intents on lots that meet their criteria.

## The Two-Tier Model

Strategies fall into exactly two disjoint tiers:

- **EntrySignal** — scores ticker bullishness in `[0, 1]`. Pure buy-side. Returning 0 means "no opinion" (not "sell"). Returning `None` means "abstain" (insufficient data, missing context). Multiple entry signals are blended into a single conviction per ticker, then softmaxed into target weights.

- **ExitRule** — looks at a position's lots and emits zero or more `ExitIntent` objects. Each intent is a dollar amount to liquidate from a specific ticker, with the source strategy and reason attached. Exit rules never go through the allocator and never participate in target-weight construction; they go straight to the order sizer's sell path.

The two tiers are enforced at the type level (`EntrySignal` and `ExitRule` are separate base classes in `strategies/base.py`) and at runtime by an `isinstance` partition in the optimizer, config loader, and backtest engine. The two share a thin `Strategy` base for shared bookkeeping (`name`, `warmup_period`, `suitability`, `description`), but the scoring interfaces (`EntrySignal.score` and `ExitRule.evaluate_exit`) are completely disjoint. There is no third tier; entry-signal logic cannot accidentally produce a sell, and exit-rule logic cannot accidentally inflate a buy.

This split exists because the old "single conviction score in `[-1, +1]`" model conflated three different concerns — bullishness, bearishness, and protective vetoes — into one number that no allocator could cleanly separate. Sell-side strategies had to be encoded as negative scores that the allocator then had to decide whether to interpret as "trim" or "liquidate", and protective stops had to be hand-coded with a veto threshold escape hatch. Splitting the buy and sell paths removes the coupling: entry signals only ever push budget toward buy candidates, and exit rules only ever pull lots out of the portfolio.

### How this compares to LEAN

QuantConnect LEAN splits the same problem across **AlphaModel** (signals), **PortfolioConstructionModel** (weights), and **RiskManagementModel** (overrides like stop loss). Midas's `EntrySignal` is the alpha + portfolio-construction half, and `ExitRule` is the risk-management half. The difference is that LEAN's risk model emits *target adjustments* that get folded back into the construction model on the next bar, whereas midas's exit rules emit `ExitIntent` objects that bypass the allocator entirely and go straight to the order sizer. This is why midas can have lot-aware FIFO exits without distorting the target-weight math: lot-level decisions never have to be projected back into a per-ticker target weight.

## Core Engine

The engine follows a linear pipeline on every tick (one simulated day in backtesting, one poll interval in live mode):

1. **Entry signals** score every ticker
2. **Allocator** blends those scores into target weights (softmax construct-to-budget)
3. **Exit rules** evaluate per-ticker lots and emit `ExitIntent` objects
4. **Order sizer** turns target-weight deltas into buy orders and exit intents into sell orders

Sells are sized first, freeing up cash for the buy pass to use in the same tick.

Every component is stateless within a single tick. State management (positions, cash, lot list, trade log) is the responsibility of whichever execution mode is driving the engine.

### Entry Signals

Entry signals are stateless, ticker-agnostic scorers that return a number in `[0, 1]` (or `None` to abstain). All entry signals inherit from `EntrySignal` and are registered by name in `strategies/__init__.py`. The CLI, optimizer, and config loader all use this registry to instantiate entry signals by name from YAML.

#### Precomputation

For backtest performance, entry signals can optionally compute scores for every day of the price series in a single vectorized pass. The allocator caches these results and looks them up by day index during simulation, avoiding per-day function calls.

#### Warmup

Each strategy declares a `warmup_period` — the bars of price history it needs before it produces a valid score. The CLI fetches a lookback buffer equal to the maximum warmup across configured entry signals *and* exit rules (plus slack for weekends/holidays) so signals are valid from day one of the simulation rather than spending the first N days in cold start. For recursive indicators (RSI, MACD), the nominal period is multiplied by a TA-Lib-style unstable-period factor so the indicator has room to converge. Live mode derives the same history window, and the walk-forward optimizer prefetches a single buffer sized for the upper bound of its parameter search space.

See [Strategies](strategies.md) for the full reference.

### Allocator

The allocator turns entry-signal scores into target portfolio weights. It runs in three phases.

#### Phase 1: Score and Blend

For each ticker, the allocator collects scores from all entry signals and computes a weighted average. Each entry signal has a configurable `weight` (default 1.0) that controls its influence. A signal returning `None` is excluded entirely — it doesn't pull the average toward zero, it simply doesn't participate.

Tickers fall into two buckets:

- **Active** — at least one entry signal scored > 0. The blended score is positive.
- **Held** — every entry signal returned 0 or `None`. The allocator treats this as "no opinion" and holds the ticker at its current weight (or the equal-weight base on the very first allocation).

#### Phase 2: Softmax Budget Allocation

The active tickers' blended scores need to become target portfolio weights that sum to the *active* budget — that is, the investable budget (`1 − min_cash_pct`) minus whatever the held tickers consume. The allocator uses softmax — the same construct-to-budget operator used by QuantConnect LEAN's `InsightWeightingPortfolioConstructionModel`, mean-variance optimizers, and risk-parity libraries.

```
target_i = active_budget * exp(blended_i / T) / sum_j(exp(blended_j / T))
```

By construction, `sum(active targets) == active_budget` exactly, always. There is no separate normalize step — oversubscription is mathematically impossible. The `softmax_temperature` parameter `T` follows the standard ML softmax convention: low `T` concentrates budget on the highest-conviction ticker (winner-take-most, `T → 0` is argmax), `T = 1` is the unscaled softmax over raw scores, and high `T` approaches a uniform split. Midas defaults to `T = 0.5`, a mild concentration bias.

**Neutral = hold.** When all entry signals abstain or score 0 for a ticker, the allocator holds that ticker's current weight rather than dragging it back to equal-weight. This avoids churn from drift-correction trades on days when no signal is firing on a held position.

#### Phase 3: Soft Position Cap

Any active ticker whose softmax target exceeds `max_position_pct` is pinned at the cap, and the freed budget is redistributed to the uncapped survivors by re-running softmax over them with the reduced budget. The loop runs until no survivor exceeds the cap.

The cap is **soft**: it can refuse to allocate *more* budget to an over-target ticker, but it never forces a sell. If a ticker drifts above the cap because of price appreciation, the allocator simply stops buying more — it does not generate a corrective sell. Sells are exclusively the domain of `ExitRule` strategies. This is the cleanest way to keep the buy-only allocator from accidentally producing exits.

If `max_position_pct` is not configured, it's auto-computed as 2.5x the equal-weight baseline (capped at 25%), which allows meaningful overweighting without extreme concentration.

### Exit Rules

Exit rules run *outside* the allocator on a per-ticker basis. Each rule's `evaluate_exit(ticker, lots, prices)` method receives the ticker symbol, its current lot list, and price history; it returns a list of `ExitIntent` objects. An intent is a dollar amount to liquidate, plus the source strategy name and reason for the trade log.

Exit rules are inherently lot-aware — `StopLoss` and `TrailingStop` evaluate each lot's individual cost basis and high-water mark, so a recently-bought lot at a higher cost basis can be liquidated while older, profitable lots in the same ticker stay untouched. The lot list is maintained by the execution mode (backtest builds it from buy fills; the live engine synthesizes a single lot from the YAML `cost_basis` until proper lot tracking is added).

### Order Sizer

`OrderSizer` (formerly `Rebalancer`) is a stateless converter from intents to concrete orders. It exposes two methods:

- **`size_buys(allocation, positions, prices, cash, constraints)`** — diffs the allocator's target weights against the current weights and emits buy orders for any underweight ticker whose delta exceeds `min_buy_delta`. **Buy-only.** A ticker that has drifted *above* its target never produces a sell — that's the soft-cap principle from the allocator carried through to the order sizer.

- **`size_exits(intents, positions, prices)`** — converts each `ExitIntent` into a whole-share sell order, capped at the actual shares held in the lot list.

Sells run before buys in every tick so that exit proceeds become available cash for the buy pass.

#### Slippage and Circuit Breaker

All orders include a slippage estimate (default 0.05%) to model realistic execution costs. Buy prices are adjusted slightly upward, sell prices slightly downward. Share counts are floored to whole numbers since fractional shares aren't supported.

A daily deployment cap limits total buy value to 25% of portfolio value per day, preventing the engine from going all-in during a single volatile session.

### Trading Restrictions

The restriction tracker enforces round-trip rules. When `round_trip_days` is configured (e.g., 30 days), you cannot buy then sell (or sell then buy) the same ticker within that window. This prevents wash sales and models real-world brokerage restrictions. Orders violating this constraint are filtered out before execution.

## Execution Modes

The core engine doesn't run itself -- it needs a driver that feeds it price data, manages portfolio state, and decides what to do with the resulting orders. Midas provides three execution modes, and the typical workflow follows them in order: optimize to find good parameters, backtest to validate performance, and live to act on real-time signals.

### Optimizer

The optimizer is usually the starting point. Rather than hand-tuning parameters, you let the optimizer search for a combination that performs well on historical data. It outputs a strategies YAML that you can then feed into backtest or live mode.

The optimizer uses Bayesian optimization (Optuna's TPE sampler) to search jointly over:

| Layer | What it controls | Search range |
|-------|-----------------|--------------|
| Entry signal parameters | When a signal fires and how strong | `window`, `threshold`, etc. |
| Entry signal weights | How much influence each signal has in the blend | 0.5 to 3.0 |
| Exit rule parameters | When an exit triggers | `loss_threshold`, `trail_pct`, `gain_threshold`, etc. |
| Softmax temperature | How aggressively the allocator concentrates budget | 0.2 to 1.0 |
| Min buy delta | Minimum weight diff to trigger a buy | 0.01 to 0.05 |
| Max position % | Maximum weight for any single position | 0.15 to 0.50 |

Default search ranges are defined in `PARAM_RANGES` in `optimizer.py`. Exit rules don't get a `_weight` field — they fire on their own conditions, not as a contributor to a blended score.

**Standard Mode** -- Runs a configurable number of trials (default 200). Each trial suggests a parameter combination, runs a full backtest with train/test split, and returns the training return as the optimization objective. Trials are distributed across CPU cores via multiprocessing for parallel evaluation.

#### Walk-Forward Optimization

[Walk-forward optimization](https://en.wikipedia.org/wiki/Walk_forward_optimization) is considered the gold standard for validating trading strategies. It determines optimal parameters while testing their robustness against overfitting.

Standard optimization can overfit -- parameters that look great on historical data may not work going forward. Walk-forward fixes this by repeatedly optimizing on in-sample data, then testing on out-of-sample data that was never used during optimization. The time window rolls forward and the process repeats until all available data is used.

```
Fold 1: train [2020───2023.01]  test [2023.01───2023.04]  → 4.3%
Fold 2: train [2020───2023.04]  test [2023.04───2023.07]  → 2.1%
Fold 3: train [2020───2023.07]  test [2023.07───2023.10]  → 3.8%
```

The optimizer only sees training data when picking parameters -- it has no access to the test window. So when you evaluate those parameters on the test window, the results tell you how the strategy would have performed on data it wasn't tuned for. A robust strategy shows consistent positive out-of-sample results across multiple folds. The summary reports annualized CAGR, per-fold OOS mean/std, best/worst fold, and an efficiency ratio (how much of the training performance holds up out-of-sample).

Parameters written to the output YAML come from the last fold (trained on the most data). Each fold is warm-started with the previous fold's best parameters to exploit correlation between adjacent time periods.

See [CLI Reference](cli.md#optimize) for all optimizer options.

### Backtest

Once you have a set of parameters (from the optimizer or hand-tuned), backtesting lets you see exactly how they would have performed over a historical period. It produces a detailed trade log, return metrics, and a comparison against a buy-and-hold baseline.

The backtest engine simulates the full pipeline over historical data, stepping through one trading day at a time. On each trading day, the engine runs the complete pipeline: cash infusion (if scheduled), entry-signal scoring, allocation, exit-rule evaluation, sell pass, buy pass, restriction filtering, and order execution. After execution, it updates positions, the lot list, cash balance, and the trade log.

**Lot-Aware Cost Basis** -- The engine tracks individual purchase lots as a `list[PositionLot]` per ticker. Every buy fill appends a new lot at the execution price; every sell consumes lots FIFO (first-in, first-out). Exit rules see the live lot list, so per-lot stops and per-lot trailing peaks evaluate independently. Trades are classified as short-term (held less than 365 days) or long-term for tax awareness.

> **Note on initial cost basis.** The backtest seeds each starting position's cost basis from the *start-day market price*, not the YAML `cost_basis`. The YAML value is the user's real purchase basis (used by the live engine and for display), but using it inside a backtest would let exit rules fire on pre-window gains, distorting strategy performance.

**Train/Test Split** -- By default, the backtest splits the date range 70/30 into training and test periods. Returns are reported separately for each. The optimizer uses train return as its objective and test return to measure how well the parameters generalize to unseen data.

**Time-Weighted Return** -- TWR accounts for external cash infusions (e.g., biweekly contributions) by breaking the simulation into sub-periods at each infusion point and compounding the sub-period returns. This gives an accurate measure of strategy performance independent of when cash enters the portfolio.

**Deferred Holdings** -- If a portfolio ticker has no price data at the backtest start date (e.g., the company IPO'd mid-backtest), the position is deferred and automatically activated when data becomes available.

See [CLI Reference](cli.md#backtest) for all backtest options.

### Live

After optimizing and backtesting, live mode puts the strategy to work on real-time market data. It polls current prices and tells you what trades to make right now based on your portfolio's actual holdings.

The live engine polls real-time prices on a configurable interval (default 60 seconds) and emits order alerts for manual execution. On each tick, it fetches the last 120 days of price history, runs the full allocation and exit-rule pipeline, and compares the resulting order set to the previous tick. If nothing changed, it stays quiet. If new orders appear or existing ones change, it emits an alert with the ticker, price, reason, source strategy, and suggested share count.

The live engine is fully stateless -- it reads current holdings from the portfolio config each tick and re-derives everything from scratch. Because the live YAML only tracks a single weighted-average `cost_basis` per ticker, exit rules see a synthesized one-element lot list (the full holding at that average basis). True per-lot tracking is a planned follow-up. The live engine does not execute trades; it's designed for operators who execute manually through their broker.

See [CLI Reference](cli.md#live) for all live options.
