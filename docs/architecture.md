# Architecture

Midas is a target-weight portfolio allocation engine. Strategies emit continuous conviction scores, an allocator blends them into target portfolio weights, and a rebalancer diffs against current holdings to generate trades.

## Core Engine

The engine follows a linear pipeline on every tick (one simulated day in backtesting, one poll interval in live mode):

1. **Strategies** evaluate price history and emit conviction scores between -1 (bearish) and +1 (bullish)
2. **Allocator** blends those scores into a target weight for each ticker in the portfolio
3. **Rebalancer** compares target weights to current holdings and generates buy/sell orders

Every component is stateless within a single tick. State management (positions, cash, trade log) is the responsibility of whichever execution mode is driving the engine.

### Strategies

Strategies are stateless, ticker-agnostic scorers. Each receives a price history array and returns a conviction score in [-1, +1], where positive is bullish, negative is bearish, and None means abstain (insufficient data, missing context, etc.).

All strategies inherit from a common base class and are registered by name in a central registry. The CLI, optimizer, and config loader all use this registry to instantiate strategies by name from YAML configuration.

#### Tiers

Each strategy belongs to one of three tiers that determine how it participates in the pipeline:

**CONVICTION** strategies are the core signal generators. Their scores are blended together via weighted average into a single conviction per ticker, which is then transformed into a target weight. Most strategies (10 of 13) are conviction-tier.

**PROTECTIVE** strategies act as safety valves. They are evaluated after conviction blending. If a protective strategy's score falls at or below its configured veto threshold, the target weight for that ticker is forced to zero -- a hard liquidation override regardless of what conviction strategies say.

**MECHANICAL** strategies bypass the allocator entirely. They generate independent order intents on a fixed schedule (e.g., dollar-cost averaging buys). The rebalancer sizes these intents into concrete orders separately from the conviction/allocation flow.

#### Precomputation

For backtest performance, strategies can optionally compute scores for every day of the price series in a single vectorized pass. The allocator caches these results and looks them up by day index during simulation, avoiding per-day function calls. Strategies that need runtime context like cost basis (ProfitTaking, StopLoss, TrailingStop) cannot precompute and fall back to per-day evaluation.

See [Strategies](strategies.md) for a reference of all available strategies.

### Allocator

The allocator takes conviction scores and produces target portfolio weights. It runs in four phases, governed by the following knobs (all configurable in the strategies YAML):

| Field | Scope | Default | Description |
|-------|-------|---------|-------------|
| `sigmoid_steepness` | Global | 2.0 | Controls how aggressively the allocator responds to conviction scores. Higher = more extreme position sizes |
| `rebalance_threshold` | Global | 0.02 | Minimum weight diff to trigger a rebalance trade. Higher = fewer, larger trades |
| `min_cash_pct` | Global | 0.05 | Minimum cash allocation as a fraction of portfolio value. Higher = more conservative |
| `max_position_pct` | Global | auto | Maximum weight for any single position. Omit to auto-compute from portfolio size |
| `weight` | CONVICTION only | 1.0 | How much influence this strategy has in the blended score. Ignored for PROTECTIVE and MECHANICAL strategies |
| `veto_threshold` | PROTECTIVE only | -0.5 | Score at or below which the strategy forces target weight to 0. Ignored for CONVICTION and MECHANICAL strategies |
| `params` | All | `{}` | Strategy-specific parameters (window sizes, thresholds, etc.) |

All knobs except `min_cash_pct` are tunable by the optimizer. `min_cash_pct` is a user risk preference.

#### Phase 1: Score and Blend

For each ticker, the allocator collects scores from all CONVICTION strategies and computes a weighted average. Each strategy has a configurable weight (default 1.0) that controls its influence. A strategy returning None is excluded entirely -- it doesn't pull the average toward zero, it simply doesn't participate.

#### Phase 2: Sigmoid Transform

The blended score (a number between -1 and +1) needs to become a target portfolio weight. A raw linear mapping would work but produces extreme allocations -- a blended score of +1 would mean "put everything in this ticker." Instead, the allocator uses a sigmoid function to smoothly map scores to weights.

The sigmoid is centered so that a blended score of 0 produces the equal-weight baseline (total investable weight divided evenly across tickers). Positive scores push above the baseline (overweight), negative scores push below (underweight). The `sigmoid_steepness` parameter controls how aggressively the curve responds -- higher values mean small conviction differences produce larger weight swings.

#### Phase 3: Protective Vetoes

Each PROTECTIVE strategy is evaluated per ticker. If its score falls at or below its veto threshold, the target weight is forced to zero. This is a hard override -- no amount of bullish conviction can prevent a protective liquidation. This is how stop-losses and trailing stops work: they don't reduce the position, they eliminate it.

#### Phase 4: Constraints

Finally, the allocator enforces portfolio-level constraints. Each ticker's weight is capped at `max_position_pct` to prevent excessive concentration. If `max_position_pct` is not configured, it's auto-computed as 2.5x the equal-weight baseline (capped at 25%), which allows meaningful overweighting without extreme concentration. Then all weights are normalized so their sum doesn't exceed the investable portion of the portfolio (1 minus the minimum cash reserve).

### Rebalancer

The rebalancer compares target weights from the allocator against the portfolio's current holdings and generates concrete buy/sell orders to close the gap.

#### Order Generation

The rebalancer computes the current weight of each ticker (its market value as a fraction of total portfolio value) and diffs it against the target weight. If the difference is smaller than the `rebalance_threshold` (default 2%), it's ignored -- this prevents excessive trading on tiny weight fluctuations.

Sells are processed before buys. This is intentional: sell proceeds become available cash for subsequent buy orders. Each sell is capped at the actual shares held, and each buy is constrained by available cash.

#### Slippage

All orders include a slippage estimate (default 0.05%) to model realistic execution costs. Buy prices are adjusted slightly upward, sell prices slightly downward. Share counts are floored to whole numbers since fractional shares aren't supported.

#### Circuit Breaker

A daily deployment cap limits total buy value to 25% of portfolio value per day. This prevents the engine from going all-in during a single volatile session. If the allocator says "buy everything," the circuit breaker spreads that deployment across multiple days.

#### Mechanical Order Sizing

MECHANICAL strategy intents are sized separately. Each intent specifies a target dollar amount (e.g., "buy $500 of VOO"). The rebalancer converts this to a share count at the current price, constrained by available cash. Mechanical orders run after rebalancing orders, using whatever cash remains.

### Trading Restrictions

The restriction tracker enforces round-trip rules. When `round_trip_days` is configured (e.g., 30 days), you cannot buy then sell (or sell then buy) the same ticker within that window. This prevents wash sales and models real-world brokerage restrictions. Orders violating this constraint are filtered out before execution.

## Execution Modes

The core engine doesn't run itself -- it needs a driver that feeds it price data, manages portfolio state, and decides what to do with the resulting orders. Midas provides three execution modes, and the typical workflow follows them in order: optimize to find good parameters, backtest to validate performance, and live to act on real-time signals.

### Optimizer

The optimizer is usually the starting point. Rather than hand-tuning strategy parameters, you let the optimizer search for a combination that performs well on historical data. It outputs a strategies YAML that you can then feed into backtest or live mode.

The optimizer uses Bayesian optimization (Optuna's TPE sampler) to search over all tunable parameters. It tunes the following layers jointly:

| Layer | What it controls | Search range |
|-------|-----------------|--------------|
| Strategy parameters | When a strategy fires and how strong | `window`, `threshold`, `loss_threshold`, etc. |
| Strategy weights | How much influence each conviction strategy has in the blend | 0.5 to 3.0 |
| Veto thresholds | When a protective strategy overrides the blend | -0.8 to -0.2 |
| Sigmoid steepness | How aggressively the allocator responds to conviction | 1.0 to 5.0 |
| Rebalance threshold | Minimum weight diff to trigger a trade | 0.01 to 0.05 |
| Max position % | Maximum weight for any single position | 0.15 to 0.50 |

Default search ranges are defined in `PARAM_RANGES` in `optimizer.py`. The optimizer outputs a strategies YAML with optimized `params`, `weight`, and `veto_threshold` per strategy. MECHANICAL strategies (DCA) are excluded -- their parameters are user preferences, not performance parameters.

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

| Option | Default | Description |
|--------|---------|-------------|
| `--walk-forward` | off | Enable walk-forward optimization |
| `--wf-min-train-pct` | 0.60 | Minimum initial training window as fraction of data |
| `--wf-min-test-days` | 63 | Minimum trading days per test fold (~3 months) |

### Backtest

Once you have a set of parameters (from the optimizer or hand-tuned), backtesting lets you see exactly how they would have performed over a historical period. It produces a detailed trade log, return metrics, and a comparison against a buy-and-hold baseline.

The backtest engine simulates the full pipeline over historical data, stepping through one trading day at a time. On each trading day, the engine runs the complete pipeline: cash infusion (if scheduled), allocation, rebalancing, mechanical strategies, restriction filtering, and order execution. After execution, it updates positions, cost basis, cash balance, and the trade log.

**Train/Test Split** -- By default, the backtest splits the date range 70/30 into training and test periods. Returns are reported separately for each. The optimizer uses train return as its objective and test return to measure how well the parameters generalize to unseen data.

**Time-Weighted Return** -- TWR accounts for external cash infusions (e.g., biweekly contributions) by breaking the simulation into sub-periods at each infusion point and compounding the sub-period returns. This gives an accurate measure of strategy performance independent of when cash enters the portfolio. Without TWR, a large cash infusion right before a market rally would inflate the apparent return.

**Holding Period Tracking** -- The engine tracks individual purchase lots using FIFO (first-in, first-out) matching. When shares are sold, the earliest lots are consumed first. Each trade is classified as short-term (held less than 365 days) or long-term for tax awareness.

**Deferred Holdings** -- If a portfolio ticker has no price data at the backtest start date (e.g., the company IPO'd mid-backtest), the position is deferred and automatically activated when data becomes available.

### Live

After optimizing and backtesting, live mode puts the strategy to work on real-time market data. It polls current prices and tells you what trades to make right now based on your portfolio's actual holdings.

The live engine polls real-time prices on a configurable interval (default 60 seconds) and emits order alerts for manual execution. On each tick, it fetches the last 120 days of price history, runs the full allocation and rebalancing pipeline, and compares the resulting order set to the previous tick. If nothing changed, it stays quiet. If new orders appear or existing ones change, it emits an alert with the ticker, price, reason, strategy scores, and suggested share count.

The live engine is fully stateless -- it reads current holdings from the portfolio config each tick and re-derives everything from scratch. It does not execute trades; it's designed for operators who want signal alerts and execute manually through their broker.
