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

The allocator takes conviction scores and produces target portfolio weights. It runs in four phases.

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

The optimizer uses Bayesian optimization (Optuna's TPE sampler) to search over all tunable parameters jointly. This includes strategy-specific parameters (lookback windows, thresholds), strategy weights, protective veto thresholds, and global allocator settings (sigmoid steepness, rebalance threshold, max position size). MECHANICAL strategies are excluded since their parameters are user preferences, not performance-tunable.

**Standard Mode** -- Runs a configurable number of trials (default 200). Each trial suggests a parameter combination, runs a full backtest with train/test split, and returns the training return as the optimization objective. Trials are distributed across CPU cores via multiprocessing for parallel evaluation.

**Walk-Forward Mode** -- Standard optimization can overfit -- parameters that look great on historical data may fail going forward. Walk-forward optimization addresses this by repeatedly expanding the training window and testing on the next unseen slice of data.

The training window starts at 60% of the data and grows with each fold. Each fold re-optimizes parameters on its training data, then evaluates on the subsequent test window (minimum ~3 months). The test window rolls forward until all data is used. Each fold is warm-started with the previous fold's best parameters to exploit correlation between adjacent time periods.

The summary reports annualized CAGR across all out-of-sample windows, per-fold mean and standard deviation, best/worst fold, and an efficiency ratio measuring how much of the training performance holds up out-of-sample. A robust strategy shows consistent positive out-of-sample results across folds. The final output YAML uses parameters from the last fold, which was trained on the most data.

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
