# Strategies

All strategies are stateless and ticker-agnostic. They receive a price history array and return a conviction score between -1 (bearish) and +1 (bullish), or None to abstain.

Strategies are registered by name in `strategies/__init__.py`. To add a new strategy, implement the `Strategy` base class and add it to `STRATEGY_REGISTRY`.

## Summary

| Strategy | Tier | Direction | Score Range | What it does |
|----------|------|-----------|-------------|--------------|
| MeanReversion | CONVICTION | Bidirectional | [-1, +1] | Buys below MA, sells above |
| Momentum | CONVICTION | Buy-only | [0, +1] | Buys when price trends above MA |
| ProfitTaking | CONVICTION | Sell-only | [-1, 0] | Sells when unrealized gains exceed threshold |
| RSIOversold | CONVICTION | Buy-only | [0, +1] | Buys when RSI indicates oversold conditions |
| RSIOverbought | CONVICTION | Sell-only | [-1, 0] | Sells when RSI indicates overbought conditions |
| BollingerBand | CONVICTION | Bidirectional | [-1, +1] | Buys/sells based on volatility-adjusted distance from MA |
| MACDCrossover | CONVICTION | Bidirectional | [-1, +1] | Follows trend via MACD/signal line relationship |
| VWAPReversion | CONVICTION | Bidirectional | [-1, +1] | Mean reversion around average price |
| MovingAverageCrossover | CONVICTION | Bidirectional | [-1, +1] | Golden cross / death cross signals |
| GapDownRecovery | CONVICTION | Buy-only | [0, +1] | Buys gap-down events that start recovering |
| StopLoss | PROTECTIVE | -- | -- | Vetoes position when loss exceeds threshold |
| TrailingStop | PROTECTIVE | -- | -- | Vetoes position when drawdown from peak exceeds threshold |
| DollarCostAveraging | MECHANICAL | Buy-only | -- | Buys on a fixed schedule regardless of price |

## Composing Strategy Files

A strategy file doesn't need every strategy -- it needs a coherent combination where the pieces complement each other. The best compositions pair entry signals with exit signals, use strategies that measure different things, and include at least one protective strategy as a safety net.

**Principles for picking strategies:**

- **Pair entries with exits.** Buy-only strategies (Momentum, RSIOversold, GapDownRecovery) need sell-side counterparts (RSIOverbought, ProfitTaking) or protective strategies (StopLoss, TrailingStop) to close positions. Without exits, the engine accumulates positions indefinitely.

- **Mix signal types.** Strategies that measure the same thing (e.g., MeanReversion and VWAPReversion both compare price to a moving average) provide redundant signals. Strategies that measure different things (e.g., BollingerBand measures distance from MA in volatility units, RSIOversold measures the ratio of up-days to down-days) provide independent confirmation. When two independent signals agree, the conviction is more trustworthy.

- **Match your market thesis.** Trend-following strategies (Momentum, MovingAverageCrossover, MACDCrossover) and mean-reversion strategies (MeanReversion, BollingerBand, VWAPReversion) have opposing views of how markets work. Using both isn't wrong -- their signals will partially cancel, producing more moderate positions -- but you should be intentional about it.

- **Always include protection.** StopLoss and TrailingStop serve different purposes. StopLoss caps absolute losses from cost basis. TrailingStop protects accumulated profits from drawdowns. Using both gives you a floor on losses and a ratchet on gains.

**Pre-built examples** in `example-strategies/` demonstrate these principles:

- **[Trend-Following](../example-strategies/trend-following.yaml)** -- Rides sustained price trends and exits when momentum fades. Uses MovingAverageCrossover and MACDCrossover for two independent trend signals that confirm each other on different timescales, RSIOverbought to detect overextended rallies, and ProfitTaking as a fixed exit target.

- **[Dip-Buying](../example-strategies/dip-buying.yaml)** -- Buys when prices are statistically cheap relative to recent history. Pairs BollingerBand (volatility-adjusted cheapness) with RSIOversold (selling exhaustion) for independent confirmation that an asset is oversold, protected by a StopLoss floor to limit damage when dips keep dipping.

- **[Balanced Growth](../example-strategies/balanced-growth.yaml)** -- Two entry strategies and two exit strategies for a balanced risk profile. Momentum buys strength while RSIOversold buys exhaustion, giving entries in both trending and recovering markets. ProfitTaking harvests gains at a fixed target while TrailingStop dynamically protects profits that have accumulated beyond that target.

## Conviction Strategies

Conviction strategies contribute scores that are blended via weighted average and transformed into target portfolio weights. Each has a configurable `weight` (default 1.0) that controls its influence in the blend.

Some conviction strategies are intentionally unidirectional -- they only produce bullish or bearish signals, never both. This prevents double-counting when complementary strategies (e.g., RSIOversold and RSIOverbought) are both active. If both sides of an indicator contributed to the blend, the signal would be counted twice.

---

### MeanReversion

**Tier**: CONVICTION | **Direction**: Bidirectional | **Score range**: [-1, +1]

Buys when price drops below its moving average, expecting it to revert back up. Sells when price stretches above the average, expecting it to come back down. The score ramps linearly with distance from the MA, reaching full conviction at the `threshold` percentage. This is a contrarian strategy -- it bets against recent price movement.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 30 | Moving average lookback period (trading days) |
| `threshold` | 0.10 | Percentage distance from MA at which score reaches full conviction |

**Suited for**: Broad market ETFs, large caps

**Interactions**: Natural counterweight to Momentum -- MeanReversion buys dips while Momentum buys strength, so using both produces more moderate positions. Overlaps significantly with VWAPReversion and BollingerBand since all three compare price to a moving average; using MeanReversion alongside either provides redundant rather than independent signals. Pairs well with RSIOversold, which measures a different dimension (up-day/down-day ratio vs. price distance from average).

---

### Momentum

**Tier**: CONVICTION | **Direction**: Buy-only | **Score range**: [0, +1]

The opposite of mean reversion -- bullish when price is above its moving average, riding the trend. Returns 0 when price is below the MA rather than going bearish, since bearish-below-MA signals are handled by dedicated strategies like MeanReversion and RSIOverbought. The score scales with how far above the MA the price has moved, reaching +1 at `momentum_scale` distance.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 20 | Moving average lookback period (trading days) |
| `momentum_scale` | 0.05 | Distance above MA at which score reaches +1 |

**Suited for**: All asset classes

**Interactions**: Works well with RSIOversold as a complementary entry pair -- Momentum buys trending assets while RSIOversold buys beaten-down ones, covering both market conditions. Conflicts with MeanReversion since they have opposing theses (trend continuation vs. trend reversal). Aligns with MovingAverageCrossover and MACDCrossover, which are also trend-following but measure trend on different timescales.

---

### ProfitTaking

**Tier**: CONVICTION | **Direction**: Sell-only | **Score range**: [-1, 0]

Encourages trimming positions that have appreciated significantly. Once unrealized gains exceed the threshold, the strategy generates a bearish score that increases with further gains. This is a conviction strategy, not a protective one -- it nudges the allocator toward reducing the position rather than forcing a liquidation. Requires cost basis context to compute unrealized gain.

| Param | Default | Description |
|-------|---------|-------------|
| `gain_threshold` | 0.20 | Unrealized gain percentage at which selling pressure begins |

**Suited for**: All asset classes

**Interactions**: Complements TrailingStop -- ProfitTaking provides a fixed exit target while TrailingStop provides a dynamic one. They work on different triggers (absolute gain vs. drawdown from peak) so there's no redundancy. Pairs naturally with any buy-side strategy as the exit mechanism. Works alongside RSIOverbought, which sells based on momentum exhaustion rather than unrealized gain.

---

### RSIOversold

**Tier**: CONVICTION | **Direction**: Buy-only | **Score range**: [0, +1]

Uses the Relative Strength Index to detect oversold conditions. Bullish when RSI is below 50, with the score increasing as RSI drops toward the oversold threshold. Neutral when RSI is at or above 50 -- the overbought side is handled separately by RSIOverbought.

The split at 50 (rather than at the threshold) means this strategy starts contributing a mild bullish signal as soon as selling pressure outweighs buying pressure, not just in extreme oversold territory.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 14 | RSI calculation period (trading days) |
| `oversold_threshold` | 30.0 | RSI level at which the score reaches +1 |

**Suited for**: All asset classes

**Interactions**: Designed as a pair with RSIOverbought -- they split the RSI indicator into buy-only and sell-only halves to avoid double-counting. Provides independent confirmation alongside BollingerBand since they measure different things (up-day/down-day ratio vs. volatility-adjusted distance from MA). Good complement to Momentum since they buy in different scenarios (trending vs. oversold).

---

### RSIOverbought

**Tier**: CONVICTION | **Direction**: Sell-only | **Score range**: [-1, 0]

The bearish counterpart to RSIOversold. Generates a sell signal when RSI is above 50, becoming more bearish as RSI approaches the overbought threshold. Neutral when RSI is at or below 50.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 14 | RSI calculation period (trading days) |
| `overbought_threshold` | 70.0 | RSI level at which the score reaches -1 |

**Suited for**: All asset classes

**Interactions**: Designed as a pair with RSIOversold. Works well alongside trend-following entry strategies (MovingAverageCrossover, MACDCrossover) as an exit signal that detects when a rally is overextended. Complements ProfitTaking since they sell on different triggers (momentum exhaustion vs. unrealized gain).

---

### BollingerBand

**Tier**: CONVICTION | **Direction**: Bidirectional | **Score range**: [-1, +1]

A volatility-aware mean reversion strategy. Computes how many standard deviations the current price is from its moving average (the z-score) and maps that to a conviction score. When price touches the lower band (negative z-score), the strategy is bullish -- it expects a bounce. When price touches the upper band (positive z-score), it's bearish. The `num_std` parameter controls band width; at exactly +/- `num_std` standard deviations, the score reaches full conviction.

Unlike plain MeanReversion, BollingerBand adapts to the stock's recent volatility. In calm markets the bands are narrow and small moves trigger conviction; in volatile markets the bands widen and it takes a larger move to reach the same score.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 20 | Moving average and standard deviation lookback (trading days) |
| `num_std` | 2.0 | Number of standard deviations defining the band width |

**Suited for**: Broad market ETFs, large caps

**Interactions**: Overlaps with MeanReversion and VWAPReversion (all are MA-based mean reversion); prefer one of the three rather than stacking them. Provides strong independent confirmation when paired with RSIOversold, since they measure fundamentally different things. Conflicts with trend-following strategies (Momentum, MovingAverageCrossover) -- BollingerBand sells rallies while trend-followers buy them.

---

### MACDCrossover

**Tier**: CONVICTION | **Direction**: Bidirectional | **Score range**: [-1, +1]

A trend-following strategy based on the convergence and divergence of two exponential moving averages. The MACD line (fast EMA minus slow EMA) measures the trend's strength and direction. The signal line (an EMA of the MACD line itself) smooths out noise. When the MACD line is above the signal line, momentum is bullish; when below, bearish.

The raw difference between MACD and signal is normalized by the current price so the score is comparable across tickers at different price levels.

Requires more historical data than most strategies -- at minimum `slow_period + signal_period` trading days (default 35).

| Param | Default | Description |
|-------|---------|-------------|
| `fast_period` | 12 | Fast EMA period |
| `slow_period` | 26 | Slow EMA period |
| `signal_period` | 9 | Signal line EMA period |

**Suited for**: All asset classes

**Interactions**: Confirms MovingAverageCrossover on a different timescale -- both are trend-following but MACD uses exponential averages and a signal line, making it more sensitive to recent price changes. Using both together gives higher confidence when they agree. Conflicts with mean-reversion strategies (MeanReversion, BollingerBand). Pairs well with RSIOverbought as an exit signal for overextended trends.

---

### VWAPReversion

**Tier**: CONVICTION | **Direction**: Bidirectional | **Score range**: [-1, +1]

Mean reversion around the average price over a lookback window. Bullish when price is below the average (expecting reversion up), bearish when above. Functionally similar to MeanReversion but uses a different threshold scale and is intended to approximate volume-weighted average price behavior.

Currently uses a simple moving average as a VWAP proxy since volume data is not available from the data provider.

| Param | Default | Description |
|-------|---------|-------------|
| `window` | 20 | Average price lookback (trading days) |
| `threshold` | 0.02 | Deviation from average at which score reaches full conviction |

**Suited for**: Large caps, broad market ETFs

**Interactions**: Highly redundant with MeanReversion and BollingerBand -- all three compare price to a moving average. Choose one rather than stacking. The tighter default threshold (2% vs. 10% for MeanReversion) makes it more sensitive to small deviations, which suits stable large caps but may generate excessive signals on volatile assets.

---

### MovingAverageCrossover

**Tier**: CONVICTION | **Direction**: Bidirectional | **Score range**: [-1, +1]

The classic golden cross / death cross strategy. Tracks two moving averages of different lengths. When the short-term MA crosses above the long-term MA (golden cross), it signals an uptrend and the score goes bullish. When the short-term crosses below (death cross), it signals a downtrend.

The score doesn't just capture the crossover moment -- it scales continuously with the spread between the two averages. A wider spread means stronger trend conviction.

| Param | Default | Description |
|-------|---------|-------------|
| `short_window` | 20 | Short-term moving average period (trading days) |
| `long_window` | 50 | Long-term moving average period (trading days) |
| `spread_scale` | 0.05 | Spread between MAs at which score reaches full conviction |

**Suited for**: All asset classes

**Interactions**: Confirms MACDCrossover on a different timescale -- both measure trend but with different calculation methods (simple vs. exponential averages), so agreement between them is meaningful. Aligns with Momentum (all trend-following). Conflicts with mean-reversion strategies. Pairs well with RSIOverbought or ProfitTaking as exit mechanisms.

---

### GapDownRecovery

**Tier**: CONVICTION | **Direction**: Buy-only | **Score range**: [0, +1]

A short-term opportunistic strategy that looks for gap-down events followed by recovery. When a stock opens significantly below the previous close (a gap-down) and then starts recovering, it generates a bullish signal. The score reflects how much of the gap has been recovered -- partial recovery produces a partial score.

Requires at least 3 days of price data to evaluate the pattern (previous close, gap open, current price). Returns 0 when there's no qualifying gap-down or no recovery.

| Param | Default | Description |
|-------|---------|-------------|
| `gap_threshold` | 0.03 | Minimum gap-down size (as fraction of previous close) to trigger |

**Suited for**: Individual equities, high-volatility stocks

**Interactions**: Fires rarely and on specific events, so it doesn't conflict with any other strategy. Pairs well with StopLoss since gap-down buying is inherently risky -- a stop loss provides a safety net if the recovery doesn't materialize. Independent from all other strategies since no other strategy measures gap patterns.

---

## Protective Strategies

Protective strategies act as safety valves. They are evaluated after conviction blending, and if their score drops to or below the configured `veto_threshold`, they force the target weight to zero -- a hard liquidation regardless of what conviction strategies say.

Both protective strategies require cost basis context to compute losses or drawdowns. They return None (abstain) when cost basis is unavailable.

The key difference between protective and conviction strategies: conviction strategies *nudge* the allocator (e.g., ProfitTaking reduces weight gradually), while protective strategies *override* it (StopLoss eliminates the position entirely).

---

### StopLoss

**Tier**: PROTECTIVE | **Default veto threshold**: -0.5

A fixed-percentage loss limiter. Computes the unrealized loss relative to cost basis. Once the loss exceeds `loss_threshold`, the score goes negative and continues dropping as the loss deepens. If the score reaches the veto threshold, the position is liquidated.

For example, with the defaults (loss_threshold=0.10, veto_threshold=-0.5), a 10% loss starts generating a negative score. The veto triggers when the loss is severe enough that the score hits -0.5, which happens at a 15% loss.

| Param | Default | Description |
|-------|---------|-------------|
| `loss_threshold` | 0.10 | Unrealized loss percentage at which the score starts going negative |

**Suited for**: All asset classes

**Interactions**: Complements TrailingStop -- StopLoss caps absolute losses while TrailingStop protects accumulated gains. They don't overlap: StopLoss fires on losing positions, TrailingStop only fires on profitable ones (it checks that current price is above cost basis). Using both gives a floor on losses and a ratchet on gains. Essential alongside dip-buying strategies (BollingerBand, RSIOversold, MeanReversion) where you're buying into price declines.

---

### TrailingStop

**Tier**: PROTECTIVE | **Default veto threshold**: -0.5

A drawdown-based exit strategy. Tracks the high-water mark (the highest price seen since purchase, or the cost basis if higher) and measures the current drawdown from that peak. Once the drawdown exceeds `trail_pct`, the score goes negative.

Unlike StopLoss which measures loss from cost basis, TrailingStop measures decline from the peak. This protects profits that have accumulated -- if a stock rallies 50% then drops 10% from the peak, TrailingStop can trigger even though the position is still profitable overall.

TrailingStop only fires when the current price is above cost basis (the position is profitable). This prevents it from compounding with StopLoss on losing positions, which would make both strategies trigger simultaneously.

| Param | Default | Description |
|-------|---------|-------------|
| `trail_pct` | 0.10 | Drawdown from high-water mark at which the score starts going negative |

**Suited for**: All asset classes

**Interactions**: Complements StopLoss (see above) and ProfitTaking. ProfitTaking nudges toward selling at a fixed gain threshold; TrailingStop dynamically protects whatever gains have accumulated, even beyond the ProfitTaking threshold. Together they create layered exits: ProfitTaking gradually reduces the position, and TrailingStop catches sharp reversals that ProfitTaking's gradual pressure can't handle.

---

## Mechanical Strategies

Mechanical strategies bypass the allocator entirely. They don't produce conviction scores -- instead they generate independent order intents on a fixed schedule. The rebalancer sizes these intents into concrete orders using available cash.

Mechanical strategy parameters are user preferences (how much to invest, how often), not performance parameters. The optimizer excludes them.

---

### DollarCostAveraging

**Tier**: MECHANICAL

Generates buy intents on a fixed trading-day interval regardless of market conditions. Each intent specifies a target dollar amount, and the rebalancer converts it to shares at the current price, constrained by available cash.

DCA fires when the number of trading days in the price history is evenly divisible by `frequency_days`. This means it triggers based on how many trading days have elapsed, not calendar dates.

| Param | Default | Description |
|-------|---------|-------------|
| `frequency_days` | 14 | Buy interval in trading days |
| `amount` | 500.0 | Dollar amount per buy |

**Suited for**: Broad market ETFs, large caps

**Interactions**: Fully independent from all other strategies since it bypasses the allocator. Can be added to any strategy composition without affecting conviction-based signals. Pairs well with protective strategies (StopLoss, TrailingStop) that can exit positions DCA builds if they go wrong.
