# Example Strategy Combinations

Pre-built strategy configurations that pair entry signals with exit rules. Every composition needs at least one of each — entry signals decide *what to buy*, exit rules decide *what to sell*. Buys and sells live in completely separate code paths and never blend.

## [Trend-Following](trend-following.yaml)

Rides sustained price trends and exits when the same trend reverses.

- **Moving Average Crossover** (20/50-day, entry) — computes two moving averages of the stock price: one over the last 20 days (short-term) and one over the last 50 days (long-term). When the short-term average crosses above the long-term average, the price is gaining upward momentum and the strategy goes bullish. The score scales with the spread between the two averages.

- **MACD Crossover** (12/26/9, entry) — a more sensitive trend indicator that compares two exponential moving averages and smooths the difference with a 9-day signal line. When the main line is above the signal line, momentum is building to the upside. It confirms trend strength on a different timescale than the Moving Average Crossover, so the two strategies agreeing gives higher confidence.

- **Moving Average Crossover Exit** (20/50-day, exit) — the symmetric exit counterpart. When the short-term average crosses *below* the long-term average (the "death cross"), the trend has reversed and this rule liquidates the position. Pairing the entry and exit on the same timescale keeps the strategy internally consistent.

- **MACD Exit** (12/26/9, exit) — fires when the MACD line crosses below the signal line, indicating that the trend the MACD entry rode in on has fizzled. Like its sibling above, it provides a cleanly symmetric exit for its entry counterpart.

- **Profit Taking** (20% threshold, exit) — a price-aware exit that triggers when the unrealized gain on a lot exceeds 20% of its cost basis. Locks in profits independently of trend signals, so a trending position that's already up 20% will be trimmed before the trend has a chance to reverse.

Best for: assets in clear uptrends. Less effective in sideways or choppy markets where trends don't sustain.

## [Dip-Buying](dip-buying.yaml)

Buys when prices are statistically cheap relative to recent history, with a hard floor on losses.

- **Bollinger Band** (20-day, 2 standard deviations, entry) — calculates the average price over the last 20 days and draws an upper and lower "band" two standard deviations above and below that average. When the current price drops below the lower band, the asset is unusually cheap relative to its recent trading range, and the strategy goes bullish.

- **RSI Oversold** (14-period, threshold 30, entry) — when the Relative Strength Index drops below 50, sellers may be exhausted and a bounce is likely. The score scales as the index falls toward the 30 threshold. RSI measures something fundamentally different from Bollinger Band (the ratio of up-days to down-days versus distance from a price average), so when both agree, the signal is more trustworthy.

- **Stop Loss** (10% threshold, exit) — a protective exit that triggers when a lot has lost more than 10% from its cost basis. Essential in a dip-buying strategy because buying assets that have dropped sometimes means buying into continued declines. The stop loss puts a hard floor on how much you can lose on any single lot.

Best for: broad market index funds and large, established companies that tend to bounce back after dips. Riskier on individual stocks in long-term decline.

## [Balanced Growth](balanced-growth.yaml)

Moderately aggressive entries with disciplined exits. Two entries push you in, two exits pull you out.

- **Momentum** (20-day, entry) — buys when the current price is above its 20-day moving average. The score scales with how far above the average the price has moved. A clean buy-only signal — when the price drops below the average, the strategy stays neutral rather than producing a sell, leaving exits entirely to the exit rules.

- **Mean Reversion** (30-day, 10% threshold, entry) — buys when the price drops below its 30-day moving average, expecting it to revert back up. Complements Momentum by buying in the opposite scenario: Momentum buys strength while Mean Reversion buys weakness, so the portfolio finds entries in both trending and recovering markets.

- **Profit Taking** (20% threshold, exit) — trims lots whose unrealized gain exceeds 20%. A fixed target that ensures profits get harvested from winners.

- **Chandelier Stop** (22-day window, 3.0× ATR, exit) — a volatility-adjusted trailing stop. Tracks the highest close over the last 22 days and sells if the current price falls more than 3× the recent average true range below that high. Unlike a fixed-percent trailing stop, the stop distance breathes with realized volatility — tighter in calm markets, wider in choppy ones — and the rolling-window reference high keeps the stop responsive to recent price action instead of anchoring to a months-old peak. Fires on both profitable and underwater positions, so it subsumes the role of a separate stop loss.

Best for: a general-purpose portfolio that balances growth capture with downside protection.

## [Turtle Breakout](turtle-breakout.yaml)

The classic "buy new highs" system from Richard Dennis's Turtle Trading experiment, updated with a modern trailing-stop exit.

- **Donchian Breakout** (20-day, entry) — buys when the current close exceeds the highest close of the prior 20 days. This is the short Turtle system — it fires on fresh breakouts and is silent the rest of the time. The score ramps linearly once the breakout happens, so a small poke above the prior high gets partial conviction while a 2%+ push gets full conviction.

- **Keltner Channel** (20-day, 2.0× ATR, entry) — confirms the breakout from a different angle. Rather than comparing the current price to a prior high, Keltner compares it to the moving average plus two times recent average true range. Donchian fires on raw price level; Keltner fires on volatility-adjusted distance from the average. When both agree, the breakout is both "new" and "large relative to normal noise" — independent confirmation that isn't just two copies of the same signal.

- **Parabolic SAR Exit** (AF 0.02 → 0.20, exit) — Wilder's self-accelerating trailing stop. Starts loose right after the breakout (small acceleration factor, SAR trails far below price) and ratchets tighter every time the price makes a new high, until the stop is hugging price late in the trend. The result is a stop that gives breakouts room to develop but locks down the exit once the move has had time to play out — a natural pairing with breakout entries.

- **Profit Taking** (30% threshold, exit) — an absolute ceiling on how long to hold a single breakout. Even with a trailing stop still loose, any lot up 30% gets trimmed so winners don't round-trip back to the basis.

Best for: trending, high-momentum assets (growth stocks, momentum ETFs, commodities-like assets). Poor fit for range-bound assets where repeated false breakouts chop the strategy.

## [Volatility Momentum](volatility-momentum.yaml)

A trend-following system built entirely on volatility-adjusted signals. Every component scales with realized ATR, so the strategy adapts its sensitivity as market conditions change.

- **Keltner Channel** (20-day, 2.0× ATR, entry) — bullish when the price breaks above the SMA + 2 × ATR upper band. In calm markets the band is tight and small moves trigger conviction; in volatile markets the band is wide and it takes a bigger move to fire. This auto-calibration keeps the entry from getting trigger-happy during chop and from missing real breakouts during high-volatility regimes.

- **Momentum** (20-day, entry) — buys when the price is above its 20-day moving average. Provides continuation support: once Keltner fires on a breakout, Momentum keeps scoring bullish as long as the price stays elevated, so the allocator won't prematurely drain the position just because the initial breakout excess has compressed back into the band.

- **Chandelier Stop** (22-day, 3.0× ATR, exit) — a volatility-matched trailing stop. Since the entries are calibrated in ATR units, it only makes sense for the exit to use the same yardstick: a 3× ATR trailing stop sits at roughly the same "distance in normal-noise terms" regardless of the ticker. The whole strategy speaks one consistent volatility language end-to-end.

- **Profit Taking** (25% threshold, exit) — a fixed-gain harvest to complement the volatility-relative trailing stop. Even if Chandelier is trailing at a wide ATR multiple, Profit Taking still ensures any lot up 25% is trimmed on schedule.

Best for: assets with stable, measurable volatility (index ETFs, large-cap tech). Less suited to micro-caps or post-earnings-spike situations where ATR is noisy or non-stationary.

## [Long Trend](long-trend.yaml)

A patient system built for capturing multi-month trends without getting shaken out by short-term pullbacks.

- **Donchian Breakout** (55-day, 3% breakout scale, entry) — the long Turtle system. A 55-day window demands a genuine regime change before firing, so the strategy sits out noise and false starts and only engages when a meaningful trend has formed. The wider 3% breakout scale means the score takes a larger push to reach full conviction, reflecting that long-window breakouts are higher-stakes commitments than short-window ones.

- **MACD Crossover** (12/26/9, entry) — trend confirmation on a completely different timescale and indicator family. Donchian is a binary event (new N-bar high, yes or no); MACD is a continuous measure of momentum strength. Together they fire when both the level (new multi-month high) and the acceleration (bullish MACD) agree the trend is real.

- **Parabolic SAR Exit** (AF 0.01 → 0.15, exit) — a deliberately slow version of Wilder's SAR. The smaller starting and step AFs mean the stop acceptance far more slack in the early weeks of a trend — exactly when Donchian has just fired and the trend needs room to breathe — and only tightens well after the position is deep in profit. The capped 0.15 max AF keeps the stop from ever getting so tight that normal pullbacks close the position.

- **Profit Taking** (35% threshold, exit) — a generous ceiling that reflects the long-trend thesis: these are meant to be multi-month holds, so trimming at 20% would cut winners off just as they're getting started.

Best for: secular bull markets and long-duration uptrends where the thesis is measured in months rather than weeks. A poor fit for fast-moving, mean-reverting assets where a 55-day lookback means you're always buying the top.
