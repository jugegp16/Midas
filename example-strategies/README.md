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

- **Trailing Stop** (10% trail, exit) — tracks the highest price each lot reaches after purchase (the "high-water mark") and sells if the price drops 10% from that peak — but only while above cost basis. Dynamically adjusts upward as the price rises, protecting gains without capping upside. Unlike a regular stop loss, it never forces a sell at a loss.

Best for: a general-purpose portfolio that balances growth capture with downside protection.
