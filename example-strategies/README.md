# Example Strategy Combinations

Pre-built strategy configurations that pair well together based on complementary signal types and risk profiles.

## [Trend-Following](trend-following.yaml)

Rides sustained price trends and exits when momentum fades.

- **Moving Average Crossover** (20/50-day) — computes two moving averages of the stock price: one over the last 20 days (short-term) and one over the last 50 days (long-term). When the short-term average crosses above the long-term average, it suggests the price is gaining momentum and trending upward. When it crosses below, the trend may be reversing. This is one of the most widely used signals for identifying whether an asset is in an uptrend or downtrend.

- **Moving Average Convergence Divergence Crossover** (12/26/9) — a more sensitive trend indicator that works by comparing two exponential moving averages (12-day and 26-day) and then smoothing the difference with a 9-day signal line. When the main line is above the signal line, momentum is building to the upside. When it falls below, momentum is fading. It confirms trend strength on a different timescale than the Moving Average Crossover, so the two strategies agreeing gives higher confidence.

- **Relative Strength Index Overbought** (14-period, threshold 70) — the Relative Strength Index measures how much of an asset's recent price movement has been upward versus downward over the last 14 days, producing a score from 0 to 100. When the score climbs above 70, the asset is considered "overbought" — it has risen so much, so fast, that a pullback is likely. This strategy generates a sell signal that scales in strength as the score rises past 50, reaching full conviction at 70 and above.

- **Profit Taking** (20% threshold) — a simple, price-aware exit rule. When your unrealized gain on a position exceeds 20% of what you originally paid for it (your cost basis), this strategy signals to sell. It ensures you lock in profits rather than watching them evaporate during a reversal.

Best for: assets in clear uptrends or downtrends. Less effective in sideways or choppy markets where trends don't sustain.

## [Dip-Buying](dip-buying.yaml)

Buys when prices are statistically cheap relative to recent history, with a hard floor on losses.

- **Bollinger Band** (20-day, 2 standard deviations) — calculates the average price over the last 20 days, then draws an upper and lower "band" two standard deviations above and below that average. Standard deviation measures how spread out prices have been — wider bands mean more volatility. When the current price drops below the lower band, the asset is unusually cheap relative to its recent trading range, which triggers a buy signal. When the price stretches above the upper band, it's unusually expensive and triggers a sell signal.

- **Relative Strength Index Oversold** (14-period, threshold 30) — the counterpart to the overbought strategy. When the Relative Strength Index drops below 30, the asset is considered "oversold" — it has fallen so much that sellers may be exhausted and a bounce is likely. This strategy generates a buy signal that scales in strength as the score falls below 50, reaching full conviction at 30 and below. It measures something fundamentally different from the Bollinger Band (the ratio of up-days to down-days versus distance from a price average), so when both agree that an asset is cheap, the signal is more trustworthy.

- **Stop Loss** (10% threshold) — a protective exit that triggers when your position has lost more than 10% from your cost basis. This is essential in a dip-buying strategy because buying assets that have dropped in price means you are sometimes buying into continued declines (known as "catching a falling knife"). The stop loss puts a hard floor on how much you can lose on any single position.

Best for: broad market index funds and large, established companies that tend to bounce back after dips. Riskier on individual stocks in long-term decline, where cheap prices may just keep getting cheaper.

## [Balanced Growth](balanced-growth.yaml)

Moderately aggressive entries with disciplined exits. Two strategies push you in, two pull you out.

- **Momentum** (20-day) — a buy-only signal that compares the current price to the 20-day moving average. When the price is above the average, it means the asset has been trending upward recently, and this strategy signals to buy more. When the price is below the average, it stays neutral rather than signaling to sell — that job is left to the exit strategies below. This one-sided design makes it a clean entry signal that doesn't conflict with the exit rules.

- **Relative Strength Index Oversold** (14-period, threshold 30) — generates buy signals when the Relative Strength Index falls below 50, indicating that recent selling pressure has outweighed buying pressure. This complements Momentum by buying in a different scenario: Momentum buys strength (price trending up), while this strategy buys exhaustion (price has been beaten down and may recover). Together, they give you entries in both trending and recovering markets.

- **Profit Taking** (20% threshold) — trims positions when your unrealized gain exceeds 20% of your cost basis. This is a fixed target that ensures you systematically harvest profits from your winners. Without it, profitable positions can ride all the way up and back down again.

- **Trailing Stop** (10% trail) — instead of using a fixed price to trigger a sell, this strategy tracks the highest price an asset reaches after you buy it (the "high-water mark") and sells if the price drops 10% from that peak — but only while you're still above your cost basis. This means it dynamically adjusts upward as the price rises, protecting more and more of your gains without capping your upside. Unlike a regular stop loss, it never forces you to sell at a loss.

Best for: a general-purpose portfolio that balances growth capture with downside protection. The symmetry of two entry and two exit signals keeps the risk profile balanced.
