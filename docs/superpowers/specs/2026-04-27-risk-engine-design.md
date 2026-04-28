# Risk Engine Design

Closes #64. Replaces the abandoned PR #63 (compose-style 5-stage pipeline) with a smaller, sharper scope tuned to a personal long-only retail equity portfolio with a fixed user-chosen universe.

## Goal

Add risk discipline to the allocator without rebuilding it. The allocator's job is still "blend conviction into target weights"; the risk engine sits around it as a small set of well-bounded modifications. No solver dependency, no compose-pipeline gymnastics, no MVO. The four pieces below cover ~90% of the safety wins from #51–#58 with ~5% of PR #63's surface area.

## Scope

In:

1. **CPPI-style drawdown overlay** — scales the gross investable budget down as realized drawdown deepens, recovers as the portfolio heals.
2. **Per-ticker volatility scaling** — pluggable `weighting` policy. `equal` (default, current behavior) or `inverse_vol` (score offset of `-log(vol)` inside the softmax).
3. **Portfolio volatility target** — Ledoit-Wolf-shrunk covariance, predicted `sqrt(w'Σw)`. If predicted annualized vol exceeds target, scale the weight vector down; slack goes to cash.
4. **Risk telemetry** — read-only `RiskMetrics` (realized vol, drawdown, rolling Sharpe, per-strategy P&L attribution, per-ticker vol contribution). Surfaced through the existing output layer.

Out:

- IDM (#55) — futures-with-leverage tool, no value for long-only retail capped at 100% gross.
- FDM / forecast scaling (#56, #57) — signal-side calibration, separate concern.
- Position buffering (#54) — turnover concern, downstream of construction.
- MVO via cvxpy — overkill for a fixed conviction universe of 5–21 tickers; the user is not solving a stock-picking problem.
- Optimizer searching risk knobs — risk is policy, not parameter. Optimizer keeps searching strategy params and the existing top-level allocator knobs only.
- CLI flag to bypass risk discipline — the YAML is the toggle. To run risk-off, omit the `risk:` block (or keep a second strategies file in the repo for A/B comparisons).

## YAML Interface

The `risk:` block is entirely optional. Omit it for current behavior — every field has a conservative default and the engine reduces to today's softmax + soft cap when nothing is configured.

```yaml
softmax_temperature: 0.5
min_buy_delta: 0.02
min_cash_pct: 0.05

risk:
  weighting: inverse_vol     # equal | inverse_vol; default equal
  vol_lookback_days: 60      # used by inverse_vol AND vol_target

  vol_target: 0.20           # null/omit disables portfolio vol target

  drawdown_penalty: 1.5      # exposure = max(1 - penalty * dd, floor)
  drawdown_floor: 0.5        # null/omit either disables CPPI overlay

strategies:
  - name: BollingerBand
    weight: 1.0
    params:
      window: 20
```

Disable rules:

- No `risk:` block → all features off, current behavior bit-for-bit.
- `weighting:` defaults to `equal`. Setting `inverse_vol` uses `vol_lookback_days` (default 60).
- `vol_target` is independently nullable. Omit/`null` skips the predicted-vol scaler. When set, it shares `vol_lookback_days` with inverse-vol.
- `drawdown_penalty` and `drawdown_floor` must both be set or both omitted; either missing disables CPPI.

The optimizer does not search `risk.*`. Risk knobs are policy and easily overfit; setting them is a deliberate user act, not a search problem. There is no `--no-risk` flag — versioning two YAMLs is the right way to A/B.

## Allocator Phases

The new phases bracket the existing softmax + cap. Numbering keeps existing phases stable; new phases are inserted at the boundaries.

```
Per bar:
  Phase 0  CPPI overlay         [new]   modifies investable budget
  Phase 1  score entry signals  [exists]
  Phase 2  softmax over scores  [exists, +inverse-vol offset if configured]
  Phase 3  soft cap + redistribute [exists]
  Phase 4  portfolio vol target [new]   scales weight vector down if needed

After bar (driver responsibility):
  update running peak; recompute drawdown; update RiskMetrics
```

### Phase 0 — CPPI overlay

```
exposure_scale = max(1 - drawdown_penalty * current_drawdown, drawdown_floor)
investable     = (1 - min_cash_pct) * exposure_scale
```

`current_drawdown` is passed into `allocate()` by the driver. Backtest engine tracks the running peak of total portfolio value (already computed daily for TWR). Live mode does *not* support CPPI v1 — peak persistence requires a state file that's out of scope for this pass; `LiveEngine` warns at startup if `drawdown_penalty` is configured, similar to the existing `TrailingStop` inert-in-live warning.

### Phase 2 — inverse-vol offset

When `weighting: inverse_vol`, modify the softmax exponent for every active ticker:

```
target_i = budget * exp(blended_i / T + offset_i) / sum_j(exp(blended_j / T + offset_j))

offset_i = -log(max(vol_i, vol_floor))      where vol_floor = 0.005
```

Crucially: the offset is added *after* the `/T` divider, not inside it. PR #63 used `(1/vol)^(1/T)` (equivalently, `offset/T`), which couples inverse-vol intensity to softmax temperature — at low `T` the vol effect blows up, at high `T` it vanishes. The form above keeps the inverse-vol contribution as a `T`-independent factor: a 10× vol gap is always a 10× weight gap from vol alone, regardless of how concentrated the conviction softmax is.

Equal weighting (`offset_i = 0`) reduces to current behavior bit-for-bit.

Tickers with insufficient history (`len(prices) < vol_lookback_days`) fall back to Option A (held at current weight, excluded from the softmax), matching how the existing allocator handles missing scores.

### Phase 4 — portfolio vol target

After cap-with-redistribution converges:

```
Σ            = LedoitWolf().fit(log_returns).covariance_   # daily
predicted    = sqrt(w' Σ w) * sqrt(252)                    # annualized
if predicted > vol_target:
    scale     = vol_target / predicted
    w         = w * scale                                  # slack → cash
```

If any active ticker has fewer than `vol_lookback_days` of history, the entire Phase 4 step is skipped for that bar (covariance over a short common window is unreliable; partial coverage would silently bias the estimate). Cap is *not* re-applied after scaling — scaling can only shrink weights, so caps that bound from above remain satisfied.

## Pure-Function Discipline

The hard lesson from PR #63: risk math precomputed once at backtest start uses future information. Avoid by construction.

- New module `src/midas/risk.py` exposes pure functions only. Inputs: sliced arrays (`prices[:t+1]`). Outputs: scalars or arrays. No state, no caches.
- The allocator does not cache vol or covariance across bars. Recompute per bar against sliced history. For 5–21 tickers and a 60-day lookback this is sub-millisecond.
- `precompute_signals` is left untouched — it caches *signal scores* (which are pure functions of price up to bar T) but not risk quantities.
- A regression test compares two backtests: one truncated to bar T, one full. The risk metrics produced at bar T must be identical. Lookahead can never sneak in silently.

## Optimizer Propagation

PR #63's second blocking bug: the optimizer silently dropped the `risk:` block when constructing trial backtests. Add a regression test:

- Load a strategies YAML with a non-default `risk:` block.
- Run `midas optimize` for two trials.
- Assert that both trials' allocator instances received the configured `RiskConfig` (not the dataclass defaults).

The optimizer code does not search risk knobs (no entries added to `PARAM_RANGES`), but it must thread the configured values through to every trial backtest unchanged.

## Risk Telemetry

New module `src/midas/risk_metrics.py`:

```python
@dataclass(frozen=True)
class RiskMetrics:
    realized_vol_60d: float           # annualized stdev of portfolio log returns
    vol_target: float | None          # configured target (None if disabled)
    drawdown_from_peak: float
    rolling_sharpe_252d: float
    per_strategy_pnl: dict[str, float]
    per_ticker_vol_contribution: dict[str, float]
```

Computed each bar in backtest, each tick in live. Backtest threads it through `BacktestResult`; live threads it through the existing periodic output. Strictly observational — no feedback into construction.

Per-strategy attribution: when a buy fills, attribute its share of P&L by the strategy's contribution to the blended score that produced the order (data already on `OrderContext.contributions`). Closing P&L flows back to the same attribution buckets.

## Files to Create or Modify

New:

- `src/midas/risk.py` — pure functions: `realized_vol`, `covariance_matrix`, `predict_portfolio_vol`, `apply_drawdown_overlay`, `inverse_vol_offset`.
- `src/midas/risk_metrics.py` — `RiskMetrics` dataclass and per-bar computation.
- `tests/test_risk.py` — unit tests for the risk module.
- `tests/test_risk_metrics.py` — unit tests for telemetry.
- `tests/test_allocator_risk.py` — allocator integration tests (CPPI, inverse_vol, vol_target).
- `tests/test_lookahead_regression.py` — sliced-vs-full bar-T equality test.
- `tests/test_optimizer_risk_propagation.py` — risk config survives the optimizer trial loop.

Modified:

- `src/midas/models.py` — `RiskConfig` dataclass with frozen defaults.
- `src/midas/config.py` — parse the `risk:` block from YAML; validate consistency (e.g., `inverse_vol` requires `vol_lookback_days`; `drawdown_*` are both-or-neither).
- `src/midas/allocator.py` — accept `RiskConfig`, optional `current_drawdown`; new Phase 0 (CPPI) and Phase 4 (vol target); inverse-vol offset in softmax.
- `src/midas/backtest.py` — track running peak, pass `current_drawdown` into `allocate()`, populate `RiskMetrics` per bar, attach to `BacktestResult`.
- `src/midas/live.py` — log warning at startup if `drawdown_*` is configured (CPPI inert in live v1); populate `RiskMetrics` per tick.
- `src/midas/optimizer.py` — thread `RiskConfig` through trial backtests; add regression test.
- `src/midas/output.py`, `src/midas/results.py` — surface `RiskMetrics` in backtest and live output.
- `pyproject.toml` — add `scikit-learn` (Ledoit-Wolf shrinkage).
- `docs/architecture.md` — document Phases 0 and 4, the inverse-vol offset, telemetry.
- `docs/strategies.md` — document the `risk:` YAML block alongside existing allocator knobs.

## Test Plan

Unit (`test_risk.py`):

- `realized_vol`: known synthetic series → known annualized vol.
- `covariance_matrix`: shape, symmetry, positive semi-definite.
- `predict_portfolio_vol`: equal-weight uncorrelated portfolio with σ=10% each → predicted ≈ 10% / sqrt(N).
- `apply_drawdown_overlay`: 0% DD → scale 1.0; 20% DD with penalty 1.5 → 0.70; 50% DD with floor 0.5 → 0.5 (floor binds).
- `inverse_vol_offset`: `-log(vol)`, with floor at `vol_floor`.

Allocator integration (`test_allocator_risk.py`):

- `weighting: equal` matches current allocator output bit-for-bit (regression).
- `weighting: inverse_vol`: high-vol ticker gets less weight at same blended score.
- Phase 0: synthetic 20% drawdown shrinks `investable` to `(1 - min_cash_pct) * 0.7`.
- Phase 4: predicted vol exceeds target → all weights scale by `target/predicted`; sum drops below investable, slack to cash.
- Insufficient history: vol targeting and inverse-vol both fall back gracefully (no crash, no silent zeros).

Telemetry (`test_risk_metrics.py`):

- Synthetic equity curve with known DD → expected `drawdown_from_peak`.
- Synthetic returns with known mean/stdev → expected `rolling_sharpe_252d`.
- Per-strategy attribution: two-strategy synthetic backtest with known contributions → expected P&L split.

Regression:

- `test_lookahead_regression.py`: backtests truncated at bar T and full both produce identical `RiskMetrics[T]`.
- `test_optimizer_risk_propagation.py`: non-default `risk:` block survives `midas optimize` trial construction.

## Migration

No migration required. Every existing YAML continues to work with no behavior change (no `risk:` block → no risk features). The architecture and strategies docs gain a new section; existing sections are unchanged.
