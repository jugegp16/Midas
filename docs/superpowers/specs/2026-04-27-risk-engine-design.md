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
- `vol_target` is independently nullable. Omit/`null` skips the predicted-vol scaler. When set, it shares `vol_lookback_days` with inverse-vol — a deliberate consolidation, since inverse-vol is a relative tilt (where responsiveness matters less than stability) and Ledoit-Wolf already shrinks the covariance toward stable estimates. If a future user genuinely needs separate windows, split the knob then; don't pre-build the option.
- `drawdown_penalty` and `drawdown_floor` must both be set or both omitted; either missing disables CPPI.

The optimizer does not search `risk.*`. Risk knobs are policy and easily overfit; setting them is a deliberate user act, not a search problem. There is no `--no-risk` flag — versioning two YAMLs is the right way to A/B.

All vol quantities in this spec (`vol_target`, `vol_floor`, internal `vol_i`) are annualized via `* sqrt(252)`. Daily log-return stdev is annualized at the `realized_vol` helper boundary, and never appears in daily units anywhere in this design.

Risk telemetry is **always populated** regardless of `risk:` block presence. With features disabled, fields like `vol_target` report as `None` while `drawdown_from_peak`, `realized_vol_60d`, `rolling_sharpe_252d`, and per-strategy attribution still compute against actual portfolio history. The "bit-for-bit current behavior" guarantee covers allocation outputs (target weights, fills); `BacktestResult.risk_metrics` is a new always-present field independent of `risk:` configuration.

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

`current_drawdown` is a **non-negative magnitude**: 0 at a new peak, 0.20 at 20% below peak. The backtest driver computes it as `max(0, 1 - portfolio_value / running_peak)` and passes it into `allocate()`. The allocator additionally clamps `exposure_scale` to `[drawdown_floor, 1.0]` to defend against future drift in the input contract — a negative `current_drawdown` (a "drawdown above peak", which can only result from a driver bug) saturates `exposure_scale` at `1.0`, never above. Backtest engine tracks the running peak of total portfolio value (already computed daily for TWR). Live mode does *not* support CPPI v1 — peak persistence requires a state file that's out of scope for this pass; `LiveEngine` warns at startup if `drawdown_penalty` is configured, similar to the existing `TrailingStop` inert-in-live warning.

`min_cash_pct` and `drawdown_floor` compose multiplicatively: a user choosing `drawdown_floor: 0.0` still holds `min_cash_pct` in cash because Phase 0 multiplies `(1 - min_cash_pct)` by `exposure_scale`, never the other way around. With `min_cash_pct: 0.05, drawdown_floor: 0.5`, deep drawdowns leave `investable = 0.95 * 0.5 = 0.475`.

This is a documented backtest/live divergence: a strategy backtested with `drawdown_penalty: 1.5` will scale exposure down during drawdowns in backtest but not in live deployment. Users setting `drawdown_penalty` should expect backtest numbers to overstate live behavior during drawdown periods. Resolving this is tracked for v2 (peak persistence via state file). See [Live Mode v1 Limitations](#live-mode-v1-limitations) for the consolidated list.

### Phase 2 — inverse-vol offset

When `weighting: inverse_vol`, modify the softmax exponent for every active ticker:

```
target_i = budget * exp(blended_i / T + offset_i) / sum_j(exp(blended_j / T + offset_j))

offset_i = -log(max(vol_i, vol_floor))      where vol_floor = 1e-8
```

Crucially: the offset is added *after* the `/T` divider, not inside it. PR #63 used `(1/vol)^(1/T)` (equivalently, `offset/T`), which couples inverse-vol intensity to softmax temperature — at low `T` the vol effect blows up, at high `T` it vanishes. The form above keeps the inverse-vol contribution as a `T`-independent factor: a 10× vol gap is always a 10× weight gap from vol alone, regardless of how concentrated the conviction softmax is.

Equal weighting (`offset_i = 0`) reduces to current behavior bit-for-bit.

Tickers with insufficient history (`len(prices) < vol_lookback_days`), zero realized vol, or all-NaN closes over the lookback window fall back to Option A (held at current weight, excluded from the softmax), matching how the existing allocator handles missing scores. The `vol_floor` is a numerical `log(0)` guard set well below any realistic annualized vol; it never binds for assets with normal price activity. If **every** active ticker hits the fallback, Phase 2 holds all positions at current weight (the softmax has no inputs and produces no new targets) and Phase 4 is skipped for the same bar.

### Phase 4 — portfolio vol target

After cap-with-redistribution converges:

```
Σ            = LedoitWolf().fit(log_returns).covariance_   # daily
predicted    = sqrt(w' Σ w) * sqrt(252)                    # annualized
if predicted > vol_target:
    scale     = vol_target / predicted
    w         = w * scale                                  # slack → cash
```

Phase 4 operates on the **active subset's weights only**. Held positions (excluded from softmax in Phase 2 or carried forward via the fallback) pass through unchanged — their weights are not scaled and their covariance rows do not enter the `sqrt(w'Σw)` calculation. The vol target therefore caps risk from *new conviction*, not total portfolio risk; held positions reflect prior allocation decisions whose risk was already evaluated at entry, and re-targeting their size based on present vol would introduce path-dependent re-allocation drift.

If any active ticker has fewer than `vol_lookback_days` of history, or zero realized vol over the window, the entire Phase 4 step is skipped for that bar (covariance over a short common window is unreliable; partial coverage would silently bias the estimate; a constant-price ticker produces a singular row in `Σ`). Phase 4 runs for any `N ≥ 1` active tickers — a `1×1` covariance is well-defined and the math reduces to scaling the single active weight by `vol_target / annualized_realized_vol`. Cap is *not* re-applied after scaling — scaling can only shrink weights, so caps that bound from above remain satisfied.

Phase 0 and Phase 4 stack multiplicatively. A 20% drawdown with `drawdown_penalty: 1.5` shrinks `investable` to 70% of its base value; if the resulting weight vector still has predicted vol above target, Phase 4 scales it down further. With aggressive settings on both knobs (e.g., `drawdown_penalty: 2.0, vol_target: 0.10` during a 25% drawdown on a high-vol portfolio) gross exposure can drop well below 50%. This is intentional — both phases are reduce-only — but worth being aware of when configuring both.

## Pure-Function Discipline

The hard lesson from PR #63: risk math precomputed once at backtest start uses future information. Avoid by construction.

- New module `src/midas/risk.py` exposes pure functions only. Inputs: sliced arrays (`prices[:t+1]`). Outputs: scalars or arrays. No state, no caches.
- The allocator does not cache vol or covariance across bars. Recompute per bar against sliced history. For 5–21 tickers and a 60-day lookback this is sub-millisecond.
- `precompute_signals` is left untouched — it caches *signal scores* (which are pure functions of price up to bar T) but not risk quantities.
- A regression test compares two backtests: one truncated to bar T, one full. **Both the allocation outputs (target weights at bar T) and the risk metrics produced at bar T must be identical.** Comparing allocation outputs (not just metrics) is the strictly stronger invariant — it catches any future shortcut that adds risk quantities to `precompute_signals`, even when those quantities don't surface in `RiskMetrics`. Lookahead can never sneak in silently.

## Optimizer Propagation

PR #63's second blocking bug: the optimizer silently dropped the `risk:` block when constructing trial backtests. Add two complementary regression tests:

**Structural** — catches a fully-dropped block:
- Load a strategies YAML with a non-default `risk:` block.
- Run `midas optimize` for two trials with identical strategy params.
- Assert both trials' allocator instances received the configured `RiskConfig` (not the dataclass defaults).

**Behavioral** — catches a partially-wired pipeline (allocator gets the config but the backtester ignores `current_drawdown`, or telemetry uses defaults):
- Run `midas optimize` twice with materially different `risk:` blocks (e.g., `weighting: equal` vs `weighting: inverse_vol` with `vol_target: 0.10`) on identical strategy params and identical input data.
- Assert the resulting equity curves differ.

The structural test confirms config plumbing; the behavioral test confirms config actually changes execution. The optimizer code does not search risk knobs (no entries added to `PARAM_RANGES`), but it must thread the configured values through to every trial backtest unchanged.

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

Per-strategy attribution uses a **cost-basis-weighted** attribution dict per position. On each buy, the position's attribution dict is updated as a size-weighted blend of the prior dict (weighted by existing basis) and the order's contribution dict (weighted by buy size, sourced from `OrderContext.contributions`):

```python
new_basis = position.basis + buy_size
for strat, contrib in order.contributions.items():
    position.attribution[strat] = (
        position.attribution.get(strat, 0.0) * (position.basis / new_basis)
        + contrib * (buy_size / new_basis)
    )
position.basis = new_basis
```

On **partial sell**, attribution shares are preserved and `position.basis` is scaled by the surviving quantity ratio (`basis *= remaining_qty / prior_qty`). Realized P&L from the sell flows to per-strategy buckets in the current attribution shares. On **full close**, both attribution dict and basis reset to empty; any future re-entry into the ticker starts fresh.

On mark-to-market, unrealized P&L is split into per-strategy buckets by the current attribution shares. This is consistent with midas's existing aggregate-position philosophy (`docs/architecture.md:187`) — no lot-level tracking required. The accuracy loss vs lot-level is bounded and only material when a position grows over many bars with rapidly shifting attribution mixes; the typical case (one strategy entering, optionally adding) is exact.

**Worked example.** Position starts empty (`basis=0`, `attribution={}`):

1. Buy $100, contributions `{A: 1.0}` → `basis=100`, `attribution={A: 1.0}`.
2. Buy $50, contributions `{B: 1.0}` → `basis=150`, `attribution={A: 0.667, B: 0.333}`.
3. Mark to market with $200 unrealized P&L → `pnl_buckets = {A: $133.33, B: $66.67}`.
4. Partial sell of 50% of shares → `basis=75`; attribution unchanged; realized P&L from the sell splits 0.667/0.333 into A/B.
5. Full close → remaining P&L splits 0.667/0.333; `basis=0`, `attribution={}`.

**Live-mode telemetry caveats.** Without peak persistence (deferred to v2 alongside CPPI), live-mode `drawdown_from_peak` is computed against a session-start peak. Restarting the live engine resets the peak tracker — drawdown reads as 0 immediately after restart regardless of the portfolio's actual drawdown state. `realized_vol_60d`, `rolling_sharpe_252d`, per-strategy attribution, and per-ticker vol contribution are unaffected, since they're computed from rolling windows of in-session data. See [Live Mode v1 Limitations](#live-mode-v1-limitations).

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
- `apply_drawdown_overlay`: 0% DD → scale 1.0; 20% DD with penalty 1.5 → 0.70; 50% DD with floor 0.5 → 0.5 (floor binds); negative `current_drawdown` (driver bug) clamps `exposure_scale` to `1.0` (defensive).
- `inverse_vol_offset`: `-log(vol)`, with floor at `vol_floor`.

Allocator integration (`test_allocator_risk.py`):

- `weighting: equal` matches current allocator output bit-for-bit (regression).
- `weighting: inverse_vol`: high-vol ticker gets less weight at same blended score.
- `weighting: inverse_vol` is **T-independent**: two tickers with identical blended scores and 2:1 vol ratio produce ≈ 2:1 weight ratio at both `softmax_temperature=0.2` and `softmax_temperature=2.0`. Targets the PR #63 `1/vol^(1/T)` regression directly — that bug would produce ~32:1 and ~1.4:1 instead.
- Phase 0: synthetic 20% drawdown shrinks `investable` to `(1 - min_cash_pct) * 0.7`.
- Phase 4: predicted vol exceeds target → all weights scale by `target/predicted`; sum drops below investable, slack to cash.
- Phase 0 + Phase 4 composition: synthetic 20% drawdown plus vol-target-binding portfolio → final exposure ≈ `(1 - min_cash_pct) * 0.7 * (target / predicted)`. Confirms the two reduce-only phases stack as expected.
- Phase 0 + `min_cash_pct` composition: `min_cash_pct=0.05, drawdown_floor=0.5`, 50% drawdown → `investable = 0.95 * 0.5 = 0.475`. Confirms the cash-side floors compose multiplicatively.
- Phase 4 single-ticker (`N=1`): vol-target-binding with one active ticker scales the single weight by `vol_target / annualized_realized_vol`; no degenerate-singular-cov fallback fires.
- Phase 4 with held positions: held tickers' weights pass through unchanged when Phase 4 binds; only active weights scale.
- All-fallback degenerate: every active ticker hits Phase 2 fallback (insufficient history or all-NaN closes) → all positions held at current weight, Phase 4 skipped, no crash.
- Insufficient history or zero realized vol: vol targeting skips the bar entirely; inverse-vol falls back to Option A for the affected ticker (no crash, no silent zeros).

Telemetry (`test_risk_metrics.py`):

- Synthetic equity curve with known DD → expected `drawdown_from_peak`.
- Synthetic returns with known mean/stdev → expected `rolling_sharpe_252d`.
- Per-strategy attribution: two-strategy synthetic backtest with known contributions → expected P&L split.

Regression:

- `test_lookahead_regression.py`: backtests truncated at bar T and full both produce identical `RiskMetrics[T]`.
- `test_optimizer_risk_propagation.py`: non-default `risk:` block survives `midas optimize` trial construction.

## Live Mode v1 Limitations

Two backtest/live divergences in v1, both rooted in the absence of peak persistence:

1. **CPPI overlay (Phase 0)** is inert in live; backtests with `drawdown_penalty` set will overstate live exposure-scaling behavior during drawdowns. `LiveEngine` warns at startup when configured.
2. **Telemetry `drawdown_from_peak`** in live is computed against a session-start peak; engine restarts reset the peak tracker.

Both are tracked for v2 (state-file-based peak persistence). Until then, treat backtest CPPI behavior as a planning aid rather than a live performance prediction, and recognize that live `drawdown_from_peak` reflects only the current session.

## Migration

No migration required. Every existing YAML continues to work with no behavior change (no `risk:` block → no risk features). The architecture and strategies docs gain a new section; existing sections are unchanged.
