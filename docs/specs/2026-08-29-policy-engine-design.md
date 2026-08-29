# Policy engine: ensemble deployment, validated re-fit, live monitoring — design

**Date:** 2026-08-29 (v2 after external review, 2026-08-29)
**Status:** Draft — for review
**Supersedes:** the 2026-08-23 comparison-harness draft (absorbed here)
**Delivery:** one PR, phased commits (decision: no sub-issues)
**Evidence:** `docs/experiments/2026-08-24-seed-budget-grid/` (lab notes +
regenerable results), the multi-seed table on issue #49, `docs/cli.md`
"Trial budget and search noise", commit `ee0c93e`.

## Problem

Two incoherences, one conceptual and one statistical.

**Conceptual.** Walk-forward validation does not validate parameters; it
validates a *procedure* — "at each fold boundary, re-fit on all history to
date with a given budget and objective, then trade the result." The OOS
numbers it produces measure that procedure. But live runs frozen
parameters indefinitely: optimize-once-and-hold is a policy nobody
validated, and the walk-forward evidence stops describing the live engine
the day after the next fold would have re-fit.

**Statistical.** Every number the optimizer prints is a point estimate
with no error bar, and the measured error bars are larger than the effects
being read from them. What is recorded (sources above; per-claim
limitations in the lab notes):

- The objective landscape is rugged and multi-modal at the current search
  dimensionality; TPE settles into a basin per seed (same fold, seeds
  reached 1324% vs 2077% training at 666 trials, diverging further at
  2,000 — and one seed stalled terminally, identical at both budgets).
- Aggregate OOS CAGR is far more seed-stable than its folds: spread
  7.6 → 2.9 pts (30 → 100 trials/fold, etf) while per-fold spread stayed
  ~27–30 pts with zero folds seed-agreeing within a point. The test
  basket's seed spread was ~27 pts at 30 trials/fold (no higher-budget
  counterpart recorded).
- Paired per-fold comparison has no discriminating power here: a variant
  against itself across seeds produces the same paired spread and "win
  rates" as a real variant comparison (the null control in the lab notes).
- **Directional hints, not results** (n too small, and measured on
  `gross` while the reference policy is `calmar`): best-of-seeds selection
  by training score looked ~+2.7 pts OOS over an average seed; the
  train→OOS rank correlation (+0.25, SE ≈ 0.2) is not distinguishable
  from zero. The landing experiments below are what turn these into
  answers.
- The noise is a property of a rugged, ~47-dimensional objective searched
  with 10³–10⁴ evaluations — not of TPE specifically; a grid is
  impossible at this dimensionality and random search would show the same
  variance. That makes dimensionality itself a design target (Step 0).

**Practice context.** Established products converge on walk-forward plus
robustness batteries (TradeStation cluster analysis, StrategyQuant Monte
Carlo suites). For deployment, two relevant precedents — neither of which
we have found *measured* against single-best in the literature we
located, which is why both are hypotheses to test here rather than
results to import:

- **Forecast combination** (Carver / systematic-CTA practice): combine
  the forecasts of rule *variations* before sizing, rather than picking
  one variation.
- **System Parameter Permutation** (Walton): the opposite of selection —
  evaluate the whole parameter space and take its median, precisely to
  remove selection bias.

## The unifying object: the deployment policy

```yaml
policy:
  objective: calmar          # any of the six
  budget: 2000               # trials per restart
  restarts: 4                # independent seeded searches per fit
  base_seed: 42              # restart seeds: base_seed + i
  deployment: ensemble       # ensemble | best
  ensemble_size: 20          # top-K, stratified across restarts
  refit_cadence: fold        # fits scheduled at the validator's fold length
  adopt_trigger:             # adoption is degradation-gated, not score-gated
    oos_percentile_below: 10   # trailing window under this fold percentile, or
    drawdown_breach: true      # trailing drawdown beyond the validated worst
```

**The coherence principle (hard constraint).** One function,
`fit_as_of(price_data, date, policy, incumbent)`, executes a fit. The
walk-forward validator calls it per fold — *including the adoption rule
and carried positions* — and live's scheduled re-fit calls it going
forward. The validator's OOS distribution is a prediction about live only
to the extent the two run the same procedure; every known gap between
them is listed in "Coherence gaps" below and either closed or disclosed.

## Step 0 — shrink the search space

The optimizer currently searches ~47 dimensions: entry parameters,
weights, allocator globals (`softmax_temperature`, `min_buy_delta`,
`max_position_pct`), and exit-rule parameters. This is the root of the
basin problem, and systematic practice (fixed lookback ladders, minimal
fitting of slow rules) argues most of those dimensions should not be
searched at all.

New split, enforced by the fitter:

- **Searched (per member):** entry-signal parameters and entry weights.
- **Policy-owned (never searched):** allocator globals, exit-rule
  parameters, risk config, `min_cash_pct`, `forecast_scaling`. Set in the
  policy/strategies file by the operator; defaults unchanged.

This roughly triples search density at fixed budget, makes ensemble
membership well-defined (members differ only in forecasts, not in sizing
or exits), and removes the reviewer-identified failure modes of
per-member exits and per-member risk scaling before they can exist. The
optimizer keeps the ability to search globals behind an explicit
`search_globals: true` escape hatch for experiments.

## Component 1 — ensemble deployment (forecast averaging)

**What an ensemble is.** K entry-parameter sets. Selection: pool trials
per restart, take the top K/R distinct sets *from each restart*
(stratified — a single pooled top-K from TPE trials would be
near-duplicates from the winning basin, collapsing the ensemble into
`best` and poisoning the landing experiment).

**Blending rule — forecasts, not targets.** Each member produces blended
per-ticker entry scores (through quantile normalization). The ensemble
averages those scores; then **one** policy-level pass of softmax, caps,
risk overlay, and exit clamps runs on the blended scores. Consequences:

- **Stops keep their guarantee.** One exit layer with operator-owned
  parameters reads real portfolio state (cost basis, high-water marks),
  exactly as today. No graded 1/K exits; a stop is still a stop.
- **Risk overlay applies once, on the blend** — averaging K
  independently vol-targeted vectors would systematically under-risk
  whenever members disagree.
- `clamp_attribution`, order sizing, trade log, confirm alerts: all
  unchanged; they see one target vector.

**Identity property (load-bearing).** An ensemble of one is byte-identical
to today's engine given the same policy-owned knobs. Existing YAML loads
as ensemble-of-one; the golden scenario is untouched, **and a second
golden scenario with K=2 is added** so the blend path is pinned too.

**YAML form.**

```yaml
policy: { ... }
fitted_as_of: 2026-08-28
members:                      # entry params + weights only
  - strategies: [ {name: Momentum, weight: 3.0, params: {...}}, ... ]
  - ...
# policy-owned, shared:
softmax_temperature: 0.5
min_buy_delta: 0.02
max_position_pct: 0.40
min_cash_pct: 0.05
forecast_scaling: quantile
risk: { ... }
exits: [ {name: ChandelierStop, params: {...}}, ... ]
```

Unknown keys fail loudly — a behavior change: the current loader ignores
unknown top-level keys; this PR tightens it (single-user project, loud
config errors preferred).

## Component 2 — the fitter

`fit_as_of(price_data, as_of, policy, incumbent=None) -> FitResult`

- `policy.restarts` independent searches (seeds `base_seed + i`) on data
  strictly before `as_of`, each `policy.budget` trials, deterministic
  ask/tell.
- **Warm start without contaminating restarts:** the incumbent's members
  are enqueued into **restart 0 only**. Enqueuing them everywhere would
  seed every restart's TPE startup phase with the incumbent basin and
  destroy restart independence — the property the restarts exist for.
- **Selection:** `best` → highest training objective across all restarts;
  `ensemble` → stratified top-K/R per restart.
- `FitResult`: members, per-member scores, per-restart bests (search-noise
  record), policy hash.

**No training-score adoption gate.** The v1 draft gated adoption on the
challenger beating the incumbent's re-scored training objective by a
margin. Review killed it correctly: the max over thousands of fresh
trials virtually always clears that bar (winner's curse — the max is
inflated by noise, not filtered by it), so the gate passes everything.
Adoption is instead **degradation-triggered** (Component 5), which gates
on evidence the search was not selected on.

## Component 3 — validator and sweep

**Validator.** `walk_forward_optimize` re-based on the fitter, executing
the full policy per fold:

- Each fold calls `fit_as_of(fold_train_end, incumbent=deployed)`.
- **Adoption modeled:** the fold deploys the challenger only if the
  adopt trigger fired for the incumbent during the previous fold;
  otherwise the incumbent carries forward — exactly live's behavior.
- **Positions and lots carry across folds.** Today each fold's test
  window starts a fresh portfolio, so the turnover, slippage, and
  short-term-gains cost of *switching parameters* is invisible —
  precisely where re-fitting hurts. Carrying state across folds is a
  prerequisite for the coherence claim and is in the build order. If it
  proves too invasive, the disclosed fallback is a synthetic switching
  charge (implied trades × slippage + tax rate) per adoption — but
  carried state is the target.

**Baselines artifact** (`<strategies>.baselines.json`): per-fold OOS
returns (pre-tax; after-tax alongside when configured — labeled), fold
drawdowns measured within the fold window, aggregate OOS CAGR, policy
hash, validation range, per-restart training spreads.

**Sweep.** `midas sweep` compares policies along one axis (`--vary` /
`--value`), across one or more portfolios, with `--seeds` replication of
`base_seed`.

Statistics, revised per review:

- Aggregate OOS CAGR per variant with seed-replication spread, as before.
- **Uncertainty via stationary block bootstrap** of the concatenated OOS
  return path, not a cross-portfolio SE — two ETF baskets over one decade
  share regime and are nowhere near independent samples, and the output
  must not pretend otherwise. Multiple portfolios are still reported
  side-by-side (consistency across baskets is evidence), just not pooled
  into an SE.
- **Deflated Sharpe Ratio** computed from the recorded trial counts and
  score variance — it is a formula, not a project, and thousands of
  trials with max-selection is exactly its use case.
- The verdict line states what the bootstrap supports and qualifies
  single-portfolio results ("on these windows").
- Budget guard: the sweep refuses budgets that differ from the policy's
  own (`--force` to override) — validating at a different budget than
  deployment breaks coherence by construction.

**Landing experiments (run before merge, results in the PR):**
1. `deployment: best` vs `ensemble` vs a `spp`-style median-of-space
   mode if cheap — the un-measured hypothesis this PR exists to measure.
2. `refit`: adopt-on-degradation vs never-adopt (frozen parameters) —
   whether re-fitting adds value *at all* is itself unvalidated, and the
   machinery can now ask.

Compute honesty at the reference policy (measured 0.16s/trial on a
7-year window, worker pool of ~5; full-history trials run longer): one
fit ≈ 8,000 trials ≈ 5–10 min wall; a 16-fold validation ≈ 128k trials
≈ 1.5–2.5 h per variant per portfolio. The landing experiments are
overnight jobs, not coffee breaks; the sweep prints its cost estimate up
front.

## Component 4 — live monitor

Nightly lines in the close-of-day report when a baselines artifact
matches the running strategies file:

```
trailing fold-window: +4.1% ann. (pre-tax) — within validation range
drawdown (window peak): 8.2% vs validated worst-fold 15.1% — normal
```

- The trailing window and its drawdown anchor use the **fold-window
  peak**, matching how fold drawdowns were measured — not the all-time
  peak the CPPI state tracks (apples to apples, per review).
- "Within validation range" comes from the block-bootstrap distribution
  of fold-length returns, not a 16-fold percentile (1/16 granularity on
  autocorrelated folds over-reads position; the bootstrap smooths it).
- Breaching the validated worst-fold drawdown escalates to a warning.
  Notify-and-act: the engine informs; de-risking is the operator's call.
- Pre-tax basis stated on the line itself.

## Component 5 — scheduled re-fit with degradation-gated adoption

- At `refit_cadence` (the validator's fold length, derived from the same
  helper), the engine runs `fit_as_of(today)` overnight. The fit always
  runs (it is cheap and its result is informative); **adoption is
  proposed only when the adopt trigger has fired** — the incumbent's
  trailing fold-window below the configured bootstrap percentile, or a
  worst-fold drawdown breach. Quiet periods keep the incumbent: minimal
  fitting for slow signals is the default posture, and "parameters
  changed" is the exception that needs evidence, not the routine.
- A proposal posts to Discord — old vs new members summary, the
  degradation evidence that triggered it, and the immediate trades
  implied — confirmable with ✅ like an order. Parameters never change
  silently. The operator's veto is a coherence gap the validator cannot
  model; it is disclosed, not pretended away.
- Adoption writes the strategies file (`fitted_as_of`) and reloads it —
  **hot-reload is new machinery** (config loads once at startup today)
  and is costed as such in the build order.
- `--no-refit` disables; dry-run never refits. Enablement (not code) is
  gated on #54 landing, since adoption events are turnover events.

## Coherence gaps (disclosed)

1. The operator can decline a proposal; the validator assumes proposals
   are adopted. Directionally conservative (live churns less than
   validated), but real.
2. Live's latest fit trains on slightly more history than the last
   validated fold — the natural continuation of the anchored scheme.
3. Slippage in live is whatever fills the operator confirms; the
   validator uses the configured slippage model.

## Build order inside the single PR

1. **Step 0** — search-space split (entry-only search; policy-owned
   globals/exits); optimizer + loader + tests. Independently valuable.
2. **Ensemble engine support** — forecast averaging, K=1 identity test,
   K=2 golden scenario.
3. **Fitter** — restarts, restart-0 warm start, stratified selection,
   determinism tests.
4. **Validator on the fitter** — carried positions/lots across folds,
   adoption modeling, baselines artifact.
5. **Sweep** — variants, block bootstrap, DSR, cost preflight; pure
   aggregation unit-tested on synthetic data.
6. **Monitor** — report lines, fold-window anchors, threshold tests.
7. **Scheduled re-fit** — cadence, degradation trigger, Discord proposal
   flow, hot-reload, state persistence.

Landing experiments run after 5; their results go in the PR description
and decide nothing is defaulted-on without measurement.

## Out of scope

- Making `ensemble` (or `spp`) the default deployment — earned via the
  landing experiment.
- Drift-triggered *fitting* (the trigger currently gates adoption only).
- Daily re-fit (testable later as a policy variant; not asserted either
  way here).
- Multiple simultaneous sweep axes; fold-geometry cluster analysis;
  trade-order Monte Carlo (execution realism belongs with broker-sync).
- #54/#89/#90 (revisited after this epic; #54 gates Component 5's
  enablement only).

## Open questions

1. Ensemble member weighting: flat mean vs training-score-weighted.
   Flat drafted; the sweep can compare later.
2. `ensemble_size` default (20 drafted) and the right K for 6–10 ticker
   baskets — measurable, not decidable a priori.
3. Whether the SPP-style median-of-space mode is cheap enough to include
   in the landing experiment (it requires scoring the whole trial pool's
   deployments, not just top-K).
4. Carried-state validator: if implementation cost explodes, is the
   synthetic switching charge an acceptable interim, or does the
   coherence claim get withdrawn until carried state lands?
