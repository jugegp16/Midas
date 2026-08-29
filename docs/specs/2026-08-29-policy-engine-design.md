# Policy engine: ensemble deployment, validated re-fit, live monitoring — design

**Date:** 2026-08-29
**Status:** Draft — for review
**Supersedes:** the 2026-08-23 comparison-harness draft (absorbed here)
**Delivery:** one PR, phased commits (decision: no sub-issues)

## Problem

Two incoherences, one conceptual and one statistical.

**Conceptual.** Walk-forward validation does not validate parameters; it
validates a *procedure* — "every ~63 trading days, re-optimize on all
history to date with budget B and objective O, then trade the result." The
OOS numbers it produces measure that procedure. But live runs frozen
parameters indefinitely: optimize-once-and-hold is a policy nobody
validated, and the walk-forward evidence stops describing the live engine
the day after the next fold would have re-fit.

**Statistical.** Every number the optimizer prints is a point estimate
with no error bar, and the measured error bars are larger than the effects
being read from them. Four conclusions drawn from single runs during the
objective-framework work were later shown to be sampler noise. The
measured landscape (all figures from this repo's own experiments,
2026-08-23/24):

- **Rugged and multi-modal.** TPE settles into a basin per seed and stays:
  same fold, seeds reached training scores of 1324% vs 2077% at 666
  trials, diverging further (1457% vs 2862%) at 2,000.
- **Seed beats budget** in 3 of 4 folds tested: the best seed at 666
  trials outran the worst seed at 2,000.
- **Restart selection helps out of sample**: deploying the best-of-3-seeds
  by training score averaged +2.7 pts OOS over an average seed
  (rank correlation train→OOS +0.25 across 24 searches).
- **The aggregate converges; folds never do.** Aggregate OOS CAGR spread
  across seeds: 7.6 pts at 30 trials/fold → 2.9 at 100. Per-fold spread:
  29.6 → 26.9 (0 of 16 folds agree within 1 pt at either budget).
- **Paired per-fold comparison has no power**: a variant compared against
  *itself* across seeds produces the same paired-fold spread (sd 17.6,
  "wins" 6–10/16) as a real variant comparison.
- **Search noise is an artifact of using TPE.** Grid-search products don't
  have it; we bought sampler efficiency and pay in a noise source that
  must be managed explicitly.

**The market context** (2026-08-26 survey): established products converge
on walk-forward + robustness batteries (TradeStation cluster analysis,
StrategyQuant Monte Carlo suites, System Parameter Permutation). The
relevant literature result: averaging the *positions* of the top-K
parameter sets, rather than deploying the single best, reduces train→test
degradation — high performers agree on direction and disagree on
threshold placement, so averaging keeps the consensus and cancels the
noise (arXiv 2602.10785; SPP philosophy).

## The unifying object: the deployment policy

The thing built, validated, deployed, and monitored is a **policy**:

```yaml
policy:
  objective: calmar        # any of the six
  budget: 2000             # trials per fit
  restarts: 4              # independent seeded searches per fit
  base_seed: 42            # restart seeds derived: base_seed + i
  deployment: ensemble     # ensemble | best
  ensemble_size: 20        # top-K distinct parameter sets across restarts
  refit_cadence_days: 63   # trading days between live re-fits (fold length)
  adopt_margin: 0.05       # hysteresis: challenger must beat incumbent by 5%
```

**The coherence principle (hard constraint):** one function,
`fit_as_of(price_data, date, policy)`, executes a fit. The walk-forward
validator calls it per fold; live's scheduled re-fit calls it going
forward. Same budget, same restarts, same selection, same code path. This
is what makes the validator's OOS distribution a *prediction about live*
rather than a description of a cousin procedure.

## Component 1 — ensemble deployment in the engine

**What an ensemble is.** K complete parameter sets (each: per-strategy
params + weights + global allocation knobs), produced by pooling all
trials across the fit's restarts, deduplicating by parameter tuple, and
taking the top K by training objective. Ensemble costs no extra search —
the trials already exist.

**Blending rule.** Each member runs the full existing pipeline —
scoring, quantile normalization, blending, risk overlay, exit clamps —
producing its own target-weight vector per bar. Final targets are the
arithmetic mean across members. One `OrderSizer` pass on the blended
targets; everything downstream (trades, confirm alerts, tax accounting,
trade log) is unchanged and sees a single target vector.

**Exits become graded, deliberately.** A member whose stop fires zeroes
its own target, dropping the blend by 1/K. Positions de-risk in
twentieths as evidence accumulates across members instead of
all-or-nothing at one threshold. This composes with the future no-trade
band (#54), which absorbs small blend drifts.

**The identity property (load-bearing for testing and compatibility).**
An ensemble of one is byte-identical to today's engine. Existing
strategies YAML files load as ensembles of one; the golden test is
untouched; `deployment: best` produces a one-member file.

**YAML form.** A strategies file is either the existing single-set form
(loaded as ensemble-of-one) or:

```yaml
policy: { ... }            # the policy that produced this file
fitted_as_of: 2026-08-28
members:
  - softmax_temperature: 0.4
    min_buy_delta: 0.04
    max_position_pct: 0.42
    strategies: [ {name: Momentum, weight: 3.0, params: {...}}, ... ]
  - ...
min_cash_pct: 0.05         # user preference, shared across members
forecast_scaling: quantile # policy-level, shared
risk: { ... }              # policy-level, shared
```

Per-member knobs are those the optimizer searches; policy-level knobs are
user preferences the optimizer never touches. Unknown keys and malformed
members fail loudly (single-user project — no migration shims).

**Cost.** K× the allocator's per-bar scoring and K× signal precompute.
At current trial cost (~0.16s for a 7-year window), a K=20 backtest is
~3s and a live tick is trivial.

## Component 2 — the fitter

`fit_as_of(price_data, as_of, policy, incumbent=None) -> FitResult`

- Runs `policy.restarts` independent searches (seeds `base_seed + i`) on
  data strictly before `as_of`, each at `policy.budget` trials, using the
  existing deterministic ask/tell loop.
- **Warm start:** when an incumbent exists, its members' parameters are
  enqueued as the first trials of every restart — the mechanism the
  walk-forward already uses across folds (`prev_best_flat`), extended to
  ensembles.
- **Selection:** `best` → the single highest-training-objective trial
  across all restarts (the validated +2.7 pt rule). `ensemble` → top-K
  distinct trials across all restarts.
- **Hysteresis:** the incumbent's members are re-scored on current data
  (scores go stale as data accrues); the challenger is adopted only if
  its mean member training objective exceeds the incumbent's re-scored
  mean by `adopt_margin`. Because incumbent members are enqueued, the
  challenger is ≥ incumbent by construction; the margin acts on the
  improvement, killing noise-level swaps. **The validator applies the
  same rule across folds** — fold N's deployment is the outcome of fold
  N's adoption decision against fold N−1's deployment, exactly as live
  will behave — otherwise the OOS distribution describes a procedure
  live does not run. (`adopt_margin: 0` recovers always-adopt.)
- `FitResult` carries members, per-member training scores, per-restart
  bests (for the sweep's noise reporting), and the policy hash.

## Component 3 — validator and sweep

**Validator.** `walk_forward_optimize` is re-based on the fitter: each
fold calls `fit_as_of(fold_train_end)` and evaluates the resulting
deployment (ensemble or best) on the fold's test window. This replaces
the current one-seed-per-fold search and makes the validator execute the
policy exactly as live will.

Outputs, alongside the existing report: a **baselines artifact**
(`<strategies>.baselines.json`) — per-fold OOS returns (raw and
annualized, pre- and after-tax when configured), aggregate OOS CAGR,
worst-fold drawdown, fold length, policy hash, validation range, and the
per-restart training-score spread per fold (the recorded search-noise
floor).

**Sweep.** `midas sweep` compares policies:

```bash
midas sweep -p etf.yaml [-p test.yaml ...] -s candidates/ --start ... --end ... \
  --vary deployment --value best --value ensemble
```

- Variants along one axis (`--vary`: any policy field, or whole strategy
  files); each variant is validated by the walk-forward above on every
  portfolio given; `-p` is repeatable and pooling portfolios is the only
  defence against period risk.
- Reporting: aggregate OOS CAGR per variant with mean ± SE across
  portfolios (and across `--seeds` replications of `base_seed` when
  requested), the noise floor, and an explicit verdict — a difference
  under 2 SE prints as *not resolvable*, and a single-portfolio verdict
  prints qualified ("on these windows").
- Budget guard: refuses under 100 trials/fold without `--force`.
- The selected variant's baselines artifact is written for the monitor.
- First experiment on landing: `best` vs `ensemble` on both baskets —
  the literature's degradation claim, measured here before ensemble can
  become a default.

## Component 4 — live monitor

Nightly, inside the existing close-of-day report (terminal + Discord),
when a baselines artifact exists for the running strategies file:

```
quarter-to-date: +4.1% ann. — 38th percentile of validation folds (normal)
drawdown: 8.2% vs worst validation fold 15.1% (normal)
```

- "Quarter" = trailing `refit_cadence_days` trading days, aligned to the
  last re-fit — i.e., the live analog of a fold.
- Percentile against the per-fold OOS distribution; drawdown against the
  worst validation-fold drawdown. Breaching the worst-ever fold drawdown
  escalates the line to a warning (notify-and-act: the engine informs,
  de-risking stays the operator's call).
- Statistical honesty is the point in both directions: fold returns
  ranged −22% to +57% annualized in validation, so a bad quarter prints
  as *normal* when it is, and a genuine breach prints as the red line it
  is.
- State: `last_report_equity`-style anchors extended with the re-fit
  epoch date; baselines are read-only input.

## Component 5 — scheduled re-fit

- After the close report, when `refit_cadence_days` have elapsed since
  the last fit (dates persisted in live state), the engine runs
  `fit_as_of(today)` in the overnight window. Cost at the reference
  policy: 4 restarts × 2,000 trials ≈ 5–10 minutes at current speed —
  comfortably inside the overnight window the market-hours gating
  already provides.
- Hysteresis applies. If the challenger is not adopted: a one-line note
  in the next report. If adopted: a **parameter-change proposal** to
  Discord — old vs new summary, mean training-objective delta, and the
  immediate trades the new targets would imply — confirmable with ✅
  exactly like an order (consistent with confirm-mode: parameters never
  change silently under the operator). On confirmation, the new members
  are written to the strategies file (with `fitted_as_of`) and
  hot-reloaded; declining keeps the incumbent and re-proposes only after
  the next fit.
- `--no-refit` disables; dry-run never refits.
- **Sequencing note:** this component is gated on the no-trade band
  (#54) landing, since re-fit amplifies turnover; ensemble continuity
  reduces but does not eliminate that concern.

## Build order inside the single PR

Phased commits, each green and each measurable before the next:

1. **Ensemble engine support** — loader, allocator blending, graded
   exits; identity test (ensemble-of-one ≡ current engine, byte-for-byte
   on the golden scenario).
2. **Fitter** — `fit_as_of`, restarts, warm start, selection rules,
   hysteresis; determinism tests.
3. **Validator on the fitter + baselines artifact** — walk-forward folds
   call `fit_as_of`; existing walk-forward tests adapted; baselines
   round-trip tests.
4. **Sweep command** — variant comparison, SE/verdict reporting, CSV;
   aggregation is pure and unit-tested on synthetic results.
5. **Monitor** — report lines from baselines; formatting and
   normal/warning threshold tests.
6. **Scheduled re-fit** — cadence, hysteresis, Discord proposal flow with
   `FakeConfirmer`-style tests; state persistence.

Then the landing experiment (best vs ensemble, both baskets) runs before
merge and its result goes in the PR description.

## Out of scope

- Changing the default deployment to `ensemble` (earned via the sweep, a
  one-line follow-up if the data supports it).
- Drift-*triggered* re-fit (the monitor's percentile line is the future
  trigger input; scheduled-only in this PR).
- Daily re-fit (testable later as a policy variant: walk-forward with
  1-day test windows; prior expectation is that tax drag and search noise
  swamp the 0.04%/day information gain).
- Multiple simultaneous sweep axes; formal significance tests beyond SE;
  DSR/PBO; fold-geometry cluster analysis; trade-order Monte Carlo
  (execution realism belongs with the broker-sync work).
- #54/#89/#90 themselves (unchanged scope, revisited after this epic;
  #54 gates component 5's *enablement*, not its code).

## Open questions

1. Ensemble member weighting: flat mean (drafted, literature-backed) vs
   training-score-weighted mean. Flat unless the sweep says otherwise.
2. Should the monitor's warning also fire on percentile (e.g. trailing
   quarter below every validation fold), or drawdown-only at first?
3. `ensemble_size` default: 20 follows the literature; the right K for
   6-10 ticker portfolios is measurable by sweep later.
