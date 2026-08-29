# Policy engine: ensemble deployment, validated re-fit, live monitoring — design

**Date:** 2026-08-29 (v3.1 — open questions resolved)
**Status:** Approved
**Supersedes:** the 2026-08-23 comparison-harness draft (absorbed here)
**Delivery:** one PR, phased commits (decision: no sub-issues)
**Evidence:** `docs/experiments/2026-08-24-seed-budget-grid/` (lab notes +
verified-regenerable results), the multi-seed table on issue #49,
`docs/cli.md` "Trial budget and search noise", commit `ee0c93e`.

## Problem

Two incoherences, one conceptual and one statistical.

**Conceptual.** Walk-forward validation does not validate parameters; it
validates a *procedure* — "at each cadence boundary, re-fit on all history
to date with a given budget and objective, then trade the result." The OOS
numbers it produces measure that procedure. But live runs frozen
parameters indefinitely: optimize-once-and-hold is a policy nobody
validated.

**Statistical.** Every number the optimizer prints is a point estimate
with no error bar, and the measured error bars exceed the effects read
from them. Four conclusions from single runs during the objective work
were retracted as sampler noise — specifically: "calmar never loses badly
to buy-and-hold", "sharpe guts right-skewed strategies", "these strategies
lose to buy-and-hold", and "the inflow-adjustment fix flipped the
ranking" (history on issue #49). What is recorded (sources above;
per-claim limitations in the lab notes):

- The objective landscape is rugged and multi-modal at the current search
  dimensionality; TPE settles into a basin per seed (1324% vs 2077%
  training at 666 trials on one fold, diverging further at 2,000; one
  seed stalled terminally).
- Aggregate OOS CAGR is much more seed-stable than its folds — spread
  7.6 → 2.9 pts (30 → 100 trials/fold, etf) while per-fold spread stayed
  ~27–30 pts. Part of that stability is arithmetic (an average over 16
  folds must be steadier than its terms); the budget-dependence of the
  aggregate spread is the evidential part. The test basket's seed spread
  was ~27 pts at 30 trials/fold, with no higher-budget counterpart
  recorded.
- Paired per-fold comparison has no discriminating power here (null
  control in the lab notes).
- **Directional hints, not results** (small n, measured on `gross` while
  the reference policy is `calmar`): best-of-seeds selection looked
  ~+2.7 pts OOS; the train→OOS rank correlation (+0.25, SE ≈ 0.2) is not
  distinguishable from zero. The landing experiments decide.
- The noise is a property of a rugged ~47-dimensional objective under
  10³–10⁴ evaluations, not of TPE per se — hence Step 0.

**Practice context.** Established products converge on walk-forward plus
robustness batteries. For deployment, two precedents — neither located
*measured* against single-best in the literature we found, so both are
hypotheses to test here: **forecast combination** (Carver /
systematic-CTA practice: combine rule variations before sizing) and
**System Parameter Permutation** (Walton: no selection at all — evaluate
the space, take its median).

## The unifying object: the deployment policy

```yaml
policy:
  objective: calmar          # any of the six
  budget: 2000               # trials per restart
  restarts: 4                # independent seeded searches per fit
  base_seed: 42
  deployment: ensemble       # ensemble | best
  ensemble_size: 20          # stratified top-K across restarts
  cadence_days: 63           # trading days: BOTH the validator's fold length
                             # and live's re-fit interval (single source)
  adopt_trigger:
    oos_percentile_below: 10 # trailing fold-window under this bootstrap pctile
    drawdown_breach: true    # trailing DD beyond the validated worst fold
```

- **Cadence flows from the policy to the folds, not from the date range.**
  Today fold length is derived from the range (`remaining // n_folds`), so
  the validated procedure silently changes with `--start`/`--end`. The
  validator now carves test windows of `cadence_days` and derives the fold
  count; live re-fits on the same constant.
- **Seeding:** the search seed for (fold *f*, restart *r*) is
  `hash(base_seed, f, r)` — the current `seed + fold` and `base_seed + i`
  schemes collide across folds/restarts and would correlate the recorded
  noise floors.

**The coherence principle (hard constraint).** One function executes a
fit; the validator calls it per fold — including the adoption rule and a
carried portfolio — and live's re-fit calls it going forward. Known gaps
are listed in "Coherence gaps" and either closed or disclosed.

## Step 0 — shrink the search space

The optimizer currently searches ~47 dimensions. New split, enforced by
the fitter:

- **Searched (per member):** entry-signal parameters and entry weights.
- **Policy-owned (never searched):** allocator globals
  (`softmax_temperature`, `min_buy_delta`, `max_position_pct`), exit-rule
  parameters, risk config, `min_cash_pct`, `forecast_scaling`.

This roughly triples search density at fixed budget, makes ensemble
membership well-defined (members differ only in forecasts), and prevents
per-member exits/risk/sizing by construction — there is exactly one
`OrderSizer` pass and its knobs have exactly one source.

**Consequence for #54:** the no-trade band must be policy-level here, not
a per-member searched knob as that issue drafted. Searchability survives
at the right altitude: band values are compared as *policy variants* via
the sweep, not sampled per-member by TPE. #54 gets a comment when this
PR lands.

Escape hatch for experiments: `search_globals: true` restores the old
space, incompatible with `deployment: ensemble`.

## Component 1 — ensemble deployment (forecast averaging)

**Membership.** K entry-parameter sets (default `ensemble_size: 20` —
a weak prior; a K ∈ {4, 12, 20, 40} sweep is queued as post-landing
follow-up), stratified top-K/R per restart. **Dedupe is behavioral, not
textual:** candidate members are deduplicated by the hash of their
training-window target series **rounded to 4 decimals** (decided: exact
float hashing would treat 1e-12 jitter as distinct members), since two
tuples differing only in a never-binding parameter are the same member
and tuple-dedupe would double-weight them. Selection walks down each
restart's ranking until K/R behaviorally distinct members are found; a
test pins that near-identical members collapse.

**Blending.** Each member produces blended per-ticker entry scores
(through quantile normalization); the ensemble averages the scores —
**flat mean** (decided: score-weighting would re-trust the in-sample
score differences the ensemble exists to distrust, and top-K scores are
compressed together anyway; a sweep can revisit); then **one** policy-level pass of softmax, caps, risk
overlay, and exit clamps runs on the blend. Stops keep their max-loss
guarantee; risk targets are hit exactly once; `clamp_attribution`,
sizing, trade log, and alerts are unchanged.

**Turnover is a measured risk, not an assumed benefit.** A mean of K
drifting score vectors has fewer flat stretches than one member, and the
sizer's band is buy-side only — so ensembles may *raise* small-trade
frequency. Turnover and after-tax return are first-class metrics of the
landing experiment, not a footnote.

**Identity property.** An ensemble of one is byte-identical to today's
engine given the same policy-owned knobs. Existing YAML loads as
ensemble-of-one; the golden scenario is untouched; a second K=2 golden
scenario pins the blend path.

## Deployment artifacts — who writes what

The strategies YAML stays **human-owned** (policy block, policy-owned
knobs, comments, ordering). Fitted members live in a **sidecar**,
`<strategies>.members.yaml`, exactly like the live-state sidecar:

```
strategies.yaml            # human-owned: policy + policy-owned knobs
strategies.members.yaml    # machine-owned: current members, fitted_as_of,
                           # input hash, per-restart bests
strategies.fits/           # last N FitResults (history + rollback)
strategies.baselines.json  # validator output for the monitor
```

- **`midas fit`** (evolves `optimize`): runs the fitter at `--as-of`
  (default today) and writes the members sidecar. The *initial*
  deployment is this command run by the operator — no incumbent, no
  trigger, deployed directly. `optimize` remains as the low-level
  single-search tool; `fit` is the policy-level entry point.
- **`midas refit --rollback [n]`** restores a previous FitResult from
  `strategies.fits/`. The monitor's "worse than everything validated"
  moment is exactly when rollback is wanted; without history it cannot
  exist.
- `deployment: best` still records the other restarts' bests in the
  sidecar — the deployed file alone cannot serve later hysteresis or
  noise reporting.

## Component 2 — the fitter

`fit_as_of(portfolio, price_data, as_of, policy, incumbent=None) -> FitResult`

(one signature; it needs the portfolio — tickers, tax, risk — not just
prices.)

- `policy.restarts` searches, seeds `hash(base_seed, fold, r)`, data
  strictly before `as_of`, deterministic ask/tell.
- Incumbent members enqueued into restart 0 only (restart independence).
- Selection: `best` (highest training objective overall) or stratified
  behavioral-deduped top-K/R.
- `FitResult`: members, per-member scores, per-restart bests, `as_of`,
  policy hash, **input hash** (see Monitor), engine version.

No training-score adoption gate (winner's curse — the max of thousands of
fresh trials always clears it). Adoption is degradation-triggered
(Component 5); the validator models the same rule.

## Component 3 — validator and sweep

**Validator.** Folds of `cadence_days`; each fold calls `fit_as_of`
(incumbent = currently deployed members) and deploys the challenger only
if the adopt trigger fired during the previous fold.

- **The portfolio carries across folds**: positions, lots, cost bases,
  high-water marks, cash. Fresh-start folds are ruled out — they ramp in
  from cash, can't fire exits on prior gains, and understate drawdown, so
  a baseline built from them is systematically mild versus the
  fully-invested continuing book live actually runs. (This also charges
  parameter switches their real turnover and tax cost.) The synthetic
  switching-charge fallback from v2 is demoted to a rejected alternative:
  it fixes switching cost but not the ramp-in bias.
- **Path dependence, disclosed:** with adoption modeled, every fold's
  deployment depends on all earlier adoption decisions. Fold 0 has no
  incumbent and always deploys. Baselines from different validation
  ranges are therefore not comparable (the input hash enforces this), and
  fold-level parallelism is off the table — the chain is sequential by
  design, as live is.

**Baselines artifact** (`<strategies>.baselines.json`): per-fold OOS
returns *and per-day fold return paths* (pre-tax; after-tax alongside
when configured, labeled), fold-window drawdowns, aggregate OOS CAGR,
the full **input hash** (policy + risk + constraints + tax config +
tickers + validation range + price-cache identity), per-seed OOS spread
when `--seeds` was used.

**Sweep.** `midas sweep` compares policy variants along one axis across
one or more portfolios, with `--seeds` replication.

- Uncertainty via stationary block bootstrap of the OOS return path;
  Deflated Sharpe Ratio from recorded trial counts; portfolios reported
  side-by-side, never pooled into an SE (same decade ≠ independent
  samples); single-portfolio verdicts qualified.
- **Noise floor reported in OOS units** — the spread of aggregate OOS
  CAGR across `--seeds` — never in objective units, which don't compare
  across variants.
- Budget guard: refuses budgets differing from the policy's own
  (`--force` to override).
- Cost preflight printed before running.

**Landing experiments — criteria pre-registered here, before running:**

1. `deployment: best` vs `ensemble` vs a **sampled SPP approximation**
   (median deployment of a random sample of the trial pool — full SPP
   requires evaluating the whole pool and is dropped; the sample rides
   only if it fits the overnight budget). **Ensemble becomes the default only if** its block-
   bootstrap OOS CAGR interval is at least as good as `best`'s on both
   baskets **and** it does not increase turnover-driven after-tax drag by
   more than 1 pt on either. Equal-return-lower-drag also adopts. Any
   other outcome: `best` stays default, results recorded.
2. Re-fit policy: degradation-gated adoption vs never-adopt (frozen).
   **Scheduled re-fit ships enabled only if** adoption does not lose to
   frozen on after-tax OOS CAGR on both baskets; otherwise Component 5
   ships behind a flag, default off, with the numbers in the PR.

Compute honesty (measured 0.16 s/trial, 7-year window, ~5-worker pool;
full-history trials run longer): one fit ≈ 8,000 trials ≈ 5–10 min wall;
a 16-fold validation ≈ 1.5–2.5 h per variant per portfolio; the landing
experiments are overnight jobs.

## Component 4 — live monitor

Nightly lines in the close-of-day report when the baselines artifact's
**input hash matches the running configuration** — on mismatch the
monitor prints one explicit "baselines stale (input mismatch)" line and
refuses percentages rather than comparing against a different engine.

```
fold-window day 23/63: +2.8% TWR (pre-tax) — 41st pctile of fold paths at day 23
drawdown (window peak): 8.2% vs validated worst-fold 15.1% — normal
```

- **Same-day-offset comparison, no annualization of partial windows:**
  the trailing window's cumulative TWR is ranked against the stored
  per-day fold paths *at the same day offset*. Nothing is annualized
  until the window is complete.
- **TWR, not equity-over-anchor** — cash infusions must not read as
  return; the engine's inflow-adjusted machinery already exists.
- The window anchors on the **fit date regardless of adoption outcome**
  (a declined proposal still closed a fold-analog window).
- Drawdown uses the fold-window peak, matching how fold drawdowns were
  measured — not the all-time CPPI peak.
- **Warnings are edge-triggered** with a persisted flag (like
  `last_report_date`): a breach alerts once, recovery resets it; no
  nightly repeats.
- Notify-and-act: the engine informs; de-risking is the operator's call.

## Component 5 — scheduled re-fit, degradation-gated adoption

- **The fit runs as its own command, `midas refit`, never inside the
  live loop.** Live is single-threaded, holds the state lock, and must
  keep polling Discord reactions; a 5–20 minute in-process fit would
  stall confirmations and couple a fit crash to the trading loop. The
  loop's role: detect that `cadence_days` have elapsed, spawn
  `midas refit` as a subprocess during the overnight window, and read
  its FitResult from the sidecar next tick. `refit` is thereby a
  standalone, testable unit and independently croneable.
- The fit always runs at cadence (cheap, informative, recorded to
  `strategies.fits/`). **Adoption is proposed only when the adopt
  trigger fired.** Quiet periods keep the incumbent — minimal fitting is
  the default posture; "parameters changed" is the exception that needs
  evidence.
- A proposal posts to Discord: old vs new member summary, the
  degradation evidence, and the trades implied **at proposal-time
  prices, marked as estimates** — a parameter change is bigger than one
  order and its preview goes staler faster; the text says so. Proposals
  **expire after `pending_ttl_hours`** like orders; an expired or
  declined proposal is re-raised only after the next fit at cadence.
- **Confirmation machinery is a real build, sized as such:** the
  `OrderConfirmer` protocol is order-shaped, so this adds a generalized
  pending-proposal type (persisted in `LiveState`, TTL, decision
  polling) beside pending orders — the largest single piece of the final
  phase, together with reload-on-adoption (config currently loads once
  at startup; member reload is new machinery; policy-owned knobs still
  require a restart, members do not).
- Adoption writes the members sidecar (never the human YAML), records
  the outgoing members to `strategies.fits/`, and reloads.
- `--no-refit` disables; dry-run never refits. Enablement gated on the
  landing experiment 2 criterion and on #54.

## Coherence gaps (disclosed)

1. The operator can decline or let a proposal expire; the validator
   assumes trigger ⇒ adoption. Directionally conservative, but real.
2. Live's latest fit sees slightly more history than the last validated
   fold — the natural continuation of the anchored scheme.
3. Live fills at operator-confirmed prices; the validator uses the
   configured slippage model.

## Build order inside the single PR

1. **Step 0** — search-space split; optimizer + loader (+ loud unknown
   keys) + tests.
2. **Ensemble engine** — forecast averaging, behavioral dedupe helper,
   K=1 identity test, K=2 golden scenario.
3. **Fitter + artifacts** — `fit_as_of`, per-(fold,restart) seeding,
   restart-0 warm start, sidecar/fits-history writers, `midas fit`,
   rollback; determinism tests.
4. **Validator** — cadence-driven folds, carried portfolio, adoption
   modeling, baselines with per-day paths + input hash.
5. **Sweep** — variants, block bootstrap, DSR, OOS-unit noise floor,
   cost preflight.
6. **Monitor** — day-offset comparison, TWR windows, edge-triggered
   warnings, input-hash gate.
7. **Re-fit** — `midas refit`, live-loop spawning, generalized proposal
   confirmation, reload-on-adoption.

Landing experiments run after 5; their pre-registered criteria decide
the defaults; results go in the PR description.

## Out of scope

Unchanged from v2: default flips without measurement, drift-triggered
*fitting*, daily re-fit, multi-axis sweeps, cluster analysis, trade-order
Monte Carlo, #54/#89/#90 themselves (#54's altitude change gets a
comment; its build stays out).

## Resolved decisions (2026-08-29)

1. Member weighting: **flat mean** (rationale inline in Component 1).
2. `ensemble_size`: **20**, held as a weak prior; K sweep queued
   post-landing.
3. SPP in landing experiment 1: **sampled approximation only**, and only
   if it fits the overnight budget; full SPP dropped.
4. Behavioral dedupe: **target series rounded to 4 decimals** before
   hashing, with a fallback to tolerance comparison if under-dedupe
   shows up in practice; collapse of near-identical members is pinned by
   test.
