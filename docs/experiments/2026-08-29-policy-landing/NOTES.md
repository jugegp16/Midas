# Policy landing experiments

Pre-registered criteria (spec: docs/specs/2026-08-29-policy-engine-design.md,
"Landing experiments"), recorded here before results existed.

## Experiment 1 — hold vs re-fit (the premise test)

**Criterion (pre-registered):** scheduled re-fit ships enabled only if
adoption does not lose to frozen on after-tax OOS CAGR on both baskets.

**Setup**

- Variants: `adopt_trigger: none` (frozen — fold 0 fits once, later folds
  hold) vs `default` (10th-percentile fold-midrank arm + drawdown-breach
  arm, degradation-gated adoption).
- Policy under test (both variants, both baskets): `objective: sharpe`,
  `budget: 1000`, `restarts: 4`, `base_seed: 42`, `deployment: best`,
  `cadence_days: 63`. `deployment: best` isolates the re-fit question;
  ensemble is Experiment 2. Budget was lowered from 2000 to 1000 mid-run
  on 2026-08-29 after watching training bests plateau well before 1000
  trials — before any OOS comparison existed; the run restarted from
  scratch at the new budget.
- Baskets: `sample-portfolios/etf.yaml` (10 ETFs, 35.8/18.8 tax rates)
  and `sample-portfolios/stonks.yaml` (11 stocks, 37/20 tax rates), each
  with its own optimized globals/exits (copied from the deployed
  `*-optimized-strategy.yaml` files into `exp1-*-strategies.yaml`).
- Range: 2016-01-04 → 2026-08-27 with the default 730-day holdout
  reserved (sweep trims to 2024-08-27). ~14 folds of cadence 63 after a
  60% initial train reserve.
- Seeds: 2 replications per (variant, basket) — chosen for wall-clock
  (two full-budget replications ≈ 14h with both baskets in parallel);
  the sweep's seed spread plus the 2026-08-24 seed-budget grid provide
  the noise floor. Deviation from the sweep's default of 3 seeds,
  recorded here.
- Late-listed tickers (ETH/IBIT list 2024, QQQM 2020-10, CEG 2022): the
  fitter searches only tickers listed before each fold's as_of; the OOS
  engine holds them at cost basis until listed. Fixed in-tree the day
  the experiment launched (commit "Fitter searches only tickers listed
  before as_of").

**Commands**

```
midas sweep -p sample-portfolios/etf.yaml -s exp1-etf-strategies.yaml \
  --start 2016-01-04 --end 2026-08-27 \
  --vary adopt_trigger --value none --value default --seeds 2
midas sweep -p sample-portfolios/stonks.yaml -s exp1-stonks-strategies.yaml \
  --start 2016-01-04 --end 2026-08-27 \
  --vary adopt_trigger --value none --value default --seeds 2
```

Logs: `exp1-etf.log`, `exp1-stonks.log` (committed with results).

**Results** (2026-08-30; logs `exp1-etf.log`, `exp1-stonks.log`;
per-cell data `exp1-*-strategies.sweep-cells.jsonl`)

etf (after-tax fold mean is the criterion column):

| variant | OOS CAGR (mean of 2 seeds) | seed spread | after-tax fold mean | per-seed after-tax |
|---|---|---|---|---|
| re-fit (default) | +11.56% | 2.67% | +2.50% | +2.72% / +2.28% |
| frozen (none)    |  +8.97% | 1.03% | +2.00% | +1.86% / +2.13% |

Sweep verdict line: variant spread 2.59% vs seed noise 2.67% — **not
resolvable at this budget** (on these windows).

stonks:

| variant | OOS CAGR (mean of 2 seeds) | seed spread | after-tax fold mean | per-seed after-tax |
|---|---|---|---|---|
| re-fit (default) | +21.45% | 3.29% | +3.91% | +3.31% / +4.51% |
| frozen (none)    | +25.66% | 2.38% | +4.78% | +4.93% / +4.64% |

Sweep verdict line: variant spread 4.21% **exceeds** seed noise 3.29%.

**Pre-registered judgment:** re-fit ships enabled only if adoption does
not lose to frozen on after-tax OOS on both baskets. Re-fit **loses on
stonks in both seeds** (frozen ahead by ~0.9 pt after-tax per fold,
with the CAGR gap beyond the seed-noise floor), and its etf edge is not
resolvable from noise. **Criterion not met → `--refit` stays opt-in and
off by default.** No polarity flip.

Observations recorded for future work (not acted on here):

- The default trigger adopted in 6–8 of 13 folds on every re-fit cell —
  far more often than "degradation-gated" suggests. The drawdown arm
  arms after a single prior fold, so early folds breach the prior-worst
  easily; trigger sensitivity is the first suspect for the stonks loss
  (churn + short-term tax drag without OOS benefit).
- Run mechanics: both original two-at-once sweeps were killed by
  system-wide memory pressure (no leak — sweep tree RSS flat ~1.2 GB);
  results above come from sequential re-runs with per-cell JSONL
  checkpointing added in response.

## Experiment 2 — best vs ensemble vs sampled SPP

Pending Experiment 1. Criterion (pre-registered): ensemble becomes the
default only if its block-bootstrap OOS interval is at least as good as
best's on both baskets AND after-tax drag rises ≤ 1 pt.
