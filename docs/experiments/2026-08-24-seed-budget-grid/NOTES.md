# Seed × budget experiments — lab notes (2026-08-23/24)

Provenance: these runs were executed during the objective-framework work
(PR #91) with explicit seeds on the deterministic ask/tell search, so every
number is re-derivable. The original raw logs were lost to tmp cleanup;
the tables below are transcribed from the working session, and
`fold_grid.py` in this directory regenerates `results.json` for the grid.
Related recorded evidence: the 30-run multi-seed study table lives in the
correction comment on issue #49; the trial-budget table is in
`docs/cli.md` ("Trial budget and search noise") and commit `ee0c93e`.

Configuration for everything below: `sample-portfolios/test.yaml` +
`example-strategies/balanced-growth.yaml`, **objective `gross`**,
2016-04-20 → 2026-04-20, anchored walk-forward folds.

## 1. Seed × budget grid (folds 1, 5, 10, 15 × seeds 42/7/123 × 666/2000 trials)

| fold | seed | n=666 train | OOS | n=2000 train | OOS |
|---|---|---|---|---|---|
| 1 | 42 | 2077.27% | −7.78% | 2862.45% | +7.22% |
| 1 | 7 | 1323.92% | −9.29% | 1456.52% | −14.91% |
| 1 | 123 | 1330.64% | −4.15% | 1619.24% | +8.53% |
| 5 | 42 | 1874.25% | 33.03% | 2062.86% | 30.27% |
| 5 | 7 | 1981.84% | 32.25% | 2954.75% | 31.97% |
| 5 | 123 | 2421.41% | 33.24% | 2565.06% | 31.87% |
| 10 | 42 | 3812.64% | 42.47% | 4919.14% | 40.40% |
| 10 | 7 | 4151.35% | 36.34% | 4852.03% | 39.22% |
| 10 | 123 | 4485.22% | 56.71% | 7104.86% | 53.90% |
| 15 | 42 | 12859.96% | −21.86% | 12859.96% | −21.86% |
| 15 | 7 | 8433.76% | −16.97% | 11235.03% | −19.22% |
| 15 | 123 | 7335.99% | −16.94% | 10685.32% | −14.72% |

Derived: best-seed@666 beat worst-seed@2000 in 3/4 folds; Spearman
rank correlation of training score vs OOS across the 24 (seed, budget,
fold) searches: +0.25; OOS of the best-training pick averaged 16.44% vs
13.74% for a mean seed (+2.70 pts). Fold 15 / seed 42 returned identical
results at both budgets (a terminal search stall).

## 2. Budget escalation on one stalled seed (fold 1 training window)

| seed | trials | best train |
|---|---|---|
| 7 | 666 | 1323.92% |
| 7 | 2000 | 1456.52% |
| 7 | 4000 | 1476.53% |
| 42 | 666 | 2077.27% |
| 42 | 2000 | 2862.45% |

## 3. Per-fold convergence across seeds (etf.yaml + balanced-growth, calmar)

| statistic | 30 trials/fold | 100 trials/fold |
|---|---|---|
| aggregate OOS CAGR spread across 5 seeds | 7.6 pts | 2.9 pts |
| mean per-fold OOS spread across seeds | 29.6 pts | 26.9 pts |
| folds where all seeds agree within 1 pt | 0/16 | 0/16 |

## 4. Paired-fold null control (etf.yaml + balanced-growth)

calmar vs **itself** across seed pairs at 100 trials/fold: paired-fold
difference sd 17.6 pts (range 15.2–22.7), "wins" 6–10 of 16 — the same
range as the calmar-vs-sharpe treatment (sd 21.9–27.7, 7/16), so paired
per-fold comparison cannot distinguish a real variant difference from the
same variant run twice at this data scale.

## Limitations (read before citing)

- The grid ran objective `gross`; conclusions transferred to other
  objectives are extrapolation.
- n is small everywhere: the rank correlation +0.25 over 24
  non-independent searches has SE ≈ 0.2 (not distinguishable from zero);
  the +2.70 pt restart benefit sits inside per-fold noise that section 4
  shows has no discriminating power. Treat both as directional hints, not
  results.
- The 100-trials/fold per-fold-spread comparison exists for the etf
  basket only; the test basket's 30-trials/fold seed spread was ~27 pts
  (issue #49 table) with no higher-budget counterpart.
- All runs share one decade of market history; nothing here measures
  period risk.
