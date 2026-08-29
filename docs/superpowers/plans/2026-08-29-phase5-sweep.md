# Phase 5: Sweep and Validation CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `midas sweep` compares policy variants across portfolios and seed replications with honest statistics (stationary block bootstrap, Deflated Sharpe Ratio, OOS-unit noise floor, qualified verdicts, holdout reservation); `midas validate` runs the strategies file's own policy and writes the baselines artifact the monitor consumes.

**Architecture:** `src/midas/sweep.py` holds the pure statistics (`stationary_bootstrap`, `deflated_sharpe_ratio`) and the orchestration (`run_sweep`) that fans `validate_policy` over variants × portfolios × seed replications and aggregates into a `SweepReport`. The CLI layer adds the two commands, the cost preflight, and the holdout trim. Splitting baselines *production* into `midas validate` (rather than the sweep writing a "selected" variant's baselines) is a recorded deviation: the sweep compares, the validate command deploys evidence.

**Tech Stack:** Python 3.14, `statistics.NormalDist` (no scipy), pytest.

**Spec:** `docs/specs/2026-08-29-policy-engine-design.md` (Component 3 sweep; holdout; Definitions)

## Global Constraints

Same as prior phases. Additions:

- **Seed replication rule:** replication *i* runs the policy with `base_seed + i` (the restart seeds then hash from that base with `as_of`, so replications never collide with restarts).
- **Holdout:** default 730 calendar days trimmed from `--end`; `--use-holdout` is allowed only with a single variant and prints the one-shot holdout report. The sweep always echoes what it trimmed.
- **DSR proxy is documented where computed:** trial count = budget × restarts × folds × seeds; SR variance proxied by the per-fold OOS Sharpe variance. Both stated in the output line, not hidden.

---

### Task 1: Pure statistics — stationary bootstrap and DSR

**Files:** Create `src/midas/sweep.py`; test `tests/test_sweep.py` (create).

**Interfaces:**
- `stationary_bootstrap(daily_returns: Sequence[float], *, n_resamples: int = 1000, mean_block: int = 21, seed: int = 0) -> list[float]` — Politis–Romano stationary bootstrap (geometric block lengths, wrap-around), returning one **annualized return** per resample; deterministic given `seed`; `[]` for fewer than 2 observations.
- `deflated_sharpe_ratio(observed_sharpe: float, *, n_trials: int, sharpe_variance: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0) -> float` — Bailey/López de Prado: probability the observed SR beats the expected max of `n_trials` noise SRs. Returns in [0, 1]; 0.5-ish for SR ≈ expected max; monotonically decreasing in `n_trials`.

- [ ] **Step 1: failing tests**

```python
"""Tests for sweep statistics."""

import numpy as np
import pytest

from midas.sweep import deflated_sharpe_ratio, stationary_bootstrap


def test_bootstrap_is_deterministic_and_centered() -> None:
    rng = np.random.default_rng(1)
    returns = list(rng.normal(0.0004, 0.01, 500))
    a = stationary_bootstrap(returns, n_resamples=200, seed=7)
    b = stationary_bootstrap(returns, n_resamples=200, seed=7)
    assert a == b
    assert len(a) == 200
    point = (np.prod([1 + r for r in returns]) ** (252 / len(returns))) - 1
    lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
    assert lo < point < hi  # the CI contains the point estimate


def test_bootstrap_degenerate_inputs() -> None:
    assert stationary_bootstrap([]) == []
    assert stationary_bootstrap([0.01]) == []


def test_dsr_decreases_with_trial_count() -> None:
    common = dict(sharpe_variance=0.25, n_obs=504, skew=0.0, kurtosis=3.0)
    few = deflated_sharpe_ratio(1.0, n_trials=10, **common)
    many = deflated_sharpe_ratio(1.0, n_trials=10_000, **common)
    assert 0.0 <= many < few <= 1.0


def test_dsr_zero_sharpe_is_below_half_once_trials_exist() -> None:
    dsr = deflated_sharpe_ratio(0.0, n_trials=100, sharpe_variance=0.25, n_obs=504)
    assert dsr < 0.5
```

- [ ] **Step 2: RED.** **Step 3: implement** (bootstrap: geometric restart probability `1/mean_block`, indices modulo n; DSR: `sr0 = sqrt(v) * ((1-g)*z(1-1/N) + g*z(1-1/(N*e)))` with `g = 0.5772...` Euler–Mascheroni, then `Φ(((sr-sr0)*sqrt(n_obs-1)) / sqrt(1 - skew*sr + (kurtosis-1)/4*sr**2))`, clamped into [0,1]). **Step 4: GREEN.** **Step 5: commit** ("Sweep statistics: stationary block bootstrap and Deflated Sharpe Ratio").

---

### Task 2: Variants and orchestration

**Files:** Modify `src/midas/sweep.py`; extend `tests/test_sweep.py`.

**Interfaces:**
- `make_variants(policy: Policy, vary: str, values: Sequence[str]) -> list[tuple[str, Policy]]` — `dataclasses.replace` with per-field parsing (`objective`/`deployment` strings validated by `Policy.__post_init__`; int fields via `int()`); `vary` must be a Policy field (ValueError otherwise, naming the valid fields); empty `values` ⇒ `[("base", policy)]`.
- `@dataclass SweepCell(variant: str, portfolio: str, seed_base: int, aggregate_oos_cagr: float, oos_sharpe: float, daily_returns: list[float], n_folds: int)`
- `@dataclass SweepReport(cells: list[SweepCell], holdout_trimmed_to: date | None)` plus method `summary_lines() -> list[str]` producing, per (variant, portfolio): mean aggregate CAGR across seeds, seed spread (the OOS-unit noise floor), bootstrap 95% CI of the concatenated path, DSR line with its proxy note, and the verdict: differences under the widest noise floor print **"not resolvable at this budget"**; single-portfolio reports append **"(on these windows)"**.
- `run_sweep(portfolios: Mapping[str, PortfolioConfig], price_data_by_portfolio, start, end, policy, vary, values, seeds: int, *, constraints, exit_params, ..., validate=validate_policy) -> SweepReport` — the `validate` parameter is the test seam.

- [ ] **Step 1: failing tests** — `make_variants` parsing (objective values, bad field name, int field); `run_sweep` with a stubbed `validate` returning synthetic `Baselines` (constructed per call from a counter): produces `variants × portfolios × seeds` cells; seed bases are `base_seed + i`; summary contains "not resolvable" when stub spreads exceed differences and "(on these windows)" for one portfolio. Write the stub concretely: it fabricates `Baselines` with `aggregate_oos_cagr` drawn from a dict keyed by (variant objective, seed) the test controls.
- [ ] **Step 2: RED.** **Step 3: implement.** **Step 4: GREEN.** **Step 5: commit** ("Sweep orchestration: variants, seed replication, qualified verdicts").

---

### Task 3: `midas validate`

**Files:** Modify `src/midas/cli.py`; test `tests/test_cli_validate.py` (create, reusing the `fake_prices` fixture pattern and the `STRATS_WITH_POLICY` fixture from `tests/test_cli_fit.py`).

**Interfaces:** `midas validate -p PORTFOLIO -s STRATEGIES --start DATE --end DATE [--min-train-pct 0.6]` — requires a `policy:` block; assembles inputs exactly as `fit` does; runs `validate_policy`; writes `<strategies>.baselines.json`; prints per-fold table (fold, window, return, drawdown, adopted) and the aggregate.

- [ ] **Step 1: failing tests** — baselines file written with matching `input_hash` (recompute via `input_fingerprint` in the test); command refuses without a policy block; output mentions fold count and aggregate.
- [ ] **Step 2: RED.** **Step 3: implement.** **Step 4: GREEN + full suite.** **Step 5: commit** ("midas validate: run the policy and write the monitor's baselines").

---

### Task 4: `midas sweep` CLI + docs

**Files:** Modify `src/midas/cli.py`, `docs/cli.md`; test `tests/test_cli_sweep.py` (create).

**Interfaces:** `midas sweep -p PORTFOLIO [-p ...] -s STRATEGIES --start --end [--vary FIELD --value V ...] [--seeds 3] [--holdout-days 730] [--use-holdout] [--budget N --force]`:

- Requires a policy block. Trims `end` by `--holdout-days` and echoes the trimmed boundary; `--use-holdout` requires exactly one variant and skips the trim, labeling the output "HOLDOUT REPORT — one shot; do not iterate on this".
- `--budget` overrides the policy's budget only together with `--force` ("validating at a different budget than deployment breaks coherence").
- Preflight line before running: variants × portfolios × seeds × folds × restarts × budget = N trials, with folds estimated from the range and cadence.
- Multiple `-p` label cells by portfolio filename stem.

- [ ] **Step 1: failing tests** — with `midas.cli.run_sweep` monkeypatched to a recorder returning an empty `SweepReport`: holdout trim passed through (recorder sees `end` shifted); `--use-holdout` with two values errors; `--budget` without `--force` errors; preflight text printed. One slow real path: single variant, tiny policy, 240-bar fake prices → exit 0 and a summary line.
- [ ] **Step 2: RED.** **Step 3: implement.** **Step 4: GREEN + full suite.** **Step 5: docs** — `docs/cli.md` sections for `validate` and `sweep` (options tables; the noise-floor/verdict semantics; the holdout rule; DSR proxy note), and the spec's Deviations section gains the validate/sweep split. **Step 6: commit + push.**

---

## Self-review record

- Spec coverage: block bootstrap (T1), DSR-as-formula with documented proxies (T1), OOS-unit noise floor and qualified single-portfolio verdicts (T2), seed replication (T2), portfolios side-by-side never pooled (T2 summary shows per-portfolio rows), holdout reservation + one-shot report (T4), budget guard = policy budget with `--force` (T4), cost preflight (T4), baselines written for the monitor (T3 — recorded deviation: `validate` writes them, the sweep stays a comparator).
- Per-seed OOS spread lands in the baselines *report line*, not the artifact (artifact stays per-validation-run; the spread is a sweep-level statistic across runs) — consistent with phase 4's schema.
- Types: `SweepCell`/`SweepReport`/`make_variants`/`run_sweep` consistent across T2–T4.
