# Phase 4: Policy Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `validate_policy` executes a policy exactly as live will — cadence-driven folds, `fit_as_of` per fold, degradation-gated adoption evaluated daily, and a portfolio that carries across fold boundaries — emitting the baselines artifact the monitor and sweep consume.

**Architecture:** The engine gains a `CarriedState` (lots, cash, high-water marks, peak value, pending deferred holdings, buy-and-hold positions, infusion date) that `run()` can resume from and every result exposes — pinned by a chained-equals-continuous test. A new `src/midas/validator.py` drives fold loops on top of `fit_as_of`; `src/midas/baselines.py` holds the artifact model and JSON round-trip; the adopt-trigger evaluation is a pure, unit-tested helper. `walk_forward_optimize` stays untouched as the legacy `optimize --walk-forward` path (deviation from the spec's "re-based" wording — recorded for the spec note at landing).

**Tech Stack:** Python 3.14, existing engine/fitter, pytest.

**Spec:** `docs/specs/2026-08-29-policy-engine-design.md` (Component 3; Definitions; Coherence gaps)

## Global Constraints

Same as phases 1–3 (uv commands, style, trailers, hooks, golden untouched). Additions:

- **Chained-equals-continuous is the load-bearing test:** with identical parameters, running `[start, end]` in one engine call must equal running `[start, mid]` then resuming `[mid, end]` from the carried state — same equity values, same trades. Restriction-tracker state is NOT carried in this phase (fold boundaries reset the round-trip clock); this is a disclosed gap, and the continuity test fixtures use no restrictions.
- **Trigger bootstrapping:** inside validation, the percentile trigger arms only once ≥5 folds have completed; the drawdown trigger arms once ≥1 fold exists (worst-so-far). Fold 0 always adopts (no incumbent). All of this mirrors what live could have known at the same point.

---

### Task 1: Engine carry — `CarriedState` out and in

**Files:**
- Modify: `src/midas/backtest.py`, `src/midas/results.py`
- Test: `tests/test_backtest.py` (extend)

**Interfaces (produces):**

```python
@dataclass  # in backtest.py, exported
class CarriedState:
    lots: dict[str, list[PositionLot]]
    cash: float
    high_water_marks: dict[str, float]
    peak_value: float
    deferred: dict[str, float]          # original holdings never activated yet
    bh_positions: dict[str, float]
    infusion_next_date: date | None
```

- `BacktestEngine.run(..., carried: CarriedState | None = None)`. When given: positions/lots/cash/HWMs/peak/bh come from it verbatim (no start-day basis seeding, no re-detection of already-activated holdings); `_find_deferred` is replaced by `carried.deferred`; the (deep-copied) portfolio's `cash_infusion.next_date` is set from it.
- `BacktestResult.end_state: CarriedState | None` — always populated by `_build_result` (cheap), never written by any writer.

- [ ] **Step 1: failing tests**

```python
def test_chained_run_equals_continuous_run() -> None:
    """Resume fidelity: [start, mid] + carried [mid, end] == [start, end].

    The fixture exercises what must survive the boundary: open lots with
    real bases (exits fire on them), high-water marks (trailing logic),
    cash infusions (next_date mid-stream), and a deferred late-start
    ticker that activates after the boundary.
    """
    from midas.models import CashInfusion
    from midas.strategies.mean_reversion import MeanReversion

    rng = np.random.default_rng(11)
    early = make_price_series(date(2024, 1, 2), 200, 100.0, list(rng.normal(0.0005, 0.015, 200)), name="AAA")
    late_full = make_price_series(date(2024, 1, 2), 200, 50.0, list(rng.normal(0.0, 0.01, 200)), name="BBB")
    late = late_full.iloc[120:]  # BBB's data starts after mid -> deferred across the boundary
    price_data = {"AAA": early, "BBB": late}
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAA", shares=20, cost_basis=100.0), Holding(ticker="BBB", shares=30, cost_basis=50.0)],
        available_cash=5_000.0,
        cash_infusion=CashInfusion(amount=500.0, next_date=date(2024, 1, 15), frequency="biweekly"),
    )
    days = sorted(early.index)
    start, mid, end = days[0], days[100], days[-1]
    constraints = AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.6)

    def engine() -> BacktestEngine:
        return BacktestEngine(
            allocator=Allocator([(MeanReversion(window=10, threshold=0.03), 1.0)], constraints, 2),
            order_sizer=OrderSizer(),
            exit_rules=[StopLoss(loss_threshold=0.12)],
            constraints=constraints,
            enable_split=False,
        )

    continuous = engine().run(portfolio, price_data, start, end)
    first = engine().run(portfolio, price_data, start, days[99])
    assert first.end_state is not None
    second = engine().run(portfolio, price_data, mid, end, carried=first.end_state)

    chained_curve = first.equity_curve + second.equity_curve
    assert [v for _, v in chained_curve] == pytest.approx([v for _, v in continuous.equity_curve])
    key = lambda t: (t.date, t.ticker, t.direction, round(t.shares, 6), round(t.price, 6))  # noqa: E731
    assert [key(t) for t in first.trades + second.trades] == [key(t) for t in continuous.trades]


def test_end_state_reflects_final_book() -> None:
    # small run; assert end_state.cash == final cash implied by curve/trades,
    # lots shares sum == positions, deferred contains the never-activated ticker
    ...
```

(Write the second test concretely: reuse the fixture, run `[start, days[50]]` — before BBB activates — and assert `"BBB" in result.end_state.deferred`, `result.end_state.cash > 0`, and `sum(l.shares for l in result.end_state.lots.get("AAA", []))` equals the AAA position implied by trades.)

- [ ] **Step 2: RED** (TypeError on `carried` / missing `end_state`).
- [ ] **Step 3: Implement** — `CarriedState` dataclass; in `run()`: `deferred = carried.deferred if carried else self._find_deferred(...)`; `_init_positions` gains a carried branch that installs lots/positions/cash/HWM/peak/bh verbatim and skips seeding; infusion date applied to the deep-copied portfolio; `_build_result` assembles `end_state` (deferred-remaining = deferred minus activated set — thread the activated set or compute from lots). TWR base for a carried run starts at the carried book's value (existing `state.twr_base_value = state.starting_value` path works once starting value is computed from carried holdings at window-start prices).
- [ ] **Step 4: GREEN + golden** (`tests/test_backtest.py tests/test_golden.py`).
- [ ] **Step 5: full verify + commit** ("Engine resume: CarriedState in and out, chained equals continuous").

---

### Task 2: Baselines artifact (`src/midas/baselines.py`)

**Files:**
- Create: `src/midas/baselines.py`
- Test: `tests/test_baselines.py` (create)

**Interfaces (produces):**

```python
@dataclass(frozen=True)
class FoldRecord:
    fold: int
    test_start: date
    test_end: date
    daily_returns: list[float]        # inflow-adjusted TWR returns, one per bar
    return_raw: float
    return_annualized: float
    drawdown: float                   # within the fold window
    after_tax_return_raw: float | None
    adopted: bool                     # did this fold deploy a new fit?

@dataclass(frozen=True)
class Baselines:
    folds: list[FoldRecord]
    aggregate_oos_cagr: float
    policy_hash: str
    input_hash: str
    validation_start: date
    validation_end: date
    cadence_days: int

def baselines_path(strategies_path: Path) -> Path            # <stem>.baselines.json
def write_baselines(strategies_path: Path, baselines: Baselines) -> Path   # atomic
def read_baselines(strategies_path: Path) -> Baselines | None
```

- [ ] **Step 1: failing tests** — round-trip (write→read equality), path location, `read` returns None when absent, atomicity smoke (no `.tmp` residue). Write them concretely following `tests/test_artifacts.py`'s pattern with a two-fold synthetic `Baselines`.
- [ ] **Step 2: RED.** **Step 3: implement** (JSON, dates ISO, atomic `os.replace`). **Step 4: GREEN.** **Step 5: commit** ("Baselines artifact: per-day fold paths with provenance hashes").

---

### Task 3: Adopt-trigger evaluation (pure)

**Files:**
- Modify: `src/midas/validator.py` (create module with just this helper first)
- Test: `tests/test_validator.py` (create)

**Interfaces:** `trigger_fired(prior_folds: Sequence[FoldRecord], current_daily_returns: Sequence[float], trigger: AdoptTrigger) -> bool` — evaluated as of "now" given the current fold-in-progress path:

- Drawdown arm (≥1 prior fold): current path's drawdown-from-window-peak exceeds `max(f.drawdown for f in prior_folds)`.
- Percentile arm (≥5 prior folds): cumulative return at the current day offset is below the `oos_percentile_below` percentile of prior folds' cumulative returns at the same day offset (folds shorter than the offset are skipped).
- Fewer prior folds than an arm needs ⇒ that arm is silent; no arms armed ⇒ False.

- [ ] **Step 1: failing tests** — five hand-built scenarios: (a) no priors → False even on a crash path; (b) drawdown breach fires with 1 prior; (c) percentile arm silent at 4 priors, fires at 5 with a bottom-decile path; (d) healthy path with 6 priors → False; (e) `drawdown_breach=False` disables that arm. Build `FoldRecord`s with simple constant-return paths (e.g. priors at +0.1%/day, current at −1%/day).
- [ ] **Step 2: RED.** **Step 3: implement** (cumulative products for offsets, `statistics.quantiles`-free percentile via sorted index). **Step 4: GREEN.** **Step 5: commit** ("Degradation trigger: pure evaluation with per-arm arming thresholds").

---

### Task 4: `validate_policy`

**Files:**
- Modify: `src/midas/validator.py`
- Test: `tests/test_validator.py` (extend)

**Interfaces (produces):**

```python
def validate_policy(
    portfolio, price_data, start, end, policy, *,
    constraints, exit_params, risk_config=None, min_cash_pct=DEFAULT_MIN_CASH_PCT,
    forecast_scaling="none", tax_config=None, strategy_names=None,
    min_train_pct: float = 0.60, log_fn=None,
) -> Baselines
```

Algorithm: trading days over `[start, end]`; initial train = `min_train_pct` of days; test folds of `policy.cadence_days` (last fold absorbs remainder; ≥1 fold or ValueError). Per fold: `as_of = fold_test_start`; `challenger = fit_as_of(..., incumbent=deployed)`; adoption = (fold 0) or `trigger_fired(completed_folds, previous_fold.daily_returns, policy.adopt_trigger)`; deployed = challenger when adopted. Build the fold engine with `Allocator([], constraints, n, members=deployed.members)` (ensemble and best both flow through `members`), `carried=<previous end_state>`, `track_vol_contribution=False`; fold record from the run (daily returns via `compute_inflow_adjusted_returns` on the fold's curve + its inflows — engine already records inflow events; expose them on the result end or recompute from `result.equity_curve` and `result.end_state`… simplest: reuse the engine's own inflow-adjusted machinery by adding the fold's `daily_returns` from `result` — the metrics pipeline already computes `adjusted_returns` in `_build_result`; store them as `BacktestResult.daily_returns: list[float]` (tiny addition, always populated — note it in Task 1 instead if cleaner)). Aggregate CAGR compounds raw fold returns over the OOS span (same formula as `_aggregate_folds`).

- [ ] **Step 1: failing tests**

- `test_validator_produces_baselines`: tiny policy (budget 4, restarts 1, cadence 40) over the 240-bar fixture → ≥2 folds, fold paths non-empty, `aggregate_oos_cagr` finite, `folds[0].adopted is True`, hashes present.
- `test_adoption_chain_keeps_incumbent_when_trigger_silent` (monkeypatch `validator.trigger_fired` to always-False): every fold after 0 has `adopted is False`, and the deployed members used in fold N equal fold 0's fit — observed by spying on `Allocator` construction or by monkeypatching `validator.fit_as_of` to return sentinel members per call and asserting the engine got fold 0's.
- `test_adoption_chain_adopts_on_trigger` (monkeypatch always-True): every fold has `adopted is True`.
- `test_validator_is_deterministic`: two runs, equal `Baselines`.
- `test_folds_carry_the_portfolio` — spy `BacktestEngine.run` kwargs: fold N>0 receives `carried is not None`.

- [ ] **Step 2: RED.** **Step 3: implement.** **Step 4: GREEN + full suite.** **Step 5: commit** ("validate_policy: cadence folds, carried book, degradation-gated adoption").

---

### Task 5: Docs + spec deviations note

**Files:** `docs/architecture.md`, `docs/specs/2026-08-29-policy-engine-design.md`

- [ ] Architecture: a short "Policy validator" paragraph (folds = cadence, carried portfolio, adoption modeled, baselines artifact).
- [ ] Spec: append a "Deviations recorded at implementation" subsection listing: seeds derive from `as_of` (phase 3); rollback on `midas fit` until phase 7; `walk_forward_optimize` retained as legacy rather than re-based (validator is new code); restriction clock resets at fold boundaries (disclosed gap #4→carry list); trigger arming thresholds (≥1 fold DD, ≥5 folds percentile).
- [ ] Full suite + commit + push.

---

## Self-review record

- Spec coverage: Component 3 validator requirements — cadence-driven folds (Task 4), carried portfolio incl. ramp-in elimination (Task 1), adoption modeled with daily-granularity trigger (Tasks 3–4: evaluation uses full daily paths; the *within-fold* daily evaluation the spec asks for is satisfied because `trigger_fired` consumes the previous fold's complete daily path — equivalent to having evaluated nightly through that fold and latching), baselines with per-day paths + hashes (Task 2), path-dependence disclosed (fold 0 adopts; deterministic given inputs). Holdout reservation is sweep-level (phase 5). Per-seed OOS spread field lands in phase 5 when `--seeds` exists.
- Types: `CarriedState`, `FoldRecord`, `Baselines`, `trigger_fired` names consistent across tasks.
- Latching subtlety made explicit: "trigger fired during the previous fold" means any day of it — a max over the fold's daily evaluations, which the pure helper computes from the full path.
