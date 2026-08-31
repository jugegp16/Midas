# Phase 3: Fitter and Deployment Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One `fit_as_of` code path produces deployable members (best-of-restarts or stratified deduped ensemble) from a policy, deterministically, with sidecar artifacts, history, and rollback — the function the validator (phase 4) and live re-fit (phase 7) will share.

**Architecture:** A `Policy` dataclass parsed from the strategies YAML's `policy:` block (`src/midas/policy.py`, pure: parsing, hashing, seed derivation). `fit_as_of` (`src/midas/fitter.py`) runs `policy.restarts` seeded `optimize()` calls over data strictly before `as_of`, warm-starts restart 0 from the incumbent, and selects members. The engine grows an off-by-default `record_targets` flag so candidate members' target series feed `behavioral_key` dedupe. Artifacts: machine-owned `<strategies>.members.yaml` sidecar, `<strategies>.fits/` history, `midas fit` CLI with `--rollback`.

**Tech Stack:** Python 3.14, Optuna ask/tell (existing), hashlib, yaml, pytest.

**Spec:** `docs/specs/2026-08-29-policy-engine-design.md` (Deployment artifacts; Component 2; Definitions)

## Global Constraints

Same as phase 1/2 plans (uv commands, Google style, no ticket refs, trailers, hooks green, golden untouched). Additions:

- **`as_of` is exclusive** everywhere: a fit sees bars strictly before `as_of`. Implemented as `end = as_of - timedelta(days=1)` into the existing date-filtered engines.
- **Seeds derive from `(base_seed, as_of, restart)`** via SHA-256, not from fold indices: the validator's fold fit at date D and a standalone `fit_as_of(D)` must draw identical seeds or the coherence principle dies at the RNG. (Deliberate refinement of the spec's `hash(base_seed, f, r)` wording — as-of-based derivation is what actually makes the two call sites coincide; note this in the spec when the phase lands.)
- **Rollback lives on `midas fit --rollback N`** in this phase; the spec's `midas refit` arrives in phase 7 and will delegate. Note in spec at landing.

---

### Task 1: `Policy` model, hashing, and seed derivation (`src/midas/policy.py`)

**Files:**
- Create: `src/midas/policy.py`
- Modify: `src/midas/config.py` (add `"policy"` to `load_strategies` known keys; new `load_policy(path) -> Policy | None`)
- Test: `tests/test_policy.py` (create)

**Interfaces (produces):**
- `@dataclass(frozen=True) AdoptTrigger(oos_percentile_below: float = 10.0, drawdown_breach: bool = True)`
- `@dataclass(frozen=True) Policy(objective: Objective = DEFAULT_OBJECTIVE, budget: int = 2000, restarts: int = 4, base_seed: int = 42, deployment: Literal["best", "ensemble"] = "best", ensemble_size: int = 20, cadence_days: int = 63, adopt_trigger: AdoptTrigger = AdoptTrigger())` with `__post_init__` validation: budget/restarts/ensemble_size/cadence_days ≥ 1, objective validated via `get_args`, deployment in {"best","ensemble"}, `ensemble_size >= restarts` when deployment is ensemble (need ≥1 member per restart stratum).
- `parse_policy(raw: Mapping[str, object]) -> Policy` — unknown keys inside the block raise ValueError naming them.
- `policy_hash(policy: Policy) -> str` — SHA-256 hex of canonical JSON (sorted keys) of behavior-affecting fields; **`ensemble_size` is excluded when `deployment == "best"`** (spec: editing an inert field never invalidates artifacts).
- `derive_seed(base_seed: int, as_of: date, restart: int) -> int` — first 4 bytes of `sha256(f"{base_seed}:{as_of.isoformat()}:{restart}")`, big-endian. Stable across processes and machines.
- `input_fingerprint(*, policy, tickers: Sequence[str], constraints, exit_params, risk_config, min_cash_pct, forecast_scaling, tax_config) -> str` — SHA-256 of canonical JSON over all of it (dataclasses via `dataclasses.asdict`, tickers sorted, None-safe).
- `load_policy(path) -> Policy | None` in config.py — `None` when the file has no `policy:` block.

- [ ] **Step 1: failing tests** (`tests/test_policy.py`)

```python
"""Tests for the deployment policy model."""

from datetime import date

import pytest

from midas.config import load_policy
from midas.models import AllocationConstraints, RiskConfig
from midas.policy import AdoptTrigger, Policy, derive_seed, input_fingerprint, parse_policy, policy_hash


def test_defaults_match_the_spec() -> None:
    policy = Policy()
    assert policy.objective == "calmar"
    assert policy.deployment == "best"
    assert (policy.budget, policy.restarts, policy.ensemble_size, policy.cadence_days) == (2000, 4, 20, 63)


def test_parse_policy_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="cadence"):
        parse_policy({"cadence": 63})


def test_parse_policy_nested_trigger() -> None:
    policy = parse_policy({"adopt_trigger": {"oos_percentile_below": 5, "drawdown_breach": False}})
    assert policy.adopt_trigger == AdoptTrigger(oos_percentile_below=5.0, drawdown_breach=False)


def test_ensemble_size_must_cover_restarts() -> None:
    with pytest.raises(ValueError, match="ensemble_size"):
        Policy(deployment="ensemble", restarts=8, ensemble_size=4)


def test_policy_hash_ignores_inert_ensemble_size_under_best() -> None:
    a = Policy(deployment="best", ensemble_size=20)
    b = Policy(deployment="best", ensemble_size=40)
    assert policy_hash(a) == policy_hash(b)
    c = Policy(deployment="ensemble", ensemble_size=20)
    d = Policy(deployment="ensemble", ensemble_size=40)
    assert policy_hash(c) != policy_hash(d)


def test_derive_seed_is_stable_and_distinct() -> None:
    s = derive_seed(42, date(2026, 8, 29), 0)
    assert s == derive_seed(42, date(2026, 8, 29), 0)  # deterministic
    assert s != derive_seed(42, date(2026, 8, 29), 1)  # restart-distinct
    assert s != derive_seed(42, date(2026, 8, 28), 0)  # date-distinct
    assert s != derive_seed(7, date(2026, 8, 29), 0)  # base-distinct
    assert 0 <= s < 2**32


def test_input_fingerprint_moves_with_every_input() -> None:
    base = dict(
        policy=Policy(), tickers=["AAA", "BBB"], constraints=AllocationConstraints(),
        exit_params={"StopLoss": {"loss_threshold": 0.1}}, risk_config=RiskConfig(),
        min_cash_pct=0.05, forecast_scaling="quantile", tax_config=None,
    )
    ref = input_fingerprint(**base)
    assert input_fingerprint(**{**base, "tickers": ["BBB", "AAA"]}) == ref  # order-insensitive
    assert input_fingerprint(**{**base, "tickers": ["AAA"]}) != ref
    assert input_fingerprint(**{**base, "min_cash_pct": 0.06}) != ref
    assert input_fingerprint(**{**base, "exit_params": {}}) != ref


def test_load_policy_roundtrip(tmp_path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(
        "strategies:\n  - name: MeanReversion\n"
        "policy:\n  objective: gross\n  budget: 100\n  restarts: 2\n"
    )
    policy = load_policy(path)
    assert policy is not None and policy.objective == "gross" and policy.budget == 100
    path.write_text("strategies:\n  - name: MeanReversion\n")
    assert load_policy(path) is None
```

- [ ] **Step 2: RED** — collection ImportError.
- [ ] **Step 3: Implement** `src/midas/policy.py` exactly per the interface block (canonical JSON: `json.dumps(payload, sort_keys=True, default=str)`); config.py adds `"policy"` to `known_keys` and `load_policy` re-reading `_load_yaml(path).get("policy")` through `parse_policy`.
- [ ] **Step 4: GREEN**, plus `uv run pytest tests/test_config.py -q --no-cov` (known-keys change).
- [ ] **Step 5: full verify + commit** ("Policy model: parsing, mode-aware hash, as-of seed derivation, input fingerprint").

---

### Task 2: Engine `record_targets`

**Files:**
- Modify: `src/midas/backtest.py`, `src/midas/results.py`
- Test: `tests/test_backtest.py` (extend)

**Interfaces:** `BacktestEngine(..., record_targets: bool = False)`; when on, `BacktestResult.target_series: list[dict[str, float]]` holds each bar's allocation targets in bar order (empty when off — default, so writers, golden, and every existing caller are untouched).

- [ ] **Step 1: failing tests**

```python
def test_record_targets_captures_per_bar_allocations() -> None:
    from midas.strategies.mean_reversion import MeanReversion

    rng = np.random.default_rng(4)
    prices = make_price_series(date(2024, 1, 2), 60, 100.0, list(rng.normal(0, 0.01, 60)))
    portfolio = PortfolioConfig(holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)], available_cash=2000.0)
    constraints = AllocationConstraints(min_buy_delta=0.01)
    engine = BacktestEngine(
        allocator=Allocator([(MeanReversion(window=10, threshold=0.03), 1.0)], constraints, 1),
        order_sizer=OrderSizer(), exit_rules=[], constraints=constraints,
        enable_split=False, record_targets=True,
    )
    result = engine.run(portfolio, {"TEST": prices}, min(prices.index), max(prices.index))
    assert len(result.target_series) == len(result.equity_curve)
    assert all(set(bar) <= {"TEST"} for bar in result.target_series)


def test_record_targets_off_is_empty_and_free() -> None:
    # reuse the setup above with record_targets omitted
    ...
    assert result.target_series == []
```

(Write the second test fully — same construction, flag omitted.)

- [ ] **Step 2: RED** (TypeError on `record_targets`).
- [ ] **Step 3: Implement** — engine stores the flag; in the per-day loop where `allocation = self._allocator.allocate(...)` returns, append `dict(allocation.targets)` to `state.target_series` when enabled; `_build_result` passes it through. `results.py`: `target_series: list[dict[str, float]] = field(default_factory=list)` — **not** added to any writer output.
- [ ] **Step 4: GREEN + golden** — `uv run pytest tests/test_backtest.py tests/test_golden.py -q --no-cov`.
- [ ] **Step 5: full verify + commit** ("Engine can record per-bar allocation targets (off by default)").

---

### Task 3: `optimize()` warm start and top-trial capture

**Files:**
- Modify: `src/midas/optimizer.py` (`optimize`, `_run_trials_batched`, `OptimizeResult`)
- Test: `tests/test_optimizer_search_space.py` (extend)

**Interfaces:**
- `optimize(..., enqueue: list[dict[str, float]] | None = None, record_top: int = 0)`.
- `enqueue`: flat Optuna param dicts (`{Strategy}__{param}` naming, exact grid values) enqueued before the batched loop — mirrors walk-forward's `prev_best_flat` mechanism.
- `record_top > 0` ⇒ `OptimizeResult.top_trials: list[tuple[float, dict[str, dict[str, float]]]]` — (objective value, nested params), best-first, capped at `record_top`, ties broken by trial order.
- New helper `flatten_params(params: dict[str, dict[str, float]]) -> dict[str, float]` (nested → flat naming) so the fitter can enqueue incumbents.

- [ ] **Step 1: failing tests**

```python
def test_enqueued_params_run_first_and_flatten_round_trips() -> None:
    from midas.optimizer import flatten_params, optimize

    portfolio, price_data, start, end = _data()
    fixed = {"MeanReversion": {"window": 20.0, "threshold": 0.05, "_weight": 1.0}}
    result = optimize(
        portfolio=portfolio, price_data=price_data, start=start, end=end,
        strategy_names=["MeanReversion"], n_trials=3, objective="gross",
        exit_params={}, constraints=AllocationConstraints(),
        enqueue=[flatten_params(fixed)], record_top=3,
    )
    # The enqueued combination must appear among the recorded trials.
    assert any(params == fixed for _score, params in result.top_trials)


def test_record_top_caps_and_orders_best_first() -> None:
    from midas.optimizer import optimize

    portfolio, price_data, start, end = _data()
    result = optimize(
        portfolio=portfolio, price_data=price_data, start=start, end=end,
        strategy_names=["MeanReversion"], n_trials=6, objective="gross",
        exit_params={}, constraints=AllocationConstraints(), record_top=4,
    )
    scores = [score for score, _ in result.top_trials]
    assert len(result.top_trials) == 4
    assert scores == sorted(scores, reverse=True)
    assert result.top_trials[0][0] == result.best_objective_value
```

- [ ] **Step 2: RED** (TypeError on `enqueue`).
- [ ] **Step 3: Implement** — `flatten_params` inverts `_suggest_params` naming (`_weight` included; ints as floats). In `optimize`: after `create_study`, `for flat in enqueue or []: study.enqueue_trial(flat)`. After the loop, when `record_top`: collect `(t.value, t.user_attrs["params"])` from completed trials, sort desc by value, cap. `OptimizeResult.top_trials: list[tuple[float, dict[str, dict[str, float]]]] = field(default_factory=list)`.
- [ ] **Step 4: GREEN + optimizer suites.**
- [ ] **Step 5: full verify + commit** ("optimize() gains warm-start enqueue and top-trial capture").

---

### Task 4: `fit_as_of` — best mode

**Files:**
- Create: `src/midas/fitter.py`
- Test: `tests/test_fitter.py` (create)

**Interfaces (produces):**

```python
@dataclass(frozen=True)
class FitResult:
    members: list[dict[str, dict[str, float]]]   # nested optimizer param dicts, one per member
    member_scores: list[float]                   # training objective per member, same order
    restart_bests: list[float]                   # best objective per restart (search-noise record)
    as_of: date
    policy_hash: str
    input_hash: str

def fit_as_of(
    portfolio, price_data, as_of, policy, *,
    constraints, exit_params, risk_config=None, min_cash_pct=DEFAULT_MIN_CASH_PCT,
    forecast_scaling="none", tax_config=None, incumbent: FitResult | None = None,
    strategy_names: list[str] | None = None, log_fn=None,
) -> FitResult
```

Semantics: `policy.restarts` calls to `optimize(seed=derive_seed(policy.base_seed, as_of, r), n_trials=policy.budget, train_pct=1.0, start=<earliest bar>, end=as_of - 1 day, objective=policy.objective, record_top=<0 for best mode>, enqueue=<incumbent members flattened, restart 0 only>)`. Best mode: member = argmax restart's `best_params`; `member_scores=[max]`; `restart_bests` = each restart's `best_objective_value`.

- [ ] **Step 1: failing tests**

```python
"""Tests for fit_as_of (best mode)."""

from datetime import date, timedelta

import numpy as np
from conftest import make_price_series

from midas.fitter import FitResult, fit_as_of
from midas.models import AllocationConstraints, Holding, PortfolioConfig
from midas.policy import Policy

def _data():
    rng = np.random.default_rng(5)
    prices = make_price_series(date(2023, 1, 2), 240, 100.0, list(rng.normal(0.0005, 0.012, 240)), name="TEST")
    portfolio = PortfolioConfig(holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)], available_cash=2000.0)
    return portfolio, {"TEST": prices}

SMALL = Policy(objective="gross", budget=6, restarts=2, base_seed=42, deployment="best")


def test_fit_is_deterministic() -> None:
    portfolio, price_data = _data()
    kwargs = dict(constraints=AllocationConstraints(), exit_params={}, strategy_names=["MeanReversion"])
    a = fit_as_of(portfolio, price_data, date(2023, 10, 2), SMALL, **kwargs)
    b = fit_as_of(portfolio, price_data, date(2023, 10, 2), SMALL, **kwargs)
    assert a.members == b.members
    assert a.restart_bests == b.restart_bests
    assert a.policy_hash == b.policy_hash and a.input_hash == b.input_hash


def test_as_of_is_exclusive() -> None:
    """Shifting as_of past the last bar the fit already saw changes nothing;
    a fit at bar t must not see bar t."""
    portfolio, price_data = _data()
    kwargs = dict(constraints=AllocationConstraints(), exit_params={}, strategy_names=["MeanReversion"])
    bars = sorted(price_data["TEST"].index)
    on_bar = fit_as_of(portfolio, price_data, bars[100], SMALL, **kwargs)
    day_after_prev_bar = fit_as_of(portfolio, price_data, bars[99] + timedelta(days=1), SMALL, **kwargs)
    assert on_bar.members == day_after_prev_bar.members  # both see bars[0..99]


def test_best_mode_selects_the_best_restart() -> None:
    portfolio, price_data = _data()
    result = fit_as_of(
        portfolio, price_data, date(2023, 10, 2), SMALL,
        constraints=AllocationConstraints(), exit_params={}, strategy_names=["MeanReversion"],
    )
    assert isinstance(result, FitResult)
    assert len(result.members) == 1
    assert len(result.restart_bests) == 2
    assert result.member_scores == [max(result.restart_bests)]


def test_incumbent_members_are_enqueued_into_restart_zero_only(monkeypatch) -> None:
    """Observed via the seams: capture optimize() calls and inspect enqueue."""
    import midas.fitter as fitter_module

    calls = []
    real_optimize = fitter_module.optimize

    def spy(*args, **kwargs):
        calls.append(kwargs.get("enqueue"))
        return real_optimize(*args, **kwargs)

    monkeypatch.setattr(fitter_module, "optimize", spy)
    portfolio, price_data = _data()
    kwargs = dict(constraints=AllocationConstraints(), exit_params={}, strategy_names=["MeanReversion"])
    first = fit_as_of(portfolio, price_data, date(2023, 8, 1), SMALL, **kwargs)
    calls.clear()
    fit_as_of(portfolio, price_data, date(2023, 10, 2), SMALL, incumbent=first, **kwargs)
    assert calls[0] is not None and len(calls[0]) == 1  # restart 0: the incumbent member
    assert all(c is None for c in calls[1:])  # other restarts stay independent
```

- [ ] **Step 2: RED** — ImportError.
- [ ] **Step 3: Implement** `src/midas/fitter.py`: derive data start as the earliest bar across `price_data`; `end = as_of - timedelta(days=1)`; loop restarts calling `optimize` (imported at module level — the spy seam); best mode as specced; hashes via `policy_hash` / `input_fingerprint` (tickers from `portfolio.holdings`).
- [ ] **Step 4: GREEN.**
- [ ] **Step 5: full verify + commit** ("fit_as_of: deterministic seeded restarts with exclusive as_of (best mode)").

---

### Task 5: `fit_as_of` — ensemble mode (stratified, behaviorally deduped)

**Files:**
- Modify: `src/midas/fitter.py`
- Test: `tests/test_fitter.py` (extend)

**Interfaces:** ensemble mode returns `members` of length ≤ `policy.ensemble_size`, at most `ceil(K/R)` from each restart's `top_trials`, walked best-first and deduplicated by `behavioral_key` of a `record_targets` backtest over the training window; `member_scores` per member. Restart runs pass `record_top=ensemble_size` (a full stratum's worth of candidates each).

- [ ] **Step 1: failing tests**

```python
def test_ensemble_mode_returns_stratified_members() -> None:
    portfolio, price_data = _data()
    policy = Policy(objective="gross", budget=8, restarts=2, ensemble_size=4, deployment="ensemble")
    result = fit_as_of(
        portfolio, price_data, date(2023, 10, 2), policy,
        constraints=AllocationConstraints(), exit_params={}, strategy_names=["MeanReversion"],
    )
    assert 1 <= len(result.members) <= 4
    assert len(result.member_scores) == len(result.members)
    # distinct by construction:
    assert len({str(m) for m in result.members}) == len(result.members)


def test_ensemble_dedupe_collapses_identical_behavior(monkeypatch) -> None:
    """Force every candidate to the same behavioral key: only one member per
    stratum may survive."""
    import midas.fitter as fitter_module

    monkeypatch.setattr(fitter_module, "behavioral_key", lambda series: ("same",))
    portfolio, price_data = _data()
    policy = Policy(objective="gross", budget=8, restarts=2, ensemble_size=4, deployment="ensemble")
    result = fit_as_of(
        portfolio, price_data, date(2023, 10, 2), policy,
        constraints=AllocationConstraints(), exit_params={}, strategy_names=["MeanReversion"],
    )
    assert len(result.members) <= 2  # one per restart stratum at most
```

- [ ] **Step 2: RED** (ensemble mode NotImplemented / wrong shape).
- [ ] **Step 3: Implement** — per restart: `record_top=policy.ensemble_size`; walk `top_trials` best-first; for each candidate run `_run_trial`-equivalent via a small local helper that builds the engine with `record_targets=True` over `[start, as_of)` and computes `behavioral_key(result.target_series)`; keep the candidate if its key is unseen globally; stop at `ceil(K/R)` per stratum. Members ordered by score desc across the final set.
- [ ] **Step 4: GREEN + determinism re-run** (`test_fit_is_deterministic` still passes; add an ensemble determinism assertion mirroring it).
- [ ] **Step 5: full verify + commit** ("fit_as_of ensemble mode: stratified top-K/R with behavioral dedupe").

---

### Task 6: Sidecar artifacts — members file, fits history, rollback

**Files:**
- Create: `src/midas/artifacts.py`
- Test: `tests/test_artifacts.py` (create)

**Interfaces (produces):**
- `members_path(strategies_path) -> Path` (`<stem>.members.yaml` beside the file), `fits_dir(strategies_path) -> Path` (`<stem>.fits/`).
- `write_fit(strategies_path, fit: FitResult) -> Path` — writes the full FitResult to `fits_dir()/<as_of>-<input_hash[:8]>.yaml` AND replaces the members sidecar (`fitted_as_of`, `policy_hash`, `input_hash`, `deployment` implied by member count? no — store `members`, scores, restart_bests verbatim). Atomic (tmp + `os.replace`), mirroring the price-cache writer.
- `read_members(strategies_path) -> FitResult | None`.
- `list_fits(strategies_path) -> list[Path]` (sorted, newest last).
- `rollback(strategies_path, steps: int = 1) -> FitResult` — re-points the members sidecar at the Nth-previous fit; raises `ValueError` when history is too short.

- [ ] **Step 1: failing tests**

```python
"""Tests for fit artifacts: members sidecar, history, rollback."""

from datetime import date
from pathlib import Path

import pytest

from midas.artifacts import fits_dir, list_fits, members_path, read_members, rollback, write_fit
from midas.fitter import FitResult


def _fit(as_of: date, score: float) -> FitResult:
    return FitResult(
        members=[{"MeanReversion": {"window": 20.0, "threshold": 0.05, "_weight": 1.0}}],
        member_scores=[score], restart_bests=[score], as_of=as_of,
        policy_hash="p" * 64, input_hash="i" * 64,
    )


def test_write_and_read_members_roundtrip(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    fit = _fit(date(2026, 8, 1), 1.5)
    write_fit(strategies, fit)
    assert members_path(strategies) == tmp_path / "strategies.members.yaml"
    assert read_members(strategies) == fit


def test_history_accumulates_and_rollback_restores(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    first = _fit(date(2026, 8, 1), 1.0)
    second = _fit(date(2026, 8, 2), 2.0)
    write_fit(strategies, first)
    write_fit(strategies, second)
    assert len(list_fits(strategies)) == 2
    assert read_members(strategies) == second
    restored = rollback(strategies)
    assert restored == first
    assert read_members(strategies) == first


def test_rollback_beyond_history_is_loud(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    write_fit(strategies, _fit(date(2026, 8, 1), 1.0))
    with pytest.raises(ValueError, match="history"):
        rollback(strategies, steps=3)


def test_read_members_none_when_absent(tmp_path: Path) -> None:
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text("strategies: []\n")
    assert read_members(strategies) is None
```

- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** with YAML serialization of the dataclass (dates ISO; scores plain floats), atomic writes, filenames `{as_of.isoformat()}-{input_hash[:8]}.yaml` (same as_of+hash overwrites — a rerun of the identical fit is one history entry). Rollback re-copies the target fit into the members sidecar without deleting history.
- [ ] **Step 4: GREEN.**
- [ ] **Step 5: full verify + commit** ("Fit artifacts: machine-owned members sidecar, fits history, rollback").

---

### Task 7: `midas fit` CLI + docs

**Files:**
- Modify: `src/midas/cli.py` (new `fit` command), `docs/cli.md`
- Test: `tests/test_cli_fit.py` (create)

**Interfaces:** `midas fit -p PORTFOLIO -s STRATEGIES --start DATE [--as-of DATE] [--rollback N]`. Requires a `policy:` block in the strategies file (ClickException otherwise). Loads the file exactly as `optimize` does (entries/exits/constraints split from phase 1), warm-starts from the current members sidecar when present, writes artifacts, prints members count, per-restart bests, and the sidecar path. `--rollback N` skips fitting entirely and restores. `--as-of` defaults to today.

- [ ] **Step 1: failing tests** (reuse `fake_prices` pattern from `tests/test_cli_optimize.py` — copy the fixture into the new file):

```python
STRATS_WITH_POLICY = """\
softmax_temperature: 0.4
min_cash_pct: 0.05
strategies:
  - name: MeanReversion
    weight: 1.0
    params: {window: 20, threshold: 0.05}
  - name: StopLoss
    params: {loss_threshold: 0.08}
policy:
  objective: gross
  budget: 4
  restarts: 2
  deployment: best
"""


def test_fit_writes_members_sidecar(tmp_path: Path, fake_prices: None) -> None:
    result, strategies = _invoke_fit(tmp_path, "--as-of", "2024-05-01")
    assert result.exit_code == 0, result.output
    sidecar = tmp_path / "strats.members.yaml"
    assert sidecar.exists()
    assert "fitted_as_of" in sidecar.read_text()
    assert (tmp_path / "strats.fits").is_dir()


def test_fit_requires_a_policy_block(tmp_path: Path, fake_prices: None) -> None:
    result, _ = _invoke_fit(tmp_path, "--as-of", "2024-05-01", strategies_text=STRATS_NO_POLICY)
    assert result.exit_code != 0
    assert "policy" in result.output


def test_fit_rollback_restores_previous_members(tmp_path: Path, fake_prices: None) -> None:
    _invoke_fit(tmp_path, "--as-of", "2024-03-01")
    first = (tmp_path / "strats.members.yaml").read_text()
    _invoke_fit(tmp_path, "--as-of", "2024-05-01")
    assert (tmp_path / "strats.members.yaml").read_text() != first
    result, _ = _invoke_fit(tmp_path, "--rollback", "1")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "strats.members.yaml").read_text() == first
```

(Define `_invoke_fit` writing portfolio + `STRATS_WITH_POLICY` (or an override) into `tmp_path` and invoking `["fit", "-p", ..., "-s", ..., "--start", "2023-01-02", *extra]`; `STRATS_NO_POLICY` = the same file minus the policy block.)

- [ ] **Step 2: RED** (no such command).
- [ ] **Step 3: Implement** the command: assemble exactly like `optimize` (loader split, tax via portfolio), `load_policy` (require), `read_members` as incumbent, `fit_as_of`, `write_fit`, summary echo. `--rollback` branch calls `rollback()` and echoes what was restored. Docs: `docs/cli.md` new `## fit` section (options table + three-sentence description referencing the policy block and sidecar; rollback note).
- [ ] **Step 4: GREEN + full suite.**
- [ ] **Step 5: full verify + commit + push** ("midas fit: policy-driven fitting with sidecar artifacts and rollback").

---

## Self-review record

- Spec coverage: Component 2 fully (restarts, warm start restart-0-only via spy test, selection both modes, FitResult fields incl. hashes and restart bests); Deployment artifacts (sidecar, fits history, rollback, `midas fit`, per-restart bests persisted outside the deployed file — Task 6 stores them in both sidecar and history); Definitions (`as_of` exclusive — pinned by `test_as_of_is_exclusive`; seed derivation; mode-aware policy hash; input fingerprint). `--objective` vs `policy.objective` precedence lands in this phase only for `fit` (which takes no `--objective` flag at all — the loud-conflict rule applies to `optimize` when a policy block exists: add that guard in Task 7 step 3 as one `if` in the optimize command + one test line in `test_cli_fit.py`).
- Deviations recorded: seed derives from `as_of` not fold index (coherence-stronger); rollback on `midas fit` until phase 7's `refit` exists. Both to be noted in the spec at phase landing.
- Types: `FitResult` field names identical across Tasks 4–7; `flatten_params` naming matches `_suggest_params`.
